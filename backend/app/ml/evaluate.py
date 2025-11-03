import argparse
from collections import defaultdict
import json
import math
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, # For regression
    accuracy_score, f1_score, log_loss, roc_auc_score # For classification
)
from xgboost import XGBRegressor
from sqlalchemy import create_engine, text

# --- Project Imports ---
try:
    from app.core.config import settings
    DATABASE_URL = settings.SQLALCHEMY_DATABASE_URI
except Exception:
    import os
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@db:5432/postgres")

from app.ml.model import LSTMSignalModel
# Assuming FusionHead is defined here or imported correctly
try:
    from app.ml.fusion_model import FusionHead, make_labels_atr_band # Import necessary components
except ImportError:
    # Define FusionHead locally if not importable (ensure consistency)
    import torch.nn as nn
    class FusionHead(nn.Module):
        def __init__(self, in_dim: int, hidden: int = 32, num_classes: int = 3, p: float = 0.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, num_classes),
            )
        def forward(self, x): return self.net(x)
    # Define make_labels_atr_band locally if not importable
    def _atr_eval(df: pd.DataFrame, length: int = 14) -> pd.Series:
        try:
            import pandas_ta as ta
            return ta.atr(df["high"], df["low"], df["close"], length=length)
        except Exception:
            high = df["high"].astype(float).values
            low = df["low"].astype(float).values
            close = df["close"].astype(float).values
            prev_close = np.roll(close, 1); prev_close[0] = close[0]
            tr1 = high - low; tr2 = np.abs(high - prev_close); tr3 = np.abs(low - prev_close)
            tr = np.maximum.reduce([tr1, tr2, tr3])
            return pd.Series(tr).rolling(window=length, min_periods=length).mean()

    def make_labels_atr_band(df: pd.DataFrame, atr_len: int, k_atr_to_bps: float) -> np.ndarray:
        df = df.copy()
        if 'target' not in df.columns: df['target'] = df['close'].shift(-1) # Ensure target exists
        df = df.dropna(subset=['target', 'close', 'high', 'low'])
        atr = _atr_eval(df, length=atr_len)
        vol = (atr / df["close"]).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.01)
        hold_band_bps = (k_atr_to_bps * vol * 10000.0).astype("float32")
        next_close = df["target"].astype("float32").to_numpy()
        close_now  = df["close"].astype("float32").to_numpy()
        ret_bps = ((next_close - close_now) / (close_now + 1e-9) * 10000.0).astype("float32")
        y = np.where(ret_bps >  hold_band_bps, 2, np.where(ret_bps < -hold_band_bps, 0, 1)).astype("int64")
        return y


from app.ml.data_preprocessor import preprocess_raw_chunk
from app.ml.artifacts import FILENAMES # Use FILENAMES mapping
from app.ml.calibration import softmax_rows # For calculating probs if needed

# --- Optional Imports ---
try:
    import shap 
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False

try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

# --- Constants ---
DEVICE = torch.device("cpu") # Evaluation typically runs on CPU for consistency/availability
EVAL_START_DATE = "2023-01-01" # Default start date for validation data
EVAL_END_DATE = "2023-12-31"   # Default end date for validation data
MIN_EVAL_ROWS = 100            # Minimum rows required after preprocessing for evaluation
SEQ_LEN_DEFAULT = 60           # Default LSTM sequence length

# --- Data Loading ---
def load_validation_data(symbol: str, table: str, start: str, end: str) -> pd.DataFrame:
    """Loads validation data from the database."""
    engine = create_engine(DATABASE_URL)
    print(f"[*] Loading validation data for {symbol} ({table}) from {start} to {end}...")
    # Adjust query based on table schema if necessary (e.g., column names)
    cols = "timestamp, open, high, low, close, volume"
    symbol_param = symbol # Assuming table expects full symbol like BTC/USDT

    q = text(f"""
        SELECT {cols}, symbol
        FROM {table}
        WHERE symbol = :symbol
        AND timestamp BETWEEN :start AND :end
        ORDER BY timestamp ASC
    """)
    df = pd.read_sql(q, engine, params={"symbol": symbol_param, "start": start, "end": end}, parse_dates=["timestamp"])
    df = df.dropna(subset=['timestamp', 'close'])
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Ensure OHLCV exist, backfill if needed
    for col in ['open', 'high', 'low']:
        if col not in df.columns: df[col] = df['close']
    if 'volume' not in df.columns: df['volume'] = 0.0

    print(f"[+] Loaded {len(df)} validation rows.")
    return df

# --- Model Loading ---
def load_artifacts_for_symbol(model_root: Path, symbol_subpath: str) -> Dict[str, Any]:
    """Loads all necessary artifacts for a specific symbol/type."""
    artifacts = {}
    symbol_dir = model_root / symbol_subpath
    symbol_name_for_artifact_lookup = symbol_subpath.split('/')[-1] # e.g., BTC_USDT

    print(f"[*] Loading artifacts from: {symbol_dir}")

    # Helper to load a single artifact
    def _load_artifact(key: str, loader_func, *args):
        fname = FILENAMES.get(key)
        if not fname: return None # Key not defined
        fpath = symbol_dir / fname
        if not fpath.exists():
             print(f"[!] Artifact not found: {fpath}")
             return None
        try:
            return loader_func(str(fpath), *args)
        except Exception as e:
            print(f"[!] Error loading artifact {fpath}: {e}")
            return None

    # Load LSTM components
    artifacts['lstm_scaler'] = _load_artifact('lstm_scaler', lambda p: joblib.load(p))
    features_data = _load_artifact('features', lambda p: json.loads(Path(p).read_text()))
    artifacts['lstm_features'] = features_data.get('lstm_features') if features_data else None
    input_dim = len(artifacts['lstm_features']) if artifacts['lstm_features'] else None

    if input_dim:
        def load_lstm_state(p: str, input_dim: int):
            ckpt = torch.load(p, map_location=DEVICE)
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt if isinstance(ckpt, dict) else None))
            if state is None: raise ValueError("Could not find state_dict")
            model = LSTMSignalModel(input_size=input_dim, hidden_layer_size=128, num_layers=2, output_size=1)
            model.load_state_dict(state, strict=False)
            return model.to(DEVICE).eval()
        artifacts['lstm_model'] = _load_artifact('lstm', load_lstm_state, input_dim)
    else:
        artifacts['lstm_model'] = None
        if artifacts['lstm_scaler']: # Only warn if scaler was found but features were not
             print("[!] Cannot load LSTM model: feature list is missing.")


    # Load XGBoost components
    xgb_features_data = _load_artifact('xgb_features', lambda p: json.loads(Path(p).read_text()))
    artifacts['xgb_features'] = xgb_features_data.get('xgb_features', xgb_features_data.get('features')) if xgb_features_data else None

    def load_xgb_model(p: str):
        model = XGBRegressor()
        model.load_model(p)
        return model
    artifacts['xgb_model'] = _load_artifact('xgb', load_xgb_model)

    # Load Fusion components
    def load_fusion_head(p: str):
         ck = torch.load(p, map_location=DEVICE)
         in_dim = ck.get("in_dim")
         if not in_dim: raise ValueError("in_dim missing")
         head = FusionHead(in_dim=in_dim) # Assumes FusionHead class is available
         head.load_state_dict(ck["state_dict"])
         return head.to(DEVICE).eval(), in_dim
    fusion_tuple = _load_artifact('fusion', load_fusion_head)
    artifacts['fusion_model'] = fusion_tuple[0] if fusion_tuple else None
    artifacts['fusion_in_dim'] = fusion_tuple[1] if fusion_tuple else None

    # Load decision params from manifest (for ATR band calc)
    manifest = _load_artifact('manifest', lambda p: json.loads(Path(p).read_text()))
    artifacts['decision_params'] = manifest.get('decision_params', {}) if manifest else {}

    print(f"[*] Artifacts loaded. LSTM: {'OK' if artifacts.get('lstm_model') else 'FAIL'}, "
          f"XGB: {'OK' if artifacts.get('xgb_model') else 'FAIL'}, "
          f"Fusion: {'OK' if artifacts.get('fusion_model') else 'FAIL'}")

    return artifacts


# --- Inference Functions ---
def run_lstm_inference(model: LSTMSignalModel, scaler: StandardScaler, features: List[str], df_processed: pd.DataFrame, seq_len: int) -> np.ndarray:
    """Runs LSTM inference on processed data."""
    if not all(f in df_processed.columns for f in features):
         print("[!] LSTM Inference: Missing features in processed data.")
         return np.full(len(df_processed), np.nan) # Return NaNs if features missing

    X_raw = df_processed[features].values.astype(np.float32)
    # Impute NaNs consistently
    if np.isnan(X_raw).any():
        X_raw = pd.DataFrame(X_raw, columns=features).ffill().bfill().values
    if np.isnan(X_raw).any(): # Check again after imputation
        print("[!] LSTM Inference: NaNs remain after imputation.")
        return np.full(len(df_processed), np.nan)

    X_scaled = scaler.transform(X_raw)
    preds_scaled = np.full(len(df_processed), np.nan, dtype=np.float32)

    # Batch inference for efficiency
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
    # Create sequences within the dataloader/inference loop if memory is a concern,
    # or pre-build sequences if feasible. Pre-building is simpler here.
    Xs = []
    valid_indices = []
    for i in range(seq_len, len(X_scaled)):
        Xs.append(X_scaled[i-seq_len:i, :])
        valid_indices.append(i) # Track original index

    if not Xs: return preds_scaled # Not enough data for a single sequence

    X_seq_tensor = torch.tensor(np.stack(Xs), dtype=torch.float32).to(DEVICE)
    dataset_seq = torch.utils.data.TensorDataset(X_seq_tensor)
    loader = torch.utils.data.DataLoader(dataset_seq, batch_size=512, shuffle=False)

    model.eval()
    batch_preds = []
    with torch.no_grad():
        for (xb,) in loader:
            pred_batch = model(xb).squeeze(-1).cpu().numpy()
            batch_preds.append(pred_batch)

    if batch_preds:
        all_preds_scaled = np.concatenate(batch_preds)
        # Place predictions back into the original array shape
        if len(valid_indices) == len(all_preds_scaled):
            preds_scaled[valid_indices] = all_preds_scaled
        else:
            print("[!] LSTM Inference: Prediction length mismatch.")

    return preds_scaled # Return SCALED predictions

def run_xgb_inference(model: XGBRegressor, features: List[str], df_processed: pd.DataFrame) -> np.ndarray:
    """Runs XGBoost inference on processed data."""
    if not all(f in df_processed.columns for f in features):
         print("[!] XGB Inference: Missing features in processed data.")
         return np.full(len(df_processed), np.nan)

    X_raw = df_processed[features].values.astype(np.float32)
    # Impute NaNs consistently
    if np.isnan(X_raw).any():
        X_raw = pd.DataFrame(X_raw, columns=features).ffill().bfill().values
    if np.isnan(X_raw).any():
        print("[!] XGB Inference: NaNs remain after imputation.")
        return np.full(len(df_processed), np.nan)

    try:
        preds_unscaled = model.predict(X_raw)
        return preds_unscaled.astype(np.float32)
    except Exception as e:
         print(f"[!] XGB Inference failed: {e}")
         return np.full(len(df_processed), np.nan)

def run_fusion_inference(model: FusionHead, in_dim: int, df_processed: pd.DataFrame, lstm_preds_scaled: np.ndarray, xgb_preds_unscaled: np.ndarray) -> np.ndarray:
    """Runs Fusion head inference."""
    # Ensure necessary columns exist
    required_cols = ['regime', 'final_sentiment']
    if not all(c in df_processed.columns for c in required_cols):
        print(f"[!] Fusion Inference: Missing required columns: {[c for c in required_cols if c not in df_processed.columns]}.")
        return np.full((len(df_processed), 3), np.nan) # Return NaNs with 3 columns for logits

    ru = (df_processed["regime"] == 1).astype(np.float32).to_numpy()
    rd = (df_processed["regime"] == 2).astype(np.float32).to_numpy()
    fs = df_processed.get("final_sentiment", pd.Series(0.0, index=df_processed.index)).astype(np.float32).to_numpy()

    # Create input vector - WATCH ORDER AND SCALING!
    # Assuming [lstm_scaled, xgb_unscaled, regime_up, regime_down, sentiment]
    X_fuse_list = [lstm_preds_scaled, xgb_preds_unscaled, ru, rd, fs]

    # Check for consistent lengths
    if not all(len(arr) == len(df_processed) for arr in X_fuse_list):
        print("[!] Fusion Inference: Length mismatch in input arrays.")
        return np.full((len(df_processed), 3), np.nan)

    X_fuse = np.stack(X_fuse_list, axis=1).astype(np.float32)

    # Check input dimension matches model
    if X_fuse.shape[1] != in_dim:
         print(f"[!] Fusion Inference: Input dimension mismatch. Expected {in_dim}, Got {X_fuse.shape[1]}.")
         return np.full((len(df_processed), 3), np.nan)

    # Handle NaNs (e.g., from failed base model predictions) before passing to model
    nan_mask = np.isnan(X_fuse).any(axis=1)
    if nan_mask.any():
        print(f"[*] Fusion Inference: Imputing NaNs for {nan_mask.sum()} rows.")
        # Simple mean imputation here; consider more robust methods
        col_means = np.nanmean(X_fuse, axis=0)
        inds = np.where(np.isnan(X_fuse))
        X_fuse[inds] = np.take(col_means, inds[1])
        if np.isnan(X_fuse).any(): # Check again
            print("[!] Fusion Inference: NaNs remain after imputation.")
            return np.full((len(df_processed), 3), np.nan)


    # Batch inference
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_fuse, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False)
    model.eval()
    all_logits = []
    with torch.no_grad():
        for (xb,) in loader:
            logits_batch = model(xb.to(DEVICE)).cpu().numpy()
            all_logits.append(logits_batch)

    if not all_logits: return np.full((len(df_processed), 3), np.nan)

    return np.concatenate(all_logits, axis=0) # Return raw logits


# --- Metrics Calculation ---
def calculate_metrics(df_eval: pd.DataFrame, artifacts: Dict[str, Any], symbol: str, model_type: str) -> Dict[str, float]:
    """Calculates metrics for a specific model type on the evaluated data."""
    metrics = {}
    target_col = 'target' # Next close price
    labels_col = 'labels_atr' # ATR band labels

    # Ensure target column exists
    if target_col not in df_eval.columns: return {"error": 1.0}

    # Regression Metrics (LSTM, XGB)
    if model_type in ['lstm', 'xgb']:
        pred_col = f'{model_type}_pred_unscaled'
        if pred_col not in df_eval.columns: return {f"{model_type}_error": 1.0}

        # Drop NaNs before calculating metrics
        valid_idx = df_eval[[target_col, pred_col]].dropna().index
        if len(valid_idx) < 10: return {f"{model_type}_error": 1.0, f"{model_type}_samples": float(len(valid_idx))}

        y_true = df_eval.loc[valid_idx, target_col].values
        y_pred = df_eval.loc[valid_idx, pred_col].values

        try: metrics[f'{model_type}_val_mse'] = float(mean_squared_error(y_true, y_pred))
        except: metrics[f'{model_type}_val_mse'] = float('inf')
        try: metrics[f'{model_type}_val_r2'] = float(r2_score(y_true, y_pred))
        except: metrics[f'{model_type}_val_r2'] = 0.0

        # Directional Accuracy
        try:
            # Need previous close for comparison
            close_col = 'close'
            if close_col in df_eval.columns:
                 prev_close = df_eval.loc[valid_idx, close_col].values
                 true_dir = (y_true - prev_close) > 0
                 pred_dir = (y_pred - prev_close) > 0
                 metrics[f'{model_type}_val_dir_acc'] = float(np.mean(true_dir == pred_dir))
            else:
                 metrics[f'{model_type}_val_dir_acc'] = 0.0
        except:
            metrics[f'{model_type}_val_dir_acc'] = 0.0

    # Classification Metrics (Fusion)
    elif model_type == 'fusion':
        logits_col = 'fusion_logits'
        if logits_col not in df_eval.columns or labels_col not in df_eval.columns:
            return {f"{model_type}_error": 1.0}

        # Drop NaNs in logits or labels
        valid_idx = df_eval[[logits_col, labels_col]].dropna().index
        if len(valid_idx) < 10: return {f"{model_type}_error": 1.0, f"{model_type}_samples": float(len(valid_idx))}

        logits = np.stack(df_eval.loc[valid_idx, logits_col].values) # Stack list of arrays
        y_true = df_eval.loc[valid_idx, labels_col].values.astype(int)
        y_pred = np.argmax(logits, axis=1)
        y_prob = softmax_rows(logits) # Calculate probabilities from logits

        try: metrics['fusion_val_accuracy'] = float(accuracy_score(y_true, y_pred))
        except: metrics['fusion_val_accuracy'] = 0.0
        try: metrics['fusion_val_f1_macro'] = float(f1_score(y_true, y_pred, average='macro'))
        except: metrics['fusion_val_f1_macro'] = 0.0
        try: metrics['fusion_val_logloss'] = float(log_loss(y_true, y_prob, labels=[0, 1, 2]))
        except: metrics['fusion_val_logloss'] = float('inf')
        # AUC requires probabilities for each class vs rest (OvR)
        try:
            auc_ovr = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro', labels=[0, 1, 2])
            metrics['fusion_val_auc_ovr'] = float(auc_ovr)
        except Exception as e:
             # AUC can fail if only one class is present in y_true
             # print(f"[*] AUC calculation skipped/failed: {e}")
             metrics['fusion_val_auc_ovr'] = 0.0

    metrics[f"{model_type}_samples"] = float(len(valid_idx))
    return metrics


# --- Main Evaluation Function ---
def evaluate_model_dir(model_root: str, eval_start: str = EVAL_START_DATE, eval_end: str = EVAL_END_DATE) -> Dict[str, Any]:
    """
    Loads models and artifacts from a structured directory (e.g., a candidate or current dir),
    evaluates them on real validation data, and returns aggregated metrics.

    Expected structure:
    model_root/
        spot/
            BTC_USDT/
                lstm_best.pt
                lstm_scaler.pkl
                features.json
                ... (xgb, fusion artifacts) ...
            ETH_USDT/
                ...
        futures/
            BTC_USDT/
                ...
            ETH_USDT/
                ...
        metrics.json (optional, for comparison)
    """
    root = Path(model_root).resolve()
    print(f"\n--- Starting Evaluation for Directory: {root} ---")
    print(f"Validation Period: {eval_start} to {eval_end}")

    all_symbols_metrics = {}
    symbols_found = []

    # Iterate through potential subdirectories (spot, futures)
    for model_type_dir in ["spot", "futures"]:
        type_path = root / model_type_dir
        if not type_path.is_dir():
            print(f"[*] Skipping directory (not found): {type_path}")
            continue

        # Find symbol directories within spot/ or futures/
        for symbol_dir in type_path.iterdir():
            if not symbol_dir.is_dir(): continue

            symbol_name = symbol_dir.name.replace("_", "/") # Convert BTC_USDT back to BTC/USDT
            db_table = "market_data" if model_type_dir == "spot" else "futures_market_data"
            symbols_found.append(symbol_name)
            print(f"\n--- Evaluating Symbol: {symbol_name} ({model_type_dir}) ---")

            # 1. Load Artifacts
            artifacts = load_artifacts_for_symbol(root, f"{model_type_dir}/{symbol_dir.name}")
            if not artifacts.get('lstm_model') and not artifacts.get('xgb_model'):
                 print("[!] No base models loaded. Skipping evaluation for this symbol.")
                 all_symbols_metrics[f"{symbol_name}_{model_type_dir}"] = {"error": 1.0, "reason": "No models found"}
                 continue

            # 2. Load and Preprocess Validation Data
            try:
                df_val_raw = load_validation_data(symbol_name, db_table, eval_start, eval_end)
                if df_val_raw.empty: raise ValueError("No validation data loaded.")
                # Use the *exact same* preprocessing
                df_val_processed, _ = preprocess_raw_chunk(df_val_raw.copy(), symbol_name, include_mtf=True)
                df_val_processed = df_val_processed.dropna(subset=['close','high','low']) # Need OHLC for ATR labels
                if len(df_val_processed) < MIN_EVAL_ROWS:
                     raise ValueError(f"Only {len(df_val_processed)} rows remain after preprocessing (min {MIN_EVAL_ROWS}).")
                # Define target (next close) for regression evaluation
                df_val_processed['target'] = df_val_processed['close'].shift(-1)
                # Define labels for classification evaluation
                atr_len = artifacts.get('decision_params', {}).get('atr_length', 14)
                k_atr = artifacts.get('decision_params', {}).get('k_atr_to_bps', 2.0)
                df_val_processed['labels_atr'] = make_labels_atr_band(df_val_processed, atr_len, k_atr)

            except Exception as e:
                print(f"[!] Error loading/preprocessing validation data for {symbol_name}: {e}")
                all_symbols_metrics[f"{symbol_name}_{model_type_dir}"] = {"error": 1.0, "reason": f"Data processing error: {e}"}
                continue

            symbol_metrics = {}

            # 3. Run Inference & Store Predictions
            seq_len = SEQ_LEN_DEFAULT # Consider loading from manifest if variable

            # --- LSTM ---
            if artifacts.get('lstm_model') and artifacts.get('lstm_scaler') and artifacts.get('lstm_features'):
                 print("[*] Running LSTM inference...")
                 lstm_preds_scaled = run_lstm_inference(
                     artifacts['lstm_model'], artifacts['lstm_scaler'], artifacts['lstm_features'],
                     df_val_processed, seq_len
                 )
                 # Unscale predictions (assuming 'close' is the first feature)
                 try:
                      close_idx = artifacts['lstm_features'].index('close')
                      scaler = artifacts['lstm_scaler']
                      z = np.zeros((len(lstm_preds_scaled), len(scaler.mean_)))
                      z[:, close_idx] = lstm_preds_scaled # Place scaled preds
                      lstm_preds_unscaled = scaler.inverse_transform(z)[:, close_idx]
                      df_val_processed['lstm_pred_unscaled'] = lstm_preds_unscaled
                      symbol_metrics.update(calculate_metrics(df_val_processed, artifacts, symbol_name, 'lstm'))
                 except Exception as e:
                      print(f"[!] Error unscaling/evaluating LSTM predictions: {e}")
                      df_val_processed['lstm_pred_unscaled'] = np.nan
                      symbol_metrics['lstm_error'] = 1.0

            else:
                 print("[!] Skipping LSTM evaluation: Missing model, scaler, or features.")
                 df_val_processed['lstm_pred_unscaled'] = np.nan # Ensure column exists

            # --- XGBoost ---
            if artifacts.get('xgb_model') and artifacts.get('xgb_features'):
                print("[*] Running XGBoost inference...")
                xgb_preds_unscaled = run_xgb_inference(
                    artifacts['xgb_model'], artifacts['xgb_features'], df_val_processed
                )
                df_val_processed['xgb_pred_unscaled'] = xgb_preds_unscaled
                symbol_metrics.update(calculate_metrics(df_val_processed, artifacts, symbol_name, 'xgb'))
            else:
                print("[!] Skipping XGBoost evaluation: Missing model or features.")
                df_val_processed['xgb_pred_unscaled'] = np.nan # Ensure column exists


            # --- Fusion ---
            if artifacts.get('fusion_model') and artifacts.get('fusion_in_dim'):
                print("[*] Running Fusion inference...")
                # Get scaled LSTM preds (might be NaN if LSTM failed)
                lstm_scaled_input = df_val_processed['lstm_pred_unscaled'].copy() # Start with unscaled
                # Re-scale LSTM preds using the loaded scaler for fusion input
                if 'lstm_scaler' in artifacts and 'lstm_features' in artifacts and not lstm_scaled_input.isnull().all():
                     try:
                        scaler = artifacts['lstm_scaler']
                        close_idx = artifacts['lstm_features'].index('close')
                        mean_close = scaler.mean_[close_idx]
                        std_close = scaler.scale_[close_idx]
                        lstm_scaled_input = (lstm_scaled_input - mean_close) / std_close
                     except Exception as e:
                          print(f"[!] Warning: Could not re-scale LSTM preds for Fusion input: {e}")
                          lstm_scaled_input.fill(np.nan) # Mark as invalid if rescaling fails
                else:
                     lstm_scaled_input.fill(np.nan) # Mark as invalid if scaler missing


                fusion_logits = run_fusion_inference(
                    artifacts['fusion_model'], artifacts['fusion_in_dim'], df_val_processed,
                    lstm_scaled_input.values, # Pass potentially NaN array
                    df_val_processed['xgb_pred_unscaled'].values # Pass potentially NaN array
                )
                # Store logits as a list of arrays in the DataFrame column
                df_val_processed['fusion_logits'] = [row for row in fusion_logits]
                symbol_metrics.update(calculate_metrics(df_val_processed, artifacts, symbol_name, 'fusion'))
            else:
                 print("[!] Skipping Fusion evaluation: Missing model.")
                 df_val_processed['fusion_logits'] = None # Indicate missing


            # Store metrics for this symbol/type
            print(f"[+] Metrics for {symbol_name} ({model_type_dir}): {symbol_metrics}")
            all_symbols_metrics[f"{symbol_name}_{model_type_dir}"] = symbol_metrics


    # 4. Aggregate Metrics (Example: Average across all evaluated symbols/types)
    print("\n--- Aggregating Metrics ---")
    aggregated_metrics = {}
    metric_values = defaultdict(list)

    for key, sym_metrics in all_symbols_metrics.items():
        if sym_metrics.get("error") == 1.0: continue # Skip failed evaluations
        for metric_name, value in sym_metrics.items():
             if isinstance(value, (int, float)) and math.isfinite(value):
                  # Use a common prefix like 'agg_' for aggregated versions
                  agg_key = f"agg_{metric_name}"
                  metric_values[agg_key].append(value)

    for agg_key, values in metric_values.items():
        if values:
            aggregated_metrics[agg_key] = float(np.mean(values))

    # Add count of evaluated symbols
    aggregated_metrics["evaluated_symbols_count"] = len([m for m in all_symbols_metrics.values() if m.get("error") != 1.0])

    # --- Optionally Add Primary Symbol Metrics ---
    # If BTC/USDT futures was evaluated, add its key metrics directly for easy comparison
    primary_key = "BTC/USDT_futures"
    if primary_key in all_symbols_metrics:
        primary_metrics = all_symbols_metrics[primary_key]
        for m_name in ['lstm_val_mse', 'xgb_val_mse', 'fusion_val_accuracy', 'fusion_val_f1_macro', 'fusion_val_logloss']:
             if m_name in primary_metrics and math.isfinite(primary_metrics[m_name]):
                  aggregated_metrics[f"primary_{m_name}"] = primary_metrics[m_name]


    print(f"\n--- Final Aggregated Metrics ---")
    print(json.dumps(aggregated_metrics, indent=2))
    print(f"---------------------------------")

    # TODO: Add SHAP calculation if needed, adapting it to loop through symbols/models

    return aggregated_metrics

# --- Command Line Interface (Example) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models in a specified directory.")
    parser.add_argument("model_dir", help="Path to the model directory (e.g., /app/models/candidate-YYYYMMDD-HHMMSS or /app/models/current)")
    parser.add_argument("--start", default=EVAL_START_DATE, help="Validation data start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=EVAL_END_DATE, help="Validation data end date (YYYY-MM-DD)")
    args = parser.parse_args()

    if not Path(args.model_dir).exists():
        print(f"[!!!] Error: Model directory not found: {args.model_dir}")
    else:
        # Run evaluation
        final_metrics = evaluate_model_dir(args.model_dir, args.start, args.end)

        # Save metrics to the evaluated directory
        try:
            metrics_path = Path(args.model_dir) / "metrics.json"
            print(f"[*] Saving evaluation metrics to {metrics_path}")
            with open(metrics_path, "w") as f:
                json.dump(final_metrics, f, indent=2)
        except Exception as e:
            print(f"[!] Error saving metrics.json: {e}")
