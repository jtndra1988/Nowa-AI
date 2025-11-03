# --- NOW WITH OPTUNA HYPERPARAMETER TUNING ---
from __future__ import annotations

import argparse
import os
import sys
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, List, Dict, Any
# --- ADD LOGGING ---
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

# --- Optuna Import ---
try:
    import optuna
    _HAVE_OPTUNA = True
except ImportError:
    _HAVE_OPTUNA = False
    optuna = None  # Placeholder

# --- Project settings ---
try:
    from app.core.config import settings
    DEFAULT_DB_URL = getattr(settings, "SQLALCHEMY_DATABASE_URI", None) or getattr(settings, "DATABASE_URL", None)
except Exception:
    DEFAULT_DB_URL = None

# --- Rich Feature Preprocessor ---
_preprocess_func = None
_preprocess_err = None
try:
    from app.ml.data_preprocessor import preprocess_raw_chunk as _pre  # rich features
    _preprocess_func = _pre
except Exception as e1:
    _preprocess_err = e1
    _preprocess_func = None
    # Use logger instead of print
    # print(f"[!] Warning: Failed to import preprocess_raw_chunk: {e1}")

# --- Canonical Model (remove duplicate; use one source of truth) ---
from app.ml.model import LSTMSignalModel

# --- Artifact Helpers (unified) ---
try:
    from .artifacts import (
        fpath, save_manifest, save_model, save_scaler, FILENAMES,  # provided
        # write_features might not exist in older trees, so import guarded below:
    )
    try:
        from .artifacts import write_features  # preferred unified writer
        _HAVE_WRITE_FEATURES = True
    except Exception:
        _HAVE_WRITE_FEATURES = False
except ImportError:
    # Basic fallbacks if artifacts.py is missing - adjust paths if needed
    # Use logger instead of print
    # print("[!] Warning: Could not import artifact helpers. Using basic fallbacks.")
    import joblib
    _DEFAULT_CHECKPOINTS = Path("./checkpoints")
    FILENAMES = {
        "lstm": "lstm_best.pt",
        "lstm_scaler": "lstm_scaler.pkl",
        "features": "features.json",
        "manifest": "manifest.json",
    }

    def _ensure_dir(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _ckpt_dir(symbol: str, out_dir: Optional[str] = None) -> Path:
        return _ensure_dir(Path(out_dir) if out_dir else (_DEFAULT_CHECKPOINTS / symbol.replace('/', '_').upper()))

    def fpath(symbol: str, key: str, out_dir: Optional[str] = None) -> Path:
        return _ckpt_dir(symbol, out_dir) / FILENAMES.get(key, f"{key}.bin")

    def save_model(model: torch.nn.Module, out_dir: Optional[str], filename: Optional[str] = None):
        out_path = _ensure_dir(Path(out_dir) if out_dir else _DEFAULT_CHECKPOINTS) / (filename or FILENAMES['lstm'])
        torch.save(model.state_dict(), out_path)

    def save_scaler(scaler, out_dir: Optional[str], filename: Optional[str] = None):
        out_path = _ensure_dir(Path(out_dir) if out_dir else _DEFAULT_CHECKPOINTS) / (filename or FILENAMES['lstm_scaler'])
        joblib.dump(scaler, out_path)

    def save_manifest(symbol: str, output_dir: Optional[str] = None, **kwargs):
        out_path = _ckpt_dir(symbol, output_dir) / FILENAMES['manifest']
        kwargs["_timestamp"] = time.time()
        out_path.write_text(json.dumps(kwargs, indent=2))

    # Fallback write_features
    def write_features(out_dir: Optional[str], *, lstm_features=None, xgb_features=None) -> Path:
        folder = _ensure_dir(Path(out_dir) if out_dir else _DEFAULT_CHECKPOINTS)
        p = folder / FILENAMES["features"]
        data: Dict[str, Any] = {}
        if p.exists():
            try:
                data = json.loads(p.read_text())
            except Exception:
                data = {}
        if lstm_features is not None:
            data["lstm_features"] = list(lstm_features)
        if xgb_features is not None:
            data["xgb_features"] = list(xgb_features)
        p.write_text(json.dumps(data, indent=2))
        return p

    _HAVE_WRITE_FEATURES = True

# --- CONFIGURE LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__) # Get logger for this module

# Log import errors discovered earlier
if _preprocess_err:
    logger.warning(f"Failed to import preprocess_raw_chunk during initial setup: {_preprocess_err}")
if not _HAVE_WRITE_FEATURES:
     logger.warning("Could not import artifact helpers during initial setup.")


# =========================================================
# Config
# =========================================================
@dataclass
class TrainConfig:
    symbol: str
    table: str
    prediction_target: Literal["next_close", "realized_volatility"] = 'next_close'
    output_dir: Optional[str] = None

    # Optuna
    tune_hyperparameters: bool = False
    optuna_trials: int = 20

    # data/sequence
    seq_len: int = 60
    batch_size: int = 128
    val_ratio: float = 0.2

    # model/optim
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 25
    hidden_size: int = 128
    lstm_layers: int = 2
    dropout: float = 0.1  # (not used by canonical model, retained for Optuna compatibility)
    early_stop_patience: int = 5

    # realized volatility
    volatility_window: int = 20
    volatility_ann_factor: float = 365.25 * 24

    # device/db
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    db_url: Optional[str] = None


# Optuna trial training length
N_OPTUNA_EPOCHS = 5
OPTUNA_EARLY_STOPPING_PATIENCE = 2


# =========================================================
# Dataset
# =========================================================
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================================================
# Helpers
# =========================================================
def build_sequences(arr: np.ndarray, targets: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    if targets.size == 0 or len(arr) < seq_len or len(targets) < seq_len:
        logger.warning(f"Insufficient data length ({len(arr)} features, {len(targets)} targets) for seq_len={seq_len}. Cannot build sequences.")
        return np.empty((0, seq_len, arr.shape[1])), np.empty((0, 1))
    loop_range = min(len(arr), len(targets)) - seq_len
    if loop_range <= 0: # Check if range is valid
         logger.warning(f"Data length ({len(arr)}) not greater than seq_len ({seq_len}). Cannot build sequences.")
         return np.empty((0, seq_len, arr.shape[1])), np.empty((0, 1))

    for i in range(loop_range):
        Xs.append(arr[i:i + seq_len, :])
        ys.append(targets[i + seq_len])
    if not Xs:
        logger.warning("Sequence building resulted in empty list.")
        return np.empty((0, seq_len, arr.shape[1])), np.empty((0, 1))
    return np.stack(Xs, axis=0), np.asarray(ys)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, last_close_seq: np.ndarray) -> float:
    if len(y_true) == 0 or len(y_pred) == 0 or len(last_close_seq) == 0:
        return 0.0
    n = min(len(y_true), len(y_pred), len(last_close_seq))
    if n == 0: return 0.0 # Prevent empty slice warning
    up_true = (y_true[:n] - last_close_seq[:n]) > 0
    up_pred = (y_pred[:n] - last_close_seq[:n]) > 0
    return float(np.mean(up_true == up_pred))


def _engine(db_url: Optional[str]):
    url = db_url or DEFAULT_DB_URL or os.environ.get("DATABASE_URL") or os.environ.get("SQLALCHEMY_DATABASE_URI")
    if not url:
        raise RuntimeError("No database URL found. Provide --db-url or set DATABASE_URL / SQLALCHEMY_DATABASE_URI.")
    return create_engine(url)


def load_df(engine, table: str, symbol: str, min_rows: int = 1000) -> pd.DataFrame:
    logger.info(f"Loading data for {symbol} from table '{table}'...")
    if table == "futures_market_data":
        cols = "timestamp, open, high, low, close, volume"; symbol_param = symbol
    elif table == "market_data": # Assuming spot data
        cols = "timestamp, open, high, low, close, volume"; symbol_param = symbol # Use full symbol if table expects it
    elif table == "options_chain": # Minimal columns for options
        cols = "timestamp, last_price AS close, volume"; symbol_param = symbol.split('/')[0] # Use base symbol
    else:
        raise ValueError(f"Unknown table name: {table}")

    # Increased limit slightly, adjust as needed based on memory/performance
    q = text(f"SELECT {cols} FROM {table} WHERE symbol = :symbol ORDER BY timestamp ASC LIMIT 300000")
    try:
        df = pd.read_sql(q, engine, params={"symbol": symbol_param}, parse_dates=["timestamp"])
    except Exception as e:
         logger.error(f"Database query failed for {symbol} on {table}: {e}", exc_info=True)
         raise RuntimeError(f"Database query failed for {symbol} on {table}.") from e

    if df.empty:
        raise RuntimeError(f"No data found in {table} for {symbol_param}.")
    if 'timestamp' not in df.columns or 'close' not in df.columns:
        raise RuntimeError(f"Required columns (timestamp, close/last_price) not found in {table}.")

    # Clean and sort
    df = df.dropna(subset=['timestamp', 'close']).sort_values("timestamp").reset_index(drop=True)

    # Ensure OHLCV exist, fill reasonably if needed
    for col in ['open', 'high', 'low']:
        if col not in df.columns:
            logger.warning(f"Column '{col}' missing in {table} for {symbol}, filling with 'close'.")
            df[col] = df['close']
    if 'volume' not in df.columns:
        logger.warning(f"Column 'volume' missing in {table} for {symbol}, filling with 0.0.")
        df['volume'] = 0.0

    logger.info(f"Loaded {len(df)} raw rows for {symbol} from {table}.")
    if len(df) < min_rows:
        logger.warning(f"Loaded data ({len(df)}) < minimum required ({min_rows}). Model quality may be affected.")
    return df


def run_preprocess_or_fallback(df: pd.DataFrame, symbol: str, engine=None) -> Tuple[pd.DataFrame, List[str]]:
    """Try rich preprocessor; fallback to simple FE if unavailable."""
    if _preprocess_func is not None:
        try:
            logger.info("Using rich feature preprocessor (preprocess_raw_chunk)...")
            proc_df, feat_cols = _preprocess_func(df, symbol, engine=engine, include_mtf=True) # Assuming include_mtf is desired
            # --- Robust column check after preprocessing ---
            required_after = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] # Base columns needed
            if not all(col in proc_df.columns for col in required_after):
                 missing = [col for col in required_after if col not in proc_df.columns]
                 logger.error(f"Rich preprocessing output missing required columns: {missing}. Falling back.")
                 raise ValueError("Rich preprocessing output incomplete.")
            # --- End robust check ---
            return proc_df, feat_cols
        except Exception as e:
            logger.warning(f"Rich preprocessing failed: {e}. Falling back.", exc_info=False) # Log traceback optionally: exc_info=True
            if _preprocess_err:
                logger.warning(f"    Initial import errors: {_preprocess_err}")
    else:
        logger.warning("Rich preprocessor import failed. Using fallback FE.")
        if _preprocess_err:
            logger.warning(f"    Initial import errors: {_preprocess_err}")

    # Fallback FE
    logger.info("Applying simple fallback feature engineering...")
    out = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    out["log_ret"] = np.log(out["close"].clip(lower=1e-9) / out["close"].shift(1).clip(lower=1e-9)) # Add clip for stability
    out["vol_20"] = out["log_ret"].rolling(20).std()
    out["ma_10"] = out["close"].rolling(10).mean()
    out["ma_50"] = out["close"].rolling(50).mean()
    out = out.fillna(method='ffill').fillna(method='bfill').fillna(0.0) # Standard fill
    feat_cols = ["log_ret", "vol_20", "ma_10", "ma_50", "close"] # Ensure 'close' is included for target calc
    return out, feat_cols


def attach_target(
    df_feat: pd.DataFrame,
    prediction_target: Literal["next_close", "realized_volatility"],
    volatility_window: int,
    volatility_ann_factor: float
) -> pd.DataFrame:
    """Add 'target' column for next_close or realized_volatility."""
    out = df_feat.copy()
    if "close" not in out.columns:
        raise ValueError("Feature DataFrame needs 'close' column to attach target.")

    if prediction_target == "next_close":
        out["target"] = out["close"].shift(-1)
    elif prediction_target == "realized_volatility":
        logger.info(f"Calculating realized volatility (window={volatility_window})...")
        log_ret = np.log(out["close"].clip(lower=1e-9) / out["close"].shift(1).clip(lower=1e-9))
        ann_sqrt = np.sqrt(volatility_ann_factor) if volatility_ann_factor > 0 else 1.0
        # Calculate rolling std, apply annualization factor, shift forward for prediction target
        realized_vol = log_ret.rolling(window=volatility_window).std() * ann_sqrt
        out["target"] = realized_vol.shift(-1).clip(lower=1e-9) # Predict next period's realized vol, clip low
        # Add realized_vol itself as a feature if desired (optional)
        # out["realized_vol"] = realized_vol
    else:
        raise ValueError(f"Unknown prediction_target: {prediction_target}")

    # Drop rows where target is NaN (typically the last row due to shift)
    original_len = len(out)
    out = out.dropna(subset=["target"])
    logger.info(f"Attached target '{prediction_target}'. Dropped {original_len - len(out)} rows with NaN target.")
    return out.reset_index(drop=True)


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, scaler_amp):
    model.train()
    total_loss = 0.0
    n = 0
    nan_batches = 0
    for i, (xb, yb) in enumerate(dataloader):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            pred = model(xb).squeeze(-1) # Ensure output matches target shape
            # Check target shape - should be [batch_size]
            loss = loss_fn(pred, yb.squeeze(-1))
        if torch.isnan(loss):
            nan_batches += 1
            # logger.warning(f"NaN loss encountered in training batch {i+1}. Skipping batch.")
            continue # Skip this batch
        scaler_amp.scale(loss).backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        total_loss += loss.item() * len(xb)
        n += len(xb)
    if nan_batches > 0:
         logger.warning(f"Encountered and skipped {nan_batches}/{i+1} batches with NaN loss during training epoch.")
    return total_loss / n if n > 0 else float("nan")


def validate_one_epoch(model, dataloader, loss_fn, device, cfg: TrainConfig, last_ref_val: np.ndarray):
    model.eval()
    total_loss = 0.0
    n = 0
    preds, targs = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(xb).squeeze(-1)
                try:
                    loss = loss_fn(pred, yb.squeeze(-1))
                except Exception as e:
                     logger.warning(f"Error calculating validation loss: {e}. Pred shape: {pred.shape}, Target shape: {yb.squeeze(-1).shape}")
                     loss = torch.tensor(float('nan')) # Assign NaN on error

            if not torch.isnan(loss):
                total_loss += loss.item() * len(xb)
                n += len(xb)
                preds.append(pred.detach().cpu().numpy())
                targs.append(yb.squeeze(-1).detach().cpu().numpy()) # Store squeezed target
            # else: logger.warning("NaN loss encountered in validation batch.")
    val_loss = total_loss / n if n > 0 else float("inf")
    metrics = {f"val_mse_{cfg.prediction_target}": val_loss}

    # Calculate directional accuracy only if predicting next_close and data is valid
    if preds and cfg.prediction_target == "next_close":
        yp = np.concatenate(preds).reshape(-1)
        yt = np.concatenate(targs).reshape(-1) # Already squeezed
        if len(last_ref_val) == len(yt):
            metrics["val_directional_acc"] = directional_accuracy(yt, yp, last_ref_val)
        else:
             logger.warning(f"Length mismatch for directional accuracy: targets ({len(yt)}), reference ({len(last_ref_val)}). Skipping metric.")
             metrics["val_directional_acc"] = 0.0
    return val_loss, metrics


# =========================================================
# Optuna Objective (uses canonical LSTMSignalModel)
# =========================================================
def objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, cfg: TrainConfig, last_ref_val: np.ndarray) -> float:
    # Hyperparameters to tune
    suggested_lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    suggested_hidden = trial.suggest_int("hidden_size", 32, 256, step=32)
    suggested_layers = trial.suggest_int("lstm_layers", 1, 3)
    # dropout is not used by LSTMSignalModel, but keep suggestion if other models might use it
    _ = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    # Optional: Tune weight decay
    suggested_wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)


    logger.info(f"Optuna Trial {trial.number}: lr={suggested_lr:.6f}, hidden={suggested_hidden}, layers={suggested_layers}, wd={suggested_wd:.6f}")

    device = torch.device(cfg.device)
    try:
        model = LSTMSignalModel(
            input_size=X_train.shape[-1],
            hidden_layer_size=suggested_hidden,
            num_layers=suggested_layers,
            output_size=1,
        ).to(device)
    except Exception as model_e:
         logger.error(f"Failed to initialize model in trial {trial.number}: {model_e}")
         return float("inf") # Return high loss if model init fails

    opt = torch.optim.Adam(model.parameters(), lr=suggested_lr, weight_decay=suggested_wd)
    loss_fn = nn.MSELoss()
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Create DataLoaders within the objective function
    try:
        train_ds = TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32)))
        val_ds = TensorDataset(torch.from_numpy(X_val.astype(np.float32)), torch.from_numpy(y_val.astype(np.float32)))
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=(len(train_ds) > cfg.batch_size))
        val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    except Exception as dl_e:
         logger.error(f"Failed to create DataLoaders in trial {trial.number}: {dl_e}")
         return float("inf")

    best_trial_val = float("inf")
    bad = 0
    for ep in range(N_OPTUNA_EPOCHS): # Use fewer epochs for Optuna trials
        try:
            tr_loss = train_one_epoch(model, train_dl, opt, loss_fn, device, scaler_amp)
            if not math.isfinite(tr_loss):
                 logger.warning(f"Trial {trial.number}, Epoch {ep+1}: Non-finite training loss encountered. Pruning.")
                 raise optuna.TrialPruned() # Prune if training diverges

            val_loss, _ = validate_one_epoch(model, val_dl, loss_fn, device, cfg, last_ref_val)
            trial.report(val_loss, ep) # Report intermediate value

            if trial.should_prune():
                 logger.info(f"Trial {trial.number} pruned at epoch {ep+1}.")
                 raise optuna.TrialPruned()

            if math.isfinite(val_loss) and val_loss < best_trial_val:
                best_trial_val = val_loss
                bad = 0
            else:
                bad += 1
                if bad >= OPTUNA_EARLY_STOPPING_PATIENCE:
                    logger.info(f"Trial {trial.number}: Early stopping at epoch {ep+1}.")
                    break
            if not math.isfinite(val_loss):
                 logger.warning(f"Trial {trial.number}, Epoch {ep+1}: Non-finite validation loss ({val_loss}). Returning inf.")
                 return float("inf") # Return high loss if validation fails

        except optuna.TrialPruned:
             raise # Re-raise prune exceptions
        except Exception as epoch_e:
             logger.error(f"Error during Optuna trial epoch {ep+1}: {epoch_e}", exc_info=False)
             return float("inf") # Return high loss on other errors

    logger.info(f"Optuna Trial {trial.number} finished. Best Val MSE: {best_trial_val:.6f}")
    return best_trial_val if math.isfinite(best_trial_val) else float("inf")


# =========================================================
# Main
# =========================================================
def main(cfg: TrainConfig):
    logger.info("--- Starting LSTM Training ---")
    logger.info(f"Symbol: {cfg.symbol}, Table: {cfg.table}, Target: {cfg.prediction_target}, Device: {cfg.device}")
    output_directory = Path(cfg.output_dir) if cfg.output_dir else None
    if output_directory:
        logger.info(f"Output Directory: {output_directory}")
        output_directory.mkdir(parents=True, exist_ok=True)
    else:
        logger.info("Output Directory: Default location based on artifact helpers")

    # --- Optuna Checks ---
    if cfg.tune_hyperparameters:
        if not _HAVE_OPTUNA or optuna is None:
            logger.warning("Optuna requested but not installed/imported correctly. Skipping hyperparameter tuning.")
            cfg.tune_hyperparameters = False
        else:
             logger.info("Optuna hyperparameter tuning enabled.")

    engine = _engine(cfg.db_url)

    # 1) Load data
    try:
        raw_df = load_df(engine, cfg.table, cfg.symbol, min_rows=cfg.seq_len + cfg.volatility_window + 100)
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        return # Exit if data loading fails

    # 2) Feature engineering
    logger.info("Running feature engineering...")
    try:
        df_feat, feat_cols = run_preprocess_or_fallback(raw_df, cfg.symbol, engine=engine)
        logger.info(f"Features generated: {len(feat_cols)}. Columns: {feat_cols[:5]}...") # Log first few features
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}", exc_info=True)
        return # Exit if FE fails

    # 3) Target
    logger.info(f"Defining prediction target: {cfg.prediction_target}")
    try:
        df_feat = attach_target(df_feat, cfg.prediction_target, cfg.volatility_window, cfg.volatility_ann_factor)
        if df_feat.empty:
            logger.error("No data remaining after attaching target (all rows dropped). Cannot train.")
            return # Exit if no data left
        logger.info(f"Target '{cfg.prediction_target}' attached. {len(df_feat)} rows remain.")
    except Exception as e:
        logger.error(f"Failed to define/attach target: {e}", exc_info=True)
        return # Exit if target fails

    # 4) Prepare features & scale
    # Ensure 'close' is always included if needed for directional accuracy or target calc
    if 'close' not in feat_cols:
        logger.error("'close' column is missing from generated features. Cannot proceed.")
        return
    feature_input = list(feat_cols) # Use all columns from FE as features initially
    # Ensure no duplicates and required columns exist
    feature_input = sorted(list(set(feature_input)))
    missing_cols = [c for c in feature_input if c not in df_feat.columns]
    if missing_cols:
        logger.error(f"Missing required features in DataFrame after preprocessing: {missing_cols}")
        return

    logger.info(f"Using {len(feature_input)} features: {feature_input[:10]}...") # Log first 10
    Xraw = df_feat[feature_input].to_numpy(dtype=np.float32)
    yraw = df_feat["target"].to_numpy(dtype=np.float32).reshape(-1, 1)

    # Impute inf/NaN before scaling
    Xraw[np.isinf(Xraw) | np.isneginf(Xraw)] = np.nan
    nan_count_before = np.isnan(Xraw).sum()
    if nan_count_before > 0:
        logger.warning(f"Input features contain {nan_count_before} NaN/inf values before scaling. Imputing with column means...")
        try:
            col_means = np.nanmean(Xraw, axis=0)
            # Handle columns that are all NaN (results in NaN mean)
            col_means[np.isnan(col_means)] = 0.0
            inds = np.where(np.isnan(Xraw))
            Xraw[inds] = np.take(col_means, inds[1])
            nan_count_after = np.isnan(Xraw).sum()
            if nan_count_after > 0:
                 logger.error(f"{nan_count_after} NaNs remain even after imputation. Stopping training.")
                 return # Stop if imputation failed
            logger.info("Imputation complete.")
        except Exception as impute_e:
             logger.error(f"Error during NaN imputation: {impute_e}", exc_info=True)
             return # Stop if imputation fails

    logger.info(f"Scaling {Xraw.shape[1]} features...")
    scaler = StandardScaler()
    n_total = len(Xraw)
    # Ensure fit_end_idx is valid and doesn't cause negative slice if n_total is small
    fit_end_idx = max(1, n_total - int(n_total * cfg.val_ratio) - cfg.seq_len - 1)
    if fit_end_idx <= 0:
         logger.warning(f"Not enough data (total={n_total}) to create a separate fitting set for scaler. Fitting on all available data.")
         fit_end_idx = n_total
    try:
        scaler.fit(Xraw[:fit_end_idx])
        Xscaled = scaler.transform(Xraw)
    except Exception as scale_e:
         logger.error(f"Error fitting/transforming scaler: {scale_e}", exc_info=True)
         return # Stop if scaling fails

    # 5) Build sequences
    logger.info(f"Building sequences (length {cfg.seq_len})...")
    Xseq, yseq = build_sequences(Xscaled, yraw, cfg.seq_len)
    if Xseq.shape[0] == 0:
        logger.error("Failed to build any sequences from the data. Check data length and seq_len.")
        return # Exit if sequence building fails
    logger.info(f"Created {Xseq.shape[0]} sequences.")

    # Reference close for directional accuracy (only for next_close)
    last_ref_seq = np.array([])
    if cfg.prediction_target == 'next_close':
        try:
            close_idx = feature_input.index("close")
            # Calculate indices needed from the original scaled data
            # Target y[i+seq_len] corresponds to features X[i:i+seq_len]
            # The reference close is the last 'close' in the input sequence X[i:i+seq_len]
            # which is Xscaled at index (i + seq_len - 1)
            ref_indices = range(cfg.seq_len - 1, len(Xscaled) - 1) # Indices corresponding to end of each sequence input

            if len(ref_indices) >= Xseq.shape[0]: # Ensure enough reference points
                last_close_seq_scaled = Xscaled[list(ref_indices)[:Xseq.shape[0]], close_idx]
                mu, sd = scaler.mean_[close_idx], scaler.scale_[close_idx]
                # Inverse transform to get actual close prices
                last_ref_seq = (last_close_seq_scaled * sd + mu).astype(np.float32)
                logger.info(f"Generated reference 'close' sequence for directional accuracy ({len(last_ref_seq)} points).")
            else:
                 logger.warning(f"Could not generate enough reference 'close' points ({len(ref_indices)}) for {Xseq.shape[0]} sequences.")
        except Exception as ref_e:
            logger.warning(f"Failed to generate reference 'close' sequence: {ref_e}")
            last_ref_seq = np.array([])

    # 6) Time split
    n = len(Xseq)
    n_val = max(1, int(n * cfg.val_ratio)) if n > 10 else 0 # Require min 10 samples for val
    if n <= n_val:
        logger.warning("Not enough sequences for validation split. Training on all data without validation.")
        X_train, y_train = Xseq, yseq
        X_val, y_val, last_ref_val = np.array([]), np.array([]), np.array([])
        val_dl = None
    else:
        X_train, X_val  = Xseq[:n - n_val], Xseq[n - n_val:]
        y_train, y_val  = yseq[:n - n_val], yseq[n - n_val:]
        # Ensure last_ref_val aligns with y_val
        last_ref_val = last_ref_seq[n - n_val:] if len(last_ref_seq) == n else np.array([])
        if len(last_ref_val) != len(y_val):
            logger.warning(f"Length mismatch for validation reference 'close': targets ({len(y_val)}), ref ({len(last_ref_val)}). Dir. accuracy may be 0.")
            last_ref_val = np.array([]) # Reset if mismatch
        if len(X_val) > 0:
            val_ds = TensorDataset(torch.from_numpy(X_val.astype(np.float32)), torch.from_numpy(y_val.astype(np.float32)))
            val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
        else:
             val_dl = None


    logger.info(f"Train sequences: {len(X_train)}, Validation sequences: {len(X_val)}")
    train_ds = TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32)))
    # Drop last batch only if training set is larger than batch size to avoid dropping all data
    drop_last_train = len(train_ds) > cfg.batch_size
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=drop_last_train)

    # 7) Optuna (optional)
    study = None
    if cfg.tune_hyperparameters and optuna and val_dl: # Ensure val_dl exists for Optuna
        logger.info(f"Starting Optuna hyperparameter search ({cfg.optuna_trials} trials)...")
        # Use a more robust storage path if needed, e.g., within output_directory
        storage_path = f"sqlite:///optuna_study_{cfg.symbol.replace('/','_')}_{cfg.prediction_target}.db"
        try:
            study = optuna.create_study(
                study_name=f"{cfg.symbol.replace('/','_')}_{cfg.prediction_target}",
                storage=storage_path,
                load_if_exists=True, # Load previous results
                direction="minimize", # Minimize validation MSE
                pruner=optuna.pruners.MedianPruner() # Add pruner
            )
            study.optimize(
                lambda trial: objective(trial, X_train, y_train, X_val, y_val, cfg, last_ref_val),
                n_trials=cfg.optuna_trials,
                timeout=3600 # 1 hour timeout for search
            )
            if study.best_trial:
                best_params = study.best_trial.params
                logger.info("Optuna finished successfully.")
                logger.info(f"  Best Value (Val MSE): {study.best_value:.6f}")
                logger.info(f"  Best Parameters: {best_params}")
                # Update cfg with best parameters found
                cfg.lr = best_params["lr"]
                cfg.hidden_size = best_params["hidden_size"]
                cfg.lstm_layers = best_params["lstm_layers"]
                cfg.weight_decay = best_params.get("weight_decay", cfg.weight_decay) # Use get for optional params
                # cfg.dropout = best_params["dropout"] # Not used by model
            else:
                 logger.warning("Optuna finished, but no best trial found (all might have failed/pruned). Using default parameters.")
                 cfg.tune_hyperparameters = False # Revert to default params if tuning failed

        except Exception as optuna_e:
            logger.error(f"Optuna optimization failed: {optuna_e}", exc_info=True)
            cfg.tune_hyperparameters = False # Revert to default params
            study = None
    elif cfg.tune_hyperparameters and not val_dl:
         logger.warning("Optuna requested but no validation set available. Skipping tuning.")
         cfg.tune_hyperparameters = False


    # 8) Final model training
    logger.info("Starting final model training...")
    device = torch.device(cfg.device)
    try:
        model = LSTMSignalModel(
            input_size=X_train.shape[-1], # Use actual shape
            hidden_layer_size=cfg.hidden_size,
            num_layers=cfg.lstm_layers,
            output_size=1, # Predicting a single value
        ).to(device)
    except Exception as model_e:
         logger.error(f"Failed to initialize final model: {model_e}", exc_info=True)
         return # Exit if model init fails

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float('inf')
    best_model_state = None
    bad_epochs = 0
    final_metrics = {f"val_mse_{cfg.prediction_target}": float('inf')} # Use specific target name
    if cfg.prediction_target == 'next_close':
        final_metrics["val_directional_acc"] = 0.0

    for epoch in range(cfg.epochs):
        t0 = time.time()
        try:
            tr_loss = train_one_epoch(model, train_dl, opt, loss_fn, device, scaler_amp)
        except Exception as train_e:
             logger.error(f"Error during training epoch {epoch+1}: {train_e}", exc_info=True)
             break # Stop training if epoch fails critically

        va_loss = float('inf'); current_metrics: Dict[str, Any] = {}
        if val_dl:
            try:
                va_loss, current_metrics = validate_one_epoch(model, val_dl, loss_fn, device, cfg, last_ref_val)
                # Update final_metrics with the latest validation results
                final_metrics.update(current_metrics)
            except Exception as val_e:
                 logger.error(f"Error during validation epoch {epoch+1}: {val_e}", exc_info=True)
                 # Continue training but don't update best model based on this epoch
                 va_loss = float('inf') # Mark validation as failed for this epoch

        # Log epoch results
        log_msg = (f"[{cfg.symbol}|{cfg.table}|{cfg.prediction_target}] "
                   f"Epoch {epoch+1:02d}/{cfg.epochs} | {time.time()-t0:5.1f}s | "
                   f"Train Loss: {tr_loss:.6f}")
        if val_dl and math.isfinite(va_loss):
             log_msg += f" | Val Loss: {va_loss:.6f}"
             if cfg.prediction_target == 'next_close':
                  acc = current_metrics.get('val_directional_acc', 0.0)
                  log_msg += f" | Val Acc: {acc:.3f}"
        elif val_dl:
             log_msg += " | Val Loss: inf/nan"
        logger.info(log_msg)

        # Early stopping logic (only if validation is happening and successful)
        if val_dl and math.isfinite(va_loss):
            if va_loss < best_val_loss - 1e-6: # Add small tolerance
                best_val_loss = va_loss
                try:
                    # Save state dict to CPU memory
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    logger.info(f"Epoch {epoch+1}: New best validation loss: {best_val_loss:.6f}")
                except Exception as state_dict_e:
                     logger.error(f"Error saving best model state dict: {state_dict_e}")
                     best_model_state = None # Invalidate state if saving fails
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.early_stop_patience:
                    logger.info(f"Early stopping triggered after {bad_epochs} epochs with no improvement.")
                    break
        elif not val_dl and epoch == cfg.epochs - 1: # No validation, save last epoch model
             try:
                 best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                 logger.info("Training finished (no validation), saving model from last epoch.")
             except Exception as state_dict_e:
                  logger.error(f"Error saving final model state dict: {state_dict_e}")
                  best_model_state = None


    # --- MODIFIED SAVING BLOCK ---
    # 9) Save best model
    best_path = fpath(cfg.symbol, "lstm", output_directory)
    model_saved = False # Flag to track success
    if best_model_state:
        try:
            logger.info(f"Attempting to save best model state to {best_path}...")
            # Load state dict back into model before saving (ensures correct model is saved)
            model.load_state_dict(best_model_state)
            save_model(model, output_directory, filename=best_path.name)
            logger.info(f"[*] Final best model successfully saved to {best_path}")
            model_saved = True # Mark as saved
        except Exception as e:
            logger.error(f"[!!!] CRITICAL: Error saving final model to {best_path}: {e}", exc_info=True)
            # DO NOT re-raise here yet, let artifact saving try
    else:
        logger.warning("[!!!] No best model state found during training. Cannot save lstm_best.pt.")

    # 10) Save artifacts (scaler, features.json, manifest)
    logger.info("[*] Saving other artifacts...")
    artifacts_saved_successfully = False
    try:
        # scaler
        scaler_fname = FILENAMES.get("lstm_scaler", "lstm_scaler.pkl")
        scaler_path = fpath(cfg.symbol, "lstm_scaler", output_directory)
        logger.info(f"Attempting to save scaler to {scaler_path}...")
        save_scaler(scaler, output_directory, filename=scaler_fname)
        logger.info(f"[*] Scaler saved successfully to {scaler_path}")

        # features.json
        lstm_features = list(feature_input)
        features_path = fpath(cfg.symbol, "features", output_directory)
        logger.info(f"Attempting to save features to {features_path}...")
        if _HAVE_WRITE_FEATURES:
            write_features(output_directory, lstm_features=lstm_features)
        else: # Fallback
             folder = Path(output_directory) if output_directory else best_path.parent
             folder.mkdir(parents=True, exist_ok=True); feat_p = folder / FILENAMES.get("features", "features.json")
             try: data = json.loads(feat_p.read_text()) if feat_p.exists() else {}
             except Exception: data = {}
             data["lstm_features"] = lstm_features
             feat_p.write_text(json.dumps(data, indent=2))
        logger.info(f"[*] Features saved successfully to {features_path}")

        # manifest
        tuned_params = None
        try: tuned_params = (study.best_trial.params if study and hasattr(study, "best_trial") and study.best_trial else None)
        except Exception: tuned_params = None
        manifest_path = fpath(cfg.symbol, "manifest", output_directory)
        logger.info(f"Attempting to save manifest to {manifest_path}...")
        save_manifest(
            symbol=cfg.symbol, output_dir=output_directory,
            lstm_present=model_saved, # Use the flag we set earlier
            lstm_trained_on_table=cfg.table, prediction_target=cfg.prediction_target,
            lstm_features=lstm_features,
            # Ensure scaler attributes exist before accessing
            scaler_mean=(scaler.mean_.tolist() if hasattr(scaler, "mean_") and scaler.mean_ is not None else None),
            scaler_std=(scaler.scale_.tolist() if hasattr(scaler, "scale_") and scaler.scale_ is not None else None),
            metrics=final_metrics, tuned_hyperparameters=tuned_params,
        )
        logger.info(f"[*] Manifest saved successfully to {manifest_path}")
        artifacts_saved_successfully = True # Mark all artifacts as saved

    except Exception as e:
        # Log the error and RE-RAISE it to make the script exit non-zero
        logger.error(f"[!!!] CRITICAL: Error saving artifacts: {e}", exc_info=True)
        raise e # <<< IMPORTANT: Re-raise the exception

    if not model_saved and not artifacts_saved_successfully:
         logger.error("[!!!] CRITICAL: Neither model nor artifacts were saved successfully.")
         # Optionally, raise an error here too if saving anything is mandatory
         raise RuntimeError("Failed to save model and artifacts.")
    elif not model_saved:
         logger.warning("[!!!] Artifacts saved, but best model state was not found or failed to save.")
         # Don't raise error here, manifest indicates lstm_present=False

    logger.info(f"--- Training Finished for {cfg.symbol} on {cfg.table} ({cfg.prediction_target}) ---")
    # --- END MODIFIED SAVING BLOCK ---


# =========================================================
# Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidated LSTM Trainer with Optuna Tuning.")
    # Required
    parser.add_argument("--symbol", type=str, required=True, help="Symbol (e.g., BTC/USDT)")
    parser.add_argument("--table", type=str, required=True, choices=["market_data", "futures_market_data", "options_chain"], help="Database table")

    # Optional / tuning
    parser.add_argument("--tune-hyperparameters", action="store_true", help="Enable Optuna hyperparameter search")
    parser.add_argument("--optuna-trials", type=int, default=20, help="Number of Optuna trials to run")

    # Target & outputs
    parser.add_argument("--prediction-target", type=str, default="next_close", choices=["next_close", "realized_volatility"], help="Target variable")
    parser.add_argument("--output-dir", type=str, default=None, help="Save location (overrides default)")

    # Model / data
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)  # kept for Optuna completeness
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--vol-ann-factor", type=float, default=365.25 * 24)
    parser.add_argument("--weight-decay", type=float, default=0.0) # Added weight decay arg

    # infra
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--db-url", type=str, default=None, help="DB URL (overrides config/env)")

    args = parser.parse_args()

    if args.tune_hyperparameters and not _HAVE_OPTUNA:
        logger.error("[!!!] Error: --tune-hyperparameters requires Optuna to be installed.")
        logger.error("      Install using: pip install optuna")
        sys.exit(1)

    forced_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TrainConfig(
        symbol=args.symbol.upper(),
        table=args.table,
        prediction_target=args.prediction_target,
        output_dir=args.output_dir,
        tune_hyperparameters=args.tune_hyperparameters,
        optuna_trials=args.optuna_trials,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden,
        lstm_layers=args.layers,
        dropout=args.dropout,
        early_stop_patience=args.patience,
        volatility_window=args.vol_window,
        volatility_ann_factor=args.vol_ann_factor,
        weight_decay=args.weight_decay, # Pass weight decay
        device=forced_device,
        db_url=args.db_url,
    )
    main(cfg)