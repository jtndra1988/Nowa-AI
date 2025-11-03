import os
import pickle
import json
from typing import List, Optional # <-- ADDED for features list
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler # <-- ADDED for type hint
from app.ml.data_preprocessor import preprocess_raw_chunk
try:
    from app.core.config import settings
    DATABASE_URL = settings.SQLALCHEMY_DATABASE_URI
except Exception:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@db:5432/postgres")

from app.ml.model import LSTMSignalModel
from app.ml.inference_ensemble import run as run_ensemble # Keep for primary path

# --- ADDED: Consistent artifact loading ---
try:
    from app.ml.artifacts import ckpt_dir, fpath
    USE_ARTIFACTS = True
except Exception:
    USE_ARTIFACTS = False
    CHECKPOINTS_ROOT = Path("/app/checkpoints")
    FILENAMES = {
        "lstm": "lstm_best.pt",
        "lstm_scaler": "lstm_scaler.pkl",
        "features": "features.json",
        "fusion": "fusion_head.pt", # Needed to check existence
        # Add other keys if needed by this script
    }
    def ckpt_dir(symbol: str) -> Path:
        p = CHECKPOINTS_ROOT / symbol.replace("/", "_").upper()
        p.mkdir(parents=True, exist_ok=True)
        return p
    def fpath(symbol: str, key: str) -> Path:
        return ckpt_dir(symbol) / FILENAMES[key]
# --- End artifact loading ---


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN_DEFAULT = 60 # Default sequence length

# --- NEW: Load Scaler ---
def _load_scaler(symbol: str) -> Optional[StandardScaler]:
    scaler_path = fpath(symbol, "lstm_scaler")
    if not scaler_path.exists():
        return None
    try:
        scaler = joblib.load(scaler_path)  # <- use joblib.load
        if not isinstance(scaler, StandardScaler):
            raise TypeError("Loaded object is not a StandardScaler")
        return scaler
    except Exception as e:
        print(f"[!] Error loading LSTM scaler for {symbol} (fallback): {e}")
        return None

# --- NEW: Load LSTM Features ---
def _load_lstm_features(symbol: str) -> Optional[List[str]]:
    features_path = fpath(symbol, "features")
    if not features_path.exists(): return None
    try:
        features_data = json.loads(features_path.read_text())
        lstm_features = features_data.get("lstm_features")
        if lstm_features is None: raise ValueError("lstm_features not found in features.json")
        return lstm_features
    except Exception as e:
        print(f"[!] Error loading LSTM features for {symbol} (fallback): {e}")
        return None


def get_prediction(symbol: str, market_type: str, current_price: float, confidence_cutoff: float = 0.55):
    """
    Hybrid prediction function with fallback, now using consistent artifact loading.
    """
    symbol_upper = symbol.upper() # Use consistent casing
    try:
        # Check for fusion model first using fpath
        fusion_ckpt = fpath(symbol_upper, "fusion")
        lstm_ckpt   = fpath(symbol_upper, "lstm") # Path to LSTM model

        # --- (A) Full ensemble path (calibrated) ---
        if fusion_ckpt.exists():
            print(f"[*] [Path A] Using ensemble inference for {symbol_upper}...")
            # run_ensemble already handles loading its own artifacts correctly
            result = run_ensemble(symbol=symbol_upper, lookback_rows=5000, base_table="futures_market_data")
            decision   = result.get("decision", "HOLD").upper()
            # Confidence cutoff is now handled *inside* run_ensemble's logic
            # confidence = float(result.get("confidence", 0.0))
            # if confidence < confidence_cutoff: decision = "HOLD"
            print(f"[*] [Path A] Ensemble Decision={decision} (Cutoff applied internally)")
            return decision

        # --- (B) LSTM fallback path (Corrected) ---
        print(f"[*] [Path B] Fusion model not found. Using LSTM fallback for {symbol_upper}...")

        # --- Load required artifacts for fallback ---
        scaler = _load_scaler(symbol_upper)
        lstm_features = _load_lstm_features(symbol_upper)
        if not (lstm_features and scaler and fpath(symbol_upper, "lstm").exists()):
          print("[!] Missing one of {lstm, scaler, features} — HOLD.")
          return "HOLD"
        if not lstm_ckpt.exists() or not scaler or not lstm_features:
            missing = []
            if not lstm_ckpt.exists(): missing.append("LSTM model (lstm_best.pt)")
            if not scaler: missing.append("LSTM scaler (lstm_scaler.pkl)")
            if not lstm_features: missing.append("LSTM features (features.json)")
            print(f"[!] [Path B] Missing required artifacts for LSTM fallback: {', '.join(missing)}. HOLD.")
            return "HOLD"

        print(f"[*] [Path B] Loading data and running LSTM inference for {symbol_upper}...")

        # Load data (same as ensemble)
        engine = create_engine(DATABASE_URL) # Use correct DB URL
        sql = text("""
            SELECT timestamp, open, high, low, close, volume, symbol
            FROM futures_market_data
            WHERE symbol = :sym
            ORDER BY timestamp DESC
            LIMIT 5000
        """)
        # Use symbol_upper for query consistency
        df = pd.read_sql(sql, engine, params={"sym": symbol_upper}, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        seq_len = SEQ_LEN_DEFAULT # Consider getting from manifest if variable

        if df.empty or len(df) < seq_len + 20: # Need buffer for preprocessing stability
            print(f"[!] [Path B] Not enough rows ({len(df)}) for LSTM fallback inference. HOLD.")
            return "HOLD"

        # Preprocess data (same function as training and ensemble)
        try:
            # Pass symbol_upper to preprocessor
            df_fe, _ = preprocess_raw_chunk(df.copy(), symbol_upper, engine=engine)
        except Exception as e:
            print(f"[!] [Path B] Preprocessing failed for LSTM fallback: {e}. HOLD.")
            return "HOLD"

        # Check if features exist and enough rows remain
        missing_runtime_feats = [f for f in lstm_features if f not in df_fe.columns]
        if missing_runtime_feats:
            print(f"[!] [Path B] Features missing after preprocessing: {missing_runtime_feats}. HOLD.")
            return "HOLD"
        if len(df_fe) < seq_len:
            print(f"[!] [Path B] Not enough rows ({len(df_fe)}) after preprocessing for sequence. HOLD.")
            return "HOLD"

        # Prepare data using LOADED features list and scaler
        X_lstm_raw = df_fe[lstm_features].values.astype(np.float32)
        # Impute NaNs before scaling
        if np.isnan(X_lstm_raw).any():
             X_lstm_raw = pd.DataFrame(X_lstm_raw, columns=lstm_features).ffill().bfill().values
        if np.isnan(X_lstm_raw).any():
            print(f"[!] [Path B] NaNs remain in LSTM features after imputation. HOLD.")
            return "HOLD"

        X_lstm_scaled = scaler.transform(X_lstm_raw)
        X_seq = np.expand_dims(X_lstm_scaled[-seq_len:], axis=0) # Use correct seq_len

        # Load LSTM Model state
        try:
            checkpoint = torch.load(lstm_ckpt, map_location=DEVICE)
            state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint if isinstance(checkpoint, dict) else None))
            if state is None: raise ValueError("Could not find state_dict in checkpoint")
        except Exception as e:
             print(f"[!] [Path B] Error loading LSTM checkpoint: {e}. HOLD.")
             return "HOLD"

        # Build & run model
        try:
            # Ensure input_size matches the loaded features
            model = LSTMSignalModel(input_size=X_seq.shape[-1], hidden_layer_size=128, num_layers=2, output_size=1)
            model.load_state_dict(state, strict=False)
            model.eval().to(DEVICE)

            with torch.no_grad():
                scaled_pred = model(torch.tensor(X_seq, dtype=torch.float32, device=DEVICE)).item()

            # Inverse transform only the 'close' dimension (assuming it's first)
            # Find the index of 'close' in the loaded features list
            try:
                close_idx = lstm_features.index('close')
            except ValueError:
                 print("[!] [Path B] 'close' feature not found in loaded lstm_features list. Cannot unscale. HOLD.")
                 return "HOLD"

            # Create zero array matching scaler's expected input features
            z = np.zeros((1, len(scaler.mean_)), dtype=np.float32)
            z[0, close_idx] = scaled_pred # Place prediction at the correct index
            unscaled_pred = scaler.inverse_transform(z)[0, close_idx] # Inverse transform and extract

        except Exception as e:
            print(f"[!] [Path B] Error during LSTM model inference or unscaling: {e}. HOLD.")
            return "HOLD"

        # --- Fallback Directional Decision ---
        # Compare unscaled prediction to the *provided* current_price
        print(f"[*] [Path B] LSTM Fallback: Predicted Price={unscaled_pred:.4f}, Current Price={current_price:.4f}")
        if unscaled_pred > current_price * 1.0005: # Add small threshold/buffer
            print("[*] [Path B] Decision: BUY")
            return "BUY"
        elif unscaled_pred < current_price * 0.9995: # Add small threshold/buffer
            print("[*] [Path B] Decision: SELL")
            return "SELL"
        else:
            print("[*] [Path B] Decision: HOLD")
            return "HOLD"

    except Exception as e:
        print(f"[!!!] Unhandled error in get_prediction for {symbol_upper}: {e}")
        import traceback
        traceback.print_exc() # Log full traceback for debugging
        return "HOLD"
