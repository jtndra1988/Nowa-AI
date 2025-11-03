# app/ml/inference_ensemble.py
from __future__ import annotations
import os
import json
import pickle # <-- ADDED for scaler
from typing import Any, Dict, List, Optional # <-- ADDED List, Optional
from app.ml.data_preprocessor import preprocess_raw_chunk
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from typing import Tuple
from sqlalchemy import create_engine, text
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler # <-- ADDED for type hint
from app.ml.artifacts import ckpt_dir, fpath
try:
    from app.core.config import settings
    DATABASE_URL = settings.SQLALCHEMY_DATABASE_URI
except Exception:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@db:5432/postgres")

from app.ml.model import LSTMSignalModel
from app.ml.data_preprocessor import preprocess_raw_chunk
from app.ml.calibration import load_temperature, load_platt, softmax_rows

# --- MODIFIED: Use fpath consistently ---
try:
    from app.ml.artifacts import ckpt_dir, fpath, load_manifest # <-- ADDED fpath
    USE_ARTIFACTS = True
except Exception:
    USE_ARTIFACTS = False
    # --- Fallback definitions ---
    CHECKPOINTS_ROOT = Path("/app/checkpoints")
    FILENAMES = {
        "lstm": "lstm_best.pt",
        "lstm_scaler": "lstm_scaler.pkl",
        "xgb": "xgb_model.json",
        "xgb_features": "xgb_features.json",
        "fusion": "fusion_head.pt",
        "cal_temp": "calibration_temp.json", # Adjusted key
        "cal_platt": "calibration_platt.json",# Adjusted key
        "features": "features.json",
        "manifest": "manifest.json",
    }
    def ckpt_dir(symbol: str) -> Path:
        p = CHECKPOINTS_ROOT / symbol.replace("/", "_").upper() # Use consistent naming
        p.mkdir(parents=True, exist_ok=True)
        return p
    def fpath(symbol: str, key: str) -> Path:
        return ckpt_dir(symbol) / FILENAMES[key]
    def load_manifest(symbol: str) -> Dict:
        manifest_path = fpath(symbol, "manifest")
        if manifest_path.exists():
            try: return json.loads(manifest_path.read_text())
            except: return {}
        return {}
    # --- End Fallbacks ---


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defaults (overridden by manifest if available)
BASE_CONF_CUTOFF_DEFAULT = 0.55
ATR_LENGTH_DEFAULT       = 14
K_ATR_TO_BPS_DEFAULT     = 2.0
SEQ_LEN_DEFAULT          = 60 # Default sequence length

# --- Helper Functions (ATR, Loaders) ---
def _atr(df: pd.DataFrame, length: int = 14) -> float:
    # (Implementation remains the same)
    try:
        import pandas_ta as ta
        # Ensure correct column names if df comes directly from DB
        high_col = "high" if "high" in df.columns else "close"
        low_col = "low" if "low" in df.columns else "close"
        close_col = "close"
        s = ta.atr(df[high_col], df[low_col], df[close_col], length=length)
        atr_val = float(s.iloc[-1])
        return atr_val if np.isfinite(atr_val) else 0.0 # Handle potential NaN/inf
    except Exception:
        # Fallback ATR approximation
        high = df["high"].astype(float).values if "high" in df.columns else df["close"].astype(float).values
        low = df["low"].astype(float).values if "low" in df.columns else df["close"].astype(float).values
        close = df["close"].astype(float).values
        prev_close = np.roll(close, 1); prev_close[0] = close[0]
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.maximum.reduce([tr1, tr2, tr3])
        atr = pd.Series(tr).rolling(window=length, min_periods=length).mean().iloc[-1]
        return float(atr) if np.isfinite(atr) else 0.0 # Handle potential NaN/inf

def _load_lstm(symbol: str, in_dim: int) -> Optional[LSTMSignalModel]:
    model_path = fpath(symbol, "lstm")
    if not model_path.exists(): return None
    try:
        st = torch.load(model_path, map_location=DEVICE)
        # Handle different checkpoint formats
        state = st.get("model_state_dict", st.get("state_dict", st if isinstance(st, dict) else None))
        if state is None: raise ValueError("Could not find state_dict in checkpoint")

        # Infer hidden size and layers if possible, otherwise use defaults
        hidden_size = 128 # Default, consider storing in manifest if varies
        num_layers = 2    # Default

        model = LSTMSignalModel(input_size=in_dim, hidden_layer_size=hidden_size, num_layers=num_layers, output_size=1).to(DEVICE)
        model.load_state_dict(state, strict=False) # Use strict=False for flexibility
        model.eval()
        return model
    except Exception as e:
        print(f"[!] Error loading LSTM for {symbol}: {e}")
        return None

def _load_xgb(symbol: str) -> Tuple[Optional[XGBRegressor], Optional[List[str]]]:
    model_path = fpath(symbol, "xgb")
    features_path = fpath(symbol, "xgb_features")
    if not model_path.exists() or not features_path.exists(): return None, None
    try:
        model = XGBRegressor()
        model.load_model(str(model_path))
        feats_data = json.loads(features_path.read_text())
        # Accommodate different potential keys for the feature list
        feats = feats_data.get("xgb_features", feats_data.get("features"))
        if feats is None: raise ValueError("Feature list key not found in xgb_features.json")
        return model, feats
    except Exception as e:
        print(f"[!] Error loading XGBoost for {symbol}: {e}")
        return None, None

def _load_fusion(symbol: str) -> Tuple[Optional[torch.nn.Module], Optional[int]]:
    model_path = fpath(symbol, "fusion")
    if not model_path.exists(): return None, None
    try:
        ck = torch.load(model_path, map_location=DEVICE)
        in_dim = ck.get("in_dim")
        if in_dim is None: raise ValueError("in_dim not found in fusion checkpoint")

        # Define class locally or import if defined elsewhere
        class FusionHead(torch.nn.Module):
            def __init__(self, in_dim: int, hidden: int = 32, num_classes: int = 3, p: float = 0.0):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, hidden),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p),
                    torch.nn.Linear(hidden, num_classes),
                )
            def forward(self, x): return self.net(x)

        head = FusionHead(in_dim=in_dim).to(DEVICE)
        head.load_state_dict(ck["state_dict"])
        head.eval()
        return head, in_dim
    except Exception as e:
        print(f"[!] Error loading Fusion head for {symbol}: {e}")
        return None, None

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
        print(f"[!] Error loading LSTM scaler for {symbol}: {e}")
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
        print(f"[!] Error loading LSTM features for {symbol}: {e}")
        return None


def _calc_dynamic_cutoff(symbol: str, df: pd.DataFrame, regime: int) -> float:
    # (Implementation remains the same - uses manifest/defaults correctly)
    base_cut = BASE_CONF_CUTOFF_DEFAULT
    atr_len  = ATR_LENGTH_DEFAULT
    k_atr    = K_ATR_TO_BPS_DEFAULT
    if USE_ARTIFACTS:
        m = load_manifest(symbol)
        dec = (m.get("decision_params") or {})
        base_cut = float(dec.get("base_confidence_cutoff", base_cut))
        atr_len  = int(dec.get("atr_length", atr_len))
        k_atr    = float(dec.get("k_atr_to_bps", k_atr))

    atr = _atr(df, length=atr_len)
    close_price = df["close"].iloc[-1]
    if close_price <= 1e-9: close_price = 1e-9 # Avoid division by zero
    vol = float(atr / close_price)
    cut = base_cut + 0.5 * vol  # tunable slope

    if regime == 0:   cut += 0.03
    elif regime == 1: cut -= 0.02
    elif regime == 2: cut -= 0.01

    return float(np.clip(cut, 0.52, 0.75))
def _try_load_temp(symbol: str, regime: int | None):
    base = ckpt_dir(symbol)
    cand = []
    if regime is not None:
        # preferred names (match FILENAMES)
        cand += [base / f"calibration_temp_{regime}.json",
                 base / f"fusion_temp_{regime}.json"]
    # global
    cand += [fpath(symbol, "cal_temp"),
             base / "fusion_temp.json"]
    for p in cand:
        try:
            return load_temperature(str(p))
        except Exception:
            continue
    return 1.0  # default T
def _try_load_platt(symbol: str, regime: int | None):
    base = ckpt_dir(symbol)
    cand = []
    if regime is not None:
        cand += [base / f"calibration_platt_{regime}.json",
                 base / f"fusion_platt_{regime}.json"]
    cand += [fpath(symbol, "cal_platt"),
             base / "fusion_platt.json"]
    for p in cand:
        try:
            return load_platt(str(p))
        except Exception:
            continue
    return None
def _apply_calibration(symbol: str, logits: np.ndarray, regime: int) -> np.ndarray:
    logits = logits.reshape(1, -1)
    T  = _try_load_temp(symbol, regime)
    pm = _try_load_platt(symbol, regime)
    logits_T = logits / max(T, 1e-6)
    probs = pm.predict_proba(logits_T) if pm else softmax_rows(logits_T)
    return probs.reshape(-1)


# --- Main Inference Function ---
def run(symbol: str, lookback_rows: int = 5000, base_table: str = "futures_market_data", df_override: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Runs the full inference pipeline for a given symbol.

    Args:
        symbol (str): The symbol to run inference for (e.g., BTC/USDT).
        lookback_rows (int): How many recent data rows to fetch.
        base_table (str): The database table to query.
        df_override (Optional[pd.DataFrame]): If provided, use this DataFrame instead of querying the DB.

    Returns:
        Dict[str, Any]: A dictionary containing the decision, confidence, and other metadata.
    """
    symbol_upper = symbol.upper() # Use consistent casing internally
    engine = create_engine(DATABASE_URL)

    # 1) Load recent data (or use override)
    if df_override is not None:
         df = df_override.copy()
         # Ensure necessary columns exist if using override
         if not all(c in df.columns for c in ['timestamp', 'open', 'high', 'low', 'close', 'volume']):
             raise ValueError("df_override missing required OHLCV columns")
         df = df.sort_values("timestamp").reset_index(drop=True)
         # Ensure enough data for lookback requirements
         if len(df) < SEQ_LEN_DEFAULT + 1: # Basic check, preprocessing might need more
              raise ValueError(f"df_override has insufficient rows ({len(df)}), need at least {SEQ_LEN_DEFAULT + 1}")

    else:
        # Fetch from DB
        sql = text(f"""
            SELECT timestamp, open, high, low, close, volume, symbol
            FROM {base_table}
            WHERE symbol = :sym
            ORDER BY timestamp DESC
            LIMIT :n
        """)
        df = pd.read_sql(sql, engine, params={"sym": symbol_upper, "n": lookback_rows}, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"No data loaded for {symbol_upper} from {base_table}")

    # 2) Feature engineering
    try:
        # Pass the full symbol for context, even if base table uses different format sometimes
        dfe, feat_cols_all = preprocess_raw_chunk(df.copy(), symbol_upper, engine=engine, include_mtf=True)
    except Exception as e:
         raise RuntimeError(f"Feature engineering failed for {symbol_upper}: {e}")

    # Ensure essential columns exist after preprocessing
    if "close" not in dfe.columns: dfe["close"] = dfe.get("last_price", df["close"]) # Handle potential renaming
    if not all(c in dfe.columns for c in ["timestamp", "open", "high", "low", "close", "volume"]):
         raise RuntimeError(f"Essential OHLCV columns missing after preprocessing for {symbol_upper}")

    # Get regime for the latest data point
    regime = int(dfe.get("regime", pd.Series([0])).iloc[-1]) # Default to 0 (chop)

    # --- 3) LSTM Prediction (Corrected) ---
    lstm_next = np.nan # Default value
    lstm_features = _load_lstm_features(symbol_upper)
    scaler = _load_scaler(symbol_upper)
    lstm = None
    if not (lstm_features and scaler and fpath(symbol_upper, "lstm").exists()):
          print("[!] Missing one of {lstm, scaler, features} — HOLD.")
          return "HOLD"
    if lstm_features and scaler:
        # Check if all required features are present in the dataframe
        missing_lstm_feats = [f for f in lstm_features if f not in dfe.columns]
        if missing_lstm_feats:
            print(f"[!] Missing LSTM features after preprocessing: {missing_lstm_feats}. Skipping LSTM.")
        else:
            # Prepare data using LOADED features list and scaler
            X_lstm_raw = dfe[lstm_features].values.astype(np.float32)
            # Impute NaNs before scaling (using simple ffill/bfill for inference)
            if np.isnan(X_lstm_raw).any():
                 X_lstm_raw = pd.DataFrame(X_lstm_raw, columns=lstm_features).ffill().bfill().values
            if np.isnan(X_lstm_raw).any():
                print(f"[!] NaNs remain in LSTM features after imputation for {symbol_upper}. Skipping LSTM.")
            else:
                X_lstm_scaled = scaler.transform(X_lstm_raw)

                # Ensure enough data for sequence length
                seq_len = SEQ_LEN_DEFAULT # Consider getting from manifest if variable
                if len(X_lstm_scaled) >= seq_len:
                    X_seq = torch.tensor(X_lstm_scaled[-seq_len:], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    lstm = _load_lstm(symbol_upper, in_dim=X_seq.shape[-1])
                    if lstm:
                        try:
                            with torch.no_grad():
                                # Get scaled prediction
                                lstm_pred_scaled = lstm(X_seq).squeeze(-1).detach().cpu().numpy()
                                lstm_next = float(lstm_pred_scaled[0]) # Store scaled prediction for fusion
                                # Note: We don't need to unscale here as fusion uses scaled inputs

                        except Exception as e:
                             print(f"[!] LSTM inference failed for {symbol_upper}: {e}")
                             lstm_next = np.nan # Mark as invalid
                    else:
                         print(f"[!] LSTM model could not be loaded for {symbol_upper}.")
                else:
                     print(f"[!] Not enough rows ({len(X_lstm_scaled)}) for LSTM sequence (need {seq_len}) for {symbol_upper}.")
    else:
        print(f"[!] Skipping LSTM for {symbol_upper}: Missing scaler or features list.")


    # --- 4) XGBoost Prediction (Corrected) ---
    xgb_next = np.nan # Default value
    xgb, xgb_features = _load_xgb(symbol_upper)
    if xgb and xgb_features:
        # Check if all required features are present
        missing_xgb_feats = [f for f in xgb_features if f not in dfe.columns]
        if missing_xgb_feats:
             print(f"[!] Missing XGBoost features after preprocessing: {missing_xgb_feats}. Skipping XGB.")
        else:
            try:
                # Use only the last row for XGB prediction
                X_xgb_raw = dfe[xgb_features].tail(1).values.astype(np.float32)
                 # Simple NaN check/fill for the single row
                if np.isnan(X_xgb_raw).any():
                    print(f"[!] NaNs found in XGB input row for {symbol_upper}. Using fallback.")
                    # Fallback: predict previous close? Or skip.
                    xgb_next = dfe['close'].iloc[-1] # Simple fallback
                else:
                    xgb_pred = xgb.predict(X_xgb_raw)
                    xgb_next = float(xgb_pred[0])
            except Exception as e:
                 print(f"[!] XGBoost inference failed for {symbol_upper}: {e}")
                 xgb_next = np.nan # Mark as invalid
    else:
        print(f"[!] Skipping XGBoost for {symbol_upper}: Missing model or features list.")

    # --- 5) Fusion Head Prediction ---
    head, head_in_dim = _load_fusion(symbol_upper)
    logits = np.array([0.0, 0.0, 0.0]) # Default HOLD logits
    fs = float(dfe.get("final_sentiment", pd.Series([0.0])).iloc[-1]) # Get latest sentiment

    if head and head_in_dim:
         # Build fusion input vector [lstm_scaled_next, xgb_unscaled_next, REGIME_UP, REGIME_DOWN, final_sentiment]
         # **Important**: Ensure the order and scaling matches fusion training!
         # Assuming fusion was trained with SCALED LSTM output and UNSCALED XGB output. Verify this!
         ru = float(1.0 if regime == 1 else 0.0)
         rd = float(1.0 if regime == 2 else 0.0)

         # Handle potential NaNs from base models before creating tensor
         lstm_input_val = 0.0 if np.isnan(lstm_next) else lstm_next
         xgb_input_val = dfe['close'].iloc[-1] if np.isnan(xgb_next) else xgb_next # Fallback to current close

         fuse_vec_list = [lstm_input_val, xgb_input_val, ru, rd, fs]

         # Check if the number of features matches the loaded model's input dimension
         if len(fuse_vec_list) == head_in_dim:
             fuse_vec = np.array(fuse_vec_list, dtype=np.float32).reshape(1, -1)
             try:
                 with torch.no_grad():
                     logits = head(torch.tensor(fuse_vec, dtype=torch.float32, device=DEVICE)).cpu().numpy().reshape(-1)
             except Exception as e:
                  print(f"[!] Fusion head inference failed for {symbol_upper}: {e}")
                  # Keep default logits (HOLD)
         else:
              print(f"[!] Dimension mismatch for Fusion head input: Expected {head_in_dim}, Got {len(fuse_vec_list)}. Using default logits.")
              # Keep default logits (HOLD)
    else:
        print(f"[!] Skipping Fusion head prediction for {symbol_upper}: Model not loaded.")
        # Keep default logits (HOLD)


    # 6) Apply Calibration
    try:
        probs = _apply_calibration(symbol_upper, logits, regime)
    except Exception as e:
        print(f"[!] Calibration failed for {symbol_upper}: {e}. Using raw softmax.")
        probs = softmax_rows(logits.reshape(1, -1)).reshape(-1) # Fallback to simple softmax

    idx = int(np.argmax(probs))
    raw_label = ["SELL", "HOLD", "BUY"][idx]
    confidence = float(probs[idx])

    # 7) Apply Dynamic Cutoff
    try:
        # Use only OHLCV from the *end* of the dataframe for ATR/volatility calc
        ohlcv_df_for_cutoff = dfe[["open", "high", "low", "close", "volume"]].tail(200).copy()
        cutoff = _calc_dynamic_cutoff(symbol_upper, ohlcv_df_for_cutoff, regime)
    except Exception as e:
        print(f"[!] Dynamic cutoff calculation failed for {symbol_upper}: {e}. Using default.")
        cutoff = BASE_CONF_CUTOFF_DEFAULT # Fallback

    # Final Decision Logic
    decision = raw_label if (raw_label != "HOLD" and confidence >= cutoff) else "HOLD"

    # --- 8) Calculate current band_bps for context (remains the same) ---
    atr_len  = ATR_LENGTH_DEFAULT
    k_atr    = K_ATR_TO_BPS_DEFAULT
    if USE_ARTIFACTS:
        m = load_manifest(symbol_upper)
        dec = (m.get("decision_params") or {})
        atr_len = int(dec.get("atr_length", atr_len))
        k_atr   = float(dec.get("k_atr_to_bps", k_atr))

    try:
        # Use same OHLCV df as cutoff calc
        atr_val = _atr(ohlcv_df_for_cutoff, length=atr_len)
        close_price = ohlcv_df_for_cutoff["close"].iloc[-1]
        if close_price <= 1e-9: close_price = 1e-9
        vol = float(atr_val / close_price)
        band_bps_now = float(k_atr * vol * 10000.0)
    except Exception as e:
        print(f"[!] Error calculating band_bps_now: {e}")
        band_bps_now = 0.0


    return {
        "symbol": symbol_upper,
        "timestamp": dfe["timestamp"].iloc[-1].isoformat(), # Add timestamp of data used
        "decision": decision,
        "confidence": confidence,
        "cutoff": cutoff,
        "raw_label": raw_label,
        "probs": {"SELL": float(probs[0]), "HOLD": float(probs[1]), "BUY": float(probs[2])},
        "regime": regime,
        "atr_len": atr_len,
        "band_bps_now": band_bps_now,
        "lstm_pred_scaled": lstm_next if not np.isnan(lstm_next) else None, # Return scaled LSTM pred
        "xgb_pred_unscaled": xgb_next if not np.isnan(xgb_next) else None, # Return unscaled XGB pred
        "sentiment": fs,
    }

# --- Command Line Interface (remains the same) ---
if __name__ == "__main__":
    import argparse
    import json as _json # Avoid conflict with module name
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--lookback", type=int, default=5000)
    p.add_argument("--base-table", default="futures_market_data")
    # New argument to allow passing a dataframe file (e.g., for debugging/testing)
    p.add_argument("--df-path", default=None, help="Path to a Parquet/CSV file to use instead of DB query")
    args = p.parse_args()

    df_override_arg = None
    if args.df_path:
        try:
            print(f"[*] Loading DataFrame override from: {args.df_path}")
            if args.df_path.endswith(".parquet"):
                df_override_arg = pd.read_parquet(args.df_path)
            elif args.df_path.endswith(".csv"):
                 df_override_arg = pd.read_csv(args.df_path, parse_dates=["timestamp"])
            else:
                 print(f"[!] Unsupported file format for df-path: {args.df_path}. Use .parquet or .csv")
            # Basic validation of override DataFrame
            if df_override_arg is not None and not all(c in df_override_arg.columns for c in ['timestamp', 'open', 'high', 'low', 'close', 'volume']):
                print("[!] df-override is missing required OHLCV columns.")
                df_override_arg = None # Invalidate it

        except Exception as e:
            print(f"[!] Error loading df-override from {args.df_path}: {e}")


    try:
        out = run(
            args.symbol,
            lookback_rows=args.lookback,
            base_table=args.base_table,
            df_override=df_override_arg
        )
        print(_json.dumps(out, indent=2))
    except Exception as e:
        print(f"[!!!] Error during inference run: {e}")
        import traceback
        traceback.print_exc()
