# app/ml/fusion_model.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
from app.ml.artifacts import fpath
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sqlalchemy import create_engine, text
from xgboost import XGBRegressor
from app.ml.model import LSTMSignalModel
# Project config
try:
    from app.core.config import settings
    DATABASE_URL = settings.SQLALCHEMY_DATABASE_URI
except Exception:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@db:5432/postgres")

# Internal modules
from app.ml.model import LSTMSignalModel
from app.ml.data_preprocessor import preprocess_raw_chunk
from app.ml.calibration import (
    fit_temperature, save_temperature, load_temperature,
    fit_platt, save_platt, load_platt,
    softmax_rows,
)

# Optional unified manifest/params (fallback to defaults if missing)
BASE_CONF_CUTOFF_DEFAULT = 0.55
ATR_LENGTH_DEFAULT = 14
K_ATR_TO_BPS_DEFAULT = 2.0

try:
    from app.ml.artifacts import ckpt_dir, fpath, load_manifest, save_manifest
    USE_ARTIFACTS = True
except Exception:
    USE_ARTIFACTS = False
    def ckpt_dir(symbol: str) -> Path:
        p = Path("/app/checkpoints") / symbol.replace("/", "_")
        p.mkdir(parents=True, exist_ok=True)
        return p
    def fpath(symbol: str, key: str) -> Path:
        files = {
            "fusion": "fusion_head.pt",
            # IMPORTANT: standardized names
            "cal_temp": "calibration_temp.json",
            "cal_platt": "calibration_platt.json",
        }
        return ckpt_dir(symbol) / files[key]
    def load_manifest(symbol: str) -> Dict: return {}
    def save_manifest(symbol: str, **kwargs): pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 60

@dataclass
class FusionConfig:
    symbol: str
    base_table: str = "futures_market_data"
    batch_size: int = 256
    hidden: int = 32
    dropout: float = 0.1
    val_split: float = 0.1
    atr_length: int = ATR_LENGTH_DEFAULT
    k_atr_to_bps: float = K_ATR_TO_BPS_DEFAULT
    # Per-regime calibration files will be saved as:
    # calibration_temp_{regime}.json / calibration_platt_{regime}.json

class FusionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, num_classes: int = 3, p: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ---------- Helpers ----------
def _load_recent(engine, table: str, symbol: str, limit: int = 50000) -> pd.DataFrame:
    cols = "timestamp, open, high, low, close, volume"
    sql = text(f"""
        SELECT {cols}, symbol
        FROM {table}
        WHERE symbol = :sym
        ORDER BY timestamp ASC
        LIMIT :lim
    """)
    df = pd.read_sql(sql, engine, params={"sym": symbol, "lim": limit}, parse_dates=["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

def _ensure_close(df: pd.DataFrame) -> pd.DataFrame:
    if "close" not in df.columns:
        if "last_price" in df.columns:
            df = df.rename(columns={"last_price": "close"})
        else:
            raise ValueError("No close/last_price column found")
    return df

def _compute_regime(df: pd.DataFrame) -> pd.Series:
    # simple regime (0=range,1=up,2=down) – data_preprocessor also adds regime; use it if exists
    if "regime" in df.columns:
        return df["regime"].astype(int)
    # fallback quick regime
    s = df["close"].pct_change().rolling(5).mean()
    adx_like = df["close"].pct_change().abs().rolling(14).mean()
    return np.where(adx_like < adx_like.median(), 0, np.where(s > 0, 1, 2))

def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    try:
        import pandas_ta as ta
        return ta.atr(df["high"], df["low"], df["close"], length=length)
    except Exception:
        # fallback ATR approximation
        tr = (df["high"] - df["low"]).abs()
        return tr.rolling(length).mean()

def _build_lstm_preds(df: pd.DataFrame, seq_len: int = SEQ_LEN) -> np.ndarray:
    # lightweight in-file LSTM runner; expects checkpoints at /app/checkpoints/{symbol}
    return df["close"].shift(-1).to_numpy(dtype=np.float32)  # fallback “naive next close” if you don’t want to load LSTM here.

def _load_lstm_checkpoint_and_predict(df: pd.DataFrame, symbol: str, seq_len: int = SEQ_LEN) -> np.ndarray:
    # locate artifacts
    lstm_path = fpath(symbol, "lstm")
    scaler_path = fpath(symbol, "lstm_scaler")
    features_path = fpath(symbol, "features")

    if not (lstm_path.exists() and scaler_path.exists() and features_path.exists()):
        # fallback: naive next close
        return df["close"].shift(-1).to_numpy(dtype=np.float32)

    # load features list
    try:
        feats_data = json.loads(features_path.read_text())
        feat_list = feats_data.get("lstm_features")
        if not feat_list:
            return df["close"].shift(-1).to_numpy(dtype=np.float32)
    except Exception:
        return df["close"].shift(-1).to_numpy(dtype=np.float32)

    # ensure columns present
    if not all(c in df.columns for c in feat_list):
        return df["close"].shift(-1).to_numpy(dtype=np.float32)

    # scale with saved scaler
    try:
        scaler = joblib.load(scaler_path)
    except Exception:
        return df["close"].shift(-1).to_numpy(dtype=np.float32)

    X_raw = df[feat_list].values.astype(np.float32)
    # simple fill for any residual NaNs
    if np.isnan(X_raw).any():
        X_raw = pd.DataFrame(X_raw, columns=feat_list).ffill().bfill().fillna(0.0).values
    X_sc = scaler.transform(X_raw)

    if len(X_sc) <= seq_len:
        return df["close"].shift(-1).to_numpy(dtype=np.float32)

    # build sequences
    X_seq = np.asarray([X_sc[i-seq_len:i] for i in range(seq_len, len(X_sc))], dtype=np.float32)
    X_seq_t = torch.tensor(X_seq, dtype=torch.float32, device=DEVICE)

    # load model
    try:
        ckpt = torch.load(str(lstm_path), map_location=DEVICE)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt if isinstance(ckpt, dict) else None))
        if state is None:
            return df["close"].shift(-1).to_numpy(dtype=np.float32)

        model = LSTMSignalModel(input_size=X_seq.shape[-1], hidden_layer_size=128, num_layers=2, output_size=1).to(DEVICE)
        model.load_state_dict(state, strict=False)
        model.eval()
        with torch.no_grad():
            preds = model(X_seq_t).squeeze(-1).detach().cpu().numpy()
    except Exception:
        return df["close"].shift(-1).to_numpy(dtype=np.float32)

    out = np.full(len(df), np.nan, dtype=np.float32)
    out[seq_len:] = preds
    return out

def _load_xgb_and_predict(df: pd.DataFrame, symbol: str) -> np.ndarray:
    # expects files under checkpoints/{symbol}/xgb_model.json and xgb_features.json
    xgb_model = ckpt_dir(symbol) / "xgb_model.json"
    xgb_feats = ckpt_dir(symbol) / "xgb_features.json"
    if not xgb_model.exists() or not xgb_feats.exists():
        return df["close"].shift(-1).to_numpy(dtype=np.float32)
    with open(xgb_feats, "r") as f:
        parsed = json.load(f)
        feats = parsed.get("xgb_features") or parsed.get("features")
    model = XGBRegressor()
    model.load_model(str(xgb_model))
    X = df[feats].values.astype(np.float32)
    out = np.full(len(df), np.nan, dtype=np.float32)
    try:
        out[:] = model.predict(X).astype(np.float32)
    except Exception:
        out[:] = df["close"].shift(-1).to_numpy(dtype=np.float32)
    return out

def _train_val_split(n: int, val_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    n_val = max(1, int(n * val_ratio))
    idx = np.arange(n)
    return idx[:-n_val], idx[-n_val:]

# ---------- ATR-band labels ----------
def make_labels_atr_band(df: pd.DataFrame, atr_len: int, k_atr_to_bps: float) -> np.ndarray:
    """
    3-class labels with a per-sample HOLD band scaled by ATR:
       SELL(0), HOLD(1), BUY(2)
       hold_band_bps_i = k * (ATR_i / close_i) * 10000
    """
    atr = _atr(df, length=atr_len)
    vol = (atr / df["close"]).replace([np.inf, -np.inf], np.nan).fillna(method="bfill").fillna(method="ffill")
    hold_band_bps = (k_atr_to_bps * vol * 10000.0).astype("float32")
    next_close = df["target"].astype("float32").to_numpy()
    close_now  = df["close"].astype("float32").to_numpy()
    ret_bps = ((next_close - close_now) / (close_now + 1e-9) * 10000.0).astype("float32")
    y = np.where(ret_bps >  hold_band_bps, 2,
        np.where(ret_bps < -hold_band_bps, 0, 1)).astype("int64")
    return y

# ---------- Main ----------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, help="e.g., BTC/USDT")
    ap.add_argument("--base-table", default="futures_market_data")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--atr-length", type=int, default=ATR_LENGTH_DEFAULT)
    ap.add_argument("--k-atr-to-bps", type=float, default=K_ATR_TO_BPS_DEFAULT)
    args = ap.parse_args()

    symbol = args.symbol.upper()
    cfg = FusionConfig(
        symbol=symbol,
        base_table=args.base_table,
        val_split=args.val_split,
        atr_length=args.atr_length,
        k_atr_to_bps=args.k_atr_to_bps,
    )

    engine = create_engine(DATABASE_URL)
    raw = _load_recent(engine, cfg.base_table, cfg.symbol, limit=200000)
    raw = _ensure_close(raw)

    # Enrich features using your preprocessor (keeps all your existing features/regime/sentiment)
    df, feat_cols = preprocess_raw_chunk(raw, cfg.symbol, engine=engine, include_mtf=True)
    df = _ensure_close(df)
    # Targets: next close
    df["target"] = df["close"].shift(-1)
    df = df.dropna(subset=["target", "close", "high", "low"]).reset_index(drop=True)

    # Regime column (0=range,1=up,2=down)
    if "regime" not in df.columns:
        df["regime"] = _compute_regime(df)

    # Predictions to feed fusion: LSTM & XGB (next-close)
    lstm_next = _load_lstm_checkpoint_and_predict(df, cfg.symbol)
    xgb_next  = _load_xgb_and_predict(df, cfg.symbol)

    # Optional signals already produced by preprocessor
    ru = (df["regime"] == 1).astype(np.float32).to_numpy()
    rd = (df["regime"] == 2).astype(np.float32).to_numpy()
    fs = df.get("final_sentiment", pd.Series(np.zeros(len(df)))).astype(np.float32).to_numpy()

    # Assemble fusion input vectors (keep order stable)
    X_fuse = np.stack([lstm_next, xgb_next, ru, rd, fs], axis=1).astype(np.float32)
    # Drop initial nans (from sequence alignment)
    mask = ~np.isnan(X_fuse).any(axis=1)
    df = df.loc[mask].reset_index(drop=True)
    X_fuse = X_fuse[mask]

    # ----- ATR-scaled labels -----
    y_cls = make_labels_atr_band(df, cfg.atr_length, cfg.k_atr_to_bps)

    # Train/Val split (time-based)
    idx_tr, idx_va = _train_val_split(len(X_fuse), cfg.val_split)
    X_tr, y_tr = X_fuse[idx_tr], y_cls[idx_tr]
    X_va, y_va = X_fuse[idx_va], y_cls[idx_va]

    # Torch training (simple CE)
    fuse = FusionHead(in_dim=X_tr.shape[1], hidden=32, num_classes=3, p=0.1).to(DEVICE)
    opt = torch.optim.Adam(fuse.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    tr_dl = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=256, shuffle=True)
    va_dl = DataLoader(TensorDataset(torch.tensor(X_va), torch.tensor(y_va)), batch_size=256, shuffle=False)

    best_va = 1e9
    for ep in range(15):
        fuse.train()
        tr_losses = []
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = fuse(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        # val
        fuse.eval()
        with torch.no_grad():
            vas, vls = [], []
            for xb, yb in va_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = fuse(xb)
                loss = loss_fn(logits, yb)
                vls.append(loss.item())
                vas.append((logits.argmax(dim=1) == yb).float().mean().item())
            va_loss = float(np.mean(vls))
            va_acc  = float(np.mean(vas))
        print(f"[FUSION:{symbol}] ep={ep+1} tr={np.mean(tr_losses):.4f} va={va_loss:.4f} acc={va_acc:.3f}")
        if va_loss < best_va:
            best_va = va_loss
            # Save best state
            torch.save({"state_dict": fuse.state_dict(), "in_dim": X_tr.shape[1]}, ckpt_dir(symbol) / "fusion_head.pt")

    # ------- Calibration (global + per-regime) -------
    # Collect val logits on entire dataset (time-safe: use idx_va)
    fuse.load_state_dict(torch.load(ckpt_dir(symbol) / "fusion_head.pt", map_location=DEVICE)["state_dict"])
    fuse.eval()
    with torch.no_grad():
        logits_all = fuse(torch.tensor(X_fuse[idx_va], dtype=torch.float32, device=DEVICE)).cpu().numpy()
        y_all = y_cls[idx_va]

    # Global temperature & platt (STANDARDIZED NAMES)
    try:
        T = fit_temperature(logits_all, y_all)
        save_temperature(T, str(ckpt_dir(symbol) / "calibration_temp.json"))
    except Exception as e:
        print(f"[WARN] Global temperature fit failed: {e}")
        T = 1.0
    try:
        pm = fit_platt(logits_all / max(T, 1e-6), y_all)
        save_platt(pm, str(ckpt_dir(symbol) / "calibration_platt.json"))
    except Exception as e:
        print(f"[WARN] Global Platt fit failed: {e}")

    # Regime-specific calibration (STANDARDIZED NAMES)
    regimes = [0, 1, 2]
    va_reg = df["regime"].to_numpy()[idx_va]
    for r in regimes:
        m = (va_reg == r)
        if m.sum() < 50:
            # too small to calibrate; skip
            continue
        logits_r = logits_all[m]
        y_r = y_all[m]
        try:
            Tr = fit_temperature(logits_r, y_r)
            save_temperature(Tr, str(ckpt_dir(symbol) / f"calibration_temp_{r}.json"))
        except Exception as e:
            print(f"[WARN] Temp fit regime={r} failed: {e}")
            Tr = 1.0
        try:
            pmr = fit_platt(logits_r / max(Tr, 1e-6), y_r)
            save_platt(pmr, str(ckpt_dir(symbol) / f"calibration_platt_{r}.json"))
        except Exception as e:
            print(f"[WARN] Platt fit regime={r} failed: {e}")

    # Update manifest (if available)
    if USE_ARTIFACTS:
        save_manifest(
            symbol,
            fusion_present=True,
            decision_params={
                "base_confidence_cutoff": BASE_CONF_CUTOFF_DEFAULT,
                "atr_length": cfg.atr_length,
                "k_atr_to_bps": float(cfg.k_atr_to_bps),
                "regime_specific_calibration": True,
            },
            metrics={
                "fusion_val_loss": float(best_va),
                "fusion_va_acc": float(va_acc),
            },
            files={
                "calibration_temp": True,
                "calibration_platt": True,
                "regime_specific": True,
            }
        )
    print(f"[FUSION:{symbol}] Saved head + calibration (global & per-regime) with standardized filenames.")

if __name__ == "__main__":
    main()
