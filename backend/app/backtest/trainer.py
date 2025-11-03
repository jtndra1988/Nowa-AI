# app/backtest/trainer.py
from __future__ import annotations
import os, json, math, uuid, tempfile, pickle, warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# Optional heavy deps
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except Exception:
    XGB_OK = False

from sklearn.preprocessing import StandardScaler

# Your existing LSTM model class
try:
    from app.ml.model import LSTMSignalModel
    HAS_LSTM_CLASS = True
except Exception:
    HAS_LSTM_CLASS = False


# =============== Feature engineering (align with strategies_ensemble) ===============
def featurize(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]
    assert {"open","high","low","close"}.issubset(set(d.columns)), "OHLC columns missing"
    d["ret_1"]   = d["close"].pct_change(1)
    d["ret_5"]   = d["close"].pct_change(5)
    d["sma_20"]  = d["close"].rolling(20).mean()
    d["sma_50"]  = d["close"].rolling(50).mean()
    d["sma_slope"] = d["sma_50"].diff()
    # ATR (fallback if pandas_ta missing)
    try:
        import pandas_ta as ta
        d["atr_14"] = ta.atr(d["high"], d["low"], d["close"], length=14)
    except Exception:
        high = d["high"].astype(float).values
        low  = d["low"].astype(float).values
        close = d["close"].astype(float).values
        prev = np.roll(close, 1); prev[0] = close[0]
        tr1 = high - low
        tr2 = np.abs(high - prev)
        tr3 = np.abs(low  - prev)
        tr  = np.maximum.reduce([tr1, tr2, tr3])
        atr = pd.Series(tr).rolling(14, min_periods=14).mean()
        d["atr_14"] = atr
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d = d.ffill().bfill()
    return d


# =============== Labels (simple next-bar direction; replace with your own) ===============
def make_labels(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Binary labels: 1 = up, 0 = down (can extend to 3-class if you want)."""
    fwd = df["close"].shift(-horizon)
    lab = (fwd > df["close"]).astype(int)
    return lab


# =============== LSTM Dataset ===============
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_sequences(dff: pd.DataFrame, feature_cols: List[str], seq_len: int, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    mat = dff[feature_cols].values
    yv = labels.values
    X, y = [], []
    for i in range(seq_len, len(mat) - 1):  # -1 because label uses shift(-1)
        X.append(mat[i - seq_len:i, :])
        y.append(yv[i])
    if not X:
        return np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(X), np.array(y)


# =============== Training config ===============
@dataclass
class TrainConfig:
    seq_len: int = 60
    lstm_epochs: int = 5
    lstm_batch: int = 128
    lstm_lr: float = 1e-3
    use_amp: bool = True
    xgb_rounds: int = 150
    xgb_lr: float = 0.05
    xgb_max_depth: int = 5
    # thresholds for adapter
    buy_threshold: float = 0.58
    sell_threshold: float = 0.58
    model_profile_buy: str = "trend"
    model_profile_sell: str = "trend"
    sl_atr_mult: float = 1.8
    tp_atr_mult: float = 3.0
    size_factor: float = 1.0

    # ensemble weights
    w_lstm: float = 0.5
    w_xgb: float  = 0.5


# =============== Segment trainer (LSTM + XGB) ===============
def train_segment(train_df: pd.DataFrame, out_root: Optional[str], cfg: Optional[TrainConfig] = None) -> str:
    """
    Trains LSTM (if available) and XGB on 'train_df' window.
    Writes artifacts under 'out_root' (created if None -> temp dir).
    Returns the folder path with artifacts.
    """
    cfg = cfg or TrainConfig()
    if out_root is None:
        out_root = os.path.join(tempfile.gettempdir(), f"wf_{uuid.uuid4().hex[:8]}")
    os.makedirs(out_root, exist_ok=True)

    # 1) FE + labels
    df = featurize(train_df)
    labels = make_labels(df, horizon=1)
    # Align/drop NaNs induced by indicators/labels
    drop_idx = df.index[df.isna().any(axis=1)].tolist()
    if drop_idx:
        df = df.drop(index=drop_idx)
        labels = labels.drop(index=drop_idx)
    labels = labels.iloc[:-1]  # final label is NaN due to shift; drop last
    df = df.iloc[:-1]

    # 2) Feature sets
    lstm_features = ["close", "ret_1", "ret_5", "sma_20", "sma_50", "atr_14"]
    xgb_features  = ["ret_1", "ret_5", "sma_20", "sma_50", "atr_14", "sma_slope"]

    # 3) Scaler (fit on full train window)
    scaler = StandardScaler()
    try:
        scaler.fit(df[lstm_features + xgb_features])
    except Exception:
        # fallback: fit on columns that exist
        cols = [c for c in (lstm_features + xgb_features) if c in df.columns]
        scaler.fit(df[cols])

    # 4) LSTM train (optional if torch + model class present)
    if TORCH_OK and HAS_LSTM_CLASS and len(df) >= cfg.seq_len + 10:
        X_seq, y_seq = make_sequences(
            pd.DataFrame(scaler.transform(df[lstm_features]), columns=lstm_features, index=df.index),
            lstm_features, cfg.seq_len, labels
        )
        ds = SeqDataset(X_seq, y_seq)
        dl = DataLoader(ds, batch_size=cfg.lstm_batch, shuffle=True, drop_last=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LSTMSignalModel()
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lstm_lr)
        loss_fn = nn.CrossEntropyLoss()
        scaler_amp = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device == "cuda"))

        model.train()
        for ep in range(cfg.lstm_epochs):
            running = 0.0
            for xb, yb in dl:
                xb = xb.to(device); yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device == "cuda")):
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(opt)
                scaler_amp.update()
                running += float(loss.item())
            # print(f"[WF/LSTM] epoch {ep+1}/{cfg.lstm_epochs} loss={running/len(dl):.4f}")
        # save
        torch.save(model.state_dict(), os.path.join(out_root, "lstm.pt"))
    else:
        warnings.warn("Skipping LSTM: Torch/LSTM class missing or not enough data.")

    # 5) XGBoost train (optional)
    if XGB_OK:
        X = scaler.transform(df[xgb_features])
        dmat = xgb.DMatrix(X, label=labels.values.astype(np.float32))
        params = {
            "objective": "multi:softprob",
            "num_class": 2,
            "eta": cfg.xgb_lr,
            "max_depth": cfg.xgb_max_depth,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "mlogloss",
            "seed": 42,
        }
        booster = xgb.train(params, dmat, num_boost_round=cfg.xgb_rounds)
        booster.save_model(os.path.join(out_root, "xgb.json"))
    else:
        warnings.warn("Skipping XGB: xgboost not installed.")

    # 6) Save scaler + meta.json
    with open(os.path.join(out_root, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "sequence_length": cfg.seq_len,
        "lstm_features": lstm_features,
        "xgb_features": xgb_features,
        "buy_threshold": cfg.buy_threshold,
        "sell_threshold": cfg.sell_threshold,
        "sl_atr_mult": cfg.sl_atr_mult,
        "tp_atr_mult": cfg.tp_atr_mult,
        "size_factor": cfg.size_factor,
        "model_profile_buy": cfg.model_profile_buy,
        "model_profile_sell": cfg.model_profile_sell,
        "w_lstm": cfg.w_lstm,
        "w_xgb": cfg.w_xgb,
    }
    with open(os.path.join(out_root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return out_root
