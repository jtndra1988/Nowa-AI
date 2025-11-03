from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

from .datasets import build_features
from .utils import out_dir, FILENAMES, ensure_dir

# ---- XGB Incremental (warm-start) ----
class XGBIncremental:
    """Pseudo-incremental updates by warm-starting from previous booster.
    Use small learning_rate and additional rounds on the latest window.
    """
    def __init__(self, base_model: Optional[XGBRegressor] = None):
        self.model = base_model or XGBRegressor(
            n_estimators=400, max_depth=8, subsample=0.8, colsample_bytree=0.8,
            learning_rate=0.03, tree_method="hist"
        )

    def fit_more(self, X_new: np.ndarray, y_new: np.ndarray, n_additional: int = 200):
        if getattr(self.model, "get_booster", None) is not None and self.model.get_booster() is not None:
            # warm-start by increasing estimators
            self.model.n_estimators += n_additional
            self.model.fit(X_new, y_new, xgb_model=self.model.get_booster())
        else:
            self.model.n_estimators = max(self.model.n_estimators, n_additional)
            self.model.fit(X_new, y_new)
        return self.model
    # ---- LSTM Fine-Tune on recent window ----
@torch.no_grad()
def _impute_nan_(X: np.ndarray) -> np.ndarray:
    if not np.isnan(X).any():
        return X
    df = pd.DataFrame(X)
    return df.ffill().bfill().fillna(0.0).values.astype(np.float32)


def finetune_lstm(model: torch.nn.Module, scaler: StandardScaler, X_recent: np.ndarray, y_recent: np.ndarray,
                  seq_len: int = 60, epochs: int = 3, lr: float = 5e-4, device: str = "cpu") -> torch.nn.Module:
    model = model.to(device)
    model.train()
    X_recent = _impute_nan_(X_recent.astype(np.float32))
    X_sc = scaler.transform(X_recent)
    if len(X_sc) <= seq_len:
        return model
    X_seq = np.stack([X_sc[i-seq_len:i] for i in range(seq_len, len(X_sc))], axis=0)
    y_seq = y_recent[seq_len:].astype(np.float32)
    ds = TensorDataset(torch.tensor(X_seq), torch.tensor(y_seq).unsqueeze(-1))
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model
