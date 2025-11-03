from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from .utils import out_dir, ensure_dir, FILENAMES

class DynamicEnsemble:
    """Stacking head that learns weights on a rolling validation window.
    Inputs: columns like [pred_lstm, pred_xgb, sentiment, regime, ...]
    Target: next_close (regression) or up/down (classification via sign).
    """
    def __init__(self):
        self.clf = LogisticRegression(max_iter=200)
        self.feats: List[str] = []

    def fit(self, df_stack: pd.DataFrame) -> None:
        # Build labels: direction of next return
        y = (df_stack["target_next_close"] - df_stack["close"]).values
        y = (y > 0).astype(int)
        X_cols = [c for c in df_stack.columns if c not in {"timestamp", "close", "target_next_close"}]
        self.feats = X_cols
        X = df_stack[X_cols].values
        self.clf.fit(X, y)

    def predict_proba(self, df_stack: pd.DataFrame) -> np.ndarray:
        X = df_stack[self.feats].values
        return self.clf.predict_proba(X)[:, 1]

    def save(self, symbol: str):
        p = out_dir(symbol, "prod") / "stacking_logreg.json"
        p.write_text(json.dumps({"coef": self.clf.coef_.tolist(), "intercept": self.clf.intercept_.tolist(), "feats": self.feats}))

    def load(self, symbol: str):
        p = out_dir(symbol, "prod") / "stacking_logreg.json"
        js = json.loads(p.read_text())
        self.feats = js["feats"]
        self.clf = LogisticRegression(max_iter=200)
        self.clf.coef_ = np.array(js["coef"]) ; self.clf.intercept_ = np.array(js["intercept"]) ; self.clf.classes_ = np.array([0,1])
        return self

