# app/backtest/strategies_ensemble.py
from __future__ import annotations
import os, json, math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

# Optional deps (XGBoost, Torch)
try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except Exception:
    XGB_OK = False


# -------------------------------
# Utilities
# -------------------------------
def _safe_last(series: pd.Series, default: float = 0.0) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return default

def _atr_np(df: pd.DataFrame, length: int = 14) -> float:
    """
    Fallback ATR if pandas_ta is unavailable. Uses simple rolling mean of TR.
    """
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum.reduce([tr1, tr2, tr3])
    if len(tr) < length:
        return 0.0
    return float(pd.Series(tr).rolling(window=length, min_periods=length).mean().iloc[-1])


# -------------------------------
# Model Loader
# -------------------------------
@dataclass
class EnsemblePaths:
    root: str
    # Common filenames under root (customize if your names differ)
    lstm_ckpt: str = "lstm.pt"            # torch checkpoint
    xgb_model: str = "xgb.json"           # xgboost booster
    scaler_pkl: str = "scaler.pkl"        # sklearn StandardScaler (optional)
    meta_json: str = "meta.json"          # contains feature list, thresholds, etc.


class EnsembleModelAdapter:
    """
    Loads your saved LSTM + XGB (or either) and exposes predict_proba() + helpers.
    Expects OHLCV-derived features produced by _featurize().
    """
    def __init__(self, paths: EnsemblePaths, device: Optional[str] = None):
        self.paths = paths
        self.device = device or ("cuda" if TORCH_OK and torch.cuda.is_available() else "cpu")
        self.loaded = False

        self.lstm = None
        self.lstm_seq_len = 60  # default; can be overridden from meta.json
        self.lstm_feature_cols: List[str] = []

        self.xgb: Optional[xgb.Booster] = None
        self.xgb_feature_cols: List[str] = []

        self.scaler = None
        self.meta = {}

    def load(self):
        # meta
        meta_fp = os.path.join(self.paths.root, self.paths.meta_json)
        if os.path.exists(meta_fp):
            with open(meta_fp, "r") as f:
                self.meta = json.load(f)
            self.lstm_seq_len = int(self.meta.get("sequence_length", self.lstm_seq_len))
            self.lstm_feature_cols = list(self.meta.get("lstm_features", []))
            self.xgb_feature_cols = list(self.meta.get("xgb_features", []))

        # scaler (optional)
        try:
            import pickle
            scal_fp = os.path.join(self.paths.root, self.paths.scaler_pkl)
            if os.path.exists(scal_fp):
                self.scaler = pickle.load(open(scal_fp, "rb"))
        except Exception as e:
            self.scaler = None

        # LSTM
        lstm_fp = os.path.join(self.paths.root, self.paths.lstm_ckpt)
        if TORCH_OK and os.path.exists(lstm_fp):
            # If your model class lives at app.ml.model:LSTMSignalModel — import it
            try:
                from app.ml.model import LSTMSignalModel  # your existing class
                self.lstm = LSTMSignalModel()
                ckpt = torch.load(lstm_fp, map_location=self.device)
                # handle state_dict nesting
                sd = ckpt.get("state_dict", ckpt)
                self.lstm.load_state_dict(sd, strict=False)
                self.lstm.to(self.device).eval()
            except Exception as e:
                self.lstm = None

        # XGB
        xgb_fp = os.path.join(self.paths.root, self.paths.xgb_model)
        if XGB_OK and os.path.exists(xgb_fp):
            self.xgb = xgb.Booster()
            self.xgb.load_model(xgb_fp)

        self.loaded = (self.lstm is not None) or (self.xgb is not None)
        if not self.loaded:
            raise RuntimeError("No ensemble components could be loaded (LSTM/XGB missing).")

    # ----- feature engineering -----
    def _featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lightweight FE to match typical live FE — extend to match your training pipeline.
        Returns a DataFrame with at least the columns specified in meta.json lists.
        """
        d = df.copy()
        d.columns = [c.lower() for c in d.columns]
        # basic indicators
        d["ret_1"] = d["close"].pct_change(1)
        d["ret_5"] = d["close"].pct_change(5)
        d["sma_20"] = d["close"].rolling(20).mean()
        d["sma_50"] = d["close"].rolling(50).mean()
        d["sma_slope"] = d["sma_50"].diff()
        d["atr_14"] = pd.Series([np.nan]*len(d))
        try:
            import pandas_ta as ta
            d["atr_14"] = ta.atr(d["high"], d["low"], d["close"], length=14)
        except Exception:
            d["atr_14"] = d.apply(lambda _: np.nan, axis=1)
            d.loc[:, "atr_14"] = d.apply(lambda row: np.nan, axis=1)
            # Fill via numpy fallback
            d["atr_14"].iloc[:] = np.nan
            if len(d) >= 16:
                d["atr_14"].iloc[-1] = _atr_np(d.tail(200), 14)

        # forward fill basics
        d = d.replace([np.inf, -np.inf], np.nan).ffill().bfill()

        return d

    # ----- predict -----
    def predict_proba(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Returns (prob_buy, prob_sell). Neutral implied as 1 - max(probs).
        If only one model is available, uses it. If both, averages probs (or meta-weighted).
        """
        d = self._featurize(df)
        # LSTM
        p_buy_lstm, p_sell_lstm = None, None
        if self.lstm is not None and len(d) >= self.lstm_seq_len:
            seq_cols = self.lstm_feature_cols or ["close", "ret_1", "ret_5", "sma_20", "sma_50", "atr_14"]
            seq = d[seq_cols].tail(self.lstm_seq_len).values.astype(np.float32)
            if self.scaler is not None:
                # scaler expects (N, F) — apply per-feature
                seq = self.scaler.transform(seq)
            x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, T, F]
            with torch.no_grad():
                logits = self.lstm(x)  # shape [1, 3] or [1,2] depending on your model
            probs = torch.softmax(logits.squeeze(0), dim=-1).detach().cpu().numpy()
            if probs.shape[-1] == 3:
                # assume order [sell, hold, buy] or similar; adjust if different
                p_sell_lstm = float(probs[0])
                p_buy_lstm  = float(probs[-1])
            elif probs.shape[-1] == 2:
                p_sell_lstm = float(probs[0]); p_buy_lstm = float(probs[1])

        # XGB
        p_buy_xgb, p_sell_xgb = None, None
        if self.xgb is not None:
            cols = self.xgb_feature_cols or ["ret_1","ret_5","sma_20","sma_50","atr_14","sma_slope"]
            row = d[cols].tail(1).values.astype(np.float32)
            if self.scaler is not None:
                row = self.scaler.transform(row)
            dmat = xgb.DMatrix(row)
            proba = self.xgb.predict(dmat)  # assumes multi:softprob or binary:logistic
            if proba.ndim == 2 and proba.shape[1] >= 2:
                p_sell_xgb = float(proba[0,0]); p_buy_xgb = float(proba[0,1])
            else:
                # binary logistic -> treat as buy prob
                p_buy_xgb = float(proba[0]); p_sell_xgb = 1.0 - p_buy_xgb

        # Combine
        probs = []
        if p_buy_lstm is not None and p_sell_lstm is not None:
            probs.append((p_buy_lstm, p_sell_lstm))
        if p_buy_xgb is not None and p_sell_xgb is not None:
            probs.append((p_buy_xgb, p_sell_xgb))
        if not probs:
            return (0.5, 0.5)
        # weights (optional from meta)
        w_lstm = float(self.meta.get("w_lstm", 0.5))
        w_xgb  = float(self.meta.get("w_xgb", 0.5))
        if len(probs) == 2:
            pb = w_lstm*probs[0][0] + w_xgb*probs[1][0]
            ps = w_lstm*probs[0][1] + w_xgb*probs[1][1]
        else:
            pb, ps = probs[0]
        return (float(pb), float(ps))


# -------------------------------
# Strategy wrapper
# -------------------------------
@dataclass
class EnsembleSignalConfig:
    buy_threshold: float = 0.55
    sell_threshold: float = 0.55
    model_profile_buy: str = "trend"
    model_profile_sell: str = "trend"
    size_factor: float = 1.0
    sl_atr_mult: float = 1.8
    tp_atr_mult: float = 3.0
    regime_ma: int = 50            # for rough regime tagging
    atr_len: int = 14


class EnsembleBacktestStrategy:
    """
    Backtest strategy that emits live-compatible signals by reading the ensemble adapter.
    Use in both run_once and walk_forward (via factory that constructs and .load()s it).
    """
    def __init__(self, adapter: EnsembleModelAdapter, symbol: str, cfg: Optional[EnsembleSignalConfig] = None):
        self.adapter = adapter
        self.symbol = symbol
        self.cfg = cfg or EnsembleSignalConfig()
        self._loaded = False

    def load(self):
        if not self._loaded:
            self.adapter.load()
            self._loaded = True
        return self

    def _regime(self, df: pd.DataFrame) -> int:
        """
        Simple regime: 1 (up), 2 (down), 0 (chop)
        based on SMA slope; you can swap to your regime detector.
        """
        d = df.tail(max(self.cfg.regime_ma + 2, 60)).copy()
        sma = d["close"].rolling(self.cfg.regime_ma).mean()
        slope = sma.diff().iloc[-1]
        if slope > 0:
            return 1
        elif slope < 0:
            return 2
        return 0

    def on_bar(self, df: pd.DataFrame, i: int) -> Dict[str, Any] | None:
        """
        Called per bar; returns a signal dict or None. Use last i-th bar window.
        """
        if i < 120:  # warm-up for features/indicators
            return None
        window = df.iloc[: i+1]
        pb, ps = self.adapter.predict_proba(window)
        regime = self._regime(window)

        # Decision
        if pb >= self.cfg.buy_threshold and pb > ps:
            side = "BUY"; profile = self.cfg.model_profile_buy; conf = float(pb)
        elif ps >= self.cfg.sell_threshold and ps > pb:
            side = "SELL"; profile = self.cfg.model_profile_sell; conf = float(ps)
        else:
            return None

        price = float(window["close"].iloc[-1])
        return {
            "symbol": self.symbol,
            "action": side,
            "price": price,
            "confidence": conf,
            "regime": regime,
            "model_profile": profile,
            "size_factor": float(self.cfg.size_factor),
            "sl_atr_mult": float(self.cfg.sl_atr_mult),
            "tp_atr_mult": float(self.cfg.tp_atr_mult),
        }
