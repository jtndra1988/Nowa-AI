from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

class BaselineATRBreakout:
    """
    Minimal example strategy:
    - Long if close > SMA + k*ATR
    - Short if close < SMA - k*ATR
    Emits the SAME signal shape your ExecutionService expects.
    """
    def __init__(self, sma=50, atr=14, k=1.0, profile="trend"):
        self.sma = sma; self.atr = atr; self.k = k; self.profile = profile

    def _atr(self, df: pd.DataFrame, length: int) -> float:
        try:
            import pandas_ta as ta
            s = ta.atr(df["high"], df["low"], df["close"], length=length)
            return float(s.iloc[-1])
        except Exception:
            high = df["high"].astype(float).values
            low = df["low"].astype(float).values
            close = df["close"].astype(float).values
            prev_close = np.roll(close, 1); prev_close[0] = close[0]
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)
            tr = np.maximum.reduce([tr1, tr2, tr3])
            atr = pd.Series(tr).rolling(window=length, min_periods=length).mean().iloc[-1]
            return float(atr)

    def on_bar(self, df: pd.DataFrame, i: int) -> Dict[str, Any] | None:
        # Need at least max(sma, atr)+1 bars
        if i < max(self.sma, self.atr) + 2: return None
        window = df.iloc[: i+1].tail(max(self.sma, self.atr) + 2)
        price = float(window["close"].iloc[-1])
        ma = float(window["close"].rolling(self.sma).mean().iloc[-1])
        atrv = self._atr(window, self.atr)
        if atrv <= 0: return None

        up = ma + self.k * atrv
        dn = ma - self.k * atrv

        if price > up:
            side = "BUY"
            conf = 0.65
            regime = 1
        elif price < dn:
            side = "SELL"
            conf = 0.65
            regime = 2
        else:
            return None

        return {
            "symbol": "BTC/USDT",   # fill at engine call
            "action": side,
            "price": price,
            "confidence": conf,
            "regime": regime,
            "model_profile": self.profile,
            "size_factor": 1.0,
            "sl_atr_mult": 1.8,
            "tp_atr_mult": 3.0,
        }
