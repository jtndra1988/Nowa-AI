from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd

# plug-in extra features on top of rich preprocessor output

def add_extra_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    out["log_ret"] = np.log(out["close"] / out["close"].shift(1))
    out["ret_5"] = out["close"].pct_change(5)
    out["ret_20"] = out["close"].pct_change(20)
    out["vol_5"] = out["log_ret"].rolling(5).std()
    out["vol_20"] = out["log_ret"].rolling(20).std()
    out["vol_60"] = out["log_ret"].rolling(60).std()
    # simple order-flow proxy if columns present
    if "bid" in out.columns and "ask" in out.columns:
        out["mid"] = (out["bid"] + out["ask"]) / 2.0
        out["spread_bps"] = (out["ask"] - out["bid"]) / (out["mid"] + 1e-9) * 1e4
    out = out.bfill().ffill()
    extra = [c for c in ["log_ret","ret_5","ret_20","vol_5","vol_20","vol_60","spread_bps"] if c in out.columns]
    return out, extra

