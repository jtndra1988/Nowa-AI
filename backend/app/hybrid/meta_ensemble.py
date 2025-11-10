from pathlib import Path
from typing import Dict, Any
import joblib

from .schemas import ExpertSignals, MarketContext

_META_MODEL_PATH = Path("./model_artifacts/meta_ensemble.joblib")
_meta_model = None


def _lazy_load():
    global _meta_model
    if _meta_model is None and _META_MODEL_PATH.exists():
        _meta_model = joblib.load(_META_MODEL_PATH)


def meta_predict(features: Dict[str, Any], expert: ExpertSignals, ctx: MarketContext) -> Dict[str, Any]:
    """
    Returns:
        p_edge: probability trade has positive edge (0-1)
        dir_raw: "long" / "short" / "flat"
        confidence: same scale as p_edge
    """
    _lazy_load()

    row = {
        "tft_price": expert.tft_price or 0.0,
        "tcn_price": expert.tcn_price or 0.0,
        "xgb_price": expert.xgb_price or 0.0,
        "tft_vol": expert.tft_vol or 0.0,
        "tcn_vol": expert.tcn_vol or 0.0,
        "xgb_vol": expert.xgb_vol or 0.0,
        "rv_24h": features.get("rv_24h", 0.0),
        "funding_1h": features.get("funding_1h", 0.0),
    }

    if _meta_model is not None:
        # Expect a sklearn-style model
        import pandas as pd
        proba = float(_meta_model.predict_proba(pd.DataFrame([row]))[0, 1])
    else:
        # Heuristic fallback based on blended expert view
        scores = [s for s in [expert.tft_price, expert.tcn_price, expert.xgb_price] if s is not None]
        if scores:
            avg = sum(scores) / len(scores)
            proba = 0.5 + 0.25 * (avg / (abs(avg) + 1e-6))  # squashed into [0.25, 0.75]
        else:
            proba = 0.5

    # Direction from blended expert signal
    scores = [s for s in [expert.tft_price, expert.tcn_price, expert.xgb_price] if s is not None]
    blended = sum(scores) / len(scores) if scores else 0.0

    if proba < 0.52:
        dir_raw = "flat"
    else:
        dir_raw = "long" if blended >= 0 else "short"

    return {
        "p_edge": max(0.0, min(1.0, proba)),
        "dir_raw": dir_raw,
        "confidence": max(0.0, min(1.0, proba)),
    }
