import joblib
from pathlib import Path
from typing import Dict, Any

from .schemas import ExpertSignals, MarketContext

_META_MODEL_PATH = Path("./model_artifacts/meta_ensemble.joblib")
_meta_model = None


def _lazy_load():
    global _meta_model
    if _meta_model is None and _META_MODEL_PATH.exists():
        _meta_model = joblib.load(_META_MODEL_PATH)


def meta_predict(
    features: Dict[str, Any],
    expert: ExpertSignals,
    ctx: MarketContext,
) -> Dict[str, Any]:
    """
    Meta-ensemble:
      - Uses expert outputs + context to produce:
        * p_edge: probability trade has positive edge
        * dir_raw: raw direction suggestion
        * confidence: same as p_edge for now
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
        "decision_net_score": expert.decision_net_score or 0.0,
        "options_vol_edge": expert.options_vol_edge or 0.0,
        "macro_onchain_bias": expert.macro_onchain_bias or 0.0,
    }

    # If you train a real meta model, it plugs in here
    if _meta_model is not None:
        proba += 0.1 * (expert.decision_net_score or 0.0)
        proba += 0.05 * (expert.options_vol_edge or 0.0)
        proba += 0.05 * (expert.macro_onchain_bias or 0.0)
        proba = max(0.0, min(1.0, proba))
    else:
        # Fallback: use agreement & magnitude of price experts as proxy edge
        scores = [
            s
            for s in (expert.tft_price, expert.tcn_price, expert.xgb_price)
            if s is not None
        ]
        if scores:
            avg = sum(scores) / len(scores)
            proba = 0.5 + 0.4 * (avg / (abs(avg) + 1e-6))
            proba = max(0.0, min(1.0, proba))
        else:
            proba = 0.5

    scores_for_dir = [
        s
        for s in (expert.tft_price, expert.tcn_price, expert.xgb_price)
        if s is not None
    ]
    blended = sum(scores_for_dir) / len(scores_for_dir) if scores_for_dir else 0.0

    if proba < 0.52:
        dir_raw = "flat"
    else:
        dir_raw = "long" if blended >= 0 else "short"

    return {
        "p_edge": proba,
        "dir_raw": dir_raw,
        "confidence": proba,
    }
