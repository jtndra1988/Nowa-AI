from typing import Dict, Any, Tuple

from .schemas import ExpertSignals, MarketContext


def bandit_weights(
    features: Dict[str, Any],
    expert: ExpertSignals,
    meta: Dict[str, Any],
    ctx: MarketContext,
) -> Tuple[Dict[str, float], str]:
    """
    Simple regime-based weighting.
    """
    rv = float(features.get("rv_24h", 0.05))
    trend = float(features.get("trend_score", 0.0))

    if trend > 0.3 and rv < 0.15:
        w = {"tft": 0.4, "tcn": 0.4, "xgb": 0.2}
        tag = "trend_perp"
    elif rv > 0.20:
        w = {"tft": 0.2, "tcn": 0.5, "xgb": 0.3}
        tag = "high_vol_momentum"
    else:
        w = {"tft": 0.2, "tcn": 0.2, "xgb": 0.6}
        tag = "mean_revert"

    s = sum(w.values()) or 1.0
    w = {k: v / s for k, v in w.items()}
    return w, tag
