from pathlib import Path
from typing import Dict, Any, Optional
import joblib

_MACRO_ONCHAIN_MODEL_PATH = Path("./model_artifacts/macro_onchain_bias.joblib")
_macro_onchain_model = None


def _lazy_load_macro_onchain_model():
    global _macro_onchain_model
    if _macro_onchain_model is None and _MACRO_ONCHAIN_MODEL_PATH.exists():
        _macro_onchain_model = joblib.load(_MACRO_ONCHAIN_MODEL_PATH)


def macro_onchain_bias(features: Dict[str, Any]) -> float:
    """
    Returns bias in [-1, 1]:
      +1 = strong risk-on (support long bias)
      -1 = strong risk-off (prefer shorts / no risk)
       0 = neutral

    Inputs may include:
      - stablecoin_flows
      - btc_exchange_reserves_change
      - perp_oi_change
      - macro_dxy_trend, macro_spx_trend, etc.

    Uses model if available; otherwise a transparent heuristic.
    """
    _lazy_load_macro_onchain_model()

    if _macro_onchain_model is not None:
        keys = sorted(features.keys())
        x = [[float(features[k]) for k in keys]]
        pred = float(_macro_onchain_model.predict(x)[0])
        return max(-1.0, min(1.0, pred))

    # Heuristic fallback:
    score = 0.0

    stbl_flow = float(features.get("stablecoin_netflow", 0.0))      # >0 inflow to exchanges or risk venues
    ex_res = float(features.get("btc_exchange_reserves_change", 0.0))
    dxy_trend = float(features.get("dxy_trend", 0.0))               # strong up = risk-off
    spx_trend = float(features.get("spx_trend", 0.0))               # strong up = risk-on

    score += 0.3 * spx_trend
    score -= 0.3 * dxy_trend
    score -= 0.2 * max(ex_res, 0.0)          # rising reserves -> sell pressure
    score += 0.2 * max(stbl_flow, 0.0)       # positive stablecoin flows -> risk-on

    return max(-1.0, min(1.0, score))
