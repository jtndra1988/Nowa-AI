from typing import Dict, Any, Tuple
from app.hybrid.schemas import ExpertSignals, MarketContext
from app.rt_adapt.event_bus import get_kv
from app.core.config import settings

def bandit_weights(
    features: Dict[str, Any],
    expert: ExpertSignals,
    meta: Dict[str, Any],
    ctx: MarketContext,
) -> Tuple[Dict[str, float], str]:
    """
    Regime-based router.
    Now fetches LIVE regime state from Redis (populated by rt_adapt).
    """
    symbol = ctx.symbol.upper()
    
    # 1. Fetch Regime from Nervous System (Redis)
    # Fallback to 'chop' if system is cold
    regime_data = get_kv(settings.CELERY_BROKER_URL, f"regime:{symbol}")
    if not regime_data:
        regime_data = {"regime": "chop", "vol": "mid_vol"}
        
    r_tag = regime_data.get("regime", "chop")
    v_tag = regime_data.get("vol", "mid_vol")

    # 2. Route Strategy based on External Truth
    if r_tag == "bull" or r_tag == "bear":
        # Strong Trend -> Trust Visionary (TFT)
        w = {"tft": 0.6, "tcn": 0.2, "xgb": 0.2}
        strategy_tag = f"trend_{r_tag}"
        
    elif v_tag == "high_vol":
        # Chaos -> Trust Reflex (TCN)
        w = {"tft": 0.2, "tcn": 0.6, "xgb": 0.2}
        strategy_tag = "volatility_reflex"
        
    else:
        # Chop/Low Vol -> Trust Analyst (XGB)
        w = {"tft": 0.2, "tcn": 0.2, "xgb": 0.6}
        strategy_tag = "mean_reversion"

    return w, strategy_tag