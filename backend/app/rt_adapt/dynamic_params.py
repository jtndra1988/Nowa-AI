from typing import Dict, Any
from sqlalchemy import create_engine, text
from .event_bus import get_kv, set_kv

def _fetch_latest_sentiment(engine_url: str, symbol: str) -> float:
    # -1..+1; adapt to your table
    eng = create_engine(engine_url)
    row = eng.execute(text("SELECT score FROM sentiment_data WHERE symbol=:s ORDER BY ts DESC LIMIT 1"), {"s": symbol}).fetchone()
    return float(row[0]) if row else 0.0

def tune(engine_url: str, redis_url: str, symbol: str) -> Dict[str, Any]:
    regime = get_kv(redis_url, f"regime:{symbol}", default={"regime":"chop","vol":"mid_vol"})
    of = get_kv(redis_url, f"orderflow:{symbol}", default={"obi":0.0,"aggr":0.5,"vpin":0.3})
    sent = _fetch_latest_sentiment(engine_url, symbol)

    # Base params
    params = {
        "min_model_conf": 0.55,
        "rsi_buy": 35,
        "rsi_sell": 65,
        "size_factor": 1.0,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 2.2,
        "cooldown_sec": 60,
        "model_profile": "default"
    }

    # Regime / vol adjustments
    if regime["regime"] == "bull":
        params["rsi_buy"] -= 3
        params["rsi_sell"] += 2
        params["size_factor"] *= 1.2
        params["model_profile"] = "trend"
    elif regime["regime"] == "bear":
        params["rsi_buy"] += 3
        params["rsi_sell"] -= 2
        params["size_factor"] *= 1.1
        params["model_profile"] = "trend"
    else:  # chop
        params["min_model_conf"] += 0.05
        params["tp_atr_mult"] -= 0.3
        params["model_profile"] = "meanrev"

    if regime.get("vol") == "high_vol":
        params["sl_atr_mult"] *= 1.3
        params["cooldown_sec"] = max(30, int(params["cooldown_sec"] * 0.7))
    elif regime.get("vol") == "low_vol":
        params["size_factor"] *= 0.8
        params["tp_atr_mult"] *= 0.9

    # Orderflow nudges
    obi = float(of.get("obi", 0.0))
    vpin = float(of.get("vpin", 0.3))
    if abs(obi) > 0.3:
        params["size_factor"] *= 1.1
    if vpin > 0.5:
        params["min_model_conf"] += 0.05
        params["sl_atr_mult"] *= 1.1

    # Sentiment gating
    if sent < -0.3:
        params["min_model_conf"] += 0.05
        params["size_factor"] *= 0.85
    elif sent > 0.3:
        params["size_factor"] *= 1.1

    set_kv(redis_url, f"dynparams:{symbol}", params, ttl=120)
    return params

def get_params(redis_url: str, symbol: str) -> Dict[str, Any]:
    return get_kv(redis_url, f"dynparams:{symbol}", default=None) or {}
