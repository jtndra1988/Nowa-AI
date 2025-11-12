import time
from typing import Dict, Any
from .event_bus import get_kv, set_kv

def update_cooldowns(redis_url: str, symbol: str):
    regime = get_kv(redis_url, f"regime:{symbol}", default={"vol":"mid_vol"})
    of = get_kv(redis_url, f"orderflow:{symbol}", default={"vpin":0.3})
    key = f"cooldowns:{symbol}"

    base = 60  # base seconds
    if regime.get("vol") == "high_vol":
        base = 30
    elif regime.get("vol") == "low_vol":
        base = 90

    if float(of.get("vpin", 0.3)) > 0.6:
        base = max(20, int(base * 0.7))

    state = {"cooldown_trade": base, "ts": int(time.time())}
    set_kv(redis_url, key, state, ttl=180)
    return state

def should_skip(redis_url: str, symbol: str, task_name: str) -> bool:
    cd = get_kv(redis_url, f"cooldowns:{symbol}", default={"cooldown_trade": 60})
    last = get_kv(redis_url, f"last_run:{task_name}:{symbol}", default={"ts": 0})
    now = int(time.time())
    if (now - int(last.get("ts", 0))) < int(cd.get("cooldown_trade", 60)):
        return True
    return False

def mark_ran(redis_url: str, symbol: str, task_name: str):
    set_kv(redis_url, f"last_run:{task_name}:{symbol}", {"ts": int(time.time())}, ttl=24*3600)
