from __future__ import annotations
from typing import Dict, List
from .state import PortfolioState
from .registry import DEFAULT_TARGETS

def suggest_rebalance() -> Dict[str, List[Dict]]:
    """
    Returns a set of suggestions to move each bucket toward its target.
    Non-invasive: you can log, publish to Redis, or create low-priority tasks to nudge.
    """
    st = PortfolioState.load()
    eq = st.equity_usd
    sug: Dict[str, List[Dict]] = {}
    for bucket, target_w in DEFAULT_TARGETS.items():
        target_usd = target_w * eq
        current_usd = abs(st.by_strategy.get(bucket, 0.0))
        delta = target_usd - current_usd
        if abs(delta) < 1e-6:
            continue
        action = "increase" if delta > 0 else "decrease"
        sug.setdefault(bucket, []).append({"action": action, "amount_usd": abs(delta)})
    return sug
