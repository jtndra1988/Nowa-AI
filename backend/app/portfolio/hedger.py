from __future__ import annotations
from typing import Dict, Any
from .state import PortfolioState

def compute_hedge(net_cap_ratio: float = 0.30) -> Dict[str, Any]:
    """
    If net long/short exceeds 'net_cap_ratio' of equity, propose offset on BTC/USDT futures.
    Return {"need": bool, "side": "sell"/"buy", "symbol": "BTC/USDT", "qty_usd": float}
    """
    st = PortfolioState.load()
    eq = max(1e-9, st.equity_usd)
    limit = net_cap_ratio * eq
    net = st.net_exposure_usd

    if abs(net) <= limit:
        return {"need": False}

    side = "sell" if net > 0 else "buy"
    qty = abs(net) - limit
    return {"need": True, "side": side, "symbol": "BTC/USDT", "qty_usd": float(qty)}
