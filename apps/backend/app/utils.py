# app/utils.py
from typing import Any, Optional
import requests
import pandas as pd 
from datetime import datetime, timezone 
# =============================================================================
# Small helpers (pure functions)
# =============================================================================
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
def _norm_futures_symbol(sym: str) -> str:
    """Normalize incoming exchange symbols to 'COIN/USDT' (trim Bybit ':USDT' suffix)."""
    if not sym:
        return sym
    if ":" in sym:
        sym = sym.split(":")[0]
    base, _, quote = sym.partition("/")
    base = base.upper()
    quote = (quote or "USDT").upper()
    return f"{base}/{quote}"

def _base_from_pair(sym: str) -> str:
    """Extract base currency (e.g., 'BTC' from 'BTC/USDT')."""
    return (sym.split("/")[0]).upper() if "/" in sym else str(sym).upper()

def _safe_float(val: Any) -> Optional[float]:
    """Safely convert any value to a float, handling None, '', etc."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
def build_top100_slug_map() -> dict[str, str]:
    """
    Fetch top-100 coins from CoinGecko and build:
      SYMBOL → slug
    Example: BTC → bitcoin
    """
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    coins = r.json()

    mapping = {}
    for coin in coins:
        symbol = coin["symbol"].upper()
        slug = coin["id"]  # CoinGecko slug
        mapping[symbol] = slug

    return mapping
# (Keep _safe_datetime_from_ms in collectors.py for now, or move it here too if needed elsewhere)
# def _safe_datetime_from_ms(val: Any) -> Optional[datetime]: ...