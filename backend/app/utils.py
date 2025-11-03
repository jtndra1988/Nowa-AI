# app/utils.py
from typing import Any, Optional
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

# (Keep _safe_datetime_from_ms in collectors.py for now, or move it here too if needed elsewhere)
# def _safe_datetime_from_ms(val: Any) -> Optional[datetime]: ...