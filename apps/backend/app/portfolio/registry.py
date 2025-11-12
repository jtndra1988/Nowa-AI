from __future__ import annotations
from typing import Dict

# Target weights (fractions of equity), soft caps
DEFAULT_TARGETS: Dict[str, float] = {
    "trend_futures":   0.40,
    "meanrev_futures": 0.25,
    "options_delta":   0.20,
    "options_premium": 0.15,
}

# Hard caps (fraction of equity)
BUCKET_MAX: Dict[str, float] = {
    "trend_futures":   0.60,
    "meanrev_futures": 0.40,
    "options_delta":   0.35,
    "options_premium": 0.30,
}

# Per-symbol cap (fraction of equity)
PER_SYMBOL_CAP = 0.20

# Portfolio-level config
MAX_LEVERAGE = 2.0
CASH_BUFFER = 0.05   # keep 5% as cash
