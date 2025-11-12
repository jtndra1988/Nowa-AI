# app/api/schemas_settings.py
from typing import Optional, Dict
from pydantic import BaseModel

class RiskSettingsGlobalIn(BaseModel):
    account_equity_usd: Optional[float] = None
    max_portfolio_leverage: Optional[float] = None
    max_concurrent_positions: Optional[int] = None
    target_daily_vol: Optional[float] = None
    min_position_usd: Optional[float] = None
    atr_length: Optional[int] = None
    sl_atr_mult_init: Optional[float] = None
    tp_atr_mult_init: Optional[float] = None
    base_confidence_cutoff: Optional[float] = None
    max_confidence_boost: Optional[float] = None
    regime_risk_multipliers: Optional[Dict[str, float]] = None
    daily_loss_limit_pct: Optional[float] = None
    rolling_max_dd_pct: Optional[float] = None
    per_trade_risk_pct_cap: Optional[float] = None
    ewma_alpha: Optional[float] = None
    min_trades_to_learn: Optional[int] = None
    bandit_lr: Optional[float] = None
    min_tp_sl_ratio: Optional[float] = None
    min_sl_atr_mult: Optional[float] = None
    max_sl_atr_mult: Optional[float] = None
    min_tp_atr_mult: Optional[float] = None
    max_tp_atr_mult: Optional[float] = None
    taker_fee_bps: Optional[float] = None
    default_adapter: Optional[str] = None  # "BYBIT" | "BINANCE"

class RiskSettingsSymbolIn(BaseModel):
    max_symbol_leverage: Optional[float] = None
    max_symbol_exposure_pct: Optional[float] = None
    atr_length: Optional[int] = None
    sl_atr_mult_init: Optional[float] = None
    tp_atr_mult_init: Optional[float] = None
    base_confidence_cutoff: Optional[float] = None
    regime_risk_multipliers: Optional[Dict[str, float]] = None
    min_position_usd: Optional[float] = None
    per_trade_risk_pct_cap: Optional[float] = None
    taker_fee_bps: Optional[float] = None
