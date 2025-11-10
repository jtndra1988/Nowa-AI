# app/api/schemas.py
from __future__ import annotations

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

# ---- Normalized instrument schema (matches your collectors/adapters/ETL) ----
class OptionInstrument(BaseModel):
    symbol: str = Field(..., description="Underlying (e.g., 'BTC')")
    expiry: Optional[datetime] = Field(None, description="Option expiry (UTC)")
    strike: Optional[float] = None
    option_type: Optional[str] = Field(None, description="'CALL' or 'PUT'")
    timestamp: Optional[datetime] = Field(None, description="Quote time (UTC)")
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_price: Optional[float] = None
    mark_price: Optional[float] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    iv: Optional[float] = Field(None, description="Implied volatility (decimal)")
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

    # Backward-compat aliases (if some older clients used these):
    # instrument_name ↔ symbol, expiration (epoch) ↔ expiry, mark_iv ↔ iv
    instrument_name: Optional[str] = Field(None, alias="instrument_name")
    expiration: Optional[int] = Field(None, alias="expiration")
    mark_iv: Optional[float] = Field(None, alias="mark_iv")

    class Config:
        orm_mode = True
        allow_population_by_field_name = True

# --- Optional: for a legacy forwarded predict route ---
class LegacyPrediction(BaseModel):
    symbol: str
    signal: str  # e.g., 'BUY_CALL', 'SELL_PUT', 'HOLD'