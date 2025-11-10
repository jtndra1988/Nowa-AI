# app/api/schemas.py
from __future__ import annotations

from typing import Optional, List
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


# ---
# --- NEW SCHEMAS ADDED FOR READ APIS ---
# ---

class SentimentResponse(BaseModel):
    symbol: str
    timestamp: datetime
    sentiment_score: Optional[float] = Field(None, alias="sent_score_1m")
    sentiment_score_15m: Optional[float] = Field(None, alias="sent_score_15m")
    
    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        
class OnchainResponse(BaseModel):
    symbol: str
    timestamp: datetime
    nvt_ratio: Optional[float]
    sopr: Optional[float]
    # Add other on-chain fields here as needed
    
    class Config:
        orm_mode = True

class DeveloperResponse(BaseModel):
    symbol: str
    timestamp: datetime
    commit_count: Optional[int]
    stars: Optional[int]
    # Add other dev fields here as needed

    class Config:
        orm_mode = True