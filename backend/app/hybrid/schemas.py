from pydantic import BaseModel
from typing import Literal, Optional, Dict

InstrumentType = Literal["spot", "perp", "future", "option"]


class MarketContext(BaseModel):
    symbol: str
    instrument_type: InstrumentType = "perp"
    exchange: str = "deribit"
    timestamp: int  # epoch ms or s


class ExpertSignals(BaseModel):
    # Core experts
    tft_price: Optional[float] = None
    tcn_price: Optional[float] = None
    xgb_price: Optional[float] = None
    tft_vol: Optional[float] = None
    tcn_vol: Optional[float] = None
    xgb_vol: Optional[float] = None

    # New specialists
    decision_net_score: Optional[float] = None     # +1 long, -1 short, 0 flat/weak
    options_vol_edge: Optional[float] = None       # >0 bullish vol/skew, <0 bearish
    macro_onchain_bias: Optional[float] = None     # >0 risk-on, <0 risk-off


class HybridDecision(BaseModel):
    symbol: str
    instrument_type: InstrumentType
    direction: Literal["long", "short", "flat"]
    p_edge: float
    confidence: float
    size_factor: float
    strategy_tag: str
    meta_execute: bool
    debug: Optional[Dict] = None
