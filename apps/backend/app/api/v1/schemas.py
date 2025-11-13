from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Optional, Literal, List

from pydantic import BaseModel, Field, ConfigDict

# ---------- Common small shapes ----------
class TimePoint(BaseModel):
    t: datetime | str  # allow ISO string or datetime
    v: float

    model_config = ConfigDict(from_attributes=True)


# ---------- Option instruments (used by options endpoints) ----------
class OptionInstrument(BaseModel):
    symbol: str = Field(..., description="Underlying, e.g., BTCUSDT")
    expiry: date = Field(..., description="YYYY-MM-DD")
    strike: float = Field(..., gt=0)
    option_type: Literal["C", "P", "call", "put"]
    exchange: Optional[str] = None
    underlying: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ---------- Market Intel responses ----------
class SentimentResponse(BaseModel):
    latest_score: float = Field(..., description="Latest sentiment score")
    label: str = Field(..., description="Human label for score, e.g., Bullish/Bearish/Neutral")
    series: List[TimePoint] = Field(default_factory=list, description="Time series for plotting")

    model_config = ConfigDict(from_attributes=True)


class OnchainResponse(BaseModel):
    latest_value: float = Field(..., description="Latest on-chain metric (e.g., active addrs)")
    label: str = Field(..., description="Human label")
    series: List[TimePoint] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class DeveloperResponse(BaseModel):
    latest_value: float = Field(..., description="Latest dev-activity metric")
    label: str = Field(..., description="Human label")
    series: List[TimePoint] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
# =============================================================================
# Market Context
# =============================================================================

class MarketContext(BaseModel):
    """
    Context passed into the decision engine.
    This is what /hybrid-signal should receive from frontend, workers, or jobs.
    """

    exchange: str = Field(..., description="Exchange, e.g. 'binance'")
    symbol: str = Field(..., description="Symbol, e.g. 'BTCUSDT'")

    # Instrument type used by InferenceService / routing
    instrument_type: Optional[str] = Field(
        "futures", description="Market type: 'spot', 'futures', 'options'"
    )

    # Timeframe is informational; not enforced by service
    timeframe: Optional[str] = Field(
        "15m", description="Candle timeframe used for features"
    )

    # Current exposure / state (used by RL)
    current_position_size: Optional[float] = Field(
        0.0,
        description="Current net position as fraction of equity (-1.0..1.0)",
    )

    current_regime: Optional[str] = Field(
        None,
        description="Regime label if available, e.g. 'bull', 'bear', 'chop'.",
    )

    current_volatility: Optional[float] = Field(
        None,
        description="Realized / implied vol snapshot.",
    )

    class Config:
        orm_mode = True


# =============================================================================
# Layer 2 Prediction (Ensemble Output)
# =============================================================================

DirectionLiteral = Literal["up", "down", "flat"]


class Layer2Prediction(BaseModel):
    """
    Output of ModelEngine (L2 ensemble).
    Consumed by RLAgent + HybridInferenceService.
    """

    asset: str
    direction: DirectionLiteral
    price_confidence: float = Field(..., ge=0.0, le=1.0)

    # Optional metadata for debugging (not required by pipeline)
    meta: Optional[Dict[str, Any]] = None

    class Config:
        orm_mode = True


# =============================================================================
# RL Action (Layer 3 Execution Policy)
# =============================================================================

class RLAction(BaseModel):
    """
    Output of RLAgent.get_optimal_action.

    This is what HybridInferenceService uses to:
      - choose LONG/SHORT/FLAT/HOLD
      - get raw size suggestion
      - get execution style hints
    """

    optimal_action: Literal["LONG", "SHORT", "FLAT", "HOLD"]
    optimal_size_pct: float = Field(..., ge=0.0, le=1.0)
    execution_style: str

    # Mode is optional so current RLAgent (which doesn't set it) is valid.
    mode: str = Field(
        "rl_live",
        description="RL mode label, e.g. 'rl_live', 'disabled', 'rule_fallback'.",
    )

    # Optional debug payload; RLAgent can populate if desired.
    debug_state: List[float] = Field(
        default_factory=list,
        description="Optional raw policy / state diagnostics.",
    )

    class Config:
        orm_mode = True


# =============================================================================
# Hybrid Decision (Final Surface)
# =============================================================================

class HybridDecision(BaseModel):
    """
    Final decision object returned by HybridInferenceService and
    exposed via /hybrid-signal.

    It merges:
      - L2 ensemble view
      - LLM narrative context
      - RL execution suggestion
      - RiskEngine-constrained sizing
      - Full debug payload for observability
    """

    # Identity
    symbol: str
    instrument_type: str
    timestamp: datetime

    # Direction & quality
    direction: DirectionLiteral
    p_edge: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)

    # Sizing & strategy
    size_factor: float = Field(
        ..., ge=0.0, le=1.0,
        description="Final suggested size factor after RL + RiskEngine."
    )
    strategy_tag: str = Field(
        "nowa_hybrid_v1",
        description="Label for the active hybrid strategy profile.",
    )
    meta_execute: bool = Field(
        ...,
        description="If True, infra is allowed to execute this decision.",
    )

    # Model / narrative context
    model_votes: Dict[str, float] = Field(
        default_factory=dict,
        description="Key model & signal scores used.",
    )
    llm_headline: Optional[str] = Field(
        default=None,
        description="Key narrative summary from LLM.",
    )

    # RL summary
    rl_action: Optional[Literal["LONG", "SHORT", "FLAT", "HOLD"]] = Field(
        default=None,
        description="RL-chosen discrete action.",
    )
    rl_mode: Optional[str] = Field(
        default=None,
        description="RL decision mode, e.g. 'rl_live', 'disabled'.",
    )
    rl_target_position: Optional[float] = Field(
        default=None,
        description="Signed net target exposure as fraction of equity (-1..1).",
    )
    rl_execution_style: Optional[str] = Field(
        default=None,
        description="Execution style hint, e.g. 'TWAP_5M', 'TWAP_15M'.",
    )

    # Full nested debug object (L2, LLM, RL, risk, features)
    debug: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full diagnostics payload.",
    )

    class Config:
        orm_mode = True


# =============================================================================
# Optional: Backwards-compat ExpertSignals
# =============================================================================

class ExpertSignals(BaseModel):
    trend: Optional[float] = None
    sentiment: Optional[float] = None
    onchain: Optional[float] = None
    options: Optional[float] = None
    macro: Optional[float] = None
    llm: Optional[float] = None

    class Config:
        orm_mode = True
