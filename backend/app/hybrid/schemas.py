# app/hybrid/schemas.py
from _future_ import annotations

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class MarketContext(BaseModel):
    """
    Context passed into the decision engine.

    This is what /hybrid-signal should receive from frontend,
    workers, or scheduled jobs.
    """
    exchange: str = Field(..., description="Exchange, e.g. 'binance'")
    symbol: str = Field(..., description="Symbol, e.g. 'BTCUSDT'")
    mode: Literal["spot", "futures", "options"] = Field(
        ..., description="Market mode"
    )

    # Optional but useful knobs:
    instrument_type: Optional[str] = Field(
        None, description="Internal instrument type if needed"
    )
    timeframe: Optional[str] = Field(
        "15m", description="Candle timeframe used for features"
    )
    position: Optional[float] = Field(
        0.0, description="Current net position (for future extensions)"
    )
    current_regime: Optional[str] = Field(
        None, description="Detected regime label if available"
    )
    current_volatility: Optional[float] = Field(
        None, description="Realized / implied vol snapshot"
    )


class HybridDecision(BaseModel):
    """
    Final decision object consumed by:
      - /hybrid-signal API
      - HybridSignal DB model
      - Frontend AI tab

    It merges:
      Layer 1: Expert votes
      Layer 2: DecisionNet-style fusion
      Layer 3: RL Head Trader execution suggestion
    """

    # Core identifiers
    symbol: str
    instrument_type: str = Field(
        "futures", description="Nowa internal instrument type"
    )

    # Layer 2: final directional view
    direction: Literal["up", "down", "flat"]
    p_edge: float = Field(
        ..., description="Estimated probability edge of being correct (0-1)"
    )
    confidence: float = Field(
        ..., description="Strength of conviction based on ensemble alignment (0-1)"
    )

    # Position sizing / routing
    size_factor: float = Field(
        ...,
        description="Recommended relative size (0-1) based on risk & RL suggestion",
    )
    strategy_tag: str = Field(
        "nowa_hybrid_v1",
        description="Label for which logic/profile produced this decision",
    )
    meta_execute: bool = Field(
        True,
        description="If True, this decision is eligible for execution by infra",
    )

    # Layer 1 / 2 introspection
    model_votes: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-expert scores, e.g. "
            "{'tft_visionary': 0.4, 'tcn_reflex': -0.1, 'xgb_analyst': 0.2, "
            "'options_psychologist': 0.1, 'macro_economist': 0.05, 'llm_narrative': 0.15}"
        ),
    )
    llm_headline: Optional[str] = Field(
        None,
        description="Key narrative summary from the LLM specialist.",
    )

    # Layer 3: RL execution layer
    rl_action: Optional[str] = Field(
        None,
        description="Human-friendly action label, e.g. LONG_50 / SHORT_100 / FLAT.",
    )
    rl_mode: Optional[str] = Field(
        None,
        description="How the decision was produced: 'rl', 'rule_fallback', etc.",
    )
    rl_target_position: Optional[float] = Field(
        None,
        description="Suggested net exposure in [-1, 1] where 1=full long, -1=full short.",
    )

    # Debug payloads (for logs / training / UI explainability)
    debug: Dict[str, Any] = Field(
        default_factory=dict,
        description="Nested diagnostics: raw expert signals, scores, RL internals, etc.",
    )

    class Config:
        orm_mode = True