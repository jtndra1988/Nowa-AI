# app/hybrid/schemas.py
from __future__ import annotations

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class MarketContext(BaseModel):
    """
    Context passed into the decision engine.

    This is what /hybrid-signal should receive from frontend,
    workers, or scheduled jobs.
    """
    exchange: str = Field(..., description="Exchange, e.g. 'binance'")
    symbol: str = Field(..., description="Symbol, e.g. 'BTCUSDT'")
    
    # --- FIX: Changed 'mode' to 'instrument_type' to align with InferenceService
    # This was a minor conflict. 'instrument_type' is used by the service.
    instrument_type: Optional[str] = Field(
        "futures", description="Market mode: 'spot', 'futures', 'options'"
    )

    timeframe: Optional[str] = Field(
        "15m", description="Candle timeframe used for features"
    )
    # --- FIX: Renamed 'position' to 'current_position_size' ---
    # This aligns with the 'MarketContext' expected by the RLAgent
    current_position_size: Optional[float] = Field(
        0.0, description="Current net position (from -1.0 to 1.0)"
    )
    current_regime: Optional[str] = Field(
        None, description="Detected regime label if available"
    )
    current_volatility: Optional[float] = Field(
        None, description="Realized / implied vol snapshot"
    )


# --- NEW: Added Layer2Prediction Schema ---
# This is the internal object passed from Layer 2 to Layer 3
class Layer2Prediction(BaseModel):
    asset: str
    direction: Literal["up", "down", "flat"]
    price_confidence: float = Field(..., ge=0.0, le=1.0)


# --- NEW: Added RLAction Schema ---
# This is the internal object returned by the RL Agent (Layer 3)
class RLAction(BaseModel):
    optimal_action: str  # e.g., "LONG", "SHORT", "HOLD"
    optimal_size_pct: float = Field(..., ge=0.0, le=1.0) # Absolute size
    execution_style: str # e.g., "TWAP_15M", "AGGRESSIVE"
    mode: str  # "rl_live" or "rule_fallback"
    debug_state: List[float] = Field(default_factory=list)


class HybridDecision(BaseModel):
    """
    Final decision object consumed by:
      - /hybrid-signal API (as response)
      - Execution services (as instruction)
      - Database (for audit)
    """
    symbol: str
    instrument_type: str
    timestamp: datetime

    # Core decision
    direction: Literal["up", "down", "flat"]
    p_edge: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of a positive-edge trade (0-1)"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence in the direction (0-1)"
    )
    size_factor: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final suggested size factor (0=none, 1=full)",
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
    
    # --- FIELDS ADDED TO MATCH INFERENCE SERVICE OUTPUT ---
    llm_headline: Optional[str] = Field(
        None,
        description="Key narrative summary from the LLM specialist.",
    )

    # Layer 3: RL execution layer
    rl_action: Optional[str] = Field(
        None,
        description="Human-friendly action label, e.g. LONG / SHORT / HOLD.",
    )
    rl_mode: Optional[str] = Field(
        None,
        description="How the decision was produced: 'rl_live', 'rule_fallback', etc.",
    )
    rl_target_position: Optional[float] = Field(
        None,
        description="Suggested net exposure, e.g. -1.0, 0.5, 1.0",
    )
    rl_execution_style: Optional[str] = Field(
        None,
        description="Suggested execution style, e.g. 'TWAP_15M'",
    )
    
    # Full debug payload
    debug: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full debug payload with L1, L2, L3 state."
    )
    # --- END OF ADDED FIELDS ---