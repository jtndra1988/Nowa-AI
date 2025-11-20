# app/hybrid/schemas.py

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Market Context
# =============================================================================

class MarketContext(BaseModel):
    """
    Snapshot of the trading environment provided to the hybrid brain.

    Only a few fields are strictly required by the current pipeline.
    Others are optional hooks for future use / RL / risk.
    """

    # Core routing
    symbol: str = Field(..., description="Trading symbol, e.g. BTCUSDT")
    exchange: str = Field("BYBIT", description="Exchange identifier")
    instrument_type: str = Field(
        "futures", description="Instrument type, e.g. 'spot' | 'futures'"
    )

    # Optional: macro / regime / vol info (used in RL or can be ignored)
    current_regime: Optional[int] = Field(
        None,
        description="Discrete regime label if available (e.g., 0=neutral,1=bull,-1=bear).",
    )
    current_volatility: Optional[float] = Field(
        None,
        description="Realized/Implied vol metric for context (0-1 or annualized).",
    )

    # Optional: position / risk context
    current_position_size: Optional[float] = Field(
        None,
        description="Current net position as fraction of equity or contracts; RL may use.",
    )
    max_position_pct: Optional[float] = Field(
        None,
        description="Per-trade or per-symbol cap hint (0-1). Not enforced here directly.",
    )

    class Config:
        orm_mode = True


# =============================================================================
# Layer 2 Prediction (Ensemble Output)
# =============================================================================

DirectionLiteral = Literal["up", "down", "flat"]


class Layer2Prediction(BaseModel):
    """
    Output of the ensemble / decision net (L2).

    This is what flows into:
      - RLAgent.get_optimal_action(...)
      - HybridInferenceService final decision logic
    """

    asset: str

    tft_vote: float = 0.0
    tcn_vote: float = 0.0
    tst_vote: float = 0.0

    xgb_price_vote: float = 0.0
    xgb_vol_vote: float = 0.0

    decision_score: float = 0.0

    options_features: Dict[str, Any] = {}
    macro_onchain_features: Dict[str, Any] = {}

    # NEW: scalar scores for RL & UI
    options_score: float = 0.0
    macro_score: float = 0.0
    llm_narrative_vote: float = 0.0

    unified_vote: float = 0.0
    
    # Keep these as flexible dicts or define sub-models
    options_features: Dict[str, Any] = Field(default_factory=dict)
    macro_onchain_features: Dict[str, Any] = Field(default_factory=dict)
    
    meta: Optional[Dict[str, Any]] = None

    class Config:
        orm_mode = True


# =============================================================================
# RL Action (Layer 3 Execution Policy)
# =============================================================================

class RLAction(BaseModel):
    """
    Output of the RL execution agent.

    This is consumed by HybridInferenceService for:
      - mapping to final direction
      - size suggestion (before risk constraints)
      - execution style & diagnostics
    """

    optimal_action: Literal["LONG", "SHORT", "FLAT", "HOLD"] = Field(
        ...,
        description="Proposed action from RL: LONG / SHORT / FLAT / HOLD.",
    )
    optimal_size_pct: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Suggested size as fraction of max/equity (0..1).",
    )
    execution_style: str = Field(
        ...,
        description="Execution style hint, e.g. 'MARKET', 'TWAP_5M', 'TWAP_15M', 'NONE'.",
    )

    # Optional: used by inference_service & for observability
    mode: Optional[str] = Field(
        default=None,
        description="RL agent mode label, e.g. 'rl_live', 'disabled', 'rule_fallback'.",
    )
    debug_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional diagnostics / raw policy state.",
    )

    class Config:
        orm_mode = True


# =============================================================================
# Hybrid Decision (Final API Surface)
# =============================================================================

class HybridDecision(BaseModel):
    """
    Final decision object returned by the HybridInferenceService
    and exposed via the /hybrid-signal (or equivalent) endpoint.

    It merges:
      - Ensemble view (L2)
      - LLM narrative context
      - RL execution suggestion
      - Applied risk constraints
      - Debug payload for transparency
    """

    # Core identifiers
    symbol: str = Field(..., description="Trading symbol, e.g. BTCUSDT.")
    instrument_type: str = Field(
        ...,
        description="Instrument type, e.g. 'spot' | 'futures'.",
    )
    timestamp: datetime = Field(
        ..., description="UTC timestamp when this decision was generated."
    )

    # View & quality
    direction: DirectionLiteral = Field(
        ..., description="Final directional stance after RL & risk."
    )
    p_edge: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability/edge proxy derived from L2 confidence.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence score (mirrors or refines p_edge).",
    )

    # Sizing & execution intent
    size_factor: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final size factor after RL + RiskEngine constraints (0..1).",
    )
    strategy_tag: str = Field(
        ...,
        description="Identifier for this hybrid strategy version.",
    )
    meta_execute: bool = Field(
        ...,
        description="Whether this decision is eligible to be executed live.",
    )

    # Model ensemble / narrative
    model_votes: Dict[str, float] = Field(
        default_factory=dict,
        description="Key model scores/signals used in the decision.",
    )
    llm_headline: Optional[str] = Field(
        default=None,
        description="Condensed narrative from the LLM specialist.",
    )

    # RL summary (flattened for convenience)
    rl_action: Literal["LONG", "SHORT", "FLAT", "HOLD"] = Field(
        ...,
        description="Flattened RL action used in this decision.",
    )
    rl_mode: str = Field(
        ...,
        description="Execution agent mode, e.g. 'rl_live', 'disabled'.",
    )
    rl_target_position: float = Field(
        ...,
        description="Signed target position as fraction of equity (-1..1).",
    )
    rl_execution_style: str = Field(
        ...,
        description="Execution style hint chosen for this decision.",
    )

    # Full debug / audit payload
    debug: Dict[str, Any] = Field(
        default_factory=dict,
        description="Nested diagnostics: L2 prediction, LLM output, RL, risk, features.",
    )

    class Config:
        orm_mode = True


# =============================================================================
# (Optional) Backwards-compat ExpertSignals stub
# =============================================================================

class ExpertSignals(BaseModel):
    """
    Lightweight container if other parts of the code still import ExpertSignals.
    Not required by the new HybridInferenceService, but safe to keep.
    """
    trend: Optional[float] = None
    sentiment: Optional[float] = None
    onchain: Optional[float] = None
    options: Optional[float] = None
    macro: Optional[float] = None
    llm: Optional[float] = None

    class Config:
        orm_mode = True
