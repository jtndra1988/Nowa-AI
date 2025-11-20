import logging
from typing import Any, Dict, Optional

from app.hybrid.schemas import (
    MarketContext,
    Layer2Prediction,
    HybridDecision,
    ExpertSignals,  # Imported for type safety
    RLAction, 
)

# L2 ensemble
from app.ml.model_engine import ModelEngine

# L3/L4 components
from app.ml.adv.llm_narrative_model import llm_engine
from app.ml.adv.rl_execution_agent import rl_agent
from app.ml.adv.model_registry import model_registry

# --- NEW: Import the Bandit and MetaLabel modules ---
from app.hybrid.bandit import bandit_weights
from app.hybrid.metalabel import metalabel_decide
# ---------------------------------------------------

logger = logging.getLogger(__name__)


class HybridInferenceService:
    """
    Orchestrates full stack:
      Data -> AI (L2) -> Meta (Bandit/LLM) -> Decision (MetaLabel) -> Execution (RL)
    """

    def __init__(self) -> None:
        self.model_engine = ModelEngine()
        self.registry = model_registry
        
        self.llm_engine = (
            self.registry.get_model("llm") if self.registry else llm_engine
        )
        self.rl_agent = (
            self.registry.get_model("rl") if self.registry else rl_agent
        )

    @property
    def is_ready(self) -> bool:
        ready_l2 = getattr(self.model_engine, "is_ready", True)
        return bool(ready_l2)

    async def build_decision(self, ctx: MarketContext) -> HybridDecision:
        """
        High-level orchestrator used by /hybrid-signal.
        """
        # 1) Get Layer-2 prediction (The AI Vote)
        layer2: Layer2Prediction = await self.model_engine.predict(ctx)

        # --- NEW: Extract Features for Meta Logic ---
        # We attempt to pull raw features if ModelEngine attached them to meta
        # Otherwise we rely on what's in Layer2Prediction
        raw_features = layer2.meta if layer2.meta else {}
        
        # If raw_features is empty, we create a minimal fallback set
        if not raw_features:
             raw_features = {
                 "roll_vol_24h": 0.02, # Default ~2% vol
                 "funding_rate": 0.0001
             }

        # 2) Run Bandit (Regime Selection)
        # This decides the "Strategy Tag" (e.g. "trend_following" vs "mean_reversion")
        # based on the market conditions found in `raw_features`.
        meta_context = {
            "confidence": layer2.price_confidence,
            "p_edge": layer2.unified_vote
        }
        
        _, strategy_tag = bandit_weights(
            features=raw_features,
            expert=ExpertSignals(), # Empty placeholder
            meta=meta_context,
            ctx=ctx
        )

        # 3) Run Meta-Labeling (Sizing & Filtering)
        # This decides "Size Factor" and "Meta Execute" (Yes/No)
        meta_decision = metalabel_decide(
            features=raw_features,
            expert=ExpertSignals(),
            meta=meta_context,
            ctx=ctx
        )
        
        size_factor = meta_decision.get("size_factor", 1.0)
        meta_execute = meta_decision.get("execute", True)
        meta_reason = meta_decision.get("reason", "default")

        # 4) Run RL Agent (Execution)
        # Pass the regime/strategy info to RL so it knows the context
        model_votes = {
            "tft_visionary": layer2.tft_vote,
            "tcn_reflex": layer2.tcn_vote,
            "xgb_analyst": layer2.xgb_price_vote,
            "xgb_vol_analyst": layer2.xgb_vol_vote,
            "options_psychologist": layer2.options_score,
            "macro_economist": layer2.macro_score,
            "llm_narrative": layer2.llm_narrative_vote,
            "regime_tag": 1.0 if "trend" in strategy_tag else 0.0 # Simple encoding
        }

        rl_action = RLAction(
            optimal_action="HOLD", 
            optimal_size_pct=0.0, 
            execution_style="NONE"
        )
        
        if self.rl_agent and self.rl_agent.is_model_loaded():
            try:
                rl_action = self.rl_agent.get_optimal_action(
                    prediction=layer2,
                    context=ctx,
                    model_votes=model_votes,
                )
            except Exception as e:
                logger.warning(f"[HybridInference] RL Agent failed: {e}")

        # 5) Construct Final Decision
        # We inject the calculated strategy_tag, size_factor, and meta_execute
        decision = HybridDecision(
            symbol=layer2.asset,
            instrument_type=ctx.instrument_type,
            timestamp=datetime.utcnow(),
            
            # Direction & Confidence from L2
            direction=layer2.direction,
            p_edge=layer2.unified_vote,
            confidence=layer2.price_confidence,
            
            # Meta-derived fields
            strategy_tag=strategy_tag,
            size_factor=size_factor,
            meta_execute=meta_execute,
            
            # Context
            model_votes=model_votes,
            llm_headline="", # Can fetch from LLM if needed
            
            # Execution
            rl_action=rl_action.optimal_action,
            rl_mode=rl_action.mode or "disabled",
            rl_target_position=0.0, # Calculated by RiskEngine usually
            rl_execution_style=rl_action.execution_style,
            
            debug={
                "meta_reason": meta_reason,
                "layer2_votes": model_votes
            }
        )
        
        return decision
        
from datetime import datetime