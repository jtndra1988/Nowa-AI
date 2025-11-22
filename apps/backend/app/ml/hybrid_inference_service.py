import logging
from datetime import datetime
from typing import Any, Dict

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

# Meta components
from app.hybrid.bandit import bandit_weights
from app.hybrid.metalabel import metalabel_decide

logger = logging.getLogger(__name__)


class HybridInferenceService:
    """
    Orchestrates full stack:
      Data -> AI (L2) -> Meta (Bandit/LLM) -> Decision (MetaLabel) -> Execution (RL)

    This class is agnostic to the concrete feature engineering details, but it
    expects that `ModelEngine.predict(...)` returns a `Layer2Prediction` whose
    `meta` dictionary may contain tabular features built via the new
    feature_builder (price + sentiment, etc.).

    Concretely, if the XGB / macro / options specialists use
    join_sentiment_features, we expect keys like:

        - 'news_score'
        - 'social_score'
        - 'global_score'
        - 'composite_score'

    plus any price/vol/funding features. These are then consumed by the
    bandit and meta-label logic.
    """

    def __init__(self) -> None:
        # L2 model ensemble (TFT/TCN/XGB/etc.)
        self.model_engine = ModelEngine()

        # Model registry allows swapping LLM / RL at runtime
        self.registry = model_registry

        self.llm_engine = (
            self.registry.get_model("llm") if self.registry else llm_engine
        )
        self.rl_agent = (
            self.registry.get_model("rl") if self.registry else rl_agent
        )

    @property
    def is_ready(self) -> bool:
        """
        System is considered 'ready' if the L2 ensemble is ready.
        L3/L4 (LLM, RL) are optional and may be disabled.
        """
        ready_l2 = getattr(self.model_engine, "is_ready", True)
        return bool(ready_l2)

    async def build_decision(self, ctx: MarketContext) -> HybridDecision:
        """
        High-level orchestrator used by /hybrid-signal.

        Flow:
          1) L2 ensemble prediction (TFT/TCN/XGB/etc.)
          2) Bandit regime selection (strategy_tag)
          3) Meta-label sizing/filtering (size_factor, meta_execute)
          4) RL execution policy (RLAction)
          5) HybridDecision assembly
        """

        # 1) Get Layer-2 prediction (The AI Vote)
        layer2: Layer2Prediction = await self.model_engine.predict(ctx)

        # 2) Extract features for meta logic
        #    Prefer the full tabular feature vector attached by ModelEngine.
        raw_features: Dict[str, Any] = layer2.meta or {}

        # If raw_features is empty, create a minimal fallback set so bandit / metalabel
        # can still run without crashing.
        if not raw_features:
            raw_features = {
                "roll_vol_24h": 0.02,    # ~2% hourly vol as a neutral default
                "funding_rate": 0.0001,  # near-flat funding
            }

        # Ensure sentiment-related keys exist (0.0 = neutral sentiment).
        # These will be filled by feature_builder-based pipelines in ModelEngine
        # when available, but we don't want meta logic to crash if they're missing.
        for key in ("news_score", "social_score", "global_score", "composite_score"):
            raw_features.setdefault(key, 0.0)

        # Build a small meta context for higher layers
        meta_context: Dict[str, Any] = {
            "confidence": layer2.price_confidence,
            "p_edge": layer2.unified_vote,
        }

        # 3) Run Bandit (Regime / Strategy selection)
        #    Returns updated weights + a human-readable strategy tag
        _, strategy_tag = bandit_weights(
            features=raw_features,
            expert=ExpertSignals(),  # Currently unused placeholder
            meta=meta_context,
            ctx=ctx,
        )

        # 4) Run Meta-Labeling (Sizing & Filtering)
        #    Decides size_factor and whether we actually execute the trade.
        meta_decision: Dict[str, Any] = metalabel_decide(
            features=raw_features,
            expert=ExpertSignals(),
            meta=meta_context,
            ctx=ctx,
        )

        size_factor: float = float(meta_decision.get("size_factor", 1.0))
        meta_execute: bool = bool(meta_decision.get("execute", True))
        meta_reason: str = str(meta_decision.get("reason", "default"))

        # 5) Run RL Agent (Execution policy)
        #    We pass in model_votes + strategy tag so RL knows the regime.
        model_votes: Dict[str, float] = {
            "tft_visionary": layer2.tft_vote,
            "tcn_reflex": layer2.tcn_vote,
            "xgb_analyst": layer2.xgb_price_vote,
            "xgb_vol_analyst": layer2.xgb_vol_vote,
            "options_psychologist": layer2.options_score,
            "macro_economist": layer2.macro_score,
            "llm_narrative": layer2.llm_narrative_vote,
            # Simple encoding: 1.0 if we are in a trend-following regime, else 0.0
            "regime_tag": 1.0 if "trend" in (strategy_tag or "").lower() else 0.0,
        }

        # Default RL action if the policy is missing or fails
        rl_action = RLAction(
            optimal_action="HOLD",
            optimal_size_pct=0.0,
            execution_style="NONE",
        )

        if self.rl_agent and getattr(self.rl_agent, "is_model_loaded", lambda: False)():
            try:
                rl_action = self.rl_agent.get_optimal_action(
                    prediction=layer2,
                    context=ctx,
                    model_votes=model_votes,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[HybridInference] RL Agent failed: %s", exc)

        # 6) Construct Final Decision
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
            llm_headline="",  # Can be populated by LLM narrative if needed

            # Execution
            rl_action=rl_action.optimal_action,
            rl_mode=getattr(rl_action, "mode", None) or "disabled",
            rl_target_position=0.0,  # Typically filled by RiskEngine
            rl_execution_style=rl_action.execution_style,

            debug={
                "meta_reason": meta_reason,
                "layer2_votes": model_votes,
                "raw_features_keys": list(raw_features.keys()),
            },
        )

        return decision
