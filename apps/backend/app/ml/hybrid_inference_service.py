import logging
from typing import Any, Dict, Optional

from app.hybrid.schemas import (
    MarketContext,
    Layer2Prediction,
    HybridDecision,
)
from app.ml.adv.model_registry import model_registry
# L2 ensemble – you already have this in your backend
from app.ml.model_engine import ModelEngine

# LLM narrative specialist (Gemini live)
from app.ml.adv.llm_narrative_model import llm_engine

# RL execution agent (loads model_artifacts/rl_agent_ppo.zip internally)
from app.ml.adv.rl_execution_agent import rl_agent
from app.hybrid.schemas import MarketContext, HybridDecision
# Options & Macro/On-chain specialists
from app.ml.adv.options_vol_model import options_vol_edge
from app.ml.adv.macro_onchain_model import macro_onchain_bias

logger = logging.getLogger(__name__)


class HybridInferenceService:
    """
    Orchestrates your full stack:

      • Data / Features → (inside ModelEngine)
      • L2: ModelEngine (TFT / TCN / XGB / DecisionNet / etc.)
      • L3: LLM narrative specialist + RL agent (PPO policy)
      • L4: Wrap everything into a HybridDecision object
    """

    def __init__(self) -> None:
        # L2 engine still does feature orchestration + ensemble
        self.model_engine = ModelEngine()

        # Use central registry for LLM & RL where possible
        self.registry = model_registry

        # Prefer registry models; fall back to direct singletons
        self.llm_engine = (
            self.registry.get_model("llm") if self.registry else llm_engine
        )
        self.rl_agent = (
            self.registry.get_model("rl") if self.registry else rl_agent
        )


    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """
        System is 'ready' only if:
          - ModelEngine reports ready (or has no flag and just exists)
          - LLM narrative engine has GEMINI_API_KEY set
          - RL agent has successfully loaded rl_agent_ppo.zip
        """
        ready_l2 = getattr(self.model_engine, "is_ready", True)
        ready_llm = getattr(self.llm_engine, "is_model_loaded", lambda: False)()
        ready_rl = getattr(self.rl_agent, "is_model_loaded", lambda: False)()

        if not (ready_l2 and ready_llm and ready_rl):
            logger.warning(
                "[HybridInferenceService] Not fully ready: "
                "L2=%s, LLM=%s, RL=%s",
                ready_l2,
                ready_llm,
                ready_rl,
            )
        return bool(ready_l2 and ready_llm and ready_rl)

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    async def build_decision(self, ctx: MarketContext) -> HybridDecision:
        # 1) Get Layer-2 prediction (the “brain votes”)
        layer2 = await self.model_engine.predict(ctx)

        # 2) Build model_votes dict for RLAgent from Layer2Prediction
        model_votes = {
            "tft_visionary": layer2.tft_vote,
            "tcn_reflex": layer2.tcn_vote,
            "xgb_analyst": layer2.xgb_price_vote,
            "xgb_vol_analyst": layer2.xgb_vol_vote,
            "options_psychologist": layer2.options_score,
            "macro_economist": layer2.macro_score,
            "llm_narrative": layer2.llm_narrative_vote,
        }

        # 3) Ask RLAgent for optimal action
        rl_action = self.rl_agent.get_optimal_action(
            prediction=layer2,
            context=ctx,
            model_votes=model_votes,
        )

        # 4) Build your final HybridDecision object however you’ve defined it
        decision = HybridDecision(
            asset=layer2.asset,
            unified_vote=layer2.unified_vote,
            rl_action=rl_action,
            layer2=layer2,
        )
        return decision

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _get_layer2_prediction(self, ctx: MarketContext) -> Layer2Prediction:
        """
        Call your L2 ensemble via ModelEngine.
        Expected to return a Layer2Prediction Pydantic model.
        """
        if not hasattr(self.model_engine, "predict"):
            raise RuntimeError(
                "[HybridInferenceService] ModelEngine has no predict() method."
            )

        pred = await self.model_engine.predict(ctx)  # type: ignore[func-returns-value]

        if not isinstance(pred, Layer2Prediction):
            raise RuntimeError(
                "[HybridInferenceService] ModelEngine.predict() did not "
                "return Layer2Prediction."
            )

        return pred

    async def _get_narrative_signal(self, asset: str) -> Optional[Dict[str, Any]]:
        """
        Calls LIVE Gemini LLM via llm_engine.get_narrative_signal(asset).

        Expected output shape:
          {
            "sentiment_score": float (-1..1),
            "key_headline": str,
          }

        If LLM is disabled/misconfigured, returns None (treated as neutral).
        """
        if not hasattr(self.llm_engine, "is_model_loaded"):
            logger.error(
                "[HybridInferenceService] llm_engine has no is_model_loaded()"
            )
            return None

        if not self.llm_engine.is_model_loaded():
            logger.warning(
                "[HybridInferenceService] LLM narrative engine disabled; "
                "using neutral narrative."
            )
            return None

        try:
            signal = await self.llm_engine.get_narrative_signal(asset)
            return signal
        except Exception as e:  # noqa: BLE001
            logger.error(
                "[HybridInferenceService] LLM narrative call failed: %s",
                e,
                exc_info=True,
            )
            return None

    def _build_model_votes(
        self,
        pred: Layer2Prediction,
        narrative: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Build model_votes used by RLAgent._build_observation().

        RLAgent's FEATURE_COLUMNS require:
          - 'tft_visionary'
          - 'tcn_reflex'
          - 'xgb_analyst'
          - 'xgb_vol_analyst'
          - 'options_psychologist'
          - 'macro_economist'
          - 'llm_narrative'
        """

        # Map from your Layer2Prediction fields to votes.
        # Adjust these getattr names to match your actual schema.
        tft_vote = float(getattr(pred, "tft_vote", 0.0) or 0.0)
        tcn_vote = float(getattr(pred, "tcn_vote", 0.0) or 0.0)
        xgb_price_vote = float(getattr(pred, "xgb_price_vote", 0.0) or 0.0)
        xgb_vol_vote = float(getattr(pred, "xgb_vol_vote", 0.0) or 0.0)

        # Options & Macro features
        options_features = getattr(pred, "options_features", {}) or {}
        macro_features = getattr(pred, "macro_onchain_features", {}) or {}

        try:
            options_score = options_vol_edge(options_features)
        except Exception:
            options_score = 0.0

        try:
            macro_score = macro_onchain_bias(macro_features)
        except Exception:
            macro_score = 0.0

        # LLM narrative score
        llm_score = float(
            narrative.get("sentiment_score", 0.0)
        ) if narrative else 0.0

        model_votes: Dict[str, float] = {
            "tft_visionary": tft_vote,
            "tcn_reflex": tcn_vote,
            "xgb_analyst": xgb_price_vote,
            "xgb_vol_analyst": xgb_vol_vote,
            "options_psychologist": options_score,
            "macro_economist": macro_score,
            "llm_narrative": llm_score,
        }

        return model_votes

    def _get_rl_action(
        self,
        pred: Layer2Prediction,
        ctx: MarketContext,
        model_votes: Dict[str, float],
    ):
        """
        Ask RLAgent for an optimal action.

        RLAgent internally:
          - loads 'model_artifacts/rl_agent_ppo.zip'
          - constructs the observation vector using:
              - decisionnet_confidence (from prediction.price_confidence)
              - decisionnet_direction (from prediction.direction)
              - the model_votes we supply here
              - context.current_position_size
        """
        if not hasattr(self.rl_agent, "is_model_loaded") or not hasattr(
            self.rl_agent, "get_optimal_action"
        ):
            logger.error(
                "[HybridInferenceService] rl_agent missing required methods."
            )
            return None

        if not self.rl_agent.is_model_loaded():
            logger.warning(
                "[HybridInferenceService] RL policy disabled; will not "
                "override size/direction."
            )
            return None

        try:
            action = self.rl_agent.get_optimal_action(
                prediction=pred,
                context=ctx,
                model_votes=model_votes,
            )
            return action
        except Exception as e:  # noqa: BLE001
            logger.error(
                "[HybridInferenceService] RLAgent.get_optimal_action failed: %s",
                e,
                exc_info=True,
            )
            return None

    def _build_hybrid_decision(
        self,
        ctx: MarketContext,
        layer2_pred: Layer2Prediction,
        narrative: Optional[Dict[str, Any]],
        model_votes: Dict[str, float],
        rl_action: Any,
    ) -> HybridDecision:
        """
        Final packaging into HybridDecision Pydantic object.

        You may need to tweak field names here to match your actual schema.
        Assumes HybridDecision has at least:

          - market_context
          - layer2_prediction
          - narrative_score
          - narrative_headline
          - model_votes
          - rl_action
        """
        narrative_score = float(
            narrative.get("sentiment_score", 0.0)
        ) if narrative else 0.0
        narrative_headline = (
            str(narrative.get("key_headline", ""))
            if narrative
            else ""
        )

        decision = HybridDecision(
            market_context=ctx,
            layer2_prediction=layer2_pred,
            narrative_score=narrative_score,
            narrative_headline=narrative_headline,
            model_votes=model_votes,
            rl_action=rl_action,
        )
        return decision
