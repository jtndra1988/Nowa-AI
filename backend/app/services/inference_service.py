# app/services/inference_service.py

import logging
from typing import Any, Dict

from app.hybrid.schemas import MarketContext, HybridDecision
from app.services.inference_service import (
    HybridInferenceService,
    HybridDecision as CoreHybridDecision,
)

logger = logging.getLogger(_name_)


class InferenceService:
    """
    Public-facing inference layer for Nowa.

    Wraps the advanced HybridInferenceService (Layer 1 + Layer 2 + Layer 3)
    and exposes a clean API used by FastAPI routes:

      - is_ready: quick health flag
      - predict(symbol): simple ensemble preview for /predict
      - build_decision(ctx): full hybrid decision for /hybrid-signal
    """

    def _init_(self) -> None:
        try:
            logger.info("[InferenceService] Initializing HybridInferenceService...")
            # Core engine: TFT + TCN + XGB + Options + Macro + LLM + RL
            self._core = HybridInferenceService(use_mock_models=True)
            self.is_ready: bool = True
            logger.info("[InferenceService] Initialization complete.")
        except Exception as e:  # noqa: BLE001
            logger.error(
                "[InferenceService] Failed to initialize core engine: %s",
                e,
                exc_info=True,
            )
            self._core = None
            self.is_ready = False

    # ------------------------------------------------------------------ #
    # /predict — simple, synchronous ensemble output
    # ------------------------------------------------------------------ #

    def predict(self, symbol: str) -> Dict[str, Any]:
        """
        Lightweight prediction for the /predict endpoint.

        Currently:
          - Uses core TFT/TCN/XGB from HybridInferenceService.
          - Returns price_prediction, volatility_prediction, feature_importance
            to match HybridPredictResponse.
        """
        if not self.is_ready or self._core is None:
            raise RuntimeError("InferenceService not ready")

        features = {
            "symbol": symbol,
            "mode": "futures",
            "exchange": "binance",
        }

        # Use underlying model wrappers directly (mock-safe if needed)
        try:
            tft = float(self._core.tft.predict(features))
        except Exception:  # noqa: BLE001
            tft = 0.0

        try:
            tcn = float(self._core.tcn.predict(features))
        except Exception:  # noqa: BLE001
            tcn = 0.0

        try:
            xgb = float(self._core.xgb.predict(features))
        except Exception:  # noqa: BLE001
            xgb = 0.0

        # Simple blended score
        ensemble_score = (tft + tcn + xgb) / 3.0

        # For demo:
        # - treat ensemble_score as directional price_prediction proxy
        # - volatility_prediction as absolute strength
        return {
            "price_prediction": ensemble_score,
            "volatility_prediction": abs(ensemble_score),
            "feature_importance": {
                "tft": tft,
                "tcn": tcn,
                "xgb": xgb,
            },
        }

    # ------------------------------------------------------------------ #
    # /hybrid-signal — full 3-layer decision
    # ------------------------------------------------------------------ #

    async def build_decision(self, ctx: MarketContext) -> HybridDecision:
        """
        Build a full hybrid decision (Layer 1 → Layer 2 → Layer 3).

        Steps:
          1) Ask core HybridInferenceService for a CoreHybridDecision.
          2) Wrap it into the public HybridDecision schema.
        """
        if not self.is_ready or self._core is None:
            raise RuntimeError("InferenceService not ready")

        # Call advanced engine (must be implemented in app/ml/adv/inference_service.py)
        core: CoreHybridDecision = await self._core.get_hybrid_decision(
            exchange=ctx.exchange,
            symbol=ctx.symbol,
            mode=ctx.mode,
        )

        payload = core.to_response()

        direction = payload.get("direction", "flat")
        confidence = float(payload.get("confidence", 0.0))
        model_votes = payload.get("model_votes", {}) or {}
        llm_headline = payload.get("llm_headline")
        rl_action = payload.get("rl_action")
        rl_mode = payload.get("rl_mode")
        rl_target = float(payload.get("rl_target_position", 0.0))

        # p_edge ~ confidence for now
        p_edge = confidence

        # If RL suggests a target, use that magnitude; otherwise use confidence
        if rl_target != 0.0:
            size_factor = min(1.0, max(0.0, abs(rl_target)))
        else:
            size_factor = min(1.0, max(0.0, confidence))

        strategy_tag = "nowa_hybrid_v1"
        meta_execute = True

        debug = {
            "core": payload,
        }

        return HybridDecision(
            symbol=ctx.symbol,
            instrument_type=ctx.instrument_type or "futures",
            direction=direction,
            p_edge=p_edge,
            confidence=confidence,
            size_factor=size_factor,
            strategy_tag=strategy_tag,
            meta_execute=meta_execute,
            model_votes=model_votes,
            llm_headline=llm_headline,
            rl_action=rl_action,
            rl_mode=rl_mode,
            rl_target_position=rl_target,
            debug=debug,
        )


# Singleton used by FastAPI routes
inference_service = InferenceService()