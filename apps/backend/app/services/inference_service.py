from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

# --- Internal Imports ---
from app.db.database import SessionLocal
from app.db import models

# --- THE BRIDGE: Import the modular ML components ---
from app.ml.model_engine import ModelEngine  # <--- Uses the robust ML/FeatureBuilder logic
from app.ml.adv.llm_narrative_model import llm_engine
from app.ml.adv.rl_execution_agent import rl_agent
from app.rt_adapt.cooldown_scheduler import should_skip, mark_ran
from app.core.config import settings
from app.hybrid.schemas import (
    MarketContext,
    Layer2Prediction,
    RLAction,
    HybridDecision,
)
from app.risk.engine import RiskEngine

logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_ACCOUNT_EQUITY = float(os.getenv("MARS_ACCOUNT_EQUITY_USD", 100000.0))

class HybridInferenceService:
    """
    Orchestrator enforcing the layer order:
      Data/AI Layer (ModelEngine) -> Meta Layer (LLM) -> Decision (RiskEngine) -> Execution (RL)
    """

    def __init__(self) -> None:
        # 1. Core AI Engine (TFT/TCN/TST/FeatureBuilder)
        self.model_engine = ModelEngine()

        # 2. Meta Layer (LLM)
        self.llm_engine = llm_engine

        # 3. Execution Layer (RL)
        self.rl_agent = rl_agent

        self._refresh_readiness()

    def _refresh_readiness(self) -> None:
        self.ai_ready = self.model_engine.is_ready
        self.llm_ready = self.llm_engine.is_model_loaded()
        self.rl_ready = self.rl_agent.is_model_loaded()
        self.is_ready = self.ai_ready  # Minimal requirement: AI must work

        logger.info(
            f"[HybridInferenceService] Status: AI={self.ai_ready}, LLM={self.llm_ready}, RL={self.rl_ready}"
        )

    def ready(self) -> bool:
        self._refresh_readiness()
        return self.is_ready

    def get_brain_health(self) -> dict:
        self._refresh_readiness()
        return {
            "is_ready": self.is_ready,
            "components": {
                "model_engine": self.ai_ready,
                "llm_engine": self.llm_ready,
                "rl_agent": self.rl_ready
            }
        }

    # ------------------------------------------------------------------ #
    # Main entrypoint
    # ------------------------------------------------------------------ #

    async def build_decision(self, ctx: MarketContext) -> HybridDecision:
        """
        Full Pipeline Execution:
        1. ModelEngine: Fetches features -> Runs Models -> Returns L2Prediction
        2. LLM: Fetches Narrative -> Returns Sentiment/Headline
        3. RL Agent: Observes L2 + Narrative -> Returns Optimal Action
        4. RiskEngine: Validates Action -> Calculates Size -> Returns Final Decision
        """
        symbol = (ctx.symbol or "BTCUSDT").upper()
        # 1. Safety Check: Are we in Cooldown?
        # If we lost money recently or volatility is insane, rt_adapt sets this flag.
        if should_skip(settings.CELERY_BROKER_URL, symbol, "inference"):
            logger.warning(f"[HybridInference] {symbol} is in COOLDOWN. Skipping.")
            return self._build_fallback_decision(ctx, "cooldown_active")
        # 2) AI LAYER (Delegated to ModelEngine)
        # This uses FeatureBuilder internally to get the correct tensor shapes
        try:
            l2_pred: Layer2Prediction = await self.model_engine.predict(ctx)
        except Exception as e:
            logger.error(f"[HybridInferenceService] AI Layer failed: {e}", exc_info=True)
            # Fallback to neutral prediction
            l2_pred = Layer2Prediction(
                asset=symbol,
                direction="flat",
                price_confidence=0.0,
                unified_vote=0.0
            )

        # 3) META LAYER (LLM)
        llm_headline = None
        llm_score = 0.0
        if self.llm_ready:
            try:
                # We run this async to not block if possible, but here we await for simplicity
                llm_res = await self.llm_engine.get_narrative_signal(symbol)
                llm_headline = llm_res.get("key_headline")
                llm_score = float(llm_res.get("sentiment_score", 0.0))
            except Exception as e:
                logger.warning(f"[HybridInferenceService] LLM Layer failed: {e}")

        # 4) PREPARE MODEL VOTES (For RL & Logging)
        model_votes = {
            "tft_visionary": l2_pred.tft_vote,
            "tcn_reflex": l2_pred.tcn_vote,
            "xgb_analyst": l2_pred.xgb_price_vote,
            "decision_net": l2_pred.decision_score,
            "llm_narrative": llm_score,
            "unified_vote": l2_pred.unified_vote
        }

        # 5) EXECUTION LAYER (RL Agent)
        # RL Agent decides "What should we do?" based on the signals
        if self.rl_ready:
            try:
                rl_action = self.rl_agent.get_optimal_action(
                    prediction=l2_pred,
                    context=ctx,
                    model_votes=model_votes
                )
            except Exception as e:
                logger.error(f"[HybridInferenceService] RL Layer failed: {e}")
                rl_action = self._get_fallback_rl_action()
        else:
            rl_action = self._get_fallback_rl_action()

        # 6) DECISION LAYER (Risk Engine)
        # Risk Engine decides "Can we actually do this?" and "How much?"
        risk_engine = RiskEngine(symbol=symbol)
        
        # We map RL Action to Risk Engine inputs
        base_direction = self._map_rl_action_to_direction(rl_action.optimal_action)
        
        # Retrieve OHLC data strictly for Risk calculations (ATR, etc)
        # Note: This is separate from Model features to keep concerns separated
        ohlc_df = risk_engine._fetch_ohlcv_for_atr(symbol) 

        (
            final_size,
            meta_execute,
            rl_target_pos,
            final_confidence,
            p_edge,
            risk_debug
        ) = risk_engine.apply(
            symbol=symbol,
            instrument_type=ctx.instrument_type,
            ohlc_df=ohlc_df,
            base_direction=base_direction,
            raw_size=rl_action.optimal_size_pct,
            llm_score=llm_score,
            model_votes=model_votes,
            ctx=ctx
        )

        # 7) PERSISTENCE & RETURN
        decision = HybridDecision(
            symbol=symbol,
            instrument_type=ctx.instrument_type,
            timestamp=datetime.utcnow(),
            direction=base_direction, # Final direction after risk checks
            p_edge=p_edge,
            confidence=final_confidence,
            size_factor=final_size,
            strategy_tag="nowa_hybrid_v2_synced",
            meta_execute=meta_execute,
            model_votes=model_votes,
            llm_headline=llm_headline,
            rl_action=rl_action.optimal_action,
            rl_mode=rl_action.mode or "unknown",
            rl_target_position=rl_target_pos,
            rl_execution_style=rl_action.execution_style,
            debug=risk_debug
        )
        
        self._persist_decision(decision)
        mark_ran(settings.CELERY_BROKER_URL, symbol, "inference")
        return decision

    def _get_fallback_rl_action(self) -> RLAction:
        return RLAction(
            optimal_action="HOLD",
            optimal_size_pct=0.0,
            execution_style="NONE",
            mode="disabled"
        )

    def _map_rl_action_to_direction(self, action: str) -> str:
        if action == "LONG": return "up"
        if action == "SHORT": return "down"
        return "flat"

    def _persist_decision(self, d: HybridDecision):
        """Fire-and-forget persistence to DB."""
        try:
            with SessionLocal() as db:
                # Map HybridDecision to your DB model (HybridSignal)
                # Ensure DB model schema matches this data
                signal = models.HybridSignal(
                    symbol=d.symbol,
                    instrument_type=d.instrument_type,
                    decided_action=d.rl_action,
                    decided_direction=d.direction,
                    decided_confidence=d.confidence,
                    size_factor=d.size_factor,
                    strategy_tag=d.strategy_tag,
                    meta_execute=d.meta_execute,
                    model_votes=d.model_votes,
                    llm_headline=d.llm_headline,
                    rl_action=d.rl_action,
                    rl_mode=d.rl_mode,
                    rl_target_position=d.rl_target_position,
                    debug_payload=d.debug,
                    created_at=d.timestamp
                )
                db.add(signal)
                db.commit()
        except Exception as e:
            logger.error(f"[HybridInferenceService] Persistence failed: {e}")

# Singleton
inference_service = HybridInferenceService()