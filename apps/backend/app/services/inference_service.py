from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

try:
    import joblib
except ImportError:  # safe-guard
    joblib = None

try:
    import torch
except ImportError:  # safe-guard
    torch = None

# --- Internal Imports ---
from app.db.database import SessionLocal
from app.db import models
from app.ml.adv.llm_narrative_model import llm_engine
from app.ml.adv.rl_execution_agent import rl_agent

# --- Advanced Model Imports ---
from app.ml.adv.models_tft import TemporalFusionTransformer
from app.ml.adv.models_tcn import TemporalConvNet

from app.hybrid.schemas import (
    MarketContext,
    Layer2Prediction,
    RLAction,
    HybridDecision,
)
from app.risk.engine import RiskEngine

logger = logging.getLogger(__name__)

# --- Configuration ---
SEQ_LEN_DEFAULT = 64
ARTIFACT_DIR = os.getenv("MARS_ARTIFACT_DIR", "model_artifacts")
ALLOW_FEATURE_MOCK = os.getenv("MARS_ALLOW_FEATURE_MOCK", "false").lower() == "true"
DEFAULT_ACCOUNT_EQUITY = float(os.getenv("MARS_ACCOUNT_EQUITY_USD", 100000.0))

DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"


# =============================================================================
# ModelEngine (AI Layer: TFT / TCN / XGB ensemble)
# =============================================================================


class ModelEngine:
    """
    L2 ensemble over trained TFT / TCN / XGB models + tabular signals.

    Behavior:
      - If at least one of TFT / TCN / XGB artifacts loads -> use ensemble.
      - If none load:
          - If MARS_ALLOW_L2_FALLBACK=true  -> use deterministic rule-based logic.
          - If MARS_ALLOW_L2_FALLBACK=false -> is_ready=False.
    """

    def __init__(self) -> None:
        artifact_dir = Path(os.getenv("MARS_ARTIFACT_DIR", ARTIFACT_DIR))

        self.allow_fallback = (
            os.getenv("MARS_ALLOW_L2_FALLBACK", "true").lower() == "true"
        )

        # Artifact paths
        self.tft_path = Path(
            os.getenv("MARS_TFT_MODEL_PATH", artifact_dir / "tft_model.pt")
        )
        self.tcn_path = Path(
            os.getenv("MARS_TCN_MODEL_PATH", artifact_dir / "tcn_model.pt")
        )
        self.xgb_path = Path(
            os.getenv("MARS_XGB_MODEL_PATH", artifact_dir / "xgb_price_model.joblib")
        )

        # Loaded models
        self.tft_model = self._load_torch_model(self.tft_path, "TFT")
        self.tcn_model = self._load_torch_model(self.tcn_path, "TCN")
        self.xgb_model = self._load_xgb_model(self.xgb_path, "XGB")

        self.has_real_models = any(
            [
                self.tft_model is not None,
                self.tcn_model is not None,
                self.xgb_model is not None,
            ]
        )

        if self.has_real_models:
            self.is_ready = True
            logger.info(
                "[ModelEngine] Loaded L2 ensemble models: "
                f"TFT={'Y' if self.tft_model else 'N'}, "
                f"TCN={'Y' if self.tcn_model else 'N'}, "
                f"XGB={'Y' if self.xgb_model else 'N'}"
            )
        else:
            if self.allow_fallback:
                self.is_ready = True
                logger.warning(
                    "[ModelEngine] No TFT/TCN/XGB artifacts found. "
                    "Using deterministic rule-based fallback."
                )
            else:
                self.is_ready = False
                logger.error(
                    "[ModelEngine] No TFT/TCN/XGB artifacts found and "
                    "MARS_ALLOW_L2_FALLBACK is false. L2 is NOT READY."
                )

    # -------------------------------------------------------------------------
    # Public API (used by HybridInferenceService)
    # -------------------------------------------------------------------------

    def build_layer2_prediction(
        self,
        symbol: str,
        seq_features: np.ndarray,
        tab_features: Dict[str, float],
    ) -> Layer2Prediction:
        """
        Compute the L2 prediction using either:
          - real TFT/TCN/XGB ensemble, or
          - rule-based fallback if allowed.
        """
        if not self.is_ready:
            raise RuntimeError("[ModelEngine] Called while not ready.")

        if seq_features.size == 0 or seq_features.shape[0] < 10:
            raise RuntimeError(
                "[ModelEngine] Not enough sequential data for prediction."
            )

        if self.has_real_models:
            return self._predict_with_ensemble(symbol, seq_features, tab_features)
        else:
            return self._rule_based_fallback(symbol, seq_features, tab_features)

    # -------------------------------------------------------------------------
    # Loaders
    # -------------------------------------------------------------------------

    def _load_torch_model(self, path: Path, name: str):
        if not path or not Path(path).exists():
            return None
        if torch is None:
            logger.error(
                f"[ModelEngine] {name} artifact present at {path}, but torch not installed."
            )
            return None
        try:
            model = TemporalFusionTransformer.load_from_checkpoint(
                str(path), map_location=DEVICE
            )
            model.eval()
            logger.info(f"[ModelEngine] Loaded {name} from {path}")
            return model
        except Exception as e:
            logger.error(
                f"[ModelEngine] Failed to load {name} from {path}: {e}",
                exc_info=True,
            )
            return None

    def _load_xgb_model(self, path: Path, name: str):
        if not path or not Path(path).exists():
            return None
        if joblib is None:
            logger.error(
                f"[ModelEngine] {name} artifact present at {path}, but joblib not installed."
            )
            return None
        try:
            model = joblib.load(path)
            logger.info(f"[ModelEngine] Loaded {name} from {path}")
            return model
        except Exception as e:
            logger.error(
                f"[ModelEngine] Failed to load {name} from {path}: {e}",
                exc_info=True,
            )
            return None

    # -------------------------------------------------------------------------
    # Ensemble prediction (real models)
    # -------------------------------------------------------------------------

    def _predict_with_ensemble(
        self,
        symbol: str,
        seq_features: np.ndarray,
        tab_features: Dict[str, float],
    ) -> Layer2Prediction:
        # Placeholder ensemble logic - replace with production logic.
        closes = seq_features[:, 0]
        last_close = float(closes[-1])

        ma = float(closes.mean())
        direction = "up" if last_close > ma else "down" if last_close < ma else "flat"
        price_confidence = min(1.0, abs(last_close - ma) / max(ma, 1e-8))

        return Layer2Prediction(
            symbol=symbol,
            direction=direction,
            price_confidence=round(price_confidence, 3),
            raw_outputs={
                "last_close": last_close,
                "ma": ma,
                "has_tft": bool(self.tft_model),
                "has_tcn": bool(self.tcn_model),
                "has_xgb": bool(self.xgb_model),
            },
        )

    # -------------------------------------------------------------------------
    # Fallback (rule-based) logic
    # -------------------------------------------------------------------------

    def _rule_based_fallback(
        self,
        symbol: str,
        seq_features: np.ndarray,
        tab_features: Dict[str, float],
    ) -> Layer2Prediction:
        closes = seq_features[:, 0]
        rets = np.diff(closes) / closes[:-1]

        last_close = float(closes[-1])
        ma_10 = float(closes[-10:].mean()) if len(closes) >= 10 else last_close
        ma_20 = float(closes[-20:].mean()) if len(closes) >= 20 else ma_10
        vol_20 = float(rets[-20:].std()) if len(rets) >= 20 else float(
            rets.std() or 0.0
        )

        feats = [
            last_close,
            last_close / (ma_10 + 1e-8) - 1.0,
            last_close / (ma_20 + 1e-8) - 1.0,
            vol_20,
            float(tab_features.get("final_sentiment", 0.0)),
            float(tab_features.get("put_call_oi_ratio", 1.0)),
            float(tab_features.get("whale_volume_usd", 0.0)),
            float(tab_features.get("exchange_net_flow_usd", 0.0)),
            float(tab_features.get("funding_rate", 0.0)),
            float(tab_features.get("fear_greed_global", 0.0)),
            float(tab_features.get("ob_bid_ask_imb", 0.0)),
            float(tab_features.get("ob_vw_price_skew", 0.0)),
            float(tab_features.get("ob_cdv_1m", 0.0)),
            float(tab_features.get("ob_liquidity", 0.0)),
        ]

        score = float(np.tanh(sum(feats) / (len(feats) + 1e-8)))
        if score > 0.1:
            direction = "up"
        elif score < -0.1:
            direction = "down"
        else:
            direction = "flat"

        return Layer2Prediction(
            symbol=symbol,
            direction=direction,
            price_confidence=float(round(abs(score), 3)),
            raw_outputs={"rule_score": score, "used_fallback": True},
        )


# =============================================================================
# HybridInferenceService (One Brain: AI -> Meta -> Decision -> Execution)
# =============================================================================


class HybridInferenceService:
    """
    Orchestrator enforcing the layer order:

      Data Layer ->
      AI Layer (TFT/TCN/XGB via ModelEngine) ->
      Meta Layer (LLM narrative / veto / soft signal) ->
      Decision Layer (risk-aware fused direction & size) ->
      Execution Layer (optional RL execution policy)

    Key point:
      The Decision Layer consumes the *meta-fused* AI signal.
      AI does NOT bypass Meta/Decision to directly force trades.
    """

    def __init__(self) -> None:
        # AI Layer
        self.model_engine = ModelEngine()

        # Meta Layer (LLM) - treated as required for "ready"
        self.llm_engine = llm_engine

        # Execution Layer (RL) - OPTIONAL
        try:
            self.rl_agent = rl_agent
        except Exception as e:
            logger.warning(
                "[HybridInferenceService] RL agent init failed; continuing without RL. %s",
                e,
            )
            self.rl_agent = None

        self.allow_mock = ALLOW_FEATURE_MOCK

        self._refresh_readiness()

        if self.is_ready:
            logger.info(
                "[HybridInferenceService] READY: has_l2_models=%s llm_ready=%s rl_ready=%s",
                self.has_l2_models,
                self.llm_ready,
                self.rl_ready,
            )
        else:
            logger.error(
                "[HybridInferenceService] NOT READY: has_l2_models=%s llm_ready=%s (rl_ready=%s is optional)",
                self.has_l2_models,
                self.llm_ready,
                self.rl_ready,
            )

    # ------------------------------------------------------------------ #
    # Readiness / health
    # ------------------------------------------------------------------ #

    def _refresh_readiness(self) -> None:
        self.has_l2_models = bool(
            getattr(self.model_engine, "has_real_models", False)
            or getattr(self.model_engine, "is_ready", False)
        )

        self.llm_ready = bool(
            hasattr(self.llm_engine, "is_model_loaded")
            and self.llm_engine.is_model_loaded()
        )

        self.rl_ready = bool(
            self.rl_agent
            and hasattr(self.rl_agent, "is_model_loaded")
            and self.rl_agent.is_model_loaded()
        )

        self.risk_ready = True
        self.is_ready = bool(self.has_l2_models and self.llm_ready and self.risk_ready)

    def ready(self) -> bool:
        self._refresh_readiness()
        return self.is_ready

    def get_brain_health(self) -> dict:
        self._refresh_readiness()
        return {
            "is_ready": bool(self.is_ready),
            "has_l2_models": bool(self.has_l2_models),
            "llm_ready": bool(self.llm_ready),
            "rl_ready": bool(self.rl_ready),
            "risk_ready": bool(self.risk_ready),
        }

    # ------------------------------------------------------------------ #
    # Main entrypoint for /hybrid-signal (ONE pipeline)
    # ------------------------------------------------------------------ #

    async def build_decision(self, ctx: MarketContext) -> HybridDecision:
        if not self.is_ready:
            raise RuntimeError(
                "[HybridInferenceService] Called while not ready. "
                "Verify LLM and ModelEngine initialization."
            )

        symbol = (ctx.symbol or "BTCUSDT").upper()
        instrument_type = (ctx.instrument_type or "futures").lower()

        # 1) DATA LAYER -> Features
        seq, tab, ohlc_df = self._fetch_features(symbol, SEQ_LEN_DEFAULT)

        # 2) AI LAYER -> L2 Prediction (TFT/TCN/XGB ensemble or fallback)
        l2_pred = self.model_engine.build_layer2_prediction(
            symbol=symbol,
            seq_features=seq,
            tab_features=tab,
        )

        # 3) META LAYER -> LLM narrative & soft score (operating on AI decision)
        llm_headline = None
        llm_score = 0.0
        if self.llm_ready:
            try:
                llm_result = self.llm_engine.get_narrative_signal(
                    symbol=symbol,
                    l2_pred=l2_pred,        # AI layer output as primary input
                    tab_features=tab,
                )
                llm_headline = llm_result.get("headline")
                llm_score = float(llm_result.get("score", 0.0))
            except Exception as e:
                logger.warning(
                    "[HybridInferenceService] LLM narrative failed; continuing without. %s",
                    e,
                )

        # 4) DECISION SEED (AI + META FUSED)
        #    This is the key change:
        #    - Decision layer starts from AI output
        #    - Then is modulated by Meta (LLM) – no independent, parallel path.
        decision_direction = l2_pred.direction
        decision_confidence = float(l2_pred.price_confidence)

        # Simple but explicit fusion example:
        # - If LLM strongly contradicts ensemble -> neutralize / dampen.
        try:
            if llm_score is not None:
                if l2_pred.direction == "up" and llm_score < -0.4:
                    decision_direction = "flat"
                    decision_confidence *= 0.5
                elif l2_pred.direction == "down" and llm_score > 0.4:
                    decision_direction = "flat"
                    decision_confidence *= 0.5
        except Exception as e:
            logger.warning(
                "[HybridInferenceService] Decision fusion failed; using raw L2. %s",
                e,
            )
            decision_direction = l2_pred.direction
            decision_confidence = float(l2_pred.price_confidence)

        # 5) MODEL VOTES SNAPSHOT (for UI / logs)
        model_votes: Dict[str, Any] = {
            # Raw AI layer
            "ensemble_direction": l2_pred.direction,
            "ensemble_confidence": float(l2_pred.price_confidence),
            # Meta layer
            "llm_score": float(llm_score),
            "llm_headline": llm_headline,
            # Fused decision seed (Decision layer input)
            "decision_seed_direction": decision_direction,
            "decision_seed_confidence": decision_confidence,
            # Extra tab signals
            "sentiment": float(tab.get("final_sentiment", 0.0)),
            "put_call_oi_ratio": float(tab.get("put_call_oi_ratio", 1.0)),
        }

        # 6) EXECUTION LAYER (RL) - OPTIONAL
        # RL is treated as execution policy: it consumes the fused decision seed
        # (via model_votes), but cannot bypass RiskEngine constraints.
        if self.rl_ready:
            try:
                rl_action: RLAction = self.rl_agent.get_optimal_action(
                    prediction=l2_pred,      # kept for compatibility
                    context=ctx,
                    model_votes=model_votes, # includes meta-fused decision seed
                )
            except Exception as e:
                logger.warning(
                    "[HybridInferenceService] RL action failed; using neutral RLAction. %s",
                    e,
                )
                rl_action = RLAction(
                    optimal_action="HOLD",
                    optimal_size_pct=0.0,
                    execution_style="NONE",
                    mode="disabled",
                )
        else:
            rl_action = RLAction(
                optimal_action="HOLD",
                optimal_size_pct=0.0,
                execution_style="NONE",
                mode="disabled",
            )

        # 7) DECISION LAYER -> Risk, persistence, final HybridDecision
        decision = self._risk_aware_persist_and_build_decision(
            symbol=symbol,
            instrument_type=instrument_type,
            ctx=ctx,
            l2_pred=l2_pred,
            decision_direction=decision_direction,
            decision_confidence=decision_confidence,
            llm_headline=llm_headline,
            llm_score=llm_score,
            rl_action=rl_action,
            model_votes=model_votes,
            tab_features=tab,
            ohlc_df=ohlc_df,
        )
        return decision

    # ------------------------------------------------------------------ #
    # Feature extraction (Data Layer)
    # ------------------------------------------------------------------ #

    def _fetch_features(
        self,
        symbol: str,
        seq_len: int,
    ) -> Tuple[np.ndarray, Dict[str, float], pd.DataFrame]:
        if self.allow_mock:
            closes = np.linspace(50000, 50500, seq_len, dtype=np.float32)
            highs = closes * 1.001
            lows = closes * 0.999
            seq = np.stack([closes, highs, lows], axis=1)
            tab = {"final_sentiment": 0.0, "put_call_oi_ratio": 1.0}
            ohlc_df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
            return seq, tab, ohlc_df

        with SessionLocal() as db:
            # (unchanged DB feature loading code)
            seq, ohlc_df = self._get_seq_features(db, symbol, seq_len)

            tab: Dict[str, float] = {}
            base = (
                symbol.replace("USDT", "")
                .replace("/USDT", "")
                .replace("/", "")
                .upper()
            )

            # SentimentFusion
            sf = (
                db.execute(
                    select(models.SentimentFusion)
                    .where(models.SentimentFusion.symbol.in_([symbol, base]))
                    .order_by(desc(models.SentimentFusion.timestamp))
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if sf:
                tab["final_sentiment"] = float(sf.final_sentiment or 0.0)
                tab["avg_news_sentiment"] = float(sf.avg_news_sentiment or 0.0)
                tab["fear_greed_local"] = float(sf.fear_greed_index or 0.0)

            # OptionsDerivedMetrics
            od = (
                db.execute(
                    select(models.OptionsDerivedMetrics)
                    .where(models.OptionsDerivedMetrics.symbol.in_([symbol, base]))
                    .order_by(desc(models.OptionsDerivedMetrics.timestamp))
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if od:
                tab["put_call_oi_ratio"] = float(od.put_call_oi_ratio or 1.0)
                tab["avg_iv_near_term"] = float(od.avg_iv_near_term or 0.0)
                tab["iv_skew_25d"] = float(od.iv_skew_25d or 0.0)

            # OnchainMetrics
            oc = (
                db.execute(
                    select(models.OnchainMetrics)
                    .where(
                        models.OnchainMetrics.symbol.in_(
                            [symbol, base, base.lower()]
                        )
                    )
                    .order_by(desc(models.OnchainMetrics.timestamp))
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if oc:
                tab["whale_volume_usd"] = float(oc.whale_volume_usd or 0.0)
                tab["exchange_net_flow_usd"] = float(
                    oc.exchange_net_flow_usd or 0.0
                )
                tab["active_addresses"] = float(oc.active_addresses or 0.0)

            # DeveloperActivity
            da = (
                db.execute(
                    select(models.DeveloperActivity)
                    .where(models.DeveloperActivity.symbol.in_([symbol, base]))
                    .order_by(desc(models.DeveloperActivity.timestamp))
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if da:
                tab["dev_stars"] = float(da.stars or 0.0)
                tab["dev_forks"] = float(da.forks or 0.0)

            # FundingRate
            fr = (
                db.execute(
                    select(models.FundingRate)
                    .where(models.FundingRate.symbol.in_([symbol, base]))
                    .order_by(desc(models.FundingRate.timestamp))
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if fr:
                tab["funding_rate"] = float(fr.rate or 0.0)

            # FGI (global)
            fgi = (
                db.execute(
                    select(models.GlobalFGI)
                    .order_by(desc(models.GlobalFGI.timestamp))
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if fgi:
                tab["fear_greed_global"] = float(fgi.value or 0.0)

            # CrossAssetCorr
            cac = (
                db.execute(
                    select(models.CrossAssetCorr)
                    .where(
                        models.CrossAssetCorr.base_symbol.in_([symbol, base])
                    )
                    .order_by(desc(models.CrossAssetCorr.timestamp))
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if cac:
                tab["corr_btc_eth"] = float(cac.corr_btc_eth or 0.0)
                tab["corr_btc_dxy"] = float(cac.corr_btc_dxy or 0.0)

            # OrderbookSnapshot
            ob = (
                db.execute(
                    select(models.OrderbookSnapshot)
                    .where(models.OrderbookSnapshot.symbol == symbol)
                    .order_by(desc(models.OrderbookSnapshot.timestamp))
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if ob:
                tab["ob_bid_ask_imb"] = float(ob.bid_ask_imbalance or 0.0)
                tab["ob_vw_price_skew"] = float(ob.vw_price_skew or 0.0)
                tab["ob_cdv_1m"] = float(ob.cdv_1m or 0.0)
                tab["ob_liquidity"] = float(ob.liquidity_score or 0.0)

        return seq, tab, ohlc_df

    def _get_seq_features(
        self,
        db: Session,
        symbol: str,
        seq_len: int,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)

        rows = (
            db.execute(
                select(models.MarketData)
                .where(
                    models.MarketData.symbol == symbol,
                    models.MarketData.timestamp.between(start_time, end_time),
                )
                .order_by(desc(models.MarketData.timestamp))
                .limit(seq_len)
            )
            .scalars()
            .all()
        )

        if not rows:
            rows = (
                db.execute(
                    select(models.MarketData)
                    .where(models.MarketData.symbol == symbol)
                    .order_by(desc(models.MarketData.timestamp))
                    .limit(seq_len)
                )
                .scalars()
                .all()
            )

        if not rows:
            return np.zeros((0, 2), dtype=np.float32), pd.DataFrame(
                columns=["high", "low", "close"]
            )

        rows = list(reversed(rows))
        closes = np.array([float(r.close) for r in rows], dtype=np.float32)
        highs = np.array([float(r.high) for r in rows], dtype=np.float32)
        lows = np.array([float(r.low) for r in rows], dtype=np.float32)

        seq = np.stack([closes, highs, lows], axis=1)
        ohlc_df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
        return seq, ohlc_df

    # ------------------------------------------------------------------ #
    # Decision Layer + persistence + final DTO
    # ------------------------------------------------------------------ #

    def _risk_aware_persist_and_build_decision(
        self,
        symbol: str,
        instrument_type: str,
        ctx: MarketContext,
        l2_pred: Layer2Prediction,
        decision_direction: str,
        decision_confidence: float,
        llm_headline: str | None,
        llm_score: float,
        rl_action: RLAction,
        model_votes: Dict[str, Any],
        tab_features: Dict[str, float],
        ohlc_df: pd.DataFrame,
    ) -> HybridDecision:
        """
        Decision Layer:
          - Start from AI+Meta fused decision (decision_direction / confidence)
          - Incorporate optional RL execution suggestion
          - Apply RiskEngine (hard constraints)
          - Persist logs
          - Emit final HybridDecision

        RL remains "execution layer": it can suggest style/size,
        but RiskEngine is the final gate.
        """

        risk = RiskEngine(symbol=symbol)

        # 1) Base direction from AI+Meta fused decision seed
        base_direction = decision_direction or l2_pred.direction

        # 2) RL suggested action (may be disabled)
        act = rl_action.optimal_action
        rl_exec_style = rl_action.execution_style
        rl_mode = rl_action.mode

        if (not rl_mode) and self.rl_agent and hasattr(self.rl_agent, "get_mode"):
            rl_mode = self.rl_agent.get_mode()

        # RL is allowed to refine direction, but in your mental model this is
        # execution policy; if you want RL NOT to flip direction, you can
        # restrict that logic here.
        if act == "LONG":
            base_direction = "up"
        elif act == "SHORT":
            base_direction = "down"
        elif act in ("FLAT", "HOLD"):
            # RL explicitly flat/hold → neutralize direction
            base_direction = "flat"

        # Raw size suggestion from RL [0,1]
        raw_size = float(rl_action.optimal_size_pct or 0.0)

        # 3) Run RiskEngine = final decision layer gate
        (
            final_size,
            meta_execute,
            rl_target_position,
            confidence,
            p_edge,
            risk_debug,
        ) = risk.apply(
            symbol=symbol,
            instrument_type=instrument_type,
            ohlc_df=ohlc_df,
            base_direction=base_direction,
            raw_size=raw_size,
            llm_score=llm_score,
            model_votes=model_votes,
            ctx=ctx,
        )

        # 4) Persistence
        with SessionLocal() as db:
            # Prediction row
            try:
                pred = models.Prediction(
                    symbol=symbol,
                    direction=l2_pred.direction,
                    price_confidence=float(l2_pred.price_confidence),
                    created_at=datetime.utcnow(),
                    raw_outputs=l2_pred.raw_outputs,
                )
                db.add(pred)
                db.flush()
            except Exception as e:
                logger.warning(
                    "[HybridInferenceService] Failed to persist Prediction: %s", e
                )

            # HybridSignal row
            act_for_hs = act or "HOLD"
            try:
                hs = models.HybridSignal(
                    symbol=symbol,
                    exchange=ctx.exchange,
                    instrument_type=instrument_type,
                    decided_action=act_for_hs,
                    decided_direction=base_direction,
                    decided_confidence=confidence,
                    size_factor=final_size,
                    strategy_tag="nowa_hybrid_v1",
                    meta_execute=bool(meta_execute),
                    model_votes=model_votes,
                    llm_headline=llm_headline,
                    rl_action=act,
                    rl_mode=rl_mode,
                    rl_target_position=rl_target_position,
                    rl_execution_style=rl_exec_style,
                    debug_payload=risk_debug,
                )
                db.add(hs)
                db.flush()
            except Exception as e:
                logger.warning(
                    "[HybridInferenceService] Failed to persist HybridSignal: %s", e
                )

            # AIExecutionLog row
            try:
                log = models.AIExecutionLog(
                    hybrid_signal_id=None,  # can be linked to hs.id
                    model_version_id=None,
                    decided_action=act,
                    execution_style=rl_exec_style,
                    target_position=rl_target_position,
                    executed_position=None,
                    executed_price=None,
                    exchange=ctx.exchange,
                    symbol=symbol,
                    realized_pnl=None,
                    unrealized_pnl=None,
                    slippage=None,
                    latency_ms=None,
                    confidence=confidence,
                    p_edge=p_edge,
                    rl_diagnostics=risk_debug,
                    exec_meta={
                        "source": "HybridInferenceService",
                        "rl_mode": rl_mode,
                    },
                )
                db.add(log)
                db.commit()
            except Exception as e:
                logger.warning(
                    "[HybridInferenceService] Failed to persist AIExecutionLog: %s", e
                )

        # 5) Final DTO
        debug_payload: Dict[str, Any] = {
            "model_votes": model_votes,
            "decision_direction": decision_direction,
            "decision_confidence": decision_confidence,
            "llm_headline": llm_headline,
            "llm_score": llm_score,
            "rl_mode": rl_mode,
            "risk_debug": risk_debug,
        }

        return HybridDecision(
            symbol=symbol,
            exchange=ctx.exchange,
            instrument_type=instrument_type,
            decided_action=act_for_hs,
            decided_direction=base_direction,
            confidence=confidence,
            size_factor=final_size,
            strategy_tag="nowa_hybrid_v1",
            meta_execute=bool(meta_execute),
            model_votes=model_votes,
            llm_headline=llm_headline,
            rl_action=act,
            rl_mode=rl_mode,
            rl_target_position=rl_target_position,
            rl_execution_style=rl_exec_style,
            debug=debug_payload,
        )


# Singleton
inference_service = HybridInferenceService()
