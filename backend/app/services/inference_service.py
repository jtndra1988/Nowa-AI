# app/services/inference_service.py

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db import models
from app.ml.adv.llm_narrative_model import llm_engine
from app.ml.adv.rl_execution_agent import rl_agent
from app.hybrid.schemas import (
    MarketContext,
    Layer2Prediction,
    RLAction,
    HybridDecision,
)
from app.risk.engine import RiskEngine

logger = logging.getLogger(__name__)

SEQ_LEN_DEFAULT = 64
ALLOW_FEATURE_MOCK = os.getenv("MARS_ALLOW_FEATURE_MOCK", "false").lower() == "true"
DEFAULT_ACCOUNT_EQUITY = float(os.getenv("MARS_ACCOUNT_EQUITY_USD", 100000.0))


# =============================================================================
# ModelEngine (L1 + L2 ensemble over REAL data)
# =============================================================================


class ModelEngine:
    """
    Encapsulates your L2 ensemble logic.

    Currently:
      - deterministic rule-based fusion over real features
      - no random mock; no external artifacts
      - `is_ready` is True as long as code paths exist

    When you plug TFT/TCN/XGB, only replace this class.
    """

    def __init__(self) -> None:
        self.is_ready: bool = True
        logger.info("[ModelEngine] Initialized (rule-based ensemble over live features).")

    def build_layer2_prediction(
        self,
        symbol: str,
        seq_features: np.ndarray,
        tab_features: Dict[str, float],
    ) -> Layer2Prediction:
        """
        Build a fused L2 prediction from sequential + tabular features.

        Uses:
          - Price trend (closes)
          - Vol-normalized trend strength
          - Sentiment, options, on-chain nudges

        Returns Layer2Prediction:
          - asset: symbol
          - direction: 'up' | 'down' | 'flat'
          - price_confidence: [0,1]
        """
        if not self.is_ready:
            raise RuntimeError("[ModelEngine] Called while not ready.")

        if seq_features.size == 0 or seq_features.shape[0] < 10:
            raise RuntimeError(
                "[ModelEngine] Not enough sequential data for prediction."
            )

        closes = seq_features[:, 0]

        last_close = float(closes[-1])
        ma_short = float(closes[-5:].mean())
        ma_long = float(closes[-20:].mean()) if len(closes) >= 20 else ma_short

        # Base trend direction
        if last_close > ma_short * 1.002 and ma_short >= ma_long:
            base_dir = "up"
        elif last_close < ma_short * 0.998 and ma_short <= ma_long:
            base_dir = "down"
        else:
            base_dir = "flat"

        # Vol-normalized trend strength
        if len(closes) >= 20:
            price_std = float(closes[-20:].std())
        else:
            price_std = float(closes.std())
        price_std = max(price_std, 1e-8)

        trend_strength = abs(last_close - ma_long) / price_std
        conf_trend = max(0.0, min(1.0, trend_strength / 4.0))

        # Tabular nudges
        sent = float(tab_features.get("final_sentiment", 0.0))
        put_call = float(tab_features.get("put_call_oi_ratio", 1.0))
        whale_vol = float(tab_features.get("whale_volume_usd", 0.0))
        ex_flow = float(tab_features.get("exchange_net_flow_usd", 0.0))

        conf_sent = min(0.2, abs(sent) * 0.2)
        conf_pc = 0.1 if 0.7 <= put_call <= 1.3 else 0.0
        conf_flow = 0.1 if whale_vol > 0 and ex_flow != 0 else 0.0

        price_confidence = max(
            0.0, min(1.0, conf_trend + conf_sent + conf_pc + conf_flow)
        )

        # Direction bias from sentiment & flows
        dir_bias = 0.0
        dir_bias += np.sign(sent) * min(0.5, abs(sent))
        if ex_flow < 0:
            dir_bias -= 0.15
        elif ex_flow > 0:
            dir_bias += 0.15

        final_score = {"up": 0.0, "down": 0.0, "flat": 0.0}

        if base_dir == "up":
            final_score["up"] += 1.0
        elif base_dir == "down":
            final_score["down"] += 1.0
        else:
            final_score["flat"] += 0.5

        if dir_bias > 0.1:
            final_score["up"] += 0.5
        elif dir_bias < -0.1:
            final_score["down"] += 0.5
        else:
            final_score["flat"] += 0.2

        direction = max(final_score.items(), key=lambda x: x[1])[0]

        return Layer2Prediction(
            asset=symbol,
            direction=direction,
            price_confidence=float(round(price_confidence, 3)),
        )


# =============================================================================
# HybridInferenceService (One Brain Orchestrator)
# =============================================================================


class HybridInferenceService:
    """
    Orchestrates:
      - Feature extraction from real DB tables
      - ModelEngine (L2)
      - LLM narrative specialist (L1.5)
      - RL execution agent (L3)
      - RiskEngine constraints
      - Persistence & HybridDecision assembly
    """

    def __init__(self) -> None:
        self.model_engine = ModelEngine()
        self.llm_engine = llm_engine
        self.rl_agent = rl_agent
        self.allow_mock = ALLOW_FEATURE_MOCK

        self.is_ready: bool = (
            self.model_engine.is_ready
            and self.llm_engine.is_model_loaded()
            and self.rl_agent.is_model_loaded()
        )

        if self.is_ready:
            logger.info("[HybridInferenceService] READY: all engines loaded.")
        else:
            logger.error(
                "[HybridInferenceService] NOT READY: "
                f"model={self.model_engine.is_ready}, "
                f"llm={self.llm_engine.is_model_loaded()}, "
                f"rl={self.rl_agent.is_model_loaded()}"
            )

    def ready(self) -> bool:
        return self.is_ready

    # -------------------------------------------------------------------------
    # Main entrypoint for /hybrid-signal
    # -------------------------------------------------------------------------

    async def build_decision(self, ctx: MarketContext) -> HybridDecision:
        if not self.is_ready:
            raise RuntimeError(
                "[HybridInferenceService] Called while not ready. "
                "Verify LLM, RL, and ModelEngine initialization."
            )

        symbol = (ctx.symbol or "BTCUSDT").upper()
        instrument_type = (ctx.instrument_type or "futures").lower()

        # 1) Features (seq, tabular, OHLC for risk)
        seq, tab, ohlc_df = self._fetch_features(symbol, SEQ_LEN_DEFAULT)

        # 2) L2 prediction
        l2_pred = self.model_engine.build_layer2_prediction(
            symbol=symbol,
            seq_features=seq,
            tab_features=tab,
        )

        # 3) LLM narrative (live only)
        llm_out = await self.llm_engine.get_narrative_signal(symbol)
        llm_score = float(llm_out["sentiment_score"])
        llm_headline = llm_out.get("key_headline")

        # 4) Model votes for RL + debug
        model_votes: Dict[str, float] = {
            "trend_model": (
                1.0 if l2_pred.direction == "up"
                else -1.0 if l2_pred.direction == "down"
                else 0.0
            ),
            "price_confidence": float(l2_pred.price_confidence),
            "llm_narrative": llm_score,
            "final_sentiment": float(tab.get("final_sentiment", 0.0)),
            "put_call_oi_ratio": float(tab.get("put_call_oi_ratio", 1.0)),
        }

        # 5) RL agent
        rl_action: RLAction = self.rl_agent.get_optimal_action(
            prediction=l2_pred,
            context=ctx,
            model_votes=model_votes,
        )

        # 6) Risk + persistence + final HybridDecision
        decision = self._risk_aware_persist_and_build_decision(
            symbol=symbol,
            instrument_type=instrument_type,
            ctx=ctx,
            l2_pred=l2_pred,
            llm_headline=llm_headline,
            llm_score=llm_score,
            rl_action=rl_action,
            model_votes=model_votes,
            tab_features=tab,
            ohlc_df=ohlc_df,
        )
        return decision

    # -------------------------------------------------------------------------
    # Feature extraction from REAL tables
    # -------------------------------------------------------------------------

    def _fetch_features(
        self,
        symbol: str,
        seq_len: int,
    ) -> Tuple[np.ndarray, Dict[str, float], pd.DataFrame]:
        """
        Pull:
          - Sequential OHLCV from FuturesMarketData or MarketData
          - Latest:
              SentimentFusion
              OptionsDerivedMetrics
              OnchainMetrics
              DeveloperActivity
              FundingRate
              FearAndGreedIndex
              CrossAssetCorr

        Returns:
          seq_features: [T,2] -> [close, volume]
          tab_features: dict
          ohlc_df: DataFrame ['high','low','close'] for RiskEngine
        """
        with SessionLocal() as db:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)

            seq, ohlc_df = self._load_price_and_ohlc(
                db=db,
                symbol=symbol,
                seq_len=seq_len,
                start_time=start_time,
                end_time=end_time,
            )

            if seq.size == 0:
                if self.allow_mock:
                    logger.warning(
                        "[HybridInferenceService] No price data; using DEV-ONLY mock features."
                    )
                    return self._generate_mock_features(symbol, seq_len)
                raise RuntimeError("[HybridInferenceService] No price data available.")

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
                    .where(models.OnchainMetrics.symbol.in_([symbol, base, base.lower()]))
                    .order_by(desc(models.OnchainMetrics.timestamp))
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if oc:
                tab["whale_volume_usd"] = float(oc.whale_volume_usd or 0.0)
                tab["exchange_net_flow_usd"] = float(oc.exchange_net_flow_usd or 0.0)
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
                tab["funding_rate"] = float(fr.funding_rate or 0.0)

            # Global Fear & Greed
            fgi = (
                db.execute(
                    select(models.FearAndGreedIndex)
                    .order_by(desc(models.FearAndGreedIndex.timestamp))
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
                    .where(models.CrossAssetCorr.base_symbol.in_([symbol, base]))
                    .order_by(desc(models.CrossAssetCorr.timestamp))
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if cac:
                tab["corr_btc_eth"] = float(cac.corr_btc_eth or 0.0)
                tab["corr_btc_dxy"] = float(cac.corr_btc_dxy or 0.0)
                tab["corr_btc_ndx"] = float(cac.corr_btc_ndx or 0.0)
                tab["corr_btc_gold"] = float(cac.corr_btc_gold or 0.0)

            return seq, tab, ohlc_df

    def _load_price_and_ohlc(
        self,
        db: Session,
        symbol: str,
        seq_len: int,
        start_time: datetime,
        end_time: datetime,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load OHLCV from FuturesMarketData or MarketData.

        Returns:
          seq_features: [T,2] -> [close, volume]
          ohlc_df: DataFrame with ['high','low','close']
        """
        # Prefer futures data
        fut_rows = (
            db.execute(
                select(models.FuturesMarketData)
                .where(models.FuturesMarketData.symbol == symbol)
                .where(models.FuturesMarketData.timestamp.between(start_time, end_time))
                .order_by(desc(models.FuturesMarketData.timestamp))
                .limit(seq_len)
            )
            .scalars()
            .all()
        )

        rows = fut_rows
        if not rows:
            spot_rows = (
                db.execute(
                    select(models.MarketData)
                    .where(models.MarketData.symbol == symbol)
                    .where(models.MarketData.timestamp.between(start_time, end_time))
                    .order_by(desc(models.MarketData.timestamp))
                    .limit(seq_len)
                )
                .scalars()
                .all()
            )
            rows = spot_rows

        if not rows:
            return np.zeros((0, 2), dtype=np.float32), pd.DataFrame(
                columns=["high", "low", "close"]
            )

        rows = list(reversed(rows))

        closes = [
            float(
                getattr(r, "close", None)
                if getattr(r, "close", None) is not None
                else getattr(r, "last_price", 0.0)
            )
            for r in rows
        ]
        vols = [float(getattr(r, "volume", 0.0)) for r in rows]
        highs = [
            float(getattr(r, "high", c)) for r, c in zip(rows, closes)
        ]
        lows = [
            float(getattr(r, "low", c)) for r, c in zip(rows, closes)
        ]

        seq = np.column_stack([closes, vols]).astype(np.float32)
        ohlc_df = pd.DataFrame({"high": highs, "low": lows, "close": closes})

        return seq, ohlc_df

    def _generate_mock_features(
        self,
        symbol: str,
        seq_len: int,
    ) -> Tuple[np.ndarray, Dict[str, float], pd.DataFrame]:
        """
        DEV-ONLY deterministic pseudo-features.
        Enabled only when MARS_ALLOW_FEATURE_MOCK=true.
        """
        seed = sum(ord(c) for c in symbol)
        rng = np.random.default_rng(seed)

        closes = np.cumsum(rng.normal(0, 1, size=seq_len)) + 100.0
        vols = rng.uniform(10, 50, size=seq_len)

        seq = np.column_stack([closes, vols]).astype(np.float32)
        tab = {
            "final_sentiment": 0.0,
            "put_call_oi_ratio": 1.0,
            "whale_volume_usd": 0.0,
            "exchange_net_flow_usd": 0.0,
        }
        ohlc_df = pd.DataFrame(
            {"high": closes + 1.0, "low": closes - 1.0, "close": closes}
        )
        return seq, tab, ohlc_df

    # -------------------------------------------------------------------------
    # Risk integration + persistence + HybridDecision
    # -------------------------------------------------------------------------

    def _risk_aware_persist_and_build_decision(
        self,
        symbol: str,
        instrument_type: str,
        ctx: MarketContext,
        l2_pred: Layer2Prediction,
        llm_headline: str,
        llm_score: float,
        rl_action: RLAction,
        model_votes: Dict[str, float],
        tab_features: Dict[str, float],
        ohlc_df: pd.DataFrame,
    ) -> HybridDecision:
        now = datetime.utcnow()

        # RL raw action
        act = (rl_action.optimal_action or "HOLD").upper()
        rl_mode = rl_action.mode or self.rl_agent.get_mode()

        # Base direction (for HybridDecision)
        if act == "LONG":
            base_direction = "up"
        elif act == "SHORT":
            base_direction = "down"
        else:
            base_direction = l2_pred.direction

        # RL suggested size [0,1]
        raw_size = float(rl_action.optimal_size_pct or 0.0)
        raw_size = max(0.0, min(1.0, raw_size))

        # Candidate execution?
        meta_execute = act in ("LONG", "SHORT") and raw_size > 0.0

        # ---- RiskEngine integration ----
        risk_debug: Dict[str, Any] = {}
        final_size = 0.0
        rl_target_position = 0.0

        if meta_execute and not ohlc_df.empty:
            risk_engine = RiskEngine(symbol)

            # 1) Global guard (rolling DD, etc.)
            current_equity = DEFAULT_ACCOUNT_EQUITY
            guard = risk_engine.check_global_guards(current_equity=current_equity)
            risk_debug["global_guard"] = guard

            if guard.get("halt"):
                # Hard stop: no new risk
                meta_execute = False
                final_size = 0.0
                rl_target_position = 0.0
            else:
                # 2) Side for RiskEngine
                side = "BUY" if base_direction == "up" else "SELL"

                # 3) Regime from context (int)
                regime = 0
                if ctx.current_regime is not None:
                    try:
                        regime = int(ctx.current_regime)
                    except Exception:
                        regime = 0

                # 4) Mark price from last close
                mark_price = float(ohlc_df["close"].iloc[-1])

                # 5) RiskEngine position proposal
                proposal = risk_engine.propose_position(
                    dfe=ohlc_df,
                    side=side,
                    confidence=float(l2_pred.price_confidence),
                    regime=regime,
                    mark_price=mark_price,
                    open_positions_usd=0.0,
                    portfolio_gross_exposure=0.0,
                )
                risk_debug["proposal"] = proposal

                if proposal.get("reason") == "ok":
                    max_risk_size = float(proposal["qty_usd"]) / max(
                        current_equity, 1e-9
                    )
                    max_risk_size = max(0.0, min(1.0, max_risk_size))
                    final_size = min(raw_size, max_risk_size)
                    rl_target_position = (
                        final_size if side == "BUY" else -final_size
                    )
                else:
                    meta_execute = False
                    final_size = 0.0
                    rl_target_position = 0.0

                risk_engine.save_state()
        else:
            # no exec / no data → flat
            final_size = 0.0
            rl_target_position = 0.0

        confidence = float(l2_pred.price_confidence)
        p_edge = confidence
        direction = base_direction
        rl_exec_style = rl_action.execution_style

        debug_payload: Dict[str, Any] = {
            "layer2_prediction": l2_pred.dict(),
            "llm": {"sentiment_score": llm_score, "headline": llm_headline},
            "rl_action": rl_action.dict(),
            "model_votes": model_votes,
            "tab_features": tab_features,
            "risk": risk_debug,
        }

        # ---- Persist DB artifacts ----
        with SessionLocal() as db:
            # Prediction
            pred_row = models.Prediction(
                model_version_id=None,
                symbol=symbol,
                prediction_time=now,
                prediction=p_edge,
                raw_score=p_edge,
                model_inputs={"tab": tab_features, "votes": model_votes},
            )
            db.add(pred_row)
            db.flush()

            # HybridSignal
            hs = models.HybridSignal(
                symbol=symbol,
                instrument_type=instrument_type,
                exchange=ctx.exchange,
                direction=direction,
                p_edge=p_edge,
                confidence=confidence,
                size_factor=final_size,
                strategy_tag="nowa_hybrid_v1",
                meta_execute=bool(meta_execute),
                debug_payload=debug_payload,
            )
            db.add(hs)
            db.flush()

            # AIExecutionLog
            try:
                log = models.AIExecutionLog(
                    hybrid_signal_id=hs.id,
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
                    exec_meta={"source": "HybridInferenceService", "rl_mode": rl_mode},
                )
                db.add(log)
            except Exception as e:
                logger.error(
                    "[HybridInferenceService] Failed to persist AIExecutionLog: %s",
                    e,
                    exc_info=True,
                )

            db.commit()

        # ---- Final HybridDecision (matches schemas.py) ----
        return HybridDecision(
            symbol=symbol,
            instrument_type=instrument_type,
            timestamp=now,
            direction=direction,
            p_edge=p_edge,
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


# Singleton used by routes
inference_service = HybridInferenceService()
