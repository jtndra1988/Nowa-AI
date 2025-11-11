# app/services/inference_service.py

from __future__ import annotations

import logging
import os
import joblib  # Required for loading XGBoost
import torch   # Required for TFT/TCN
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

from sqlalchemy import select, desc
from sqlalchemy.orm import Session

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
# Path where your trained .pt and .joblib files are stored
ARTIFACT_DIR = os.getenv("MARS_ARTIFACT_DIR", "/app/backend/app/ml/artifacts")
ALLOW_FEATURE_MOCK = os.getenv("MARS_ALLOW_FEATURE_MOCK", "false").lower() == "true"
DEFAULT_ACCOUNT_EQUITY = float(os.getenv("MARS_ACCOUNT_EQUITY_USD", 100000.0))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# ModelEngine (REAL L2 Ensemble: TFT + TCN + XGB)
# =============================================================================

class ModelEngine:
    """
    The 'Brain's' Pattern Recognition Layer.

    Orchestrates:
      1. Temporal Fusion Transformer (TFT)
      2. Temporal Convolutional Network (TCN)
      3. XGBoost (Gradient Boosting)

    It runs all three on live data and ensembles their outputs.
    """

    def __init__(self) -> None:
        self.is_ready: bool = False
        self.device = DEVICE

        # -- 1. Initialize Model Architectures --
        # NOTE: Ensure these params match exactly what you used in training!
        # If you trained with different dimensions, update these numbers.
        self.tft = TemporalFusionTransformer(
            feature_dims={"price": 2},  # Assuming [Close, Volume]
            seq_len=SEQ_LEN_DEFAULT,
            d_model=128,
            nhead=4
        ).to(self.device)

        self.tcn = TemporalConvNet(
            in_feat=2,  # Assuming [Close, Volume]
            channels=(64, 128, 128)
        ).to(self.device)

        self.xgb_model = None

        # -- 2. Load Trained Weights --
        try:
            logger.info(f"[ModelEngine] Loading artifacts from {ARTIFACT_DIR}...")

            # Load PyTorch models (TFT & TCN)
            tft_path = os.path.join(ARTIFACT_DIR, "tft_model.pt")
            tcn_path = os.path.join(ARTIFACT_DIR, "tcn_model.pt")

            if os.path.exists(tft_path):
                self.tft.load_state_dict(torch.load(tft_path, map_location=self.device))
                self.tft.eval()
            else:
                logger.warning(f"[ModelEngine] TFT model not found at {tft_path}")

            if os.path.exists(tcn_path):
                self.tcn.load_state_dict(torch.load(tcn_path, map_location=self.device))
                self.tcn.eval()
            else:
                logger.warning(f"[ModelEngine] TCN model not found at {tcn_path}")

            # Load XGBoost
            xgb_path = os.path.join(ARTIFACT_DIR, "xgb_price_model.joblib")
            if os.path.exists(xgb_path):
                self.xgb_model = joblib.load(xgb_path)
            else:
                logger.warning(f"[ModelEngine] XGBoost model not found at {xgb_path}")

            # We consider the engine ready if at least one model loaded,
            # but ideally you want all three.
            self.is_ready = True
            logger.info("[ModelEngine] Advanced models initialization complete.")

        except Exception as e:
            logger.error(f"[ModelEngine] Failed to load models: {e}", exc_info=True)
            self.is_ready = False

    def build_layer2_prediction(
        self,
        symbol: str,
        seq_features: np.ndarray,
        tab_features: Dict[str, float],
    ) -> Layer2Prediction:
        """
        Runs the ensemble (TFT, TCN, XGB) and returns a fused prediction.
        """
        if not self.is_ready:
            logger.warning("[ModelEngine] Models not ready, using simple fallback.")
            return self._fallback_rule_based(symbol, seq_features)

        # 1. Prepare Data for PyTorch
        # seq_features is [T, 2] (Close, Volume).
        # PyTorch expects [Batch, Seq, Features] -> [1, T, 2]
        x_tensor = torch.tensor(seq_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Input block dictionary for TFT/TCN
        x_blocks = {"price": x_tensor}

        current_price = float(seq_features[-1, 0])

        # 2. Run Inference
        preds_price = []
        preds_conf = []

        with torch.no_grad():
            # --- A. TFT Inference ---
            try:
                tft_out = self.tft(x_blocks)
                p_tft = float(tft_out["price"].item())
                # Use inverse of predicted volatility as a proxy for confidence
                v_tft = float(tft_out["vol"].item())
                preds_price.append(p_tft)
                # Simple normalization for confidence (avoid div by zero)
                preds_conf.append(1.0 / (1.0 + v_tft))
            except Exception as e:
                logger.error(f"TFT Inference failed: {e}")

            # --- B. TCN Inference ---
            try:
                tcn_out = self.tcn(x_blocks)
                p_tcn = float(tcn_out["price"].item())
                preds_price.append(p_tcn)
            except Exception as e:
                logger.error(f"TCN Inference failed: {e}")

        # --- C. XGBoost Inference ---
        if self.xgb_model:
            try:
                # Create single-row DataFrame for XGBoost
                # Note: Features here must match what you used in `train_xgb.py`
                xgb_input = pd.DataFrame([{
                    "close": current_price,
                    "volume": float(seq_features[-1, 1]),
                    "ma_5": float(seq_features[-5:, 0].mean()),
                    "volatility": float(seq_features[-20:, 0].std())
                }])
                p_xgb = float(self.xgb_model.predict(xgb_input)[0])
                preds_price.append(p_xgb)
            except Exception as e:
                logger.error(f"XGB Inference failed: {e}")

        # 3. Ensemble Fusion
        if not preds_price:
            return self._fallback_rule_based(symbol, seq_features)

        avg_pred_price = sum(preds_price) / len(preds_price)

        # Determine Direction
        # If predicted price > current price by threshold (e.g. 0.05%)
        threshold = current_price * 0.0005
        if avg_pred_price > current_price + threshold:
            direction = "up"
        elif avg_pred_price < current_price - threshold:
            direction = "down"
        else:
            direction = "flat"

        # Determine Confidence
        # Calculate disagreement (std dev) among models. Lower std = Higher confidence.
        if len(preds_price) > 1:
            disagreement = np.std(preds_price)
            rel_disagreement = disagreement / current_price
            # Heuristic: 0% diff = 1.0 conf, 1% diff = 0.0 conf
            ensemble_conf = max(0.0, 1.0 - (rel_disagreement * 100))
        else:
            # If only 1 model ran, use TFT confidence or default
            ensemble_conf = preds_conf[0] if preds_conf else 0.5
            # Tabular nudges
        sent = float(tab_features.get("final_sentiment", 0.0))
        put_call = float(tab_features.get("put_call_oi_ratio", 1.0))
        whale_vol = float(tab_features.get("whale_volume_usd", 0.0))
        ex_flow = float(tab_features.get("exchange_net_flow_usd", 0.0))

        # NEW: orderbook microstructure
        ob_imb = float(tab_features.get("ob_bid_ask_imb", 0.0))       # -1..1
        ob_skew = float(tab_features.get("ob_vw_price_skew", 0.0))    # signed skew
        ob_cdv = float(tab_features.get("ob_cdv_1m", 0.0))            # signed delta vol
        ob_liq = float(tab_features.get("ob_liquidity", 0.0))         # depth proxy

        conf_sent = min(0.2, abs(sent) * 0.2)
        conf_pc = 0.1 if 0.7 <= put_call <= 1.3 else 0.0
        conf_flow = 0.1 if whale_vol > 0 and ex_flow != 0 else 0.0

        # NEW: microstructure contributes confidence if liquidity+signal present
        conf_ob = 0.0
        if ob_liq > 0:
         # confidence scales with how directional the book is
         conf_ob = min(0.2, abs(ob_imb) * 0.2 + abs(ob_cdv) * 0.1)

         price_confidence = max(
        0.0,
        min(1.0, conf_sent + conf_sent + conf_pc + conf_flow + conf_ob),
        )

        # Direction bias from sentiment, flows, orderbook
        dir_bias = 0.0
        dir_bias += np.sign(sent) * min(0.5, abs(sent))

        # flows
        if ex_flow < 0:
          dir_bias -= 0.15
        elif ex_flow > 0:
          dir_bias += 0.15

        # NEW: orderbook tilt
        dir_bias += np.sign(ob_imb) * min(0.25, abs(ob_imb))
        dir_bias += np.sign(ob_cdv) * min(0.15, abs(ob_cdv))

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

        return Layer2Prediction(
            asset=symbol,
            direction=direction,
            price_confidence=float(round(ensemble_conf, 3)),
        )

    def _fallback_rule_based(self, symbol, seq_features):
        """Original logic kept as backup safety net."""
        if seq_features.size == 0:
             return Layer2Prediction(asset=symbol, direction="flat", price_confidence=0.0)

        closes = seq_features[:, 0]
        last_close = float(closes[-1])
        ma_short = float(closes[-5:].mean())

        if last_close > ma_short:
            d = "up"
        else:
            d = "down"

        return Layer2Prediction(
            asset=symbol,
            direction=d,
            price_confidence=0.1  # Low confidence for fallback
        )


# =============================================================================
# HybridInferenceService (One Brain)
# =============================================================================

class HybridInferenceService:
    """
    One-brain orchestrator:
      features -> L2 ensemble (TFT/TCN/XGB) -> LLM narrative -> RL -> RiskEngine -> HybridDecision
    """

    def __init__(self) -> None:
        # Initialize the REAL ModelEngine with advanced models
        self.model_engine = ModelEngine()
        self.llm_engine = llm_engine
        self.rl_agent = rl_agent
        self.allow_mock = ALLOW_FEATURE_MOCK

        # Service is ready if LLM and RL are loaded.
        # ModelEngine handles its own partial failures gracefully.
        self.is_ready: bool = (
        self.model_engine.is_ready
        and self.llm_engine.is_model_loaded()
        and self.rl_agent.is_model_loaded()
        )

        if self.is_ready:
            logger.info("[HybridInferenceService] READY: engines loaded.")
        else:
            logger.error(
                "[HybridInferenceService] NOT READY: "
                f"llm={self.llm_engine.is_model_loaded()}, "
                f"rl={self.rl_agent.is_model_loaded()}"
            )

    def ready(self) -> bool:
        return self.is_ready

    # ------------------------------------------------------------------
    # Main entrypoint for /hybrid-signal
    # ------------------------------------------------------------------

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

        # 2) L2 prediction (TFT + TCN + XGB Ensemble)
        l2_pred = self.model_engine.build_layer2_prediction(
            symbol=symbol,
            seq_features=seq,
            tab_features=tab,
        )

        # 3) LLM narrative (live)
        llm_out = await self.llm_engine.get_narrative_signal(symbol)
        llm_score = float(llm_out["sentiment_score"])
        llm_headline = llm_out.get("key_headline")

        # 4) Model votes (used by RL and debug)
        model_votes: Dict[str, float] = {
            "trend_model": (
                1.0
                if l2_pred.direction == "up"
                else -1.0
                if l2_pred.direction == "down"
                else 0.0
            ),
            "price_confidence": float(l2_pred.price_confidence),
            "llm_narrative": llm_score,
            "final_sentiment": float(tab.get("final_sentiment", 0.0)),
            "put_call_oi_ratio": float(tab.get("put_call_oi_ratio", 1.0)),
        }

        # 5) RL decision
        rl_action: RLAction = self.rl_agent.get_optimal_action(
            prediction=l2_pred,
            context=ctx,
            model_votes=model_votes,
        )

        # 6) Risk, persistence, final HybridDecision
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

    # ------------------------------------------------------------------
    # Feature extraction from REAL tables
    # ------------------------------------------------------------------

    def _fetch_features(
        self,
        symbol: str,
        seq_len: int,
    ) -> Tuple[np.ndarray, Dict[str, float], pd.DataFrame]:
        """
        Pull:
          - Sequential OHLCV from FuturesMarketData or MarketData
          - Latest: SentimentFusion, OptionsDerivedMetrics, OnchainMetrics, etc.

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
                        # OrderbookSnapshot (microstructure signal)
           
            # OrderbookSnapshot (microstructure signal)
            ob = (
                db.execute(
                    select(models.OrderbookSnapshot)
                    .where(
                        models.OrderbookSnapshot.symbol.in_(
                            [symbol, base, base + "USDT"]
                        )
                    )
                    .order_by(desc(models.OrderbookSnapshot.timestamp))
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if ob:
                # Core microstructure signals
                tab["ob_bid_ask_imb"] = float(ob.bid_ask_imb or 0.0)
                tab["ob_vw_price_skew"] = float(ob.vw_price_skew or 0.0)
                tab["ob_cdv_1m"] = float(ob.cdv_1m or 0.0)

                # Optional: liquidity proxy
                bv = float(ob.bid_volume or 0.0)
                av = float(ob.ask_volume or 0.0)
                tab["ob_liquidity"] = float(bv + av)

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
        tab: Dict[str, float] = {
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
        """
        Apply RL + RiskEngine, persist audit rows, and return HybridDecision.
        """
        now = datetime.utcnow()

        # ---- Base RL interpretation ----
        act = (rl_action.optimal_action or "HOLD").upper()
        rl_exec_style = rl_action.execution_style
        rl_mode = rl_action.mode or self.rl_agent.get_mode()

        # Map RL action to an initial directional view
        if act == "LONG":
            base_direction = "up"
        elif act == "SHORT":
            base_direction = "down"
        else:
            base_direction = l2_pred.direction

        # Raw size suggestion from RL [0,1]
        raw_size = float(rl_action.optimal_size_pct or 0.0)
        raw_size = max(0.0, min(1.0, raw_size))

        # Candidate for live execution?
        meta_execute = act in ("LONG", "SHORT") and raw_size > 0.0

        # ---- RiskEngine integration ----
        risk_debug: Dict[str, Any] = {}
        final_size = 0.0
        rl_target_position = 0.0

        if meta_execute and not ohlc_df.empty:
            # Initialize per-symbol risk engine
            risk_engine = RiskEngine(symbol)

            # 1) Global / session guards (dd, halt flags, etc.)
            current_equity = DEFAULT_ACCOUNT_EQUITY
            guard = risk_engine.check_global_guards(current_equity=current_equity)
            risk_debug["global_guard"] = guard

            if guard.get("halt"):
                # Hard stop: no new exposure
                meta_execute = False
                final_size = 0.0
                rl_target_position = 0.0
            else:
                # 2) Side for risk proposal
                side = "BUY" if base_direction == "up" else "SELL"

                # 3) Regime (optional, from context)
                regime = 0
                if ctx.current_regime is not None:
                    try:
                        regime = int(ctx.current_regime)
                    except Exception:
                        regime = 0

                # 4) Mark price from last close
                mark_price = float(ohlc_df["close"].iloc[-1])

                # 5) Ask RiskEngine for allowed position
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
                    # Convert allowed notional -> max fraction of equity
                    max_risk_size = float(proposal["qty_usd"]) / max(
                        current_equity, 1e-9
                    )
                    max_risk_size = max(0.0, min(1.0, max_risk_size))

                    # Final size is min(RL suggestion, risk cap)
                    final_size = min(raw_size, max_risk_size)
                    rl_target_position = (
                        final_size if side == "BUY" else -final_size
                    )
                else:
                    # RiskEngine vetoed
                    meta_execute = False
                    final_size = 0.0
                    rl_target_position = 0.0

                # Persist updated risk state if engine supports it
                try:
                    risk_engine.save_state()
                except Exception:
                    # Don't break decision flow on risk save failure
                    pass
        else:
            # No execution or no OHLC data -> flat
            final_size = 0.0
            rl_target_position = 0.0

        # ---- Derive high-level decision metrics ----
        confidence = float(l2_pred.price_confidence)
        p_edge = confidence  # you can refine if needed
        direction = base_direction

        debug_payload: Dict[str, Any] = {
            "layer2_prediction": l2_pred.dict(),
            "llm": {
                "sentiment_score": llm_score,
                "headline": llm_headline,
            },
            "rl_action": rl_action.dict(),
            "model_votes": model_votes,
            "tab_features": tab_features,
            "risk": risk_debug,
        }

        # ---- Persist DB artifacts ----
        with SessionLocal() as db:
            # 1) Prediction (minimal generic record)
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

            # 2) HybridSignal snapshot
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

            # 3) AIExecutionLog for auditability
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
                    exec_meta={
                        "source": "HybridInferenceService",
                        "rl_mode": rl_mode,
                    },
                )
                db.add(log)
            except Exception as e:
                logger.error(
                    "[HybridInferenceService] Failed to persist AIExecutionLog: %s",
                    e,
                    exc_info=True,
                )

            db.commit()

        # ---- Return HybridDecision (matches schemas.HybridDecision) ----
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