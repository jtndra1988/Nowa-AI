# app/ml/model_engine.py

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy import desc
from sqlalchemy.orm import Session

# Core Schemas & DB
from app.hybrid.schemas import Layer2Prediction, MarketContext
from app.db.database import SessionLocal
from app.db import models

# Registry & Feature Engineering
from app.ml.adv.model_registry import model_registry
from app.ml.adv.feature_engineering import FeatureBuilder

# Model Wrappers (for type hinting/reference, instantiated via registry)
from app.ml.adv.models_tft import TFTPredictor  # noqa: F401
from app.ml.adv.models_tcn import TCNPredictor  # noqa: F401
from app.ml.adv.models_tst import TSTPredictor  # noqa: F401

# DecisionNet & Experts
from app.ml.adv.decision_net import decision_net_score
from app.ml.adv.options_vol_model import options_vol_edge
from app.ml.adv.macro_onchain_model import macro_onchain_bias
from app.ml.adv.llm_narrative_model import llm_engine

# Ensemble (instantiated via registry)
from app.ml.ensemble import EnsembleStackerService  # noqa: F401

logger = logging.getLogger(__name__)


class ModelEngine:
    """
    Aggregates your full ML stack into one unified prediction object.

    Layer 1 (Predictors):
      - TFT (Visionary): Returns {price, vol}
      - TCN (Reflex): Returns {price, vol}
      - TST (Transformer): Returns {price, vol}
      - XGB (Tabular): Returns price only

    Layer 2 (Fusion):
      - DecisionNet (Neural Fusion)
      - EnsembleStacker (Linear Meta-Learner)

    Layer 3 (Experts):
      - Options Expert
      - Macro/On-chain Expert
      - LLM Narrative
    """

    def __init__(self) -> None:
        logger.info("[ModelEngine] Initializing...")

        # 1. Central Registry (Singleton)
        self.registry = model_registry

        # 2. Initialize Predictors via Registry
        # XGB Service
        self.xgb_service = self.registry.get_model("xgb")
        self.xgb_ready = bool(
            self.xgb_service and getattr(self.xgb_service, "is_ready", False)
        )

        # TFT
        self.tft = self.registry.get_model("tft")
        self.tft_loaded = bool(
            self.tft and getattr(self.tft, "is_model_loaded", lambda: False)()
        )

        # TCN
        self.tcn = self.registry.get_model("tcn")
        self.tcn_loaded = bool(
            self.tcn and getattr(self.tcn, "is_model_loaded", lambda: False)()
        )

        # TST
        self.tst = self.registry.get_model("tst")
        self.tst_loaded = bool(
            self.tst and getattr(self.tst, "is_model_loaded", lambda: False)()
        )

        # 3. Initialize Stacker (The "Judge")
        self.stacker = self.registry.get_model("ensemble")
        self.stacker_ready = bool(
            self.stacker and getattr(self.stacker, "is_ready", False)
        )

        # 4. Initialize Data Ingestors
        self.options_ingestor = self.registry.get_model("options_ingestor")
        self.macro_ingestor = self.registry.get_model("macro_ingestor")

        # 5. Feature Builder
        self.feature_builder = FeatureBuilder()

    # --------------------------------------------------------------
    # Data Loading & Feature Prep
    # --------------------------------------------------------------

    def _load_recent_ohlcv_for_symbol(
        self, symbol: str, limit: int = 300
    ) -> pd.DataFrame:
        """
        Load recent OHLCV for XGBoost and current price reference.
        """
        session: Session = SessionLocal()
        try:
            rows = (
                session.query(models.MarketData)
                .filter(models.MarketData.symbol == symbol.upper())
                .order_by(desc(models.MarketData.timestamp))
                .limit(limit)
                .all()
            )

            if not rows:
                logger.warning(
                    "[ModelEngine] No OHLCV rows found for symbol %s.", symbol
                )
                return pd.DataFrame()

            data = [
                {
                    "symbol": r.symbol,
                    "timestamp": r.timestamp,
                    "open": float(r.open),
                    "high": float(r.high),
                    "low": float(r.low),
                    "close": float(r.close),
                    "volume": float(r.volume),
                }
                for r in rows
            ]
            return (
                pd.DataFrame(data)
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

        except Exception as e:
            logger.error(
                "[ModelEngine] Failed to load OHLCV for %s: %s", symbol, e, exc_info=True
            )
            return pd.DataFrame()
        finally:
            session.close()

    async def _build_feature_set(self, symbol: str) -> Dict[str, Any]:
        """
        Build the full feature dictionary for all models.
        """
        try:
            # 1. Sequence Features (TFT/TCN/TST)
            raw_features = await self.feature_builder.build_features(symbol)
            if not isinstance(raw_features, dict):
                raise TypeError(
                    f"FeatureBuilder returned {type(raw_features)}, expected dict"
                )

            features: Dict[str, Any] = dict(raw_features)

            # Normalize Tensor Shapes [L, F] -> [B, L, F] if needed
            price = features.get("price")
            try:
                import torch

                if isinstance(price, torch.Tensor) and price.ndim == 2:
                    features["price"] = price.unsqueeze(0)
            except Exception:
                # If torch is missing or anything else, just skip reshape
                pass

            # 2. Expert Features (Options / Macro)
            if self.macro_ingestor:
                try:
                    ingested = self.macro_ingestor.build_features(symbol)
                    if ingested:
                        features["macro_onchain_features"] = ingested
                except Exception as e:
                    logger.warning("[ModelEngine] Macro ingest failed: %s", e)

            if self.options_ingestor:
                try:
                    ingested = self.options_ingestor.build_features(symbol)
                    if ingested:
                        features["options_features"] = ingested
                except Exception as e:
                    logger.warning("[ModelEngine] Options ingest failed: %s", e)

            # 3. Tabular Data (XGBoost & Price Reference)
            if "xgb_df" not in features or features["xgb_df"] is None:
                xgb_df = self._load_recent_ohlcv_for_symbol(symbol)
                if not xgb_df.empty:
                    features["xgb_df"] = xgb_df

            return features

        except Exception as e:
            logger.error(
                "[ModelEngine] Feature build failed for %s: %s", symbol, e, exc_info=True
            )
            raise

    # --------------------------------------------------------------
    # Readiness
    # --------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """Ready if at least one core model is loaded."""
        return any([self.tft_loaded, self.tcn_loaded, self.tst_loaded, self.xgb_ready])

    # --------------------------------------------------------------
    # Core Prediction Logic
    # --------------------------------------------------------------

    async def predict(self, ctx: MarketContext) -> Layer2Prediction:
        """
        Unified prediction call:
        1. Gathers Features.
        2. Runs Models (getting Price Return AND Volatility).
        3. Calculates Price Targets and Ranges.
        4. Returns unified Layer2Prediction.
        """
        symbol = ctx.symbol.upper()

        # --- STEP 1: Build Features ---
        try:
            features = await self._build_feature_set(symbol)
        except Exception as e:
            logger.error("[ModelEngine] Aborting predict due to feature error: %s", e)
            raise

        # --- STEP 2: Get Reference Price ---
        current_price = 0.0
        xgb_df = features.get("xgb_df")
        if isinstance(xgb_df, pd.DataFrame) and not xgb_df.empty:
            current_price = float(xgb_df["close"].iloc[-1])

        # --- STEP 3: Run Individual Models ---

        # Note: Models are expected to return Dict[str, float]: {"price": 0.0, "vol": 0.0}

        # TFT
        tft_res: Dict[str, float] = {"price": 0.0, "vol": 0.0}
        if self.tft_loaded:
            try:
                raw = self.tft.predict(features)
                if isinstance(raw, dict):
                    tft_res = raw
                else:  # backward compatibility
                    tft_res["price"] = float(raw)
            except Exception as e:
                logger.warning("[ModelEngine] TFT error: %s", e)

        # TCN
        tcn_res: Dict[str, float] = {"price": 0.0, "vol": 0.0}
        if self.tcn_loaded:
            try:
                raw = self.tcn.predict(features)
                if isinstance(raw, dict):
                    tcn_res = raw
                else:
                    tcn_res["price"] = float(raw)
            except Exception as e:
                logger.warning("[ModelEngine] TCN error: %s", e)

        # TST
        tst_res: Dict[str, float] = {"price": 0.0, "vol": 0.0}
        if self.tst_loaded:
            try:
                raw = self.tst.predict(features)
                if isinstance(raw, dict):
                    tst_res = raw
                else:
                    tst_res["price"] = float(raw)
            except Exception as e:
                logger.warning("[ModelEngine] TST error: %s", e)

        # XGBoost (Returns price only, volatility assumed similar to others or 0)
        xgb_price_vote = 0.0
        if self.xgb_ready and self.xgb_service:
            xgb_df = features.get("xgb_df")
            if isinstance(xgb_df, pd.DataFrame) and not xgb_df.empty:
                try:
                    pred_val = self.xgb_service.predict_latest_for_symbol(xgb_df, symbol)
                    if pred_val is not None:
                        xgb_price_vote = float(pred_val)
                except Exception as e:
                    logger.warning("[ModelEngine] XGB error: %s", e)

        # DecisionNet (Fusion)
        try:
            decision_score = decision_net_score(features)
        except Exception as e:
            logger.warning("[ModelEngine] DecisionNet error: %s", e)
            decision_score = 0.0

        # Options Expert
        try:
            options_score = options_vol_edge(features.get("options_features", {}))
        except Exception:
            options_score = 0.0

        # Macro Expert
        try:
            macro_score = macro_onchain_bias(
                features.get("macro_onchain_features", {})
            )
        except Exception:
            macro_score = 0.0

        # LLM Narrative
        llm_sentiment: Optional[float] = None
        try:
            narrative_result = await llm_engine.get_narrative_signal(
                asset=symbol, news_text=None
            )
            llm_sentiment = float(narrative_result.get("sentiment_score", 0.0))
        except Exception as e:
            logger.warning("[ModelEngine] LLM error: %s", e)
            llm_sentiment = None

        # --- STEP 4: Compute Consensus ---

        # A. Unified Return Vote (The "Direction")
        unified_vote = self._combine_votes(
            tft=tft_res["price"],
            tcn=tcn_res["price"],
            tst=tst_res["price"],
            xgb_price=xgb_price_vote,
            xgb_vol=0.0,
            fusion=decision_score,
            options=options_score,
            macro=macro_score,
            llm_narrative=llm_sentiment,
        )

        # Direction with a "flat" band to avoid over-confident noise
        eps = 0.001  # ~0.1% move considered noise
        if unified_vote > eps:
            direction = "up"
        elif unified_vote < -eps:
            direction = "down"
        else:
            direction = "flat"

        # Map edge magnitude to a 0..1 confidence score.
        # Example: 0% -> 0.0, 2% -> 1.0 (clamped at 1.0).
        edge_mag = abs(unified_vote)
        price_confidence = float(max(0.0, min(1.0, edge_mag / 0.02)))

        # B. Unified Volatility Vote (The "Range")
        # Gather non-zero volatilities from deep learning models
        valid_vols = [
            v for v in [tft_res["vol"], tcn_res["vol"], tst_res["vol"]] if v > 0.0
        ]
        if valid_vols:
            avg_vol = float(np.mean(valid_vols))
        else:
            # Fallback if models output 0 vol (approx 0.5% hourly movement)
            avg_vol = 0.005

        # --- STEP 5: Calculate Price Targets ---

        # We treat 'unified_vote' as the expected % return for the next hour
        if current_price > 0:
            predicted_price = current_price * (1.0 + unified_vote)
            range_delta = predicted_price * avg_vol
            predicted_range_high = predicted_price + range_delta
            predicted_range_low = predicted_price - range_delta
            predicted_range_low = max(0.01, predicted_range_low)
        else:
            # No price data available -> zero prediction, minimal range
            predicted_price = 0.0
            predicted_range_high = 0.0
            predicted_range_low = 0.01

        # --- STEP 6: Create Response Object ---

        layer2 = Layer2Prediction(
            asset=symbol,
            direction=direction,
            price_confidence=price_confidence,
            current_price=current_price,
            predicted_price=predicted_price,
            predicted_range_high=predicted_range_high,
            predicted_range_low=predicted_range_low,
            tft_vote=float(tft_res["price"]),
            tcn_vote=float(tcn_res["price"]),
            tst_vote=float(tst_res["price"]),
            xgb_price_vote=xgb_price_vote,
            xgb_vol_vote=0.0,
            decision_score=float(decision_score),
            options_score=float(options_score),
            macro_score=float(macro_score),
            llm_narrative_vote=float(llm_sentiment or 0.0),
            unified_vote=float(unified_vote),
            options_features=features.get("options_features", {}),
            macro_onchain_features=features.get("macro_onchain_features", {}),
        )

        return layer2

    # --------------------------------------------------------------
    # Ensemble Logic
    # --------------------------------------------------------------

    def _combine_votes(
        self,
        tft: Optional[float],
        tcn: Optional[float],
        tst: Optional[float],
        xgb_price: Optional[float],
        xgb_vol: Optional[float],
        fusion: float,
        options: float,
        macro: float,
        llm_narrative: Optional[float],
    ) -> float:
        """
        Smart Ensemble: Uses the Stacker if available, otherwise simpler average.
        """

        # 1. Try to use the "Judge" (Stacker)
        if self.stacker_ready:
            try:
                # Pass complete list of base model predictions.
                base_preds = {
                    "tft": tft or 0.0,
                    "tcn": tcn or 0.0,
                    "tst": tst or 0.0,
                    "xgb": xgb_price or 0.0,
                    "xgb_vol": xgb_vol or 0.0,
                    "decision_net": fusion,
                    "options": options,
                    "macro": macro,
                    "llm": llm_narrative or 0.0,
                }
                return self.stacker.blend(base_preds)
            except Exception as e:
                logger.warning(
                    "[ModelEngine] Stacker failed, reverting to mean: %s", e
                )

        # 2. Fallback to Simple Average (Legacy Logic)
        vals = [
            v
            for v in [
                tft,
                tcn,
                tst,
                xgb_price,
                fusion,
                options,
                macro,
                llm_narrative,
            ]
            if v is not None
        ]

        if not vals:
            return 0.0

        return float(np.mean(vals))
