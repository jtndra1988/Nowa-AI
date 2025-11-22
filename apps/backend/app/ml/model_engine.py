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
from app.db.database import SessionLocal
from app.db.models import (
    MarketData,
    OptionsDerivedMetrics,
    OnchainMetrics,
    DeveloperActivity,
    CrossAssetCorr,
    AggregatedSentiment,  # <-- add this
)

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
        # ------------------------------------------------------------------
    # Sentiment meta helper
    # ------------------------------------------------------------------
    def _get_latest_sentiment_meta(self, base_symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch the most recent aggregated_sentiment bucket for the given base symbol.

        This uses the hourly buckets produced by sentiment_scorer / aggregated_sentiment,
        which are already generated for the top-100 symbols + GLOBAL.

        Returns a small dict with the latest scores, or None if nothing found.
        """
        session = SessionLocal()
        try:
            sym = base_symbol.upper()

            # 1) Try asset-specific sentiment (e.g. 'BTC', 'ETH', etc.).
            row = (
                session.query(AggregatedSentiment)
                .filter(AggregatedSentiment.symbol == sym)
                .order_by(AggregatedSentiment.bucket_start.desc())
                .first()
            )

            # 2) Fallback to GLOBAL if there is no per-asset sentiment yet.
            if row is None:
                row = (
                    session.query(AggregatedSentiment)
                    .filter(AggregatedSentiment.symbol == "GLOBAL")
                    .order_by(AggregatedSentiment.bucket_start.desc())
                    .first()
                )

            if row is None:
                return None

            return {
                "symbol": row.symbol,
                "bucket_start": row.bucket_start,
                "composite_score": float(row.composite_score)
                if row.composite_score is not None
                else None,
                "news_score": float(row.news_score)
                if row.news_score is not None
                else None,
                "social_score": float(row.social_score)
                if row.social_score is not None
                else None,
                "global_score": float(row.global_score)
                if row.global_score is not None
                else None,
                "source_count": int(row.source_count)
                if row.source_count is not None
                else None,
            }
        finally:
            session.close()

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
    # Symbol normalization helper
    # --------------------------------------------------------------
    def _normalize_symbol_to_base(self, symbol: str) -> str:
        """
        Normalize trading symbols like:
          - BTCUSDT, BTCUSD, BTCUSDC
          - BTC-PERP, ETH-PERP
        into their base asset: BTC, ETH, etc.

        This is important because AggregatedSentiment is stored
        per base symbol (BTC, ETH, SOL, ...), plus GLOBAL.
        """
        s = (symbol or "").upper()

        # Handle common derivatives/perp suffixes
        perp_suffixes = ["-PERP", "PERP"]
        for suf in perp_suffixes:
            if s.endswith(suf):
                return s[: -len(suf)]

        # Handle common quote currencies
        quote_suffixes = ["USDT", "USD", "USDC", "BUSD"]
        for suf in quote_suffixes:
            if s.endswith(suf):
                return s[: -len(suf)]

        # Fallback: return as-is (e.g. already "BTC")
        return s
    # --------------------------------------------------------------
    # Core Prediction Logic
    # --------------------------------------------------------------
    async def predict(self, ctx: MarketContext) -> Layer2Prediction:
        symbol = ctx.symbol
        base_symbol = self._normalize_symbol_to_base(symbol)

        # 1. Build feature set
        features = await self._build_feature_set(symbol)

        if not features:
            logger.warning("[ModelEngine] No features available. Falling back to default.")
            return Layer2Prediction(
                asset=symbol,
                direction="flat",
                price_confidence=0.0,
                current_price=0.0,
                predicted_price=0.0,
                predicted_range_high=0.0,
                predicted_range_low=0.0,
                tft_vote=0.0,
                tcn_vote=0.0,
                tst_vote=0.0,
                xgb_price_vote=0.0,
                xgb_vol_vote=0.0,
                decision_score=0.0,
                options_score=0.0,
                macro_score=0.0,
                llm_narrative_vote=0.0,
                unified_vote=0.0,
                options_features={},
                macro_onchain_features={},
                meta=None,
            )

        # 2. Run base ensemble (TFT/TCN/TST + XGB/DecisionNet etc.)
        base_scores = await self._run_base_ensemble(symbol, features)

        # 3. Run expert models (options, macro/on-chain)
        experts = await self._run_expert_models(symbol, features)

        # 4. LLM narrative + decision net / unified vote
        # (keep your existing logic here: llm_narrative_vote, decision_score, unified_vote, etc.)

        # --- NEW: sentiment meta (per base symbol, for top-100 support) ---
        sentiment_meta = self._get_latest_sentiment_meta(base_symbol)
        meta_payload: Optional[Dict[str, Any]] = None
        if sentiment_meta is not None:
            # Flat scalar for quick access + a breakdown dict for debugging / UI
            meta_payload = {
                "sentiment": sentiment_meta["composite_score"],
                "sentiment_breakdown": sentiment_meta,
            }

        # 5. Build Layer2Prediction
        return Layer2Prediction(
            asset=symbol,
            direction=direction,             # from your existing logic
            price_confidence=price_conf,     # from your existing logic
            current_price=current_price,     # from your existing logic
            predicted_price=predicted_price,
            predicted_range_high=range_high,
            predicted_range_low=range_low,
            tft_vote=tft_vote,
            tcn_vote=tcn_vote,
            tst_vote=tst_vote,
            xgb_price_vote=xgb_price_vote,
            xgb_vol_vote=xgb_vol_vote,
            decision_score=decision_score,
            options_score=options_score,
            macro_score=macro_score,
            llm_narrative_vote=llm_vote,
            unified_vote=unified_vote,
            options_features=experts.options_features if experts else {},
            macro_onchain_features=experts.macro_onchain_features if experts else {},
            meta=meta_payload,
        )

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
