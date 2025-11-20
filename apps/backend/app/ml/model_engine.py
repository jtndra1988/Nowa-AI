# app/ml/model_engine.py

import logging
import numpy as np
from typing import Any, Dict, Optional, List
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import desc

# Core Schemas & DB
from app.hybrid.schemas import Layer2Prediction, MarketContext
from app.db.database import SessionLocal
from app.db import models

# Registry & Feature Engineering
from app.ml.adv.model_registry import model_registry
from app.ml.adv.feature_engineering import FeatureBuilder

# Model Wrappers (for type hinting if needed, though Registry handles instantiation)
from app.ml.adv.models_tft import TFTPredictor
from app.ml.adv.models_tcn import TCNPredictor
from app.ml.adv.models_tst import TSTPredictor

# DecisionNet & Experts
from app.ml.adv.decision_net import decision_net_score
from app.ml.adv.options_vol_model import options_vol_edge
from app.ml.adv.macro_onchain_model import macro_onchain_bias
from app.ml.adv.llm_narrative_model import llm_engine

# Ensemble
from app.ml.ensemble import EnsembleStackerService

logger = logging.getLogger(__name__)


class ModelEngine:
    """
    Aggregates your full ML stack into one unified prediction object.

    Layer 1 (Predictors):
      - TFT (Visionary)
      - TCN (Reflex)
      - TST (Transformer specialist)
      - XGB (Tabular analyst)

    Layer 2 (Fusion):
      - DecisionNet (Neural Fusion)
      - EnsembleStacker (Linear Meta-Learner)

    Layer 3 (Experts):
      - Options Expert
      - Macro/On-chain Expert
      - LLM Narrative
    """

    def __init__(self):
        logger.info("[ModelEngine] Initializing...")

        # 1. Central Registry (Singleton)
        self.registry = model_registry

        # 2. Initialize Predictors via Registry
        # XGB Service
        self.xgb_service = self.registry.get_model("xgb")
        self.xgb_ready = bool(self.xgb_service and getattr(self.xgb_service, "is_ready", False))

        # TFT
        self.tft = self.registry.get_model("tft")
        self.tft_loaded = bool(self.tft and getattr(self.tft, "is_model_loaded", lambda: False)())

        # TCN
        self.tcn = self.registry.get_model("tcn")
        self.tcn_loaded = bool(self.tcn and getattr(self.tcn, "is_model_loaded", lambda: False)())

        # TST
        self.tst = self.registry.get_model("tst")
        self.tst_loaded = bool(self.tst and getattr(self.tst, "is_model_loaded", lambda: False)())

        # 3. Initialize Stacker (The "Judge")
        # Ensure 'ensemble' is registered in your model_registry.py
        self.stacker = self.registry.get_model("ensemble")
        self.stacker_ready = bool(self.stacker and getattr(self.stacker, "is_ready", False))

        # 4. Initialize Data Ingestors
        self.options_ingestor = self.registry.get_model("options_ingestor")
        self.macro_ingestor = self.registry.get_model("macro_ingestor")

        # 5. Feature Builder (Direct instantiation as it handles data logic)
        self.feature_builder = FeatureBuilder()

    # --------------------------------------------------------------
    # Data Loading & Feature Prep
    # --------------------------------------------------------------

    def _load_recent_ohlcv_for_symbol(self, symbol: str, limit: int = 300) -> pd.DataFrame:
        """
        Load recent OHLCV for XGBoost.
        Mirrors schema: ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
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
                logging.warning(f"[ModelEngine] No OHLCV rows found for symbol {symbol} (XGB).")
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
            return pd.DataFrame(data).sort_values("timestamp").reset_index(drop=True)

        except Exception as e:
            logging.error(f"[ModelEngine] Failed to load OHLCV for XGB ({symbol}): {e}", exc_info=True)
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
                raise TypeError(f"FeatureBuilder returned {type(raw_features)}, expected dict")
            
            features: Dict[str, Any] = dict(raw_features)

            # Normalize Tensor Shapes [L, F] -> [B, L, F]
            price = features.get("price")
            try:
                import torch
                if isinstance(price, torch.Tensor) and price.ndim == 2:
                    features["price"] = price.unsqueeze(0)
            except Exception:
                pass

            # 2. Expert Features (Options / Macro)
            # Ingest live if available, otherwise keep defaults
            if self.macro_ingestor:
                try:
                    ingested = self.macro_ingestor.build_features(symbol)
                    if ingested: features["macro_onchain_features"] = ingested
                except Exception as e:
                    logger.warning(f"[ModelEngine] Macro ingest failed: {e}")

            if self.options_ingestor:
                try:
                    ingested = self.options_ingestor.build_features(symbol)
                    if ingested: features["options_features"] = ingested
                except Exception as e:
                    logger.warning(f"[ModelEngine] Options ingest failed: {e}")

            # 3. Tabular Data (XGBoost)
            if "xgb_df" not in features or features["xgb_df"] is None:
                xgb_df = self._load_recent_ohlcv_for_symbol(symbol)
                if not xgb_df.empty:
                    features["xgb_df"] = xgb_df
            
            return features

        except Exception as e:
            logging.error(f"[ModelEngine] Feature build failed for {symbol}: {e}", exc_info=True)
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
        Unified prediction call used by HybridInferenceService.
        """
        symbol = ctx.symbol.upper()

        # --- STEP 1: Build Features ---
        try:
            features = await self._build_feature_set(symbol)
        except Exception as e:
            logger.error(f"[ModelEngine] Aborting predict due to feature error: {e}")
            raise

        # --- STEP 2: Run Individual Models ---

        # TFT
        tft_pred = None
        if self.tft_loaded:
            try:
                tft_pred = self.tft.predict(features)
            except Exception as e:
                logger.warning(f"[ModelEngine] TFT error: {e}")

        # TCN
        tcn_pred = None
        if self.tcn_loaded:
            try:
                tcn_pred = self.tcn.predict(features)
            except Exception as e:
                logger.warning(f"[ModelEngine] TCN error: {e}")

        # TST
        tst_pred = None
        if self.tst_loaded:
            try:
                tst_pred = self.tst.predict(features)
            except Exception as e:
                logger.warning(f"[ModelEngine] TST error: {e}")

        # XGBoost
        xgb_price_vote: Optional[float] = None
        xgb_vol_vote: Optional[float] = None
        if self.xgb_ready and self.xgb_service:
            xgb_df = features.get("xgb_df")
            if xgb_df is not None:
                try:
                    pred_val = self.xgb_service.predict_latest_for_symbol(xgb_df, symbol)
                    if pred_val is not None:
                        xgb_price_vote = float(pred_val)
                        xgb_vol_vote = float(pred_val) # Placeholder for vol model
                except Exception as e:
                    logger.warning(f"[ModelEngine] XGB error: {e}")

        # DecisionNet
        try:
            decision_score = decision_net_score(features)
        except Exception as e:
            logger.warning(f"[ModelEngine] DecisionNet error: {e}")
            decision_score = 0.0

        # Options Expert
        try:
            options_score = options_vol_edge(features.get("options_features", {}))
        except Exception:
            options_score = 0.0

        # Macro Expert
        try:
            macro_score = macro_onchain_bias(features.get("macro_onchain_features", {}))
        except Exception:
            macro_score = 0.0

        # LLM Narrative
        llm_sentiment: Optional[float] = None
        try:
            narrative_result = await llm_engine.get_narrative_signal(asset=symbol, news_text=None)
            llm_sentiment = float(narrative_result.get("sentiment_score", 0.0))
        except Exception as e:
            logger.warning(f"[ModelEngine] LLM error: {e}")
            llm_sentiment = None

        # --- STEP 3: Compute Unified Votes (Ensemble) ---
        unified_vote = self._combine_votes(
            tft=tft_pred,
            tcn=tcn_pred,
            tst=tst_pred,
            xgb_price=xgb_price_vote,
            xgb_vol=xgb_vol_vote,
            fusion=decision_score,
            options=options_score,
            macro=macro_score,
            llm_narrative=llm_sentiment,
        )

        # --- STEP 4: Create Response Object ---
        layer2 = Layer2Prediction(
            asset=symbol,
            tft_vote=float(tft_pred or 0.0),
            tcn_vote=float(tcn_pred or 0.0),
            tst_vote=float(tst_pred or 0.0),
            xgb_price_vote=float(xgb_price_vote or 0.0),
            xgb_vol_vote=float(xgb_vol_vote or 0.0),
            decision_score=float(decision_score),
            options_features=features.get("options_features", {}),
            macro_onchain_features=features.get("macro_onchain_features", {}),
            options_score=float(options_score),
            macro_score=float(macro_score),
            llm_narrative_vote=float(llm_sentiment or 0.0),
            unified_vote=float(unified_vote),
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
                # The stacker's .blend() method will safely ignore keys 
                # that it was not trained on, so it is safe to send everything.
                base_preds = {
                    "tft": tft or 0.0,
                    "tcn": tcn or 0.0,
                    "tst": tst or 0.0,
                    "xgb": xgb_price or 0.0,       # Key matches DEFAULT_BASE_MODELS
                    "xgb_vol": xgb_vol or 0.0,
                    "decision_net": fusion,
                    "options": options,
                    "macro": macro,
                    "llm": llm_narrative or 0.0
                }
                return self.stacker.blend(base_preds)
            except Exception as e:
                logger.warning(f"[ModelEngine] Stacker failed, reverting to mean: {e}")

        # 2. Fallback to Simple Average (Legacy Logic)
        vals = [
            v for v in [
                tft, 
                tcn, 
                tst, 
                xgb_price, 
                fusion, 
                options, 
                macro, 
                llm_narrative
            ]
            if v is not None
        ]
        
        if not vals:
            return 0.0
            
        return float(np.mean(vals))