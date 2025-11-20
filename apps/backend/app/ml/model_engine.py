# app/ml/model_engine.py

import logging
import numpy as np
from typing import Any, Dict, Optional, List
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.hybrid.schemas import Layer2Prediction, MarketContext
from app.db.database import SessionLocal
from app.db import models
from app.ml.adv.model_registry import model_registry
# Feature engineering
from app.ml.adv.feature_engineering import FeatureBuilder  # user-provided file

# TFT / TCN / TST models
from app.ml.adv.models_tft import TFTPredictor        # user-provided
from app.ml.adv.models_tcn import TCNPredictor        # user-provided
from app.ml.adv.models_tst import TSTPredictor        # user-provided

# DecisionNet
from app.ml.adv.decision_net import decision_net_score   # user-provided

# Options / Macro-Onchain experts
from app.ml.adv.options_vol_model import (
    options_vol_edge,
    OptionsVolFeatureIngestion,
)
from app.ml.adv.macro_onchain_model import (
    macro_onchain_bias,
    MacroOnchainFeatureIngestion,
)
from app.ml.adv.llm_narrative_model import llm_engine
# XGB tabular specialist
from app.ml.adv.inference_xgb import XGBInferenceService

logger = logging.getLogger(__name__)


class ModelEngine:
    """
    Aggregates your full ML stack into one unified prediction object.

    Layer 2 includes:
      - TFT (Visionary)
      - TCN (Reflex)
      - TST (Transformer specialist)
      - DecisionNet (Fusion)
      - XGB models (tabular analyst)
      - Options model
      - Macro/On-chain model
    """

    def __init__(self):
        logger.info("[ModelEngine] Initializing...")

        # Central registry (singleton)
        self.registry = model_registry

        # XGB service (tabular analyst)
        self.xgb_service = self.registry.get_model("xgb")
        self.xgb_ready = bool(
            self.xgb_service and getattr(self.xgb_service, "is_ready", False)
        )

        # TFT
        self.tft = self.registry.get_model("tft")
        self.tft_loaded = bool(
            self.tft
            and getattr(self.tft, "is_model_loaded", lambda: False)()
        )

        # TCN
        self.tcn = self.registry.get_model("tcn")
        self.tcn_loaded = bool(
            self.tcn
            and getattr(self.tcn, "is_model_loaded", lambda: False)()
        )

        # TST
        self.tst = self.registry.get_model("tst")
        self.tst_loaded = bool(
            self.tst
            and getattr(self.tst, "is_model_loaded", lambda: False)()
        )

        # Options feature ingestor
        self.options_ingestor = self.registry.get_model("options_ingestor")

        # Macro + on-chain feature ingestor
        self.macro_ingestor = self.registry.get_model("macro_ingestor")

        # Feature builder (data-side, not in registry)
        self.feature_builder = FeatureBuilder()

    def _load_recent_ohlcv_for_symbol(
        self,
        symbol: str,
        limit: int = 300,
    ) -> pd.DataFrame:
        """
        Load recent OHLCV for a single symbol from MarketData.

        This is used to feed XGBInferenceService; it deliberately
        mirrors the schema used in train_xgb.py:
            ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']

        Returns:
            DataFrame (may be empty if no rows found).
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
                logging.warning(
                    "[ModelEngine] No OHLCV rows found for symbol %s for XGB.",
                    symbol,
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

            df = pd.DataFrame(data).sort_values("timestamp").reset_index(drop=True)
            return df

        except Exception as e:
            logging.error(
                "[ModelEngine] Failed to load OHLCV for XGB (symbol=%s): %s",
                symbol,
                e,
                exc_info=True,
            )
            return pd.DataFrame()
        finally:
            session.close()
    async def _build_feature_set(self, symbol: str) -> Dict[str, Any]:
        """
        Build the full feature dictionary for all models.

        - Uses FeatureBuilder for sequence features (TFT/TCN/TST, options, macro).
        - Ensures tensors have batch dimension where needed.
        - Attaches a raw OHLCV DataFrame at features['xgb_df'] for XGBInferenceService.
        """
        try:
            raw_features = await self.feature_builder.build_features(symbol)

            if not isinstance(raw_features, dict):
                raise TypeError(
                    f"FeatureBuilder.build_features() must return dict, got {type(raw_features)}"
                )

            # Shallow copy so we can normalize without mutating original
            features: Dict[str, Any] = dict(raw_features)

            # --- Normalize 'price' tensor shape for deep models ---
            price = features.get("price")
            try:
                import torch  # local import to avoid hard dependency at module import time

                if isinstance(price, torch.Tensor):
                    # FeatureBuilder currently returns [L, F]; models expect [B, L, F]
                    if price.ndim == 2:
                        features["price"] = price.unsqueeze(0)  # [1, L, F]
                    elif price.ndim == 3:
                        # already [B, L, F] – leave as is
                        pass
                    else:
                        logging.warning(
                            "[ModelEngine] Unexpected 'price' tensor ndim=%d; expected 2 or 3.",
                            price.ndim,
                        )
                elif price is not None:
                    logging.warning(
                        "[ModelEngine] 'price' feature is not a tensor (type=%s); "
                        "TFT/TCN/TST may be skipped.",
                        type(price),
                    )
            except Exception as e:
                logging.warning(
                    "[ModelEngine] Failed to normalize 'price' feature: %s", e, exc_info=True
                )

            # --- Macro + on-chain features ---
            macro_features = features.get("macro_onchain_features", {})
            if self.macro_ingestor is not None:
                try:
                    ingested_macro = self.macro_ingestor.build_features(symbol)
                    if ingested_macro:
                        macro_features = ingested_macro
                except Exception as e:
                    logger.warning(
                        "[ModelEngine] MacroOnchainFeatureIngestion failed for %s: %s",
                        symbol,
                        e,
                        exc_info=True,
                    )
            features["macro_onchain_features"] = macro_features


            # Options features: prefer live ingestion from OptionsDerivedMetrics.
            # If ingestion fails or no data, fall back to any precomputed features (or empty dict).
            options_features = features.get("options_features", {})
            if self.options_ingestor is not None:
                try:
                    ingested_opts = self.options_ingestor.build_features(symbol)
                    if ingested_opts:
                        options_features = ingested_opts
                except Exception as e:
                    logging.warning(
                        "[ModelEngine] OptionsVolFeatureIngestion failed for %s: %s",
                        symbol,
                        e,
                        exc_info=True,
                    )
            features["options_features"] = options_features

            # --- Attach XGB raw OHLCV DataFrame ---
            # If FeatureBuilder already added 'xgb_df', keep it; otherwise we fetch from DB.
            if "xgb_df" not in features or features["xgb_df"] is None:
                xgb_df = self._load_recent_ohlcv_for_symbol(symbol)
                if not xgb_df.empty:
                    features["xgb_df"] = xgb_df
                else:
                    # Not fatal: XGBInferenceService will skip if xgb_df is missing
                    logging.debug(
                        "[ModelEngine] No XGB OHLCV available for symbol %s; "
                        "XGB will be skipped for this request.",
                        symbol,
                    )

            return features

        except Exception as e:
            logging.error("[ModelEngine] Feature construction failed for %s: %s", symbol, e, exc_info=True)
            # In production, it's better to surface a clean error than silently continue
            raise

    # --------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """
        Engine is 'ready' if at least one of the core models (TFT/TCN/TST/XGB)
        is available.
        """
        return any([
            self.tft_loaded,
            self.tcn_loaded,
            self.tst_loaded,
            self.xgb_ready,
        ])

    # --------------------------------------------------------------

    async def predict(self, ctx: MarketContext) -> Layer2Prediction:
        """
        Unified prediction call used by HybridInferenceService.
        """

        symbol = ctx.symbol.upper()

        # -----------------------------------------
        # STEP 1 → Build feature set
        # -----------------------------------------
        try:
            # Expect FeatureBuilder to return a dict of feature blocks.
            # You can extend it to also add 'xgb_df' with raw OHLCV for XGB.
            features = await self._build_feature_set(symbol)
        except Exception as e:
            logger.error("[ModelEngine] Feature construction failed: %s", e)
            raise

        # -----------------------------------------
        # STEP 2 → Run individual models
        # -----------------------------------------

        # TFT
        tft_pred = None
        if self.tft_loaded:
            try:
                tft_pred = self.tft.predict(features)
            except Exception as e:
                logger.warning("[ModelEngine] TFT error: %s", e)

        # TCN
        tcn_pred = None
        if self.tcn_loaded:
            try:
                tcn_pred = self.tcn.predict(features)
            except Exception as e:
                logger.warning("[ModelEngine] TCN error: %s", e)

        # TST
        tst_pred = None
        if self.tst_loaded:
            try:
                tst_pred = self.tst.predict(features)
            except Exception as e:
                logger.warning("[ModelEngine] TST error: %s", e)

        # XGB – tabular analyst
        xgb_price_vote: Optional[float] = None
        xgb_vol_vote: Optional[float] = None

        if self.xgb_ready and self.xgb_service is not None:
            # We need a raw OHLCV DataFrame for XGB.
            # Convention: FeatureBuilder adds 'xgb_df' into features if available.
            xgb_df = features.get("xgb_df")
            if xgb_df is None:
                logger.debug(
                    "[ModelEngine] No 'xgb_df' found in features; "
                    "XGBInferenceService will be skipped."
                )
            else:
                try:
                    # Use latest prediction for this symbol
                    pred_val = self.xgb_service.predict_latest_for_symbol(
                        df_raw=xgb_df,
                        symbol=symbol,
                    )
                    if pred_val is not None:
                        xgb_price_vote = float(pred_val)
                        # TODO: if you later train a dedicated vol model,
                        #       set xgb_vol_vote accordingly.
                        xgb_vol_vote = float(pred_val)
                except Exception as e:
                    logger.warning("[ModelEngine] XGB error: %s", e)

        # DecisionNet (fusion of features)
        try:
            decision_score = decision_net_score(features)
        except Exception as e:
            logger.warning("[ModelEngine] DecisionNet error: %s", e)
            decision_score = 0.0

        # Options expert
        try:
            options_score = options_vol_edge(features.get("options_features", {}))
        except Exception:
            options_score = 0.0

        # Macro/On-chain expert
        try:
            macro_score = macro_onchain_bias(features.get("macro_onchain_features", {}))
        except Exception:
            macro_score = 0.0

        # LLM Narrative expert
        llm_sentiment: Optional[float] = None
        try:
            # For now we don't pass explicit news (you can later wire in aggregated news text).
            narrative_result = await llm_engine.get_narrative_signal(
                asset=symbol,
                news_text=None,
            )
            llm_sentiment = float(narrative_result.get("sentiment_score", 0.0))
        except Exception as e:
            logger.warning("[ModelEngine] LLM narrative engine error: %s", e)
            llm_sentiment = None

        # -----------------------------------------
        # STEP 3 → Compute unified “votes”
        # -----------------------------------------
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

        # -----------------------------------------
        # STEP 4 → Create Layer2Prediction
        # -----------------------------------------
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
        Weighted meta-ensemble.
        Weights can be tuned; here simple normalized mean.

        Includes:
          - tft, tcn, tst
          - xgb_price, xgb_vol
          - fusion (DecisionNet)
          - options, macro
        """

        vals: List[float] = [
            v
            for v in [
                tft,
                tcn,
                tst,
                xgb_price,
                xgb_vol,
                fusion,
                options,
                macro,
                llm_narrative,
            ]
            if v is not None
        ]

        if not vals:
            return 0.0

        arr = np.array(vals, dtype=float)
        return float(np.mean(arr))
