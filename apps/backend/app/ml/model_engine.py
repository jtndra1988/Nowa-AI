# app/ml/model_engine.py

import logging
import numpy as np
from typing import Any, Dict, Optional

from app.hybrid.schemas import Layer2Prediction, MarketContext

# Feature engineering
from app.ml.adv.feature_engineering import FeatureBuilder  # user-provided file

# TFT / TCN / TST models
from app.ml.adv.models_tft import TFTPredictor        # user-provided
from app.ml.adv.models_tcn import TCNPredictor        # user-provided
from app.ml.adv.models_tst import TSTPredictor        # user-provided

# DecisionNet
from app.ml.adv.decision_net import decision_net_score   # user-provided

# Options / Macro-Onchain experts
from app.ml.adv.options_vol_model import options_vol_edge      # user-provided
from app.ml.adv.macro_onchain_model import macro_onchain_bias  # user-provided

logger = logging.getLogger(__name__)


class ModelEngine:
    """
    Aggregates your full ML stack into one unified prediction object.

    Layer 2 includes:
      - TFT (Visionary)
      - TCN (Reflex)
      - TST (Transformer specialist)
      - DecisionNet (Fusion)
      - XGB models (if included in training pipeline)
      - Options model
      - Macro/On-chain model
    """

    def __init__(self):
        logger.info("[ModelEngine] Initializing...")

        # Load TFT
        try:
            self.tft = TFTPredictor()
            self.tft_loaded = self.tft.is_model_loaded()
        except Exception as e:
            logger.error("[ModelEngine] Failed to load TFT: %s", e)
            self.tft = None
            self.tft_loaded = False

        # Load TCN
        try:
            self.tcn = TCNPredictor()
            self.tcn_loaded = self.tcn.is_model_loaded()
        except Exception as e:
            logger.error("[ModelEngine] Failed to load TCN: %s", e)
            self.tcn = None
            self.tcn_loaded = False

        # Load TST
        try:
            self.tst = TSTPredictor()
            self.tst_loaded = self.tst.is_model_loaded()
        except Exception as e:
            logger.error("[ModelEngine] Failed to load TST: %s", e)
            self.tst = None
            self.tst_loaded = False

        # Feature builder
        self.feature_builder = FeatureBuilder()

    # --------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return any([self.tft_loaded, self.tcn_loaded, self.tst_loaded])

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
            features = await self.feature_builder.build_features(symbol)
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

        # -----------------------------------------
        # STEP 3 → Compute unified “votes”
        # -----------------------------------------
        unified_vote = self._combine_votes(
            tft_pred, tcn_pred, tst_pred, decision_score, options_score, macro_score
        )

        # -----------------------------------------
        # STEP 4 → Create Layer2Prediction
        # -----------------------------------------
        layer2 = Layer2Prediction(
            symbol=symbol,
            tft_vote=float(tft_pred or 0.0),
            tcn_vote=float(tcn_pred or 0.0),
            tst_vote=float(tst_pred or 0.0),
            decision_score=float(decision_score),
            options_features=features.get("options_features", {}),
            macro_onchain_features=features.get("macro_onchain_features", {}),
            unified_vote=float(unified_vote),
        )

        return layer2

    # --------------------------------------------------------------

    def _combine_votes(
        self,
        tft: Optional[float],
        tcn: Optional[float],
        tst: Optional[float],
        fusion: float,
        options: float,
        macro: float,
    ) -> float:
        """
        Weighted meta-ensemble.
        Weights can be tuned; here simple normalized sum.
        """

        vals = [
            v for v in [tft, tcn, tst, fusion, options, macro]
            if v is not None
        ]

        if not vals:
            return 0.0

        # Simple normalized ensemble
        arr = np.array(vals, dtype=float)
        return float(np.mean(arr))
