import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse the canonical XGB helpers from training
from app.ml.train_xgb import (
    ARTIFACT_PATH,
    load_xgb_artifact,
    build_features_for_inference,
)

logger = logging.getLogger(__name__)


class XGBInferenceService:
    """
    Production XGBoost/GBM inference service.

    Responsibilities:
      - Load the trained XGB artifact (model + scaler + feature_cols)
      - Rebuild tabular features from raw OHLCV using the SAME pipeline as training
      - Apply the SAME StandardScaler
      - Expose simple predict(...) methods for Nowa's HybridInferenceService / ModelEngine.

    The artifact is expected at:
        app.core.config.settings.MODEL_ARTIFACTS_DIR / "xgb_model.pkl"

    This file is created by app/ml/train_xgb.py.
    """

    def __init__(self, artifact_path: Optional[Path] = None) -> None:
        # Allow overriding path, but default to the training artifact location
        self.artifact_path: Path = Path(artifact_path) if artifact_path else ARTIFACT_PATH

        self.model = None
        self.scaler = None
        self.feature_cols: Optional[List[str]] = None
        self.metrics: Dict[str, Any] = {}

        self._load_artifact()

    # ------------------------------------------------------------------
    # Internal loading / readiness
    # ------------------------------------------------------------------

    def _load_artifact(self) -> None:
        """
        Load model, scaler, feature_cols, and metrics from disk.
        Uses the canonical load_xgb_artifact() from train_xgb.
        """
        try:
            artifact = load_xgb_artifact(self.artifact_path)
        except FileNotFoundError:
            logger.warning("[XGBInferenceService] Artifact not found at %s", self.artifact_path)
            return
        except Exception as e:  # noqa: BLE001
            logger.error(
                "[XGBInferenceService] Failed to load XGB artifact: %s",
                e,
                exc_info=True,
            )
            return

        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.feature_cols = artifact["feature_cols"]
        self.metrics = artifact.get("metrics", {})

        logger.info(
            "[XGBInferenceService] Loaded XGB model from %s (features=%d, metrics=%s)",
            self.artifact_path,
            len(self.feature_cols or []),
            self.metrics,
        )

    @property
    def is_ready(self) -> bool:
        """
        Service is 'ready' if:
          - model, scaler, and feature_cols are successfully loaded.
        """
        return self.model is not None and self.scaler is not None and bool(self.feature_cols)

    # ------------------------------------------------------------------
    # Public prediction APIs
    # ------------------------------------------------------------------

    def predict_from_df(
        self,
        df_raw: pd.DataFrame,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Main inference method.

        Args:
            df_raw: DataFrame with at least:
                    ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']

        Returns:
            preds:   np.ndarray [N] – model-predicted next-horizon returns
            df_feat: DataFrame aligned with preds, including:
                     - original symbol/timestamp
                     - engineered feature columns
        """
        if not self.is_ready:
            raise RuntimeError(
                "[XGBInferenceService] Service not ready – model/scaler/feature_cols missing."
            )

        # Rebuild the EXACT same features as in training
        df_feat = build_features_for_inference(df_raw, self.feature_cols)  # type: ignore[arg-type]

        if df_feat.empty:
            raise RuntimeError(
                "[XGBInferenceService] No valid rows after feature engineering for inference."
            )

        X = df_feat[self.feature_cols].values.astype(float)  # type: ignore[index]
        X_scaled = self.scaler.transform(X)

        preds = self.model.predict(X_scaled)
        return preds, df_feat

    def predict_latest_for_symbol(
        self,
        df_raw: pd.DataFrame,
        symbol: str,
    ) -> Optional[float]:
        """
        Convenience helper: returns the latest prediction for a given symbol
        (most recent timestamp row after feature engineering).

        Args:
            df_raw: DataFrame with OHLCV rows (possibly multiple symbols)
            symbol: Symbol to filter on (e.g., 'BTCUSDT')

        Returns:
            float (prediction) or None if no valid rows.
        """
        preds, df_feat = self.predict_from_df(df_raw)

        # Filter by symbol and pick last time row
        mask = df_feat["symbol"] == symbol
        symbol_df = df_feat[mask]

        if symbol_df.empty:
            logger.warning(
                "[XGBInferenceService] No engineered rows for symbol %s during inference.",
                symbol,
            )
            return None

        # Align preds with df_feat index
        symbol_idx = symbol_df.index
        symbol_preds = preds[symbol_idx]

        if len(symbol_preds) == 0:
            return None

        # Return last prediction (most recent timestamp)
        return float(symbol_preds[-1])

    def predict_from_rows(
        self,
        rows: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Convenience helper: accept a list of raw OHLCV rows (e.g. from an API),
        convert to DataFrame, and run the full XGB pipeline.

        Each row should contain at least:
          {
            "symbol": str,
            "timestamp": datetime or str,
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float
          }
        """
        df_raw = pd.DataFrame(rows)
        return self.predict_from_df(df_raw)
