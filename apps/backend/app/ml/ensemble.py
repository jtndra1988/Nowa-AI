import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import joblib
import torch
import torch.nn as nn

from app.core.config import settings

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Optional PyTorch stacker (not used by current training, but kept for future)
# -----------------------------------------------------------------------------

class StackingEnsemble(nn.Module):
    """
    PyTorch stacking blender (optional).

    Learns a convex combination of N base model predictions.
    Input:
        preds: Tensor [batch_size, n_models]
    Output:
        blended prediction: Tensor [batch_size]
    """
    def __init__(self, n_models: int):
        super().__init__()
        # Initialize with equal weights
        self.w = nn.Parameter(torch.ones(n_models) / n_models)

    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        # Softmax to ensure weights are positive and sum to 1
        w = torch.softmax(self.w, dim=0)       # [n_models]
        # Weighted sum across model dimension
        return (preds * w).sum(dim=1)          # [batch_size]


# -----------------------------------------------------------------------------
# Scikit-learn stacker runtime wrapper
# -----------------------------------------------------------------------------

# Where the "current production" ensemble artifact is stored
ARTIFACTS_DIR = Path(getattr(settings, "MODEL_ARTIFACTS_DIR", "model_artifacts"))
ENSEMBLE_PATH = ARTIFACTS_DIR / "ensemble_model.pkl"

# Default order of base model predictions.
# This MUST match the order used in train_ensemble.py
DEFAULT_BASE_MODELS: List[str] = ["tft", "tcn", "xgb"]


class EnsembleStackerService:
    """
    Production inference wrapper for the trained LinearRegression stacker.

    - Loads the joblib artifact saved by app/ml/train_ensemble.py
    - Expects base model predictions in a dict keyed by base model names,
      e.g. {'tft': 0.01, 'tcn': 0.008, 'xgb': 0.012}
    - Returns a blended scalar prediction.

    The artifact structure is:
        {
          "model": sklearn LinearRegression (or similar),
          "base_models": ["tft", "tcn", "xgb"],
          "metrics": {"rmse": float, ...}
        }
    """

    def __init__(self, artifact_path: str = "model_artifacts/ensemble_stacker.pkl") -> None:
        self.artifact_path = Path(artifact_path)

        self.model = None               # sklearn LinearRegression (or similar)
        self.base_models: List[str] = []  # e.g. ["tft", "tcn", "xgb"]
        self.metrics: Dict[str, Any] = {}  # {"rmse": ..., "r2": ...}
        self.trained_at_utc: Optional[str] = None
        self.version: str = "unknown"

        self._load_artifact()

    # ------------------------------------------------------------------
    # Loading / readiness
    # ------------------------------------------------------------------

    def _load_artifact(self) -> None:
        if not self.artifact_path.exists():
            logger.warning(
                "[EnsembleStackerService] Artifact not found at %s",
                self.artifact_path,
            )
            return

        artifact = joblib.load(self.artifact_path)

        # Required
        self.model = artifact.get("model")

        # Optional / metadata
        # train_ensemble.py already sets at least "base_models" and "metrics"
        self.base_models = artifact.get("base_models", [])
        self.metrics = artifact.get("metrics", {})

        # If you add these in train_ensemble.py, they’ll show up here:
        meta = artifact.get("metadata", {})
        self.trained_at_utc = (
            artifact.get("trained_at_utc")
            or meta.get("trained_at_utc")
            or meta.get("train_date_utc")
        )
        self.version = (
            artifact.get("version")
            or meta.get("version")
            or "unknown"
        )

        logger.info(
            "[EnsembleStackerService] Loaded ensemble artifact from %s "
            "(version=%s, base_models=%s)",
            self.artifact_path,
            self.version,
            self.base_models,
        )

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": "ensemble_stacker",
            "version": self.version,
            "trained_at_utc": self.trained_at_utc,
            "base_models": self.base_models,
            "metrics": self.metrics,
        }

    @property
    def is_ready(self) -> bool:
        return self.model is not None and bool(self.base_models)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def blend(self, base_preds: Dict[str, Any]) -> float:
        """
        Blend base model predictions using the trained stacker.

        Args:
            base_preds: dict like
                        {
                            "tft":        0.0123,
                            "tcn":        0.0110,
                            "xgb":        0.0105,
                            ...
                        }

        Returns:
            blended_pred: float
        """
        if not self.is_ready:
            raise RuntimeError(
                "[EnsembleStackerService] Stacker is not ready. "
                "Check that training ran and ensemble_model.pkl exists."
            )

        # Build input vector in the same order as training
        try:
            x_row = [
                float(base_preds.get(name, 0.0))
                for name in self.base_models
            ]
        except Exception as e:  # noqa: BLE001
            logger.error(
                "[EnsembleStackerService] Failed to coerce base predictions to float: %s",
                e,
                exc_info=True,
            )
            raise

        X = np.array([x_row], dtype=float)   # shape [1, n_base_models]
        try:
            y_hat = self.model.predict(X)    # type: ignore[call-arg]
        except Exception as e:  # noqa: BLE001
            logger.error("[EnsembleStackerService] Prediction error: %s", e, exc_info=True)
            raise

        return float(y_hat[0])


# -----------------------------------------------------------------------------
# Routing logic: when to use Stacker vs DecisionNet
# -----------------------------------------------------------------------------

def route_fusion(
    stacker_pred: Optional[float],
    decision_score: Optional[float],
    regime_volatility: Optional[float] = None,
    mode: str = "auto",
) -> float:
    """
    Routing logic between the numeric stacker output and DecisionNet score.

    Args:
        stacker_pred:   Continuous expected return (e.g., +0.01 = +1%),
                        produced by EnsembleStackerService.blend().
        decision_score: Directional score in [-1, 1], from DecisionNet.
                        (+1 = strong long, -1 = strong short, 0 = flat)
        regime_volatility:
                        Optional realized/forecast vol (e.g. 0.02 for 2% 1h vol)
                        used to adapt thresholds in 'auto' mode.
        mode:
            - "stacker_only"   → always trust stacker_pred
            - "decision_only"  → always trust decision_score
            - "average"        → simple average (after scaling decision_score)
            - "auto" (default) → pick strategy based on signal strength

    Returns:
        fused_meta_signal: float

        The typical usage is:
          - treat this fused value as a "meta" expected return / strength signal,
          - feed it into the RL agent or risk engine as one of the inputs.
    """
    # Handle missing values gracefully
    if stacker_pred is None and decision_score is None:
        return 0.0
    if stacker_pred is None:
        stacker_pred = 0.0
    if decision_score is None:
        decision_score = 0.0

    # Explicit modes
    if mode == "stacker_only":
        return float(stacker_pred)

    if mode == "decision_only":
        return float(decision_score)

    if mode == "average":
        # Map decision_score in [-1, 1] to a pseudo-return scale (~2% max)
        decision_as_return = float(decision_score) * 0.02
        return float(0.5 * stacker_pred + 0.5 * decision_as_return)

    if mode == "auto":
        # Threshold for "strong" stacker signal
        # Default: 0.5% expected move; scale up a bit in high vol regimes.
        base_thresh = 0.005  # 0.5%
        if regime_volatility is not None and regime_volatility > 0:
            base_thresh = max(base_thresh, 0.5 * float(regime_volatility))

        # If stacker has high conviction (|ret| >= thresh), prefer it.
        # Otherwise, rely more on DecisionNet's directional bias.
        if abs(stacker_pred) >= base_thresh:
            return float(stacker_pred)
        return float(decision_score)

    raise ValueError(f"[route_fusion] Unknown mode: {mode}")
