from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db import models

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model artifact + metadata
# ---------------------------------------------------------------------------

# Runtime "current" artifact used in production.
_OPTIONS_MODEL_PATH = Path("./model_artifacts/options_vol_edge.joblib")

# Optional sidecar metadata file (if your training script writes it separately).
_OPTIONS_METADATA_PATH = Path("./model_artifacts/options_vol_edge_metadata.json")

# In-memory singletons
_options_model: Optional[Any] = None
_options_metadata: Dict[str, Any] = {}


def _lazy_load_options_model() -> None:
    """
    Lazy-load the options vol model + metadata.

    Expected artifact format (recommended):

        artifact = {
            "model": trained_estimator,
            "metadata": {
                "version": "v1.0",
                "trained_at_utc": "...",
                "feature_names": ["iv_rank", "risk_reversal_25d", "term_structure_slope"],
            },
        }

        joblib.dump(artifact, _OPTIONS_MODEL_PATH)

    For backward compatibility, if the artifact is just a model object, we still
    load it and leave metadata empty (or try reading the JSON sidecar).
    """
    global _options_model, _options_metadata

    if _options_model is not None:
        return

    if not _OPTIONS_MODEL_PATH.exists():
        logger.warning(
            "[OptionsVol] Model artifact not found at %s. "
            "Falling back to heuristic scoring.",
            _OPTIONS_MODEL_PATH,
        )
        return

    try:
        artifact = joblib.load(_OPTIONS_MODEL_PATH)
    except Exception as e:  # noqa: BLE001
        logger.error(
            "[OptionsVol] Failed to load options vol model from %s: %s",
            _OPTIONS_MODEL_PATH,
            e,
            exc_info=True,
        )
        return

    # New format: dict with model + metadata
    if isinstance(artifact, dict) and "model" in artifact:
        _options_model = artifact["model"]
        _options_metadata = artifact.get("metadata", {})
    else:
        # Legacy: artifact IS the model
        _options_model = artifact
        # Try a sidecar JSON for metadata
        if _OPTIONS_METADATA_PATH.exists():
            try:
                import json

                with open(_OPTIONS_METADATA_PATH, "r", encoding="utf-8") as f:
                    _options_metadata = json.load(f)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[OptionsVol] Failed to load metadata sidecar %s: %s",
                    _OPTIONS_METADATA_PATH,
                    e,
                )


def get_options_model_metadata() -> Dict[str, Any]:
    """
    Returns a copy of the loaded options model metadata
    (version, trained_at_utc, feature_names, etc.), or {} if unavailable.
    """
    _lazy_load_options_model()
    return dict(_options_metadata) if _options_metadata else {}


# ---------------------------------------------------------------------------
# Ingestion wrapper for live options-derived data
# ---------------------------------------------------------------------------


class OptionsVolFeatureIngestion:
    """
    Ingestion wrapper for options-derived metrics.

    Responsibilities:
      - Fetch latest OptionsDerivedMetrics row for a given symbol.
      - Transform it into the feature dict expected by `options_vol_edge`.
      - Provide safe fallbacks (empty dict) when no data is present.

    This centralizes the mapping from DB schema → model feature space and
    keeps the pipeline consistent and versionable.
    """

    def __init__(self, session_factory=SessionLocal) -> None:
        self._session_factory = session_factory

    def _get_latest_row(self, symbol: str) -> Optional[models.OptionsDerivedMetrics]:
        """
        Returns the most recent OptionsDerivedMetrics row for `symbol`, or None.
        """
        db: Session = self._session_factory()
        try:
            return (
                db.query(models.OptionsDerivedMetrics)
                .filter(models.OptionsDerivedMetrics.symbol == symbol.upper())
                .order_by(models.OptionsDerivedMetrics.timestamp.desc())
                .first()
            )
        except Exception as e:  # noqa: BLE001
            logger.error(
                "[OptionsVolFeatureIngestion] Failed to load options_derived_metrics "
                "for %s: %s",
                symbol,
                e,
                exc_info=True,
            )
            return None
        finally:
            db.close()

    def build_features(self, symbol: str) -> Dict[str, float]:
        """
        Build a feature dict suitable for `options_vol_edge()` from the latest
        derived options metrics.

        Mapping (can be refined as you iterate):
          - iv_rank:
                Approximate IV rank based on avg_iv_near_term. We clamp to [0, 1]
                assuming typical near-term vol is in [0, 2].
          - risk_reversal_25d:
                Directly from iv_skew_25d (25d call IV - put IV).
          - term_structure_slope:
                Directly from iv_term_slope_near_far (mid - near).

        Returns:
            dict with numeric features; empty dict if no row is found.
        """
        row = self._get_latest_row(symbol)
        if row is None:
            logger.warning(
                "[OptionsVolFeatureIngestion] No options_derived_metrics row for %s; "
                "returning empty feature dict.",
                symbol,
            )
            return {}

        # --- iv_rank ---
        avg_iv_near = getattr(row, "avg_iv_near_term", None)
        if avg_iv_near is None:
            iv_rank = 0.5  # neutral
        else:
            try:
                iv_val = float(avg_iv_near)
                # crude normalization: assume typical near IV in [0, 2]
                iv_rank = max(0.0, min(1.0, iv_val / 2.0))
            except Exception:
                iv_rank = 0.5

        # --- risk_reversal_25d ---
        risk_reversal = getattr(row, "iv_skew_25d", None)
        try:
            risk_reversal_val = 0.0 if risk_reversal is None else float(risk_reversal)
        except Exception:
            risk_reversal_val = 0.0

        # --- term_structure_slope ---
        term_slope = getattr(row, "iv_term_slope_near_far", None)
        try:
            term_slope_val = 0.0 if term_slope is None else float(term_slope)
        except Exception:
            term_slope_val = 0.0

        features: Dict[str, float] = {
            "iv_rank": float(iv_rank),
            "risk_reversal_25d": float(risk_reversal_val),
            "term_structure_slope": float(term_slope_val),
        }

        return features


# ---------------------------------------------------------------------------
# Main scoring function (model + heuristic fallback)
# ---------------------------------------------------------------------------


def options_vol_edge(features: Dict[str, Any]) -> float:
    """
    Returns a score in [-1, 1] indicating options-based directional/vol edge.

    Positive:
      - bullish or supportive vol/skew structure
    Negative:
      - bearish / stress / downside protection bid
    0:
      - neutral or no data

    Pipeline:
      1) Try using the trained model (if present and prediction succeeds).
      2) If model is missing OR prediction errors, fall back to a light heuristic
         based on provided metrics.

    Fallback behavior when data is missing:
      - If `features` is empty, we immediately return 0.0 (neutral).
      - If certain keys are missing, they are treated as 0.0 in the heuristic.
    """
    _lazy_load_options_model()

    # --- Hard fallback if no features at all ---
    if not features:
        logger.info(
            "[OptionsVol] No options features provided; returning neutral score 0.0."
        )
        return 0.0

    # --- Try model-based prediction first ---
    if _options_model is not None:
        try:
            # Expect caller to pass a clean dict of numeric features.
            ordered_keys = sorted(features.keys())
            x = [[float(features[k]) for k in ordered_keys]]
            pred = float(_options_model.predict(x)[0])  # type: ignore[call-arg]
            # Assume model already outputs in [-1,1] or close; clip just in case.
            return max(-1.0, min(1.0, pred))
        except Exception as e:  # noqa: BLE001
            logger.error(
                "[OptionsVol] Model prediction failed; falling back to heuristic: %s",
                e,
                exc_info=True,
            )

    # --- Heuristic fallback ---
    iv_rank = float(features.get("iv_rank", 0.0))
    skew = float(features.get("risk_reversal_25d", 0.0))
    term_slope = float(features.get("term_structure_slope", 0.0))

    score = 0.0

    # Example heuristics:
    if skew < -0.05:
        score -= 0.3  # puts expensive -> downside fear
    if skew > 0.05:
        score += 0.3  # calls bid -> upside interest

    score += 0.2 * term_slope
    score += 0.1 * (iv_rank - 0.5)

    return max(-1.0, min(1.0, score))
