from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.db import models
import joblib
import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Artifact paths & versioning
# --------------------------------------------------------------------------

# Runtime "current" model artifact
_MACRO_ONCHAIN_MODEL_PATH = Path("./model_artifacts/macro_onchain_bias.joblib")

# Optional drift monitoring stats
_MACRO_ONCHAIN_DRIFT_STATS_PATH = Path("./model_artifacts/macro_onchain_bias_drift_stats.json")

# In-memory singletons
_macro_onchain_model: Optional[Any] = None
_macro_onchain_metadata: Dict[str, Any] = {}

# Recommended metadata format (training script should write this):
# artifact = {
#   "model": trained_estimator,
#   "metadata": {
#       "model_name": "macro_onchain_bias",
#       "version": "v1.0",
#       "trained_at_utc": "2025-01-01T00:00:00Z",
#       "feature_names": [
#           "stablecoin_netflow",
#           "btc_exchange_reserves_change",
#           "dxy_trend",
#           "spx_trend",
#       ],
#   },
# }
# joblib.dump(artifact, _MACRO_ONCHAIN_MODEL_PATH)


# --------------------------------------------------------------------------
# Expected schema & simple validation
# --------------------------------------------------------------------------

# Canonical macro/on-chain feature names.
# You can extend this safely; unknown keys are ignored by the model but still
# observed for drift.
EXPECTED_FEATURES = [
    "stablecoin_netflow",          # >0 = more risk-on capital flowing in
    "btc_exchange_reserves_change",# >0 = more BTC on exchanges (potential sell)
    "dxy_trend",                   # >0 = USD strength (risk-off)
    "spx_trend",                   # >0 = equities strength (risk-on)
]

# Optional: "soft" ranges for sanity checks (not enforced, only logged).
SOFT_RANGES: Dict[str, Tuple[float, float]] = {
    "stablecoin_netflow": (-1e3, 1e3),
    "btc_exchange_reserves_change": (-1e3, 1e3),
    "dxy_trend": (-5.0, 5.0),  # in % or normalized units
    "spx_trend": (-5.0, 5.0),
}


@dataclass
class DriftStats:
    """Simple running statistics per feature for drift monitoring."""
    count: int
    mean: float
    m2: float  # sum of squared deviations (for variance)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return float(np.sqrt(self.m2 / (self.count - 1)))


def _lazy_load_macro_onchain_model() -> None:
    """
    Lazy-load the macro/on-chain model and its metadata.

    Supports two artifact formats:
    1) New: dict with {"model": estimator, "metadata": {...}}
    2) Legacy: plain estimator object (no metadata).
    """
    global _macro_onchain_model, _macro_onchain_metadata

    if _macro_onchain_model is not None:
        return

    if not _MACRO_ONCHAIN_MODEL_PATH.exists():
        logger.warning(
            "[MacroOnchain] Model artifact not found at %s; "
            "will use heuristic fallback.",
            _MACRO_ONCHAIN_MODEL_PATH,
        )
        return

    try:
        artifact = joblib.load(_MACRO_ONCHAIN_MODEL_PATH)
    except Exception as e:  # noqa: BLE001
        logger.error(
            "[MacroOnchain] Failed to load model from %s: %s",
            _MACRO_ONCHAIN_MODEL_PATH,
            e,
            exc_info=True,
        )
        return

    if isinstance(artifact, dict) and "model" in artifact:
        _macro_onchain_model = artifact["model"]
        _macro_onchain_metadata = artifact.get("metadata", {})
    else:
        _macro_onchain_model = artifact
        _macro_onchain_metadata = {}

    logger.info(
        "[MacroOnchain] Loaded model from %s (version=%s)",
        _MACRO_ONCHAIN_MODEL_PATH,
        _macro_onchain_metadata.get("version", "unknown"),
    )


def get_macro_onchain_metadata() -> Dict[str, Any]:
    """Return a copy of model metadata (version, feature_names, etc.)."""
    _lazy_load_macro_onchain_model()
    return dict(_macro_onchain_metadata) if _macro_onchain_metadata else {}


# --------------------------------------------------------------------------
# Schema validation & input normalization
# --------------------------------------------------------------------------

def validate_macro_onchain_schema(
    features: Dict[str, Any],
) -> Tuple[Dict[str, float], bool, Dict[str, Any]]:
    """
    Validate and normalize incoming macro/on-chain features.

    Returns:
        clean_features:  dict with float values for all EXPECTED_FEATURES
                         (missing -> 0.0, non-numeric -> 0.0, unknown keys preserved
                         in 'extra_features' but NOT sent to the model).
        schema_ok:       bool indicating whether all expected keys were present
                         and numeric (soft ranges are *not* strict).
        info:            dict with:
                         - missing_keys: List[str]
                         - non_numeric_keys: List[str]
                         - out_of_range_keys: List[str]
                         - extra_keys: List[str]
    """
    clean: Dict[str, float] = {}
    missing: list[str] = []
    non_numeric: list[str] = []
    out_of_range: list[str] = []
    extra_keys: list[str] = []

    # Normalize expected keys
    for key in EXPECTED_FEATURES:
        val = features.get(key, 0.0)
        try:
            fval = float(val)
        except Exception:
            fval = 0.0
            non_numeric.append(key)

        # Soft range check
        if key in SOFT_RANGES:
            lo, hi = SOFT_RANGES[key]
            if not (lo <= fval <= hi):
                out_of_range.append(key)

        if key not in features:
            missing.append(key)

        clean[key] = fval

    # Track extra keys (observed but not in EXPECTED_FEATURES)
    for key in features.keys():
        if key not in EXPECTED_FEATURES:
            extra_keys.append(key)

    schema_ok = not missing and not non_numeric

    info = {
        "missing_keys": missing,
        "non_numeric_keys": non_numeric,
        "out_of_range_keys": out_of_range,
        "extra_keys": extra_keys,
    }

    if missing or non_numeric or out_of_range:
        logger.warning(
            "[MacroOnchain] Schema anomalies: missing=%s non_numeric=%s out_of_range=%s",
            missing,
            non_numeric,
            out_of_range,
        )

    if extra_keys:
        logger.info("[MacroOnchain] Extra macro/on-chain fields detected: %s", extra_keys)

    return clean, schema_ok, info


# --------------------------------------------------------------------------
# Input drift monitoring (simple EMA / Z-score detection)
# --------------------------------------------------------------------------

def _load_drift_stats() -> Dict[str, DriftStats]:
    if not _MACRO_ONCHAIN_DRIFT_STATS_PATH.exists():
        return {}
    try:
        with open(_MACRO_ONCHAIN_DRIFT_STATS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        stats: Dict[str, DriftStats] = {}
        for k, v in raw.items():
            stats[k] = DriftStats(
                count=int(v.get("count", 0)),
                mean=float(v.get("mean", 0.0)),
                m2=float(v.get("m2", 0.0)),
            )
        return stats
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[MacroOnchain] Failed to load drift stats from %s: %s",
            _MACRO_ONCHAIN_DRIFT_STATS_PATH,
            e,
            exc_info=True,
        )
        return {}


def _save_drift_stats(stats: Dict[str, DriftStats]) -> None:
    try:
        payload = {k: asdict(v) for k, v in stats.items()}
        _MACRO_ONCHAIN_DRIFT_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_MACRO_ONCHAIN_DRIFT_STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "updated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "features": payload,
                },
                f,
                indent=4,
            )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[MacroOnchain] Failed to persist drift stats to %s: %s",
            _MACRO_ONCHAIN_DRIFT_STATS_PATH,
            e,
            exc_info=True,
        )


def update_input_drift(clean_features: Dict[str, float]) -> Dict[str, Any]:
    """
    Update running statistics for each feature and detect simple drift.

    For each feature:
      - We maintain count, mean, and M2 (Welford's algorithm).
      - If the new value is > mean + 4*std or < mean - 4*std, we log a drift warning.

    Returns:
      info dict:
        {
          "drift_flags": {feature_name: bool},
          "z_scores": {feature_name: float},
        }
    """
    stats = _load_drift_stats()
    drift_flags: Dict[str, bool] = {}
    z_scores: Dict[str, float] = {}

    for key, value in clean_features.items():
        v = float(value)

        ds = stats.get(key)
        if ds is None:
            # Initialize stats
            ds = DriftStats(count=1, mean=v, m2=0.0)
        else:
            # Welford's algorithm update
            ds.count += 1
            delta = v - ds.mean
            ds.mean += delta / ds.count
            ds.m2 += delta * (v - ds.mean)

        stats[key] = ds

        # Drift detection via Z-score
        std = ds.std
        if std > 0:
            z = (v - ds.mean) / std
        else:
            z = 0.0

        z_scores[key] = float(z)
        drift_flags[key] = abs(z) >= 4.0  # 4σ threshold

        if drift_flags[key]:
            logger.warning(
                "[MacroOnchain] Input drift detected for '%s': value=%.4f mean=%.4f std=%.4f z=%.2f",
                key,
                v,
                ds.mean,
                std,
                z,
            )

    _save_drift_stats(stats)

    return {
        "drift_flags": drift_flags,
        "z_scores": z_scores,
    }

class MacroOnchainFeatureIngestion:
    """
    Ingestion wrapper for macro + on-chain metrics.

    Reads the latest row from MacroOnchainMetrics for a symbol and returns
    a feature dict compatible with macro_onchain_bias():

        {
          "stablecoin_netflow": float,
          "btc_exchange_reserves_change": float,
          "dxy_trend": float,
          "spx_trend": float,
        }

    If no row is found or any field is missing, we fall back to neutral 0.0
    for that field. This keeps schema stable and avoids hard failures.
    """

    def __init__(self, session_factory=SessionLocal) -> None:
        self._session_factory = session_factory

    def _get_latest_row(self, symbol: str):
        """
        Returns the most recent MacroOnchainMetrics row for `symbol`, or None.

        NOTE: If your ORM model is not models.MacroOnchainMetrics,
        adjust the class name and field names accordingly.
        """
        db: Session = self._session_factory()
        try:
            return (
                db.query(models.MacroOnchainMetrics)  # TODO: rename if your model is different
                .filter(models.MacroOnchainMetrics.symbol == symbol.upper())
                .order_by(models.MacroOnchainMetrics.timestamp.desc())
                .first()
            )
        except Exception as e:
            logger.error(
                "[MacroOnchainFeatureIngestion] Failed to load macro/on-chain row for %s: %s",
                symbol,
                e,
                exc_info=True,
            )
            return None
        finally:
            db.close()

    def build_features(self, symbol: str) -> Dict[str, float]:
        """
        Build macro_onchain_features dict for the given symbol.

        Mapping (aligned with FEATURE_COLS / EXPECTED_FEATURES):

          - stablecoin_netflow
          - btc_exchange_reserves_change
          - dxy_trend
          - spx_trend

        Returns:
            dict[str, float]. If no row is found, returns {} and lets
            macro_onchain_bias() treat it as neutral.
        """
        row = self._get_latest_row(symbol)
        if row is None:
            logger.warning(
                "[MacroOnchainFeatureIngestion] No MacroOnchainMetrics row for %s; "
                "returning empty feature dict.",
                symbol,
            )
            return {}

        def _to_float(val, default=0.0) -> float:
            if val is None:
                return float(default)
            try:
                return float(val)
            except Exception:
                return float(default)

        features: Dict[str, float] = {
            "stablecoin_netflow": _to_float(getattr(row, "stablecoin_netflow", None), 0.0),
            "btc_exchange_reserves_change": _to_float(
                getattr(row, "btc_exchange_reserves_change", None), 0.0
            ),
            "dxy_trend": _to_float(getattr(row, "dxy_trend", None), 0.0),
            "spx_trend": _to_float(getattr(row, "spx_trend", None), 0.0),
        }

        return features

# --------------------------------------------------------------------------
# Main scoring function
# --------------------------------------------------------------------------

def macro_onchain_bias(features: Dict[str, Any]) -> float:
    """
    Returns bias in [-1, 1]:
      +1 = strong risk-on (support long bias)
      -1 = strong risk-off (prefer shorts / no risk)
       0 = neutral

    Inputs may include (at least):
      - stablecoin_netflow
      - btc_exchange_reserves_change
      - dxy_trend
      - spx_trend
      plus optional extra keys for future expansion.

    Pipeline:
      1) Validate & normalize schema (fill missing with 0.0, log anomalies).
      2) Update drift statistics and log any 4σ outliers.
      3) If a trained model exists, use it to predict bias (clipped to [-1, 1]).
      4) Otherwise, or if model prediction fails, fall back to a transparent heuristic.
    """
    _lazy_load_macro_onchain_model()

    # 1) Schema validation / normalization
    clean_features, schema_ok, info = validate_macro_onchain_schema(features)

    if not schema_ok:
        logger.info(
            "[MacroOnchain] Using normalized features after schema anomalies: %s",
            info,
        )

    # 2) Drift monitoring
    drift_info = update_input_drift(clean_features)
    if any(drift_info["drift_flags"].values()):
        logger.info("[MacroOnchain] Drift info: %s", drift_info)

    # 3) Model-based prediction (if available)
    if _macro_onchain_model is not None:
        try:
            # Use explicit feature ordering from metadata if present
            feature_names = _macro_onchain_metadata.get("feature_names")
            if not feature_names:
                feature_names = sorted(clean_features.keys())

            x_row = [float(clean_features.get(k, 0.0)) for k in feature_names]
            x = [x_row]  # shape [1, n_features]

            pred = float(_macro_onchain_model.predict(x)[0])  # type: ignore[call-arg]
            return max(-1.0, min(1.0, pred))
        except Exception as e:  # noqa: BLE001
            logger.error(
                "[MacroOnchain] Model prediction failed; falling back to heuristic: %s",
                e,
                exc_info=True,
            )

    # 4) Heuristic fallback (original logic, but using clean_features)
    score = 0.0

    stbl_flow = float(clean_features.get("stablecoin_netflow", 0.0))
    ex_res = float(clean_features.get("btc_exchange_reserves_change", 0.0))
    dxy_trend = float(clean_features.get("dxy_trend", 0.0))
    spx_trend = float(clean_features.get("spx_trend", 0.0))

    score += 0.3 * spx_trend
    score -= 0.3 * dxy_trend
    score -= 0.2 * max(ex_res, 0.0)       # rising reserves -> sell pressure
    score += 0.2 * max(stbl_flow, 0.0)    # positive stablecoin flows -> risk-on

    return max(-1.0, min(1.0, score))
