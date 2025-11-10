from pathlib import Path
from typing import Dict, Any, Optional
import joblib

_OPTIONS_MODEL_PATH = Path("./model_artifacts/options_vol_edge.joblib")
_options_model = None


def _lazy_load_options_model():
    global _options_model
    if _options_model is None and _OPTIONS_MODEL_PATH.exists():
        _options_model = joblib.load(_OPTIONS_MODEL_PATH)


def options_vol_edge(features: Dict[str, Any]) -> float:
    """
    Returns a score in [-1, 1] indicating options-based directional/vol edge.

    Positive:
      - bullish or supportive vol/skew structure
    Negative:
      - bearish / stress / downside protection bid
    0:
      - neutral or no data

    If no model is trained yet, use a light heuristic based on provided metrics.
    """
    _lazy_load_options_model()

    if _options_model is not None:
        # Expect caller to pass a clean dict of numeric features from options surface
        ordered_keys = sorted(features.keys())
        x = [[float(features[k]) for k in ordered_keys]]
        pred = float(_options_model.predict(x)[0])
        # Assume model already outputs in [-1,1] or close; clip just in case.
        return max(-1.0, min(1.0, pred))

    # Heuristic fallback:
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
