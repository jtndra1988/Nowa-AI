import joblib
from pathlib import Path
from typing import Dict, Any

from .schemas import ExpertSignals, MarketContext

_METALABEL_PATH = Path("./model_artifacts/meta_label.joblib")
_metalabel_model = None


def _lazy_load():
    global _metalabel_model
    if _metalabel_model is None and _METALABEL_PATH.exists():
        _metalabel_model = joblib.load(_METALABEL_PATH)


def metalabel_decide(
    features: Dict[str, Any],
    expert: ExpertSignals,
    meta: Dict[str, Any],
    ctx: MarketContext,
) -> Dict[str, Any]:
    """
    Meta-label:
      - Should we execute?
      - How big? (size_factor in [0,1])
    """
    _lazy_load()

    base = {
        "p_edge": meta["p_edge"],
        "conf": meta["confidence"],
        "rv_24h": features.get("rv_24h", 0.0),
        "funding_1h": features.get("funding_1h", 0.0),
    }

    if _metalabel_model is not None:
        p_ok = float(_metalabel_model.predict_proba([base])[0, 1])
    else:
        penalty = 0.0
        if abs(base["funding_1h"]) > 0.01:
            penalty += 0.1
        if base["rv_24h"] > 0.20:
            penalty += 0.1
        p_ok = max(0.0, meta["p_edge"] - penalty)

    execute = (p_ok > 0.55) and (meta["dir_raw"] != "flat")
    size_factor = max(0.1, min(1.0, p_ok)) if execute else 0.0

    return {
        "execute": execute,
        "size_factor": size_factor,
    }
