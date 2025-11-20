import joblib
from pathlib import Path
from typing import Dict, Any

from .schemas import ExpertSignals, MarketContext

_METALABEL_PATH = Path("./model_artifacts/meta_label.joblib")
_metalabel_model = None


def _lazy_load():
    global _metalabel_model
    if _metalabel_model is None and _METALABEL_PATH.exists():
        try:
            _metalabel_model = joblib.load(_METALABEL_PATH)
        except Exception as e:
            print(f"[MetaLabel] Failed to load model: {e}")


def metalabel_decide(
    features: Dict[str, Any],
    expert: ExpertSignals,
    meta: Dict[str, Any],
    ctx: MarketContext,
) -> Dict[str, Any]:
    """
    Secondary Model (Meta-Labeling) to filter false positives.
    
    Decides:
      - execute: boolean (should we block this trade?)
      - size_factor: float (0.0 to 1.0, how much capital to deploy)
    """
    _lazy_load()

    # Extract critical signals
    # 'p_edge' is the raw model prediction (e.g. 0.005 for 0.5% return)
    p_edge = float(meta.get("p_edge", 0.0))
    conf = float(meta.get("confidence", 0.0))
    rv_24h = float(features.get("roll_vol_24h", 0.0))
    
    # Funding rate is often a good "crowdedness" signal
    # Check both feature set and tabular features
    funding = float(features.get("funding_rate", 0.0))

    # 1. Heuristic Logic (Fallback / Guardrails)
    # If volatility is extremely low, don't trade (dead market)
    if rv_24h < 0.001: 
        return {"execute": False, "size_factor": 0.0, "reason": "vol_too_low"}

    # If Machine Learning Model Exists, Use It
    if _metalabel_model is not None:
        try:
            # Feature vector must match training: [p_edge, conf, vol, funding]
            X_meta = [[p_edge, conf, rv_24h, funding]]
            # predict_proba returns [prob_0, prob_1]. We want prob_1 (Trade is Good)
            prob_success = float(_metalabel_model.predict_proba(X_meta)[0, 1])
            
            if prob_success > 0.6:
                return {"execute": True, "size_factor": 1.0, "reason": "ml_high_conviction"}
            elif prob_success > 0.5:
                return {"execute": True, "size_factor": 0.5, "reason": "ml_low_conviction"}
            else:
                return {"execute": False, "size_factor": 0.0, "reason": "ml_reject"}
        except Exception:
            # Fallback if ML fails
            pass

    # 2. Default Heuristic (if no ML model)
    # Scale size based on confidence
    if conf > 65.0:
        return {"execute": True, "size_factor": 1.0, "reason": "high_conf_heuristic"}
    elif conf > 50.0:
        return {"execute": True, "size_factor": 0.5, "reason": "med_conf_heuristic"}
    
    return {"execute": True, "size_factor": 0.25, "reason": "low_conf_heuristic"}