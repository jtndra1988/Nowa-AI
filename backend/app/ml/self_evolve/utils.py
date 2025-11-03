from __future__ import annotations
import os, json, shutil, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Reuse your artifact helpers if available
try:
    from app.ml.artifacts import (
        FILENAMES,
        default_output_dir as _default_output_dir,
        ensure_dir,
        save_model,
        save_scaler,
        save_manifest as _save_manifest,
    )
except Exception:
    FILENAMES = {
        "lstm": "lstm_best.pt",
        "lstm_scaler": "lstm_scaler.pkl",
        "features": "features.json",
        "xgb": "xgb_model.json",
        "xgb_features": "xgb_features.json",
        "fusion": "fusion_head.pt",
        "manifest": "manifest.json",
    }
    def ensure_dir(p: Path | str) -> Path:
        p = Path(p); p.mkdir(parents=True, exist_ok=True); return p
    def _default_output_dir(symbol: str) -> Path:
        base = Path(os.getenv("MODEL_OUTPUT_DIR", "./checkpoints"))
        return ensure_dir(base / symbol.replace("/", "_").upper())
    def _save_manifest(out_dir: Path | str, **info: Any) -> Path:
        out_dir = ensure_dir(out_dir)
        path = Path(out_dir) / FILENAMES.get("manifest", "manifest.json")
        data = {}
        if path.exists():
            try: data = json.loads(path.read_text())
            except Exception: data = {}
        data.update(info)
        path.write_text(json.dumps(data, indent=2))
        return path


def out_dir(symbol: str, stage: str = "prod") -> Path:
    """
    stage ∈ {"prod", "staging"} — staging is where new models are trained
    before promotion.
    """
    root = _default_output_dir(symbol)
    if stage == "prod":
        return root
    return ensure_dir(root / "staging")


def promote_if_better(symbol: str, metric_new: float, metric_old: Optional[float], higher_is_better: bool,
                      files: Dict[str, str]) -> bool:
    """If `metric_new` beats `metric_old` (or old missing), copy staging files → prod.
    Args:
        files: mapping of artifact key → filename in FILENAMES (e.g., {"lstm": "lstm_best.pt"})
    Returns:
        True if promoted.
    """
    if metric_old is None:
        better = True
    else:
        better = metric_new > metric_old if higher_is_better else metric_new < metric_old

    if not better:
        return False

    src = out_dir(symbol, "staging")
    dst = out_dir(symbol, "prod")
    for key, fname in files.items():
        src_path = src / FILENAMES.get(key, fname)
        if src_path.exists():
            shutil.copy2(src_path, dst / src_path.name)
    _save_manifest(dst, last_promotion_ts=time.time(), last_metric=float(metric_new))
    return True


def read_manifest(symbol: str, stage: str = "prod") -> Dict[str, Any]:
    p = out_dir(symbol, stage) / FILENAMES.get("manifest", "manifest.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}