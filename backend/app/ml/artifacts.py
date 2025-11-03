# app/ml/artifacts.py
from __future__ import annotations
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import torch


# -----------------------------
# Paths & FS helpers
# -----------------------------
def _sanitize_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(" ", "").upper()


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_output_dir(symbol: str) -> Path:
    base = Path(os.environ.get("MODEL_OUTPUT_DIR", "./checkpoints"))
    return ensure_dir(base / _sanitize_symbol(symbol))


# -----------------------------
# Model & Scaler persistence
# -----------------------------
def save_model(model: torch.nn.Module, out_dir: Path | str, filename: str | None = None) -> Path:
    """
    Save LSTM (or any torch nn.Module) with a consistent filename.
    Defaults to FILENAMES['lstm'].
    """
    out_dir = ensure_dir(out_dir)
    fname = filename or FILENAMES.get("lstm", "model.pt")
    path = Path(out_dir) / fname
    # Save as raw state_dict to keep backward-compat
    torch.save(model.state_dict(), path)
    return path


def load_model(
    model: torch.nn.Module,
    out_dir: Path | str,
    filename: str | None = None,
    map_location: Optional[str] = None
) -> torch.nn.Module:
    """
    Tolerant loader: supports raw state_dict or dict-wrapped checkpoints.
    """
    fname = filename or FILENAMES.get("lstm", "model.pt")
    path = Path(out_dir) / fname
    obj = torch.load(path, map_location=map_location or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Accept raw state_dict or {"state_dict": ...} / {"model_state_dict": ...}
    if isinstance(obj, dict):
        state = obj.get("state_dict", obj.get("model_state_dict", obj))
    else:
        state = obj
    model.load_state_dict(state, strict=False)
    return model

def save_scaler(scaler, out_dir: Path | str, filename: str | None = None) -> Path:
    out_dir = ensure_dir(out_dir)
    fname = filename or FILENAMES.get("lstm_scaler", "scaler.pkl")
    path = Path(out_dir) / fname
    joblib.dump(scaler, path)
    return path


def load_scaler(out_dir: Path | str, filename: str | None = None):
    fname = filename or FILENAMES.get("lstm_scaler", "scaler.pkl")
    path = Path(out_dir) / fname
    return joblib.load(path)


# -----------------------------
# Manifest (metadata)
# -----------------------------
def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def save_manifest(out_dir: Path | str, **info: Any) -> Path:
    """
    Merge/append metadata into manifest.json in out_dir.

    Example:
        save_manifest(
            out_dir,
            instrument="futures",
            table="futures_market_data",
            prediction_target="next_close",
            metrics={"val_loss": 0.0123},
        )
    """
    out_dir = ensure_dir(out_dir)
    path = Path(out_dir) / "manifest.json"
    manifest = _read_json(path)

    # always bump timestamp
    manifest["last_updated"] = datetime.utcnow().isoformat() + "Z"

    for k, v in info.items():
        manifest[k] = _to_jsonable(v)

    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return path

# Unified filenames used across train/eval/inference
FILENAMES: Dict[str, str] = {
    "lstm": "lstm_best.pt",
    "lstm_scaler": "lstm_scaler.pkl",
    "features": "features.json",          # expects {"lstm_features":[...], "xgb_features":[...]}
    "xgb": "xgb_model.json",
    "xgb_features": "xgb_features.json",  # legacy support; keep if you use separate file
    "fusion": "fusion_head.pt",
    "cal_temp": "calibration_temp.json",
    "cal_platt": "calibration_platt.json",
    "manifest": "manifest.json",
}

CHECKPOINTS_ROOT = Path(os.environ.get("MODEL_OUTPUT_DIR", "./checkpoints"))

def _symbol_dirname(symbol: str) -> str:
    return symbol.replace("/", "_").replace(" ", "").upper()

def ckpt_dir(symbol: str, base: Optional[Path] = None) -> Path:
    base = base or CHECKPOINTS_ROOT
    d = base / _symbol_dirname(symbol)
    d.mkdir(parents=True, exist_ok=True)
    return d

def fpath(symbol: str, key: str, base: Optional[Path] = None) -> Path:
    fn = FILENAMES.get(key)
    if not fn:
        raise KeyError(f"Unknown artifact key: {key}")
    return ckpt_dir(symbol, base=base) / fn

def load_manifest(symbol_or_dir: str | Path) -> Dict[str, Any]:
    """Accept a symbol ('BTC/USDT') or an absolute/relative directory path."""
    if isinstance(symbol_or_dir, (str,)):
        # Treat as path if it looks like one
        s = symbol_or_dir
        looks_like_path = any(tok in s for tok in ("/", "\\", ".", ":"))
        path = (Path(s) / FILENAMES["manifest"]) if looks_like_path else fpath(s, "manifest")
    else:
        path = Path(symbol_or_dir) / FILENAMES["manifest"]

    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

# --- Convenience helpers for features & existence checks ---

def write_features(out_dir: Path | str, *, lstm_features=None, xgb_features=None) -> Path:
    """
    Write a unified features.json: {"lstm_features": [...], "xgb_features": [...]}
    """
    out_dir = ensure_dir(out_dir)
    data = {}
    p = Path(out_dir) / FILENAMES["features"]
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except Exception:
            data = {}
    if lstm_features is not None:
        data["lstm_features"] = list(lstm_features)
    if xgb_features is not None:
        data["xgb_features"] = list(xgb_features)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return p


def read_features(out_dir: Path | str) -> Dict[str, Any]:
    p = Path(out_dir) / FILENAMES["features"]
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def artifact_exists(symbol_or_dir: str | Path, key: str) -> bool:
    """
    Check if a particular artifact exists for a symbol or in a given folder.
    """
    if isinstance(symbol_or_dir, (str,)):
        path = fpath(symbol_or_dir, key)
    else:
        path = Path(symbol_or_dir) / FILENAMES.get(key, key)
    return path.exists()
