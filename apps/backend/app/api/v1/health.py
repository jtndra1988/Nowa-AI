from __future__ import annotations

import logging
import socket
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter
from app.db.database import SessionLocal

# Optional: model registry if you have it; if missing, we just mark AI as not ready
try:
    from app.ml.adv.model_registry import model_registry  # type: ignore
except Exception:
    model_registry = None  # type: ignore

# psutil is optional – if missing, we still don't crash
try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/healthz")
def healthz() -> Dict[str, Any]:
    """Basic liveness probe."""
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
    }


@router.get("/readyz")
def readyz() -> Dict[str, Any]:
    """Readiness probe – checks DB and basic AI model registry state."""
    db_ok = True
    try:
        with SessionLocal() as db:
            db.execute("SELECT 1")
    except Exception as exc:  # pragma: no cover
        logger.exception("DB readiness check failed: %s", exc)
        db_ok = False

    ai_ready = False
    try:
        if model_registry is not None:
            models = getattr(model_registry, "models", {})
            ai_ready = bool(models)
    except Exception:
        ai_ready = False

    status = "ok" if db_ok else "degraded"
    return {
        "status": status,
        "db": db_ok,
        "ai_models_loaded": ai_ready,
        "time": datetime.now(timezone.utc).isoformat(),
    }


def get_system_resources() -> Dict[str, float]:
    """
    Return CPU / RAM / GPU metrics for the System tab.
    Uses psutil when available; otherwise returns zeros.
    """
    cpu_load = 0.0
    ram_usage = 0.0
    gpu_util = 0.0

    if psutil is not None:  # pragma: no branch
        try:
            cpu_load = float(psutil.cpu_percent(interval=0.1))
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to read CPU load: %s", exc)

        try:
            ram = psutil.virtual_memory()
            ram_usage = float(ram.percent)
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to read RAM usage: %s", exc)

        # GPU metrics can be wired later (pynvml / nvidia-smi)

    return {
        "cpu_load": cpu_load,
        "ram_usage": ram_usage,
        "gpu_util": gpu_util,
    }


@router.get("/system-resources")
def system_resources() -> Dict[str, float]:
    """
    API endpoint – just wraps get_system_resources().
    Full URL (with main.py prefix): /api/v1/system-resources
    """
    return get_system_resources()
