from __future__ import annotations

import logging
import socket
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter
from app.db.database import SessionLocal
from app.ml.adv.model_registry import model_registry
# psutil is optional – if missing, we still don't crash
try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

logger = logging.getLogger(__name__)

router = APIRouter()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Basic health endpoints
# ---------------------------------------------------------------------------

@router.get("/health/models", summary="Model versions & status")
async def health_models() -> Dict[str, Any]:
    meta = model_registry.all_metadata()
    status = "ok" if meta else "degraded"
    return {
        "status": status,
        "models": meta,
    }
@router.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {
        "status": "ok",
        "time": _utcnow_iso(),
        "host": socket.gethostname(),
        "service": "ml-trading-api",
    }


@router.get("/livez")
def livez() -> Dict[str, Any]:
    return {"status": "alive", "time": _utcnow_iso()}


@router.get("/readyz")
def readyz() -> Dict[str, Any]:
    """
    Readiness check: confirms DB connectivity.
    """
    ok = False
    error: str | None = None
    try:
        db = SessionLocal()
        try:
            db.execute("SELECT 1")
            ok = True
        finally:
            db.close()
    except Exception as e:
        error = str(e)
        logger.warning("readyz DB check failed: %s", e)

    return {
        "status": "ok" if ok else "error",
        "db_ready": ok,
        "error": error,
        "time": _utcnow_iso(),
    }


# ---------------------------------------------------------------------------
# System resources (shared helper + endpoint)
# ---------------------------------------------------------------------------


def get_system_resources() -> Dict[str, float]:
    """
    Helper used by other modules (system.py, system_stream.py).
    Safe: never raises.
    """
    cpu_load = 0.0
    ram_usage = 0.0
    gpu_util = 0.0

    if psutil is None:
        # psutil not installed – return zeros instead of crashing
        return {
            "cpu_load": cpu_load,
            "ram_usage": ram_usage,
            "gpu_util": gpu_util,
        }

    try:
        cpu_load = float(psutil.cpu_percent(interval=None))
        mem = psutil.virtual_memory()
        ram_usage = float(mem.percent)
    except Exception as e:
        logger.exception("get_system_resources failed: %s", e)

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
