# app/api/v1/system.py

from __future__ import annotations

import logging
import os
import socket
from datetime import datetime
from sqlalchemy import text
from fastapi import APIRouter

from app.db.database import SessionLocal
from app.services.inference_service import inference_service
from app.api.v1.health import get_system_resources

logger = logging.getLogger(__name__)
router = APIRouter()

# Optional Redis import (won't crash if missing)
try:
    import redis as redis_lib  # type: ignore
except ImportError:  # pragma: no cover
    redis_lib = None


def _check_db():
    try:
        db = SessionLocal()
        try:
            # SQLAlchemy 2.x style
            db.execute(text("SELECT 1"))
        finally:
            db.close()

        return {"status": "ok", "detail": "DB reachable"}
    except Exception as e:
        logger.exception("DB health check failed: %s", e)
        return {"status": "down", "detail": str(e)}
def _check_redis():
    if redis_lib is None:
        return {
            "status": "unknown",
            "detail": "redis-py not installed in this environment",
        }

    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    try:
        client = redis_lib.from_url(url)
        client.ping()
        return {"status": "ok", "detail": f"Redis reachable at {url}"}
    except Exception as e:
        logger.exception("Redis health check failed: %s", e)
        return {"status": "down", "detail": str(e)}


def _check_llm(brain: dict):
    ready = bool(brain.get("llm_ready"))
    provider = os.getenv("LLM_PROVIDER", "gemini")
    return {
        "status": "ok" if ready else "down",
        "detail": f"Provider={provider}",
    }


def _check_binance():
    # TODO: wire to real exchange client / price feed
    # For now, assume OK if backend is up – UI just needs a live signal.
    return {
        "status": "ok",
        "detail": "Price feed active (stubbed)",
    }


def _check_celery(redis_health: dict):
    # Simple heuristic: if Redis is OK, we assume workers are likely up.
    if redis_health.get("status") == "ok":
        return {"status": "ok", "detail": "Assumed healthy (Redis OK)"}
    if redis_health.get("status") == "down":
        return {"status": "degraded", "detail": "Redis down – workers impacted"}
    return {"status": "unknown", "detail": "No direct Celery health check yet"}


@router.get("/system-health")
def system_health():
    """
    Aggregated backend + AI brain + infra health.

    Frontend expects:
      - top-level fields (backend_api, ai_brain, l2_ensemble, llm_engine, rl_agent, risk_engine, resources)
      - services.{db,redis,celery,binance,llm_provider}.status
    """
    # --- Brain health ---
    try:
        brain = inference_service.get_brain_health()
    except Exception as e:
        logger.exception("get_brain_health failed: %s", e)
        brain = {
            "is_ready": False,
            "has_l2_models": False,
            "llm_ready": False,
            "rl_ready": False,
            "risk_ready": False,
        }

    # --- Resources (CPU / RAM / GPU) ---
    try:
        resources = get_system_resources()
    except Exception as e:
        logger.exception("get_system_resources failed: %s", e)
        resources = {"cpu_load": 0.0, "ram_usage": 0.0, "gpu_util": 0.0}

    # --- Service-level checks ---
    db_health = _check_db()
    redis_health = _check_redis()
    celery_health = _check_celery(redis_health)
    binance_health = _check_binance()
    llm_health = _check_llm(brain)

    services = {
        "db": db_health,
        "redis": redis_health,
        "celery": celery_health,
        "binance": binance_health,
        "llm_provider": llm_health,
    }

    return {
        "backend_api": "ok",
        "ai_brain": "ready" if brain.get("is_ready") else "offline",
        "l2_ensemble": bool(brain.get("has_l2_models")),
        "llm_engine": bool(brain.get("llm_ready")),
        "rl_agent": bool(brain.get("rl_ready")),
        "risk_engine": bool(brain.get("risk_ready")),
        "resources": resources,
        "services": services,
        "hostname": socket.gethostname(),
        "time": datetime.utcnow().isoformat(),
    }
