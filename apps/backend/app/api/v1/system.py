from __future__ import annotations

import logging
import os
import socket
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter
from sqlalchemy import text

from app.db.database import SessionLocal
from app.services.inference_service import inference_service
from app.api.v1.health import get_system_resources

logger = logging.getLogger(__name__)

router = APIRouter()


def _check_db() -> Dict[str, Any]:
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        return {"status": "ok", "detail": "DB reachable"}
    except Exception as exc:
        logger.exception("DB health check failed: %s", exc)
        return {"status": "down", "detail": str(exc)}


def _check_redis() -> Dict[str, Any]:
    redis_url = (
        os.getenv("REDIS_URL")
        or os.getenv("CELERY_BROKER_URL")
        or "redis://redis:6379/0"
    )
    try:
        import redis  # type: ignore

        client = redis.Redis.from_url(redis_url)
        client.ping()
        return {"status": "ok", "detail": f"Redis reachable @ {redis_url}"}
    except ImportError:
        return {
            "status": "unknown",
            "detail": "redis package not installed in backend image",
        }
    except Exception as exc:
        logger.exception("Redis health check failed: %s", exc)
        return {"status": "down", "detail": str(exc)}


def _check_celery(redis_health: Dict[str, Any]) -> Dict[str, Any]:
    if redis_health.get("status") != "ok":
        return {
            "status": "unknown",
            "detail": "Redis not healthy; Celery status inferred as unknown",
        }
    return {"status": "ok", "detail": "Assuming Celery healthy (Redis OK)"}


def _check_binance() -> Dict[str, Any]:
    if os.getenv("BINANCE_API_KEY"):
        return {
            "status": "ok",
            "detail": "BINANCE_API_KEY set; collectors validate depth",
        }
    return {
        "status": "degraded",
        "detail": "BINANCE_API_KEY not set; using public endpoints only",
    }


def _normalize_brain(brain: Dict[str, Any]) -> Dict[str, bool]:
    """
    Normalize whatever HybridInferenceService returns into simple flags.

    We try multiple possible key names so we stay robust:
      - AI: is_ready / ai / ai_ready
      - L2: has_l2_models / l2 / ensemble
      - LLM: llm_ready / llm / llm_online
      - RL: rl_ready / rl / rl_online
      - RISK: risk_ready / risk / risk_ok
    """
    ai_flag = bool(
        brain.get("is_ready")
        or brain.get("ai")
        or brain.get("ai_ready")
    )
    l2_flag = bool(
        brain.get("has_l2_models")
        or brain.get("l2")
        or brain.get("ensemble")
    )
    llm_flag = bool(
        brain.get("llm_ready")
        or brain.get("llm")
        or brain.get("llm_online")
    )
    rl_flag = bool(
        brain.get("rl_ready")
        or brain.get("rl")
        or brain.get("rl_online")
    )
    risk_flag = bool(
        brain.get("risk_ready")
        or brain.get("risk")
        or brain.get("risk_ok")
    )
    return {
        "ai": ai_flag,
        "l2": l2_flag,
        "llm": llm_flag,
        "rl": rl_flag,
        "risk": risk_flag,
    }


def _check_llm(brain_flags: Dict[str, bool]) -> Dict[str, Any]:
    if brain_flags.get("llm"):
        return {"status": "ok", "detail": "LLM engine ready (Gemini live)"}
    return {
        "status": "unknown",
        "detail": "LLM engine not configured or not yet ready",
    }


@router.get("/system-health")
def system_health() -> Dict[str, Any]:
    # --- Brain / AI layer health ---
    try:
        raw_brain = inference_service.get_brain_health()  # type: ignore[attr-defined]
    except Exception as exc:
        logger.exception("get_brain_health failed: %s", exc)
        raw_brain = {}

    brain_flags = _normalize_brain(raw_brain)

    # --- Resources ---
    try:
        resources = get_system_resources()
    except Exception as exc:
        logger.exception("get_system_resources failed: %s", exc)
        resources = {"cpu_load": 0.0, "ram_usage": 0.0, "gpu_util": 0.0}

    # --- Services ---
    db_health = _check_db()
    redis_health = _check_redis()
    celery_health = _check_celery(redis_health)
    binance_health = _check_binance()
    llm_health = _check_llm(brain_flags)

    services = {
        "db": db_health,
        "redis": redis_health,
        "celery": celery_health,
        "binance": binance_health,
        "llm_provider": llm_health,
    }

    # --- Overall status ---
    service_statuses = [svc.get("status") for svc in services.values()]
    any_down = any(s == "down" for s in service_statuses)
    any_degraded = any(s == "degraded" for s in service_statuses)

    # Consider the AI "ready" if either core AI or L2 or LLM are on
    ai_ready_overall = brain_flags["ai"] or brain_flags["l2"] or brain_flags["llm"]

    if any_down:
        overall_status = "DOWN"
        backend_api = "down"
    elif any_degraded or not ai_ready_overall:
        overall_status = "DEGRADED"
        backend_api = "degraded"
    else:
        overall_status = "OK"
        backend_api = "ok"

    return {
        "status": overall_status,
        "backend_api": backend_api,

        "ai_brain": "ready" if ai_ready_overall else "offline",
        "l2_ensemble": brain_flags["l2"],
        "llm_engine": brain_flags["llm"],
        "rl_agent": brain_flags["rl"],
        "risk_engine": brain_flags["risk"],

        "resources": resources,
        "services": services,
        "hostname": socket.gethostname(),
        "time": datetime.now(timezone.utc).isoformat(),
    }
