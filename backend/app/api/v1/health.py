# app/api/health.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
import socket

from fastapi import APIRouter
from app.db.database import engine  # reuse your sync engine

router = APIRouter()

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

@router.get("/healthz")
def healthz():
    """
    Lightweight liveness probe.
    """
    return {
        "status": "ok",
        "time": _utcnow_iso(),
        "host": socket.gethostname(),
        "service": "ml-trading-api",
    }

@router.get("/livez")
def livez():
    """
    Synonym for healthz (some platforms use /livez).
    """
    return healthz()

@router.get("/readyz")
def readyz():
    """
    Readiness probe: simple DB connectivity check.
    If you want deeper checks (e.g., model files on disk), add them here.
    """
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("SELECT 1")
        db_ok: Literal["ok"] = "ok"
    except Exception as e:
        db_ok = f"error: {e.__class__.__name__}"

    return {
        "status": "ok" if db_ok == "ok" else "degraded",
        "db": db_ok,
        "time": _utcnow_iso(),
        "host": socket.gethostname(),
    }
