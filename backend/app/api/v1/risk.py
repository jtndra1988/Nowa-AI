# app/api/risk.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException

router = APIRouter()

STATE_PATH = Path("/app/state/risk/state.json")
LEDGER_PATH = Path("/app/state/risk/ledger.json")

def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read {path.name}: {e}")

@router.get("/health")
def risk_health() -> Dict[str, Any]:
    return {
        "ok": True,
        "state_exists": STATE_PATH.exists(),
        "ledger_exists": LEDGER_PATH.exists(),
    }

@router.get("/state")
def risk_state() -> Dict[str, Any]:
    """
    Current risk parameters, engine mode, equity, and per-symbol performance
    (as last dumped by the RiskEngine inside the Celery process).
    """
    state = _load_json(STATE_PATH)
    if not state:
        raise HTTPException(status_code=404, detail="risk state not found yet")
    return state

@router.get("/ledger")
def risk_ledger() -> Dict[str, Any]:
    """
    Open simulated trades tracked by the ExecutionService ledger
    (or you can extend to show live trades by querying the venue).
    """
    data = _load_json(LEDGER_PATH)
    # keep shape stable
    return {"open": data}
