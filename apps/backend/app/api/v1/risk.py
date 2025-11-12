# app/api/v1/risk.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.db.database import get_db
from app.db import models

router = APIRouter(prefix="/api/v1/risk", tags=["risk"])


@router.get("/state/{symbol}")
def get_risk_state(symbol: str, db: Session = Depends(get_db)):
    symbol = symbol.upper()

    global_cfg = db.query(models.RiskSettingsGlobal).first()
    sym_cfg = (
        db.query(models.RiskSettingsSymbol)
        .filter(models.RiskSettingsSymbol.symbol == symbol)
        .one_or_none()
    )
    learned = (
        db.query(models.RiskLearnedState)
        .filter(models.RiskLearnedState.symbol == symbol)
        .one_or_none()
    )

    return {
        "symbol": symbol,
        "global": global_cfg.__dict__ if global_cfg else None,
        "symbol_settings": sym_cfg.__dict__ if sym_cfg else None,
        "learned": learned.__dict__ if learned else None,
    }
