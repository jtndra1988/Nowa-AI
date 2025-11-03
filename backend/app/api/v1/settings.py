# app/api/settings.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import RiskSettingsGlobal, RiskSettingsSymbol
from app.services.settings_service import load_risk_settings
from app.api.v1.schemas_settings import RiskSettingsGlobalIn, RiskSettingsSymbolIn

router = APIRouter(prefix="/api/settings", tags=["settings"])

@router.get("/effective/{symbol}")
def get_effective_settings(symbol: str, db: Session = Depends(get_db)):
    return load_risk_settings(db, symbol.upper())

@router.get("/global")
def get_global_settings(db: Session = Depends(get_db)):
    g = db.query(RiskSettingsGlobal).order_by(RiskSettingsGlobal.id.asc()).first()
    if not g:
        g = RiskSettingsGlobal()
        db.add(g); db.commit(); db.refresh(g)
    return g.__dict__

@router.put("/global")
def update_global_settings(payload: RiskSettingsGlobalIn, db: Session = Depends(get_db)):
    g = db.query(RiskSettingsGlobal).order_by(RiskSettingsGlobal.id.asc()).first()
    if not g:
        g = RiskSettingsGlobal(); db.add(g)
    for k, v in payload.dict(exclude_unset=True).items():
        setattr(g, k, v)
    db.commit(); db.refresh(g)
    return g.__dict__

@router.get("/symbol/{symbol}")
def get_symbol_settings(symbol: str, db: Session = Depends(get_db)):
    s = db.query(RiskSettingsSymbol).filter_by(symbol=symbol.upper()).first()
    return (s.__dict__ if s else {})

@router.put("/symbol/{symbol}")
def upsert_symbol_settings(symbol: str, payload: RiskSettingsSymbolIn, db: Session = Depends(get_db)):
    s = db.query(RiskSettingsSymbol).filter_by(symbol=symbol.upper()).first()
    if not s:
        s = RiskSettingsSymbol(symbol=symbol.upper()); db.add(s)
    for k, v in payload.dict(exclude_unset=True).items():
        setattr(s, k, v)
    db.commit(); db.refresh(s)
    return s.__dict__
