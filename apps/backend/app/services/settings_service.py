# app/services/settings_service.py
from typing import Dict, Any
from sqlalchemy.orm import Session
from app.db.models import RiskSettingsGlobal, RiskSettingsSymbol, RiskLearnedState

DEFAULT_REGIME = {0:0.8, 1:1.1, 2:1.0}
# ------------------------------------------------------------------------
# Simple class wrapper so Celery worker can use a RiskSettingsService
# without import errors.
# ------------------------------------------------------------------------
class RiskSettingsService:
    """
    Thin wrapper around the existing risk settings helpers.
    Used by the Celery worker to load all symbol-level risk configs.
    """

    def __init__(self, db: Session):
        self.db = db

    def get_all_settings(self):
        """
        Returns all RiskSettingsSymbol rows so the worker can cache them.
        """
        return self.db.query(RiskSettingsSymbol).all()

def load_risk_settings(db: Session, symbol: str) -> Dict[str, Any]:
    g: RiskSettingsGlobal = db.query(RiskSettingsGlobal).order_by(RiskSettingsGlobal.id.asc()).first()
    s: RiskSettingsSymbol = db.query(RiskSettingsSymbol).filter_by(symbol=symbol).first()

    cfg = {
        # global defaults:
        "account_equity_usd": g.account_equity_usd if g else 100000.0,
        "max_portfolio_leverage": g.max_portfolio_leverage if g else 2.0,
        "max_concurrent_positions": g.max_concurrent_positions if g else 10,
        "target_daily_vol": g.target_daily_vol if g else 0.01,
        "min_position_usd": g.min_position_usd if g else 200.0,
        "atr_length": g.atr_length if g else 14,
        "sl_atr_mult_init": g.sl_atr_mult_init if g else 1.8,
        "tp_atr_mult_init": g.tp_atr_mult_init if g else 3.0,
        "base_confidence_cutoff": g.base_confidence_cutoff if g else 0.55,
        "max_confidence_boost": g.max_confidence_boost if g else 1.6,
        "regime_risk_multipliers": (g.regime_risk_multipliers if g and g.regime_risk_multipliers else DEFAULT_REGIME),
        "daily_loss_limit_pct": g.daily_loss_limit_pct if g else 0.03,
        "rolling_max_dd_pct": g.rolling_max_dd_pct if g else 0.15,
        "per_trade_risk_pct_cap": g.per_trade_risk_pct_cap if g else 0.01,
        "ewma_alpha": g.ewma_alpha if g else 0.2,
        "min_trades_to_learn": g.min_trades_to_learn if g else 20,
        "bandit_lr": g.bandit_lr if g else 0.05,
        "min_tp_sl_ratio": g.min_tp_sl_ratio if g else 1.3,
        "min_sl_atr_mult": g.min_sl_atr_mult if g else 0.8,
        "max_sl_atr_mult": g.max_sl_atr_mult if g else 4.0,
        "min_tp_atr_mult": g.min_tp_atr_mult if g else 1.2,
        "max_tp_atr_mult": g.max_tp_atr_mult if g else 6.0,
        "taker_fee_bps": g.taker_fee_bps if g else 6.0,
        "default_adapter": g.default_adapter if g else "BYBIT",
    }

    if s:
        for k in [
            "max_symbol_leverage", "max_symbol_exposure_pct", "atr_length", "sl_atr_mult_init",
            "tp_atr_mult_init", "base_confidence_cutoff", "min_position_usd",
            "per_trade_risk_pct_cap", "taker_fee_bps"
        ]:
            v = getattr(s, k)
            if v is not None:
                cfg[k] = v
        if s.regime_risk_multipliers:
            cfg["regime_risk_multipliers"] = s.regime_risk_multipliers
    return cfg

def load_learned_state(db: Session, symbol: str) -> RiskLearnedState:
    st = db.query(RiskLearnedState).filter_by(symbol=symbol).first()
    if st is None:
        st = RiskLearnedState(symbol=symbol)
        db.add(st); db.commit(); db.refresh(st)
    return st
def refresh_cache(db: Session):
    """
    Refreshes any cached risk settings.
    Currently, this just pre-queries the global settings to warm the DB cache.
    """
    try:
        # Pre-load the global settings by executing the query
        db.query(RiskSettingsGlobal).order_by(RiskSettingsGlobal.id.asc()).first()
    except Exception as e:
        print(f"[RiskSettings] Error during cache warm-up: {e}")
        # Don't crash the task, just log
        pass
