from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.services.inference_service import inference_service
from app import db as db_pkg  # to be safe when importing models dynamically

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# DB dependency (safe)
# ---------------------------------------------------------------------------


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class BrainHealth(BaseModel):
    is_ready: bool
    has_l2_models: bool
    llm_ready: bool
    rl_ready: bool
    risk_ready: bool


class AIStats(BaseModel):
    total_decisions: int
    decisions_24h: int
    win_rate_30d: float
    sharpe_30d: float
    max_drawdown_30d: float
    pnl_30d: float


class AIStrategy(BaseModel):
    id: str
    name: str
    kind: str
    mode: str
    enabled: bool
    description: str
    sharpe_30d: Optional[float] = None
    win_rate_30d: Optional[float] = None


class AIDecision(BaseModel):
    id: str
    timestamp: datetime
    symbol: str
    market: str
    side: str
    action: str
    size: Optional[float] = None
    price: Optional[float] = None
    confidence: Optional[float] = None
    pnl: Optional[float] = None
    notes: Optional[str] = None


def _get_model(name: str) -> Any | None:
    try:
        return getattr(db_pkg.models, name)  # type: ignore[attr-defined]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# /api/v1/brain-health
# ---------------------------------------------------------------------------


@router.get("/brain-health", response_model=BrainHealth)
def brain_health() -> BrainHealth:
    """
    URL (with main.py prefix): /api/v1/brain-health
    """
    try:
        if inference_service is None:
            return BrainHealth(
                is_ready=False,
                has_l2_models=False,
                llm_ready=False,
                rl_ready=False,
                risk_ready=False,
            )

        data = inference_service.get_brain_health()
        return BrainHealth(
            is_ready=bool(data.get("is_ready")),
            has_l2_models=bool(data.get("has_l2_models")),
            llm_ready=bool(data.get("llm_ready")),
            rl_ready=bool(data.get("rl_ready")),
            risk_ready=bool(data.get("risk_ready")),
        )
    except Exception as e:
        logger.exception("brain_health failed: %s", e)
        return BrainHealth(
            is_ready=False,
            has_l2_models=False,
            llm_ready=False,
            rl_ready=False,
            risk_ready=False,
        )


# ---------------------------------------------------------------------------
# /api/v1/ai/stats
# ---------------------------------------------------------------------------


@router.get("/ai/stats", response_model=AIStats)
def ai_stats(db: Session = Depends(get_db)) -> AIStats:
    """
    High-level AI performance stats.

    Uses HybridDecisionLog / DecisionLog / AIDecisionLog if present.
    If not, returns zeros so frontend always works.
    """
    total_decisions = 0
    decisions_24h = 0
    win_rate_30d = 0.0
    sharpe_30d = 0.0
    max_drawdown_30d = 0.0
    pnl_30d = 0.0

    try:
        DecisionModel = None
        for name in ("HybridDecisionLog", "DecisionLog", "AIDecisionLog"):
            DecisionModel = _get_model(name)
            if DecisionModel is not None:
                break

        if DecisionModel is None:
            return AIStats(
                total_decisions=0,
                decisions_24h=0,
                win_rate_30d=0.0,
                sharpe_30d=0.0,
                max_drawdown_30d=0.0,
                pnl_30d=0.0,
            )

        now = datetime.now(timezone.utc)
        since_24h = now - timedelta(hours=24)
        since_30d = now - timedelta(days=30)

        total_decisions = db.query(func.count(DecisionModel.id)).scalar() or 0
        decisions_24h = (
            db.query(func.count(DecisionModel.id))
            .filter(DecisionModel.timestamp >= since_24h)
            .scalar()
            or 0
        )

        # Win rate if model has a boolean win_flag
        if hasattr(DecisionModel, "win_flag"):
            total_30 = (
                db.query(func.count(DecisionModel.id))
                .filter(DecisionModel.timestamp >= since_30d)
                .scalar()
                or 0
            )
            if total_30 > 0:
                wins_30 = (
                    db.query(func.count(DecisionModel.id))
                    .filter(
                        DecisionModel.timestamp >= since_30d,
                        DecisionModel.win_flag.is_(True),
                    )
                    .scalar()
                    or 0
                )
                win_rate_30d = round(wins_30 / total_30, 4)

        # PnL aggregation if we have a PnL-like column
        for pnl_field in ("pnl", "pnl_usd", "pnl_realized"):
            if hasattr(DecisionModel, pnl_field):
                pnl_30d = float(
                    db.query(
                        func.coalesce(
                            func.sum(getattr(DecisionModel, pnl_field)), 0.0
                        )
                    )
                    .filter(DecisionModel.timestamp >= since_30d)
                    .scalar()
                    or 0.0
                )
                break

    except Exception as e:
        logger.exception("ai_stats failed: %s", e)

    return AIStats(
        total_decisions=total_decisions,
        decisions_24h=decisions_24h,
        win_rate_30d=win_rate_30d,
        sharpe_30d=sharpe_30d,
        max_drawdown_30d=max_drawdown_30d,
        pnl_30d=pnl_30d,
    )


# ---------------------------------------------------------------------------
# /api/v1/ai/strategy
# ---------------------------------------------------------------------------


@router.get("/ai/strategy", response_model=List[AIStrategy])
def ai_strategy(db: Session = Depends(get_db)) -> List[AIStrategy]:
    """
    Returns defined strategies from DB if available,
    otherwise a static view based on loaded models.
    """
    strategies: List[AIStrategy] = []

    try:
        StrategyModel = None
        for name in ("StrategyConfig", "AIStrategyConfig", "Strategy"):
            StrategyModel = _get_model(name)
            if StrategyModel is not None:
                break

        if StrategyModel is not None:
            rows = db.query(StrategyModel).order_by(
                getattr(StrategyModel, "name", StrategyModel.id)
            )
            for r in rows:
                strategies.append(
                    AIStrategy(
                        id=str(getattr(r, "id")),
                        name=getattr(r, "name", "Unnamed"),
                        kind=getattr(r, "kind", "hybrid"),
                        mode=getattr(r, "mode", "futures"),
                        enabled=bool(getattr(r, "enabled", True)),
                        description=getattr(r, "description", ""),
                        sharpe_30d=getattr(r, "sharpe_30d", None),
                        win_rate_30d=getattr(r, "win_rate_30d", None),
                    )
                )

    except Exception as e:
        logger.exception("ai_strategy DB load failed: %s", e)
        strategies = []

    if not strategies:
        # Static fallback based on your model stack
        brain = (
            inference_service.get_brain_health()
            if inference_service is not None
            else {"is_ready": False}
        )
        ready = bool(brain.get("is_ready"))
        strategies = [
            AIStrategy(
                id="tft-vision",
                name="TFT Vision Model",
                kind="TFT",
                mode="futures",
                enabled=ready,
                description="Sequence-aware transformer forecasting short-term direction.",
            ),
            AIStrategy(
                id="tcn-reflex",
                name="TCN Reflex Model",
                kind="TCN",
                mode="futures",
                enabled=ready,
                description="Fast convolutional model capturing intraday reflex moves.",
            ),
            AIStrategy(
                id="xgb-analyst",
                name="XGBoost Analyst",
                kind="XGBoost",
                mode="spot",
                enabled=ready,
                description="Gradient-boosted model on structured factors for spot signals.",
            ),
        ]

    return strategies


# ---------------------------------------------------------------------------
# /api/v1/ai/decisions
# ---------------------------------------------------------------------------


@router.get("/ai/decisions", response_model=List[AIDecision])
def ai_decisions(
    limit: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
) -> List[AIDecision]:
    """
    Returns the most recent decisions taken by the AI.

    Uses HybridDecisionLog / DecisionLog / AIDecisionLog if present.
    Safe fallback: returns empty list on any failure.
    """
    decisions: List[AIDecision] = []

    try:
        DecisionModel = None
        for name in ("HybridDecisionLog", "DecisionLog", "AIDecisionLog"):
            DecisionModel = _get_model(name)
            if DecisionModel is not None:
                break

        if DecisionModel is None:
            return []

        rows = (
            db.query(DecisionModel)
            .order_by(desc(DecisionModel.timestamp))
            .limit(limit)
            .all()
        )

        for r in rows:
            decisions.append(
                AIDecision(
                    id=str(getattr(r, "id")),
                    timestamp=getattr(r, "timestamp"),
                    symbol=getattr(r, "symbol", "UNKNOWN"),
                    market=getattr(r, "market", "futures"),
                    side=getattr(r, "side", "flat"),
                    action=getattr(r, "action", "hold"),
                    size=getattr(r, "size", None),
                    price=getattr(r, "price", None),
                    confidence=getattr(r, "confidence", None),
                    pnl=getattr(r, "pnl", None),
                    notes=getattr(r, "reason", None),
                )
            )
    except Exception as e:
        logger.exception("ai_decisions failed: %s", e)
        decisions = []

    return decisions