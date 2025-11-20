from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app import db as db_pkg  # gives us db.models.*

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _get_model(name: str) -> Any | None:
    """Safely get a model from app.db.models, or None if it doesn't exist."""
    try:
        return getattr(db_pkg.models, name)  # type: ignore[attr-defined]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Schemas (match frontend types in api.ts)
# ---------------------------------------------------------------------------

class SentimentPoint(BaseModel):
    t: datetime
    score: float


class OnChainPoint(BaseModel):
    t: datetime
    active: float


class SimpleSeriesPoint(BaseModel):
    t: datetime
    v: float


class CorrelationPoint(BaseModel):
    name: str
    val: float


class RadarIntel(BaseModel):
    regime: Literal["MOMENTUM", "MEAN-REVERT", "BALANCED"] = "BALANCED"
    crowding: str = "Normal positioning"
    trend: Literal["UP", "DOWN", "FLAT"] = "FLAT"
    liquidity: Literal["LOW", "NORMAL", "HIGH"] = "NORMAL"
    warning: str = ""


class MarketIntelResponse(BaseModel):
    symbol: str
    mode: str
    sentimentHistory: List[SentimentPoint]
    onChainHistory: List[OnChainPoint]
    ivHistory: List[SimpleSeriesPoint]
    fundingHistory: List[SimpleSeriesPoint]
    oiHistory: List[SimpleSeriesPoint]
    cvdHistory: List[SimpleSeriesPoint]
    correlations: List[CorrelationPoint]
    radar: RadarIntel


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def _load_market_intel_from_db(
    db: Session,
    symbol: str,
    mode: str,
) -> MarketIntelResponse:
    """Pull timeseries snapshots for the Market Intel tab."""
    now = datetime.now(timezone.utc)
    lookback_hours = 24
    since = now - timedelta(hours=lookback_hours)

    base = (
        symbol.replace("USDT", "")
        .replace("/USDT", "")
        .replace("/", "")
        .upper()
    )

    sentiment_history: List[SentimentPoint] = []
    onchain_history: List[OnChainPoint] = []
    iv_history: List[SimpleSeriesPoint] = []
    funding_history: List[SimpleSeriesPoint] = []
    oi_history: List[SimpleSeriesPoint] = []
    cvd_history: List[SimpleSeriesPoint] = []
    correlations: List[CorrelationPoint] = []

    # --- SentimentFusion -> sentimentHistory ---
    SentimentFusion = _get_model("SentimentFusion")
    if SentimentFusion is not None:
        rows = (
            db.execute(
                select(SentimentFusion)
                .where(
                    SentimentFusion.symbol.in_([symbol, base]),
                    SentimentFusion.timestamp >= since,
                )
                .order_by(desc(SentimentFusion.timestamp))
                .limit(200)
            )
            .scalars()
            .all()
        )
        for r in reversed(rows):
            sentiment_history.append(
                SentimentPoint(
                    t=r.timestamp,
                    score=float(getattr(r, "final_sentiment", 0.0) or 0.0),
                )
            )

    # --- OnchainMetrics -> onChainHistory ---
    OnchainMetrics = _get_model("OnchainMetrics")
    if OnchainMetrics is not None:
        rows = (
            db.execute(
                select(OnchainMetrics)
                .where(
                    OnchainMetrics.symbol.in_([symbol, base, base.lower()]),
                    OnchainMetrics.timestamp >= since,
                )
                .order_by(desc(OnchainMetrics.timestamp))
                .limit(200)
            )
            .scalars()
            .all()
        )
        for r in reversed(rows):
            onchain_history.append(
                OnChainPoint(
                    t=r.timestamp,
                    active=float(getattr(r, "active_addresses", 0.0) or 0.0),
                )
            )

    # --- OptionsDerivedMetrics -> ivHistory & oiHistory ---
    OptionsDerivedMetrics = _get_model("OptionsDerivedMetrics")
    if OptionsDerivedMetrics is not None:
        rows = (
            db.execute(
                select(OptionsDerivedMetrics)
                .where(
                    OptionsDerivedMetrics.symbol.in_([symbol, base]),
                    OptionsDerivedMetrics.timestamp >= since,
                )
                .order_by(desc(OptionsDerivedMetrics.timestamp))
                .limit(200)
            )
            .scalars()
            .all()
        )
        for r in reversed(rows):
            ts = r.timestamp
            iv_history.append(
                SimpleSeriesPoint(
                    t=ts,
                    v=float(getattr(r, "avg_iv_near_term", 0.0) or 0.0),
                )
            )
            # We may not have a dedicated OI field; fall back gracefully
            oi_val = (
                getattr(r, "total_oi", None)
                or getattr(r, "open_interest", None)
                or getattr(r, "put_call_oi_ratio", 1.0)
            )
            oi_history.append(SimpleSeriesPoint(t=ts, v=float(oi_val or 0.0)))

    # --- FundingRate -> fundingHistory ---
    FundingRate = _get_model("FundingRate")
    if FundingRate is not None:
        rows = (
            db.execute(
                select(FundingRate)
                .where(
                    FundingRate.symbol.in_([symbol, base]),
                    FundingRate.timestamp >= since,
                )
                .order_by(desc(FundingRate.timestamp))
                .limit(200)
            )
            .scalars()
            .all()
        )
        for r in reversed(rows):
            funding_history.append(
                SimpleSeriesPoint(
                    t=r.timestamp,
                    v=float(getattr(r, "rate", 0.0) or 0.0),
                )
            )

    # --- OrderbookSnapshot -> cvdHistory ---
    OrderbookSnapshot = _get_model("OrderbookSnapshot")
    if OrderbookSnapshot is not None:
        rows = (
            db.execute(
                select(OrderbookSnapshot)
                .where(
                    OrderbookSnapshot.symbol == symbol,
                    OrderbookSnapshot.timestamp >= since,
                )
                .order_by(desc(OrderbookSnapshot.timestamp))
                .limit(200)
            )
            .scalars()
            .all()
        )
        for r in reversed(rows):
            cvd_history.append(
                SimpleSeriesPoint(
                    t=r.timestamp,
                    v=float(getattr(r, "cdv_1m", 0.0) or 0.0),
                )
            )

    # --- CrossAssetCorr -> correlations ---
    CrossAssetCorr = _get_model("CrossAssetCorr")
    if CrossAssetCorr is not None:
        cac = (
            db.execute(
                select(CrossAssetCorr)
                .where(CrossAssetCorr.base_symbol.in_([symbol, base]))
                .order_by(desc(CrossAssetCorr.timestamp))
                .limit(1)
            )
            .scalars()
            .first()
        )
        if cac:
            correlations.append(
                CorrelationPoint(
                    name="BTC vs ETH",
                    val=float(getattr(cac, "corr_btc_eth", 0.0) or 0.0),
                )
            )
            correlations.append(
                CorrelationPoint(
                    name="BTC vs DXY",
                    val=float(getattr(cac, "corr_btc_dxy", 0.0) or 0.0),
                )
            )

    # --- Simple radar summary from what we have ---
    last_sentiment = sentiment_history[-1].score if sentiment_history else 0.0
    last_funding = funding_history[-1].v if funding_history else 0.0

    regime: Literal["MOMENTUM", "MEAN-REVERT", "BALANCED"]
    trend: Literal["UP", "DOWN", "FLAT"]

    if abs(last_sentiment) > 0.4 and abs(last_funding) > 0.0005:
        regime = "MOMENTUM"
    else:
        regime = "BALANCED"

    if last_sentiment > 0.1:
        trend = "UP"
    elif last_sentiment < -0.1:
        trend = "DOWN"
    else:
        trend = "FLAT"

    radar = RadarIntel(
        regime=regime,
        trend=trend,
        liquidity="NORMAL",
        crowding="Normal positioning",
        warning="",
    )

    return MarketIntelResponse(
        symbol=symbol.upper(),
        mode=mode,
        sentimentHistory=sentiment_history,
        onChainHistory=onchain_history,
        ivHistory=iv_history,
        fundingHistory=funding_history,
        oiHistory=oi_history,
        cvdHistory=cvd_history,
        correlations=correlations,
        radar=radar,
    )


def _build_mock_market_intel(symbol: str, mode: str) -> MarketIntelResponse:
    """Safe fallback if DB is empty or errors out."""
    now = datetime.now(timezone.utc)
    points = 40

    def series(scale: float = 1.0, bias: float = 0.0) -> List[SimpleSeriesPoint]:
        out: List[SimpleSeriesPoint] = []
        for i in range(points):
            t = now - timedelta(minutes=(points - i) * 10)
            v = bias + scale * ((i - points / 2) / points)
            out.append(SimpleSeriesPoint(t=t, v=v))
        return out

    sent = [
        SentimentPoint(
            t=now - timedelta(minutes=(points - i) * 10),
            score=(i - points / 2) / points,
        )
        for i in range(points)
    ]
    onchain = [
        OnChainPoint(
            t=now - timedelta(minutes=(points - i) * 10),
            active=100_000 + i * 500,
        )
        for i in range(points)
    ]

    radar = RadarIntel(
        regime="BALANCED",
        crowding="Normal positioning",
        trend="FLAT",
        liquidity="NORMAL",
        warning="Mock data – collectors not live yet",
    )
    has_any_data = any(
        len(arr)
        for arr in (
            sentiment_history,
            onchain_history,
            iv_history,
            funding_history,
            oi_history,
            cvd_history,
            correlations,
        )
    )

    if not has_any_data:
        # Return synthetic intel so the UI is always populated
        return _build_mock_market_intel(symbol, mode)
    return MarketIntelResponse(
        symbol=symbol.upper(),
        mode=mode,
        sentimentHistory=sent,
        onChainHistory=onchain,
        ivHistory=series(0.1, 0.6),
        fundingHistory=series(0.0005, 0.0),
        oiHistory=series(10_000, 50_000),
        cvdHistory=series(1000, 0.0),
        correlations=[
            CorrelationPoint(name="BTC vs ETH", val=0.65),
            CorrelationPoint(name="BTC vs DXY", val=-0.35),
        ],
        radar=radar,
    )


# ---------------------------------------------------------------------------
# Public endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/market-intel/{symbol}",
    response_model=MarketIntelResponse,
    tags=["market-intel"],
)
def market_intel(
    symbol: str,
    mode: str = Query("futures", description="Market mode (spot/futures/perp)"),
    db: Session = Depends(get_db),
) -> MarketIntelResponse:
    """
    Used by Market Intel tab.

    Final URL (with main.py prefix): /api/v1/market-intel/{symbol}
    """
    try:
        return _load_market_intel_from_db(db, symbol=symbol, mode=mode)
    except Exception as e:
        logger.exception("market_intel failed, falling back to mock: %s", e)
        # Never break the UI – always return a sane structure
        return _build_mock_market_intel(symbol, mode)
