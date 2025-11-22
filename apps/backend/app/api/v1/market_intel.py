from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, List, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app import db as db_pkg  # gives us db.models.*

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market-intel", tags=["market-intel"])


def get_db():
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

def _get_model(name: str) -> Any | None:
    """Safely get a model from app.db.models, or None if it doesn't exist."""
    try:
        return getattr(db_pkg.models, name)  # type: ignore[attr-defined]
    except Exception:
        return None

def _symbol_candidates(symbol: str) -> List[str]:
    """Return plausible symbol keys for intel lookups.

    Accepts BTC-PERP / BTCUSDT / BTC/USDT etc and returns a small set
    like [BTC-PERP, BTC, BTCUSDT, BTCUSD] so we can match whatever
    we actually stored in the DB.
    """
    if not symbol:
        return []
    s = symbol.upper()

    candidates: set[str] = set()
    candidates.add(s)

    # Split on common separators first, e.g. BTC-PERP -> BTC
    base = s
    for sep in ("-", "/", ":"):
        if sep in base:
            base = base.split(sep)[0]
            break
    candidates.add(base)

    # Strip common suffixes
    base_no_suf = base
    for suf in ("USDT", "USD", "PERP"):
        if base_no_suf.endswith(suf):
            base_no_suf = base_no_suf[: -len(suf)]

    # Handle embedded suffixes (BTCUSDT style)
    for suf in ("USDT", "USD"):
        if suf in base_no_suf and not base_no_suf.endswith(suf):
            base_no_suf = base_no_suf.replace(suf, "")

    if base_no_suf:
        candidates.add(base_no_suf)          # BTC
        candidates.add(base_no_suf + "USDT") # BTCUSDT
        candidates.add(base_no_suf + "USD")  # BTCUSD

    return [c for c in candidates if c]

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

    # Normalize symbol like BTC-PERP / BTC/USDT / BTCUSDT into
    # a small candidate set we can use in .in_(...) filters.
    candidates = _symbol_candidates(symbol)

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
                    SentimentFusion.symbol.in_(candidates),
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
                    OnchainMetrics.symbol.in_(candidates),
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
                    active=float(
                        getattr(
                            r,
                            "active_addresses",
                            getattr(r, "total_active_addresses", 0.0),
                        )
                        or 0.0
                    ),
                )
            )

    # --- OptionsDerivedMetrics -> ivHistory & oiHistory ---
    OptionsDerivedMetrics = _get_model("OptionsDerivedMetrics")
    if OptionsDerivedMetrics is not None:
        rows = (
            db.execute(
                select(OptionsDerivedMetrics)
                .where(
                    OptionsDerivedMetrics.symbol.in_(candidates),
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
                    FundingRate.symbol.in_(candidates),
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
                    # IMPORTANT: column is funding_rate, not 'rate'
                    v=float(getattr(r, "funding_rate", 0.0) or 0.0),
                )
            )

    # --- OrderbookSnapshot -> cvdHistory ---
    OrderbookSnapshot = _get_model("OrderbookSnapshot")
    if OrderbookSnapshot is not None:
        rows = (
            db.execute(
                select(OrderbookSnapshot)
                .where(
                    OrderbookSnapshot.symbol.in_(candidates),
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
                .where(CrossAssetCorr.base_symbol.in_(candidates))
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

    if abs(last_sentiment) > 0.4 and abs(last_funding) > 0.0005:
        regime: Literal["MOMENTUM", "MEAN-REVERT", "BALANCED"] = "MOMENTUM"
    else:
        regime = "BALANCED"

    if last_sentiment > 0.1:
        trend: Literal["UP", "DOWN", "FLAT"] = "UP"
    elif last_sentiment < -0.1:
        trend = "DOWN"
    else:
        trend = "FLAT"

    radar = RadarIntel(
        regime=regime,
        crowding="Normal positioning",
        trend=trend,
        liquidity="NORMAL",
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

# ---------------------------------------------------------------------------
# Mock / safe fallback (only for catastrophic errors)
# ---------------------------------------------------------------------------


def _build_mock_market_intel(symbol: str, mode: str) -> MarketIntelResponse:
    """
    Very small synthetic fallback so the UI never completely breaks.

    This SHOULD be used only if _load_market_intel_from_db raises unexpectedly
    (for example, during a migration). It does NOT try to be "smart".
    """
    now = datetime.now(timezone.utc)
    points = 40

    def make_series(scale: float = 1.0, bias: float = 0.0) -> List[SimpleSeriesPoint]:
        out: List[SimpleSeriesPoint] = []
        for i in range(points):
            t = now - timedelta(minutes=(points - i) * 10)
            v = bias + scale * ((i - points / 2) / points)
            out.append(SimpleSeriesPoint(t=t, v=v))
        return out

    # Flat-ish sentiment and on-chain active addresses
    sentiment = [
        SentimentPoint(
            t=now - timedelta(minutes=(points - i) * 10),
            score=0.0,
        )
        for i in range(points)
    ]
    onchain = [
        OnChainPoint(
            t=now - timedelta(minutes=(points - i) * 10),
            active=100_000.0,
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

    return MarketIntelResponse(
        symbol=symbol.upper(),
        mode=mode,
        sentimentHistory=sentiment,
        onChainHistory=onchain,
        ivHistory=make_series(0.1, 0.6),
        fundingHistory=make_series(0.0005, 0.0),
        oiHistory=make_series(10_000, 50_000),
        cvdHistory=make_series(1000, 0.0),
        correlations=[
            CorrelationPoint(name="BTC vs ETH", val=0.65),
            CorrelationPoint(name="BTC vs DXY", val=-0.35),
        ],
        radar=radar,
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@router.get(
    "/{symbol}",
    response_model=MarketIntelResponse,
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
        # Main path: always try to pull real DB-backed intel.
        return _load_market_intel_from_db(db, symbol=symbol, mode=mode)
    except Exception as e:
        logger.exception("market_intel failed, falling back to mock: %s", e)
        # Never break the UI – always return a sane structure
        return _build_mock_market_intel(symbol, mode)
