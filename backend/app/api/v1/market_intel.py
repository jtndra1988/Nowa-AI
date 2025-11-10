# app/api/v1/market_intel.py

from datetime import datetime, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy import select, desc
from app.db.database import get_db
from app.db import models
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/v1/market-intel", tags=["market-intel"])


@router.get("/{symbol}")
def get_market_intel(symbol: str, db: Session = Depends(get_db)):
    symbol = symbol.upper()
    now = datetime.utcnow()
    since = now - timedelta(hours=24)

    sf = (
        db.execute(
            select(models.SentimentFusion)
            .where(models.SentimentFusion.symbol == symbol)
            .order_by(desc(models.SentimentFusion.timestamp))
            .limit(1)
        )
        .scalars()
        .first()
    )

    od = (
        db.execute(
            select(models.OptionsDerivedMetrics)
            .where(models.OptionsDerivedMetrics.symbol == symbol)
            .order_by(desc(models.OptionsDerivedMetrics.timestamp))
            .limit(1)
        )
        .scalars()
        .first()
    )

    oc = (
        db.execute(
            select(models.OnchainMetrics)
            .where(models.OnchainMetrics.symbol == symbol)
            .order_by(desc(models.OnchainMetrics.timestamp))
            .limit(1)
        )
        .scalars()
        .first()
    )

    da = (
        db.execute(
            select(models.DeveloperActivity)
            .where(models.DeveloperActivity.symbol == symbol)
            .order_by(desc(models.DeveloperActivity.timestamp))
            .limit(1)
        )
        .scalars()
        .first()
    )

    fr = (
        db.execute(
            select(models.FundingRate)
            .where(models.FundingRate.symbol == symbol)
            .order_by(desc(models.FundingRate.timestamp))
            .limit(1)
        )
        .scalars()
        .first()
    )

    return {
        "symbol": symbol,
        "sentiment": {
            "final_sentiment": getattr(sf, "final_sentiment", None) if sf else None,
            "avg_news_sentiment": getattr(sf, "avg_news_sentiment", None) if sf else None,
            "fear_greed_local": getattr(sf, "fear_greed_index", None) if sf else None,
        },
        "options": {
            "put_call_oi_ratio": getattr(od, "put_call_oi_ratio", None) if od else None,
            "avg_iv_near_term": getattr(od, "avg_iv_near_term", None) if od else None,
            "iv_skew_25d": getattr(od, "iv_skew_25d", None) if od else None,
        },
        "onchain": {
            "whale_volume_usd": getattr(oc, "whale_volume_usd", None) if oc else None,
            "exchange_net_flow_usd": getattr(oc, "exchange_net_flow_usd", None) if oc else None,
            "active_addresses": getattr(oc, "active_addresses", None) if oc else None,
        },
        "developer": {
            "stars": getattr(da, "stars", None) if da else None,
            "forks": getattr(da, "forks", None) if da else None,
            "open_issues": getattr(da, "open_issues", None) if da else None,
        },
        "funding": {
            "funding_rate": getattr(fr, "funding_rate", None) if fr else None,
        },
    }
