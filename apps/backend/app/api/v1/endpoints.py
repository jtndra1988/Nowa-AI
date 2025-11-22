# app/api/v1/endpoints.py
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, desc, distinct

from app.db import models
from app.db.database import get_db
from app.api.v1.schemas import OptionInstrument
from app.api.v1.schemas import SentimentResponse, OnchainResponse, DeveloperResponse
from app.services.market_data import MarketDataService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Market Data"])
md_service = MarketDataService()


@router.get("/options-chain", response_model=List[OptionInstrument])
def get_options_chain(
    symbol: str = Query(..., description="Underlying symbol, e.g., 'BTC'"),
    db: Session = Depends(get_db),
):
    """
    Get the latest options chain data for a given underlying.
    """
    try:
        data = md_service.get_latest_options_chain(db, symbol)
        if not data:
            return []
        return data
    except Exception as e:
        logger.error(
            f"Failed to fetch options chain for {symbol}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/assets", response_model=List[str])
def get_active_assets(
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Return at most `limit` base symbols that actually exist in the database.
    Used to populate frontend asset selectors dynamically.
    """
    try:
        # 1. Try Futures
        symbols = db.query(models.FuturesMarketData.symbol).distinct().all()

        # 2. If empty, Try Spot
        if not symbols:
            symbols = db.query(models.MarketData.symbol).distinct().all()

        # 3. If still empty, fallback
        if not symbols:
            return ["BTC", "ETH"][:limit]

        # Clean up symbols (e.g., "BTC/USDT" -> "BTC")
        cleaned_assets: list[str] = []
        seen = set()

        for (sym,) in symbols:
            base = (
                sym.replace("/USDT", "")
                .replace("USDT", "")
                .replace("-PERP", "")
                .upper()
            )
            if base and base not in seen:
                seen.add(base)
                cleaned_assets.append(base)

        # Sort and cap to `limit`
        cleaned_assets = sorted(cleaned_assets)
        return cleaned_assets[:limit]

    except Exception as e:
        logger.error(f"Failed to fetch active assets: {e}")
        return ["BTC", "ETH"][:limit]  # Fail safe


# --- EXISTING READ ENDPOINTS ---

@router.get("/data/sentiment/{symbol}", response_model=List[SentimentResponse])
def get_sentiment_data(
    symbol: str,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    stmt = (
        select(models.SentimentFusion)
        .where(models.SentimentFusion.symbol == symbol)
        .order_by(desc(models.SentimentFusion.timestamp))
        .limit(limit)
    )
    results = db.scalars(stmt).all()
    return results

@router.get("/data/onchain/{symbol}", response_model=List[OnchainResponse])
def get_onchain_data(
    symbol: str,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    stmt = (
        select(models.OnchainMetrics)
        .where(models.OnchainMetrics.symbol == symbol)
        .order_by(desc(models.OnchainMetrics.timestamp))
        .limit(limit)
    )
    results = db.scalars(stmt).all()
    return results

@router.get("/data/developer/{symbol}", response_model=List[DeveloperResponse])
def get_developer_data(
    symbol: str,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    stmt = (
        select(models.DeveloperActivity)
        .where(models.DeveloperActivity.symbol == symbol)
        .order_by(desc(models.DeveloperActivity.timestamp))
        .limit(limit)
    )
    results = db.scalars(stmt).all()
    return results