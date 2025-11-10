# app/api/v1/endpoints.py
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from app.db import models
from app.db.database import get_db
from app.api.v1.schemas import OptionInstrument
# --- NEW IMPORTS ---
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


# ---
# --- NEW "READ" ENDPOINTS ADDED ---
# ---

@router.get("/data/sentiment/{symbol}", response_model=List[SentimentResponse])
def get_sentiment_data(
    symbol: str,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """
    Get the latest sentiment data for a symbol.
    """
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
    """
    Get the latest on-chain data for a symbol.
    """
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
    """
    Get the latest developer activity data for a symbol.
    """
    stmt = (
        select(models.DeveloperActivity)
        .where(models.DeveloperActivity.symbol == symbol)
        .order_by(desc(models.DeveloperActivity.timestamp))
        .limit(limit)
    )
    results = db.scalars(stmt).all()
    return results