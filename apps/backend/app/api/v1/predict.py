# app/api/v1/predict.py

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.hybrid.schemas import Layer2Prediction, MarketContext
from app.ml.model_engine import ModelEngine

router = APIRouter()

# Single shared instance for the process
_model_engine = ModelEngine()


@router.get(
    "/predict/hourly",
    response_model=Layer2Prediction,
    summary="Hourly price & range prediction for a symbol",
)
async def predict_hourly(
    symbol: str = Query(..., description="Symbol, e.g. 'BTCUSDT'."),
    exchange: str = Query("BYBIT", description="Exchange identifier, e.g. 'BYBIT'."),
) -> Layer2Prediction:
    """
    Returns the unified hourly price prediction and range for a given symbol.

    In Phase 1 this is purely *predictive* (no trade execution):
      - direction: 'up' / 'down' / 'flat'
      - price_confidence: 0..1
      - current_price, predicted_price
      - predicted_range_low / predicted_range_high
      - all model votes & debug scores (TFT/TCN/TST/XGB/etc.)
    """

    if not _model_engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model engine is not ready (no models loaded).",
        )

    ctx = MarketContext(
        symbol=symbol,
        exchange=exchange,
        instrument_type="futures",
    )

    layer2 = await _model_engine.predict(ctx)
    return layer2
