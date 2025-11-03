# app/api/v1/predict.py
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from app.ml.inference_ensemble import run

router = APIRouter(tags=["Prediction"])

class PredictResponse(BaseModel):
    symbol: str
    decision: str
    confidence: float
    lstm_next: float
    xgb_next: float
    regime_up: float
    regime_down: float
    final_sentiment: float

@router.get("/predict", response_model=PredictResponse)
def predict(symbol: str = Query(..., description="Asset symbol, e.g. BTC/USDT")):
    """
    Run the ensemble inference (LSTM + XGB + Fusion) and return calibrated decision.
    """
    try:
        result = run(symbol=symbol, lookback_rows=5000, base_table="futures_market_data")
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
