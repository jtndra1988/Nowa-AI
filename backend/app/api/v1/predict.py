from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime
import logging

# --- Import our new service ---
from app.services.inference_service import inference_service
from app.db.database import get_db
from app.db.models import Prediction, ModelVersion
from app.hybrid.schemas import MarketContext, HybridDecision
router = APIRouter(tags=["Prediction"])
logger = logging.getLogger(__name__)

# --- New Response Model ---
class HybridPredictResponse(BaseModel):
    symbol: str
    timestamp: datetime
    price_prediction: float
    volatility_prediction: float
    model_version_id: int
    feature_importance: dict

@router.get("/predict", response_model=HybridPredictResponse)
def predict(
    symbol: str = Query(..., description="Asset symbol, e.g. BTC/USDT"),
    db: Session = Depends(get_db)
):
    """
    Run the new Hybrid Ensemble (TFT+TCN+XGB) inference
    and return the blended multi-task predictions.
    """
    if inference_service is None or not inference_service.is_ready:
        logger.error("Inference service not ready or failed to load.")
        raise HTTPException(
            status_code=503, 
            detail="InferenceService is not available. Check server logs."
        )

    try:
        # 1. Run inference
        # The service handles all data fetching and ML logic
        result = inference_service.predict(symbol=symbol)
        
        # 2. Get active model version from DB
        # TODO: You should have a way to query the *active* version
        # For now, we'll just get the latest one for the ensemble
        model_version = db.query(ModelVersion).filter(
            ModelVersion.model_name == "HybridEnsemble"
        ).order_by(ModelVersion.created_at.desc()).first()
        
        if not model_version:
            logger.warning("No 'HybridEnsemble' model version found in DB. Saving prediction without version_id.")
            model_version_id = None
        else:
            model_version_id = model_version.id

        # 3. Save prediction to DB (preserving your old logic)
        prediction_record = Prediction(
            model_version_id=model_version_id,
            symbol=symbol,
            prediction_time=datetime.utcnow(),
            prediction=result['price_prediction'], # Storing price pred
            raw_score=result['volatility_prediction'], # Storing vol pred
            model_inputs=result['feature_importance'] # Storing interpretability
        )
        db.add(prediction_record)
        db.commit()
        logger.info(f"Saved prediction {prediction_record.id} for {symbol}")

        # 4. Return response
        return HybridPredictResponse(
            symbol=symbol,
            timestamp=prediction_record.prediction_time,
            price_prediction=result['price_prediction'],
            volatility_prediction=result['volatility_prediction'],
            model_version_id=model_version_id,
            feature_importance=result['feature_importance']
        )

    except Exception as e:
        logger.error(f"Failed to run prediction for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/hybrid-signal", response_model=HybridDecision)
def hybrid_signal(ctx: MarketContext, db: Session = Depends(get_db)):
    if inference_service is None or not inference_service.is_ready:
        logger.error("Inference service not ready.")
        raise HTTPException(status_code=503, detail="InferenceService is not available")
    return inference_service.build_decision(ctx)