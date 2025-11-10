from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime
import logging

from app.services.inference_service import inference_service
from app.db.database import get_db
from app.db.models import Prediction, ModelVersion
from app.hybrid.schemas import MarketContext, HybridDecision

router = APIRouter(tags=["Prediction"])
logger = logging.getLogger(__name__)


class HybridPredictResponse(BaseModel):
    symbol: str
    timestamp: datetime
    price_prediction: float
    volatility_prediction: float
    model_version_id: int | None = None
    feature_importance: dict


@router.get("/predict", response_model=HybridPredictResponse)
def predict(
    symbol: str = Query(..., description="Asset symbol, e.g. BTC-PERP"),
    db: Session = Depends(get_db),
):
    """
    Run Hybrid Ensemble (TFT+TCN+XGB) and store prediction.
    """
    if inference_service is None or not inference_service.is_ready:
        logger.error("Inference service not ready or failed to load.")
        raise HTTPException(
            status_code=503,
            detail="InferenceService is not available.",
        )

    try:
        result = inference_service.predict(symbol=symbol)

        model_version = (
            db.query(ModelVersion)
            .filter(ModelVersion.model_name == "HybridEnsemble")
            .order_by(ModelVersion.created_at.desc())
            .first()
        )
        model_version_id = model_version.id if model_version else None

        prediction_record = Prediction(
            model_version_id=model_version_id,
            symbol=symbol,
            prediction_time=datetime.utcnow(),
            prediction=result["price_prediction"],
            raw_score=result["volatility_prediction"],
            model_inputs=result["feature_importance"],
        )
        db.add(prediction_record)
        db.commit()
        db.refresh(prediction_record)

        return HybridPredictResponse(
            symbol=symbol,
            timestamp=prediction_record.prediction_time,
            price_prediction=result["price_prediction"],
            volatility_prediction=result["volatility_prediction"],
            model_version_id=model_version_id,
            feature_importance=result["feature_importance"],
        )
    except Exception as e:
        logger.error(f"Failed to run prediction for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/hybrid-signal", response_model=HybridDecision)
def hybrid_signal(
    ctx: MarketContext,
    db: Session = Depends(get_db),
):
    """
    Returns a trade-ready hybrid decision for the given context.

    This is what your Nowa execution bot should call:
      - For spot:    instrument_type="spot"
      - For perps:   instrument_type="perp"
      - For futures: instrument_type="future"
      - For options: instrument_type="option"
    """
    if inference_service is None or not inference_service.is_ready:
        logger.error("Inference service not ready or failed to load.")
        raise HTTPException(
            status_code=503,
            detail="InferenceService is not available. Check server logs.",
        )

    try:
        decision = inference_service.build_decision(ctx)

        # Optional: log decisions to DB for monitoring / future meta-training.
        # Example (uncomment when you add a Decision model):
        # db.add(HybridDecisionORM.from_decision(decision))
        # db.commit()

        return decision
    except Exception as e:
        logger.error(f"Failed to build hybrid signal for {ctx.symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
