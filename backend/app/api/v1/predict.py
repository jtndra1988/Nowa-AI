from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime
import logging

from app.services.inference_service import inference_service
from app.db.database import get_db
from app.db.models import Prediction, ModelVersion, HybridSignal
from app.hybrid.schemas import MarketContext, HybridDecision

router = APIRouter(tags=["Prediction"])
logger = logging.getLogger(__name__)


class HybridPredictResponse(BaseModel):
    symbol: str
    timestamp: datetime
    price_prediction: float
    volatility_prediction: float
    model_version_id: Optional[int] = None
    feature_importance: dict


@router.get("/predict", response_model=HybridPredictResponse)
def predict(
    symbol: str = Query(..., description="Asset symbol, e.g. BTC/USDT or BTC-PERP"),
    db: Session = Depends(get_db),
):
    """
    Run the Hybrid Ensemble (TFT + TCN + XGBoost) inference
    and return blended price & volatility predictions.
    """
    if inference_service is None or not inference_service.is_ready:
        logger.error("Inference service not ready or failed to load.")
        raise HTTPException(
            status_code=503,
            detail="InferenceService is not available. Check server logs.",
        )

    try:
        # 1. Run inference via service
        result = inference_service.predict(symbol=symbol)

        # 2. Get latest model version for bookkeeping (if exists)
        model_version = (
            db.query(ModelVersion)
            .filter(ModelVersion.model_name == "HybridEnsemble")
            .order_by(ModelVersion.created_at.desc())
            .first()
        )
        model_version_id = model_version.id if model_version else None

        # 3. Persist prediction (optional, but useful for monitoring)
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

        logger.info(f"Saved prediction {prediction_record.id} for {symbol}")

        # 4. Return API response
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
    if inference_service is None or not inference_service.is_ready:
        logger.error("Inference service not ready or failed to load.")
        raise HTTPException(
            status_code=503,
            detail="InferenceService is not available. Check server logs.",
        )

    try:
        decision = inference_service.build_decision(ctx)

        # Log decision for future training / audit
        try:
            record = HybridSignal(
                symbol=decision.symbol,
                instrument_type=decision.instrument_type,
                exchange=ctx.exchange,
                direction=decision.direction,
                p_edge=decision.p_edge,
                confidence=decision.confidence,
                size_factor=decision.size_factor,
                strategy_tag=decision.strategy_tag,
                meta_execute=decision.meta_execute,
                debug_payload=decision.debug,
            )
            db.add(record)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(
                f"Failed to persist HybridSignal for {decision.symbol}: {e}",
                exc_info=True,
            )

        return decision

    except Exception as e:
        logger.error(
            f"Failed to build hybrid signal for {ctx.symbol}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))
