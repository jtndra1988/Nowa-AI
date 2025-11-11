# app/api/v1/predict.py

from typing import Optional
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
# --- FIX: Changed _name_ to __name__ ---
logger = logging.getLogger(__name__)

@router.post("/hybrid-signal", response_model=HybridDecision)
async def hybrid_signal(ctx: MarketContext, db: Session = Depends(get_db)):
    if inference_service is None or not inference_service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="InferenceService is not available. Check server logs.",
        )

    try:
        decision = await inference_service.build_decision(ctx)
        # Do NOT write HybridSignal here; it's already persisted inside the service.
        return decision
    except Exception as e:
        logger.error(
            f"Failed to build hybrid signal for {ctx.symbol}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


