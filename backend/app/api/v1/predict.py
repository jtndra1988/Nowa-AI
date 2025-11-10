# app/api/v1/predict.py

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import logging

from app.services.inference_service import inference_service
from app.db.database import get_db
from app.db.models import HybridSignal
from app.hybrid.schemas import MarketContext, HybridDecision

# ---
# NOTE: Renamed the file-level logger to avoid a potential `_name_` error.
# Using __name__ is the standard Python practice.
# ---
router = APIRouter(tags=["Prediction"])
logger = logging.getLogger(__name__)


# ---
# I have REMOVED the entire broken `@router.get("/predict")` route.
# It was calling a non-existent function (inference_service.predict) and
# was architecturally incompatible with your 3-layer logic, which requires
# the MarketContext body provided by the POST route below.
# ---


@router.post("/hybrid-signal", response_model=HybridDecision)
async def hybrid_signal(
    ctx: MarketContext,
    db: Session = Depends(get_db),
):
    """
    Build and persist a full hybrid decision (Layer 1 + Layer 2 + Layer 3).
    This is your main "brain" endpoint.
    """
    if inference_service is None or not inference_service.is_ready:
        logger.error("Inference service not ready or failed to load.")
        raise HTTPException(
            status_code=503,
            detail="InferenceService is not available. Check server logs.",
        )

    try:
        # This is the correct call to your 3-layer "brain"
        decision = await inference_service.build_decision(ctx)

        # Persist for audit / future training
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