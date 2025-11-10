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


# ---
# NOTE: I have removed the broken `@router.get("/predict")` route.
# It was calling a non-existent function (inference_service.predict) and
# was architecturally incompatible with your 3-layer logic, which requires
# the MarketContext body provided by this POST route.
# ---


@router.post("/hybrid-signal", response_model=HybridDecision)
async def hybrid_signal(
    ctx: MarketContext,
    db: Session = Depends(get_db),
):
    """
    Build and persist a full hybrid decision (Layer 1 + Layer 2 + Layer 3).
    This is the main "brain" endpoint.
    """
    if inference_service is None or not inference_service.is_ready:
        logger.error("Inference service not ready or failed to load.")
        raise HTTPException(
            status_code=503,
            detail="InferenceService is not available. Check server logs.",
        )

    try:
        # 1. This call is correct and returns the full HybridDecision
        decision = await inference_service.build_decision(ctx)

        # 2. Persist the *full* decision for audit / future training
        try:
            # --- FIX: Added ALL fields from the 'decision' object ---
            # This now correctly maps the Pydantic schema to the DB model.
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
                
                # --- NEWLY ADDED FIELDS FOR DB WRITE ---
                rl_action=decision.rl_action,
                rl_mode=decision.rl_mode,
                rl_target_position=decision.rl_target_position,
                llm_headline=decision.llm_headline
            )
            db.add(record)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(
                f"Failed to persist HybridSignal for {decision.symbol}: {e}",
                exc_info=True,
            )
        
        # 3. Return the full decision to the client
        return decision

    except Exception as e:
        logger.error(
            f"Failed to build hybrid signal for {ctx.symbol}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))