from fastapi import APIRouter, HTTPException

from app.services.inference_service import inference_service
from app.hybrid.schemas import MarketContext, HybridDecision

router = APIRouter()


@router.get("/predict/{symbol}")
def predict(symbol: str):
    if inference_service is None or not inference_service.is_ready:
        raise HTTPException(status_code=503, detail="Inference service not ready")
    try:
        return inference_service.predict(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hybrid-signal", response_model=HybridDecision)
def hybrid_signal(ctx: MarketContext):
    if inference_service is None or not inference_service.is_ready:
        raise HTTPException(status_code=503, detail="Inference service not ready")
    try:
        return inference_service.build_decision(ctx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
