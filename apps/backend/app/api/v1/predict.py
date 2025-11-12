from fastapi import APIRouter, Depends
from app.services.inference_service import inference_service
from app.hybrid.schemas import HybridDecision, MarketContext

router = APIRouter()

@router.post("/hybrid-signal", response_model=HybridDecision)
async def hybrid_signal(ctx: MarketContext):
    return await inference_service.build_decision(ctx)

@router.get("/predict/{symbol}", response_model=HybridDecision)
async def predict(symbol: str):
    ctx = MarketContext(symbol=symbol)
    return await inference_service.build_decision(ctx)
