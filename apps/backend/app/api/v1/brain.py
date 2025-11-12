# app/api/v1/brain.py

from fastapi import APIRouter
from app.services.inference_service import inference_service

router = APIRouter()


@router.get("/brain-health", tags=["health"])
def brain_health():
    """
    Lightweight health/status endpoint for the AI brain.

    Returns:
    - is_ready: overall readiness (L2 + LLM + Risk)
    - has_l2_models: whether TFT/TCN/XGB ensemble (or artifacts) is available
    - llm_ready: whether LLM narrative engine is configured & reachable
    - rl_ready: whether RL policy is loaded (optional, non-blocking)
    - risk_ready: whether core RiskEngine logic is available
    """
    return inference_service.get_brain_health()
