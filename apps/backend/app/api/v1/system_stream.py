from __future__ import annotations

import json
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.services.inference_service import inference_service
from app.api.v1.health import get_system_resources

router = APIRouter()


def event_stream():
    while True:
        brain = inference_service.get_brain_health()

        payload = {
            "ai_ready": brain["is_ready"],
            "llm": brain["llm_ready"],
            "rl": brain["rl_ready"],
            "l2": brain["has_l2_models"],
            "resources": get_system_resources(),
            "ts": time.time(),
        }

        yield f"data: {json.dumps(payload)}\n\n"
        time.sleep(1)


@router.get("/system-stream")
def system_stream():
    return StreamingResponse(event_stream(), media_type="text/event-stream")
