from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Any, Dict
from app.api.v1.health import router as health_router
from app.api.v1.endpoints import router as core_router
from app.api.v1.predict import router as v1_predict_router
from app.api.v1.brain import router as brain_router
from app.api.v1.risk import router as risk_router
from app.api.v1.settings import router as settings_router
from app.api.v1.market_intel import router as market_intel_router  # ✅ NEW
from app.api.v1.system import router as system_router
from app.api.v1.system_stream import router as system_stream_router
from app.core.config import settings
from starlette_exporter import PrometheusMiddleware, handle_metrics
from app.services.inference_service import inference_service
from app.db.database import init_db
from app.api.v1 import market_intel
def create_app() -> FastAPI:
    app = FastAPI(
        title="Nowa API",
        version="2.0.0",
        contact={"name": "Jiten"},
        openapi_tags=[
            {"name": "health", "description": "Liveness & readiness probes"},
            {"name": "core", "description": "Core market/options endpoints"},
            {
                "name": "predict",
                "description": "v2 ML predictions (TFT + TCN + XGB Hybrid Ensemble)",
            },
            {"name": "metrics", "description": "Prometheus metrics"},
            {"name": "risk", "description": "Risk engine health/state/ledger"},
            {"name": "settings", "description": "Risk settings (global & per-symbol)"},
            {"name": "market-intel", "description": "Cross-layer market intelligence"},  # ✅ tag
        ],
    )

    # --- CORS SETTINGS ---
    origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",           # ✅ add this
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:3002",           # ✅ and this
    # Add production domains if needed
]

    app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or specific origin: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    @app.get("/health", tags=["health"])
    async def health_check():
        return {"status": "ok"}

    app.add_middleware(
        PrometheusMiddleware, app_name="ml-trading-api", group_paths=True
    )
    app.add_route("/metrics", handle_metrics)

    # Routers
    app.include_router(health_router,tags=["health"])
    app.include_router(core_router, prefix="/api", tags=["core"])
    app.include_router(v1_predict_router, prefix="/api/v1", tags=["predict"])
    app.include_router(brain_router, prefix="/api/v1", tags=["health"])
    app.include_router(risk_router, prefix="/api/risk", tags=["risk"])
    app.include_router(settings_router)
    app.include_router(market_intel.router, prefix="/api/v1", tags=["market-intel"])
    app.include_router(system_router, prefix="/api/v1", tags=["system"])
    app.include_router(system_stream_router, prefix="/api/v1", tags=["system"])

    return app


app = create_app()


@app.on_event("startup")
async def startup_event():
    print("[*] FastAPI startup event: Application is starting.")
    print("[*] Initializing database and seeding models.")
    init_db()

    if inference_service is None or not inference_service.ready():
        logging.critical(
            "CRITICAL: ML Inference Service FAILED readiness check on startup. Check logs."
        )
    else:
        logging.info(
            "ML Inference Service loaded successfully and is ready. Brain health: %s",
            inference_service.get_brain_health(),
        )

    print("[*] FastAPI startup complete.")
