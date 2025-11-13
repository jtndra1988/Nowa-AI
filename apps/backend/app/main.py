from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.api.v1.health import router as health_router
from app.api.v1.endpoints import router as core_router
from app.api.v1.predict import router as v1_predict_router
from app.api.v1.brain import router as brain_router  # 👈 NEW import
from app.core.config import settings
from starlette_exporter import PrometheusMiddleware, handle_metrics  # type: ignore
from prometheus_client import make_asgi_app
# --- ADDED: Import our new ML service and DB seeder ---
from app.services.inference_service import inference_service
from app.db.database import init_db  # <-- NEW IMPORT
# --- Robust imports so it works with or without the "v1" package layout ---
try:
    from app.api.v1.risk import router as risk_router  # no prefix in module
except ImportError:
    from app.api.v1.risk import router as risk_router  # fallback
try:
    # settings router defines its own prefix="/api/settings" & tags=["settings"]
    from app.api.v1.settings import router as settings_router
except ImportError:
    from app.api.v1.settings import router as settings_router
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
        ],
    )
    # CORS
    allowed = getattr(settings, "CORS_ALLOW_ORIGINS", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    @app.get("/health", tags=["health"])
    async def health_check():
     return {"status": "ok"}
    # Prometheus
    app.add_middleware(
        PrometheusMiddleware, app_name="ml-trading-api", group_paths=True
    )
    app.add_route("/metrics", handle_metrics)
    # Routers
    app.include_router(health_router, tags=["health"])
    app.include_router(core_router, prefix="/api", tags=["core"])
    app.include_router(v1_predict_router, prefix="/api/v1", tags=["predict"])
    # NEW: brain health endpoint at /api/v1/brain-health
    app.include_router(brain_router, prefix="/api/v1", tags=["health"])
    # Risk router: module has no prefix, so mount it under /api/risk
    app.include_router(risk_router, prefix="/api/risk", tags=["risk"])
    # Settings router: module already declares prefix="/api/settings" & tags=["settings"]
    # Include as-is (no prefix)
    app.include_router(settings_router)
    return app


app = create_app()


@app.on_event("startup")
async def startup_event():
    print("[*] FastAPI startup event: Application is starting...")

    # --- Initialize and seed the database ---
    print("[*] Initializing database and seeding models...")
    init_db()

    # --- This loads all the ML models into memory (HybridInferenceService singleton) ---
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
