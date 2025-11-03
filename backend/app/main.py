from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.health import router as health_router
from app.api.v1.endpoints import router as core_router
from app.api.v1.predict import router as v1_predict_router
from app.core.config import settings
from starlette_exporter import PrometheusMiddleware, handle_metrics  # type: ignore

def create_app() -> FastAPI:
    app = FastAPI(
        title="ML Trading API",
        version="1.0.0",
        contact={"name": "Your Team"},
        openapi_tags=[
            {"name": "health", "description": "Liveness & readiness probes"},
            {"name": "core", "description": "Core market/options endpoints"},
            {"name": "predict", "description": "v1 ML predictions (ensemble + calibration)"},
            {"name": "metrics", "description": "Prometheus metrics"},
        ],
    )

    # CORS
    allowed = getattr(settings, "CORS_ALLOW_ORIGINS", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus
    app.add_middleware(PrometheusMiddleware, app_name="ml-trading-api", group_paths=True)
    app.add_route("/metrics", handle_metrics)

    # Routers
    app.include_router(health_router, tags=["health"])
    app.include_router(core_router, prefix="/api", tags=["core"])
    app.include_router(v1_predict_router, prefix="/api/v1", tags=["predict"])

    @app.get("/", tags=["health"])
    def root():
        return {"status": "ok", "service": "ml-trading-api"}

    return app

app = create_app()
