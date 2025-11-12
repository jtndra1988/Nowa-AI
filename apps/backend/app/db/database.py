# app/db/database.py

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from .base import Base
from .models import ModelVersion

logger = logging.getLogger(__name__)

# Engine & Session
engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency to get a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database schema and seed model_versions if empty.

    We treat ModelVersion as the single source of truth for model metadata.
    """
    db = SessionLocal()
    try:
        # Ensure all tables exist
        Base.metadata.create_all(bind=engine)

        # If we already have at least one version, don't reseed
        existing = db.query(ModelVersion).first()
        if existing:
            logger.info("ModelVersion already seeded. Skipping initial seeding.")
            return

        logger.info("Seeding initial ModelVersion entries...")

        seed_versions = [
            ModelVersion(
                model_name="TFT",
                version="1",
                metadata={
                    "description": "Temporal Fusion Transformer base model for price/vol forecasts.",
                    "artifact_path": "model_artifacts/tft_best_model.pth",
                    "type": "torch",
                },
            ),
            ModelVersion(
                model_name="TCN",
                version="1",
                metadata={
                    "description": "Temporal Convolutional Network base model.",
                    "artifact_path": "model_artifacts/tcn_best_model.pth",
                    "type": "torch",
                },
            ),
            ModelVersion(
                model_name="XGBoost",
                version="1",
                metadata={
                    "description": "XGBoost tabular models for price/vol.",
                    "artifacts": {
                        "price": "model_artifacts/xgb_price_model.joblib",
                        "vol": "model_artifacts/xgb_vol_model.joblib",
                    },
                    "type": "xgboost",
                },
            ),
            ModelVersion(
                model_name="HybridEnsemble",
                version="1",
                metadata={
                    "description": "Blender / meta-ensemble over TFT+TCN+XGB.",
                    "artifact_path": "model_artifacts/hybrid_ensemble_best.pth",
                    "type": "torch",
                },
            ),
        ]

        db.add_all(seed_versions)
        db.commit()
        logger.info("Database seeding complete.")

    except Exception:
        logger.exception("Error while initializing / seeding the database.")
        db.rollback()
    finally:
        db.close()
