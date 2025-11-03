from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
import logging

# Import the models that we need to seed
from .models import Model, ModelVersion

logger = logging.getLogger(__name__)

engine = create_engine(settings.SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Initializes the database.
    If the Model table is empty, it seeds it with the new models.
    """
    db = SessionLocal()
    try:
        # Check if the database is already seeded
        first_model = db.query(Model).first()
        if first_model:
            logger.info("Database already seeded. Skipping initial data load.")
            return

        logger.info("Database is empty. Seeding initial models...")

        # --- 1. Add Model Names ---
        model_tft = Model(
            name='TFT',
            description='Temporal Fusion Transformer (TFT) base model for price and vol.'
        )
        model_tcn = Model(
            name='TCN',
            description='Temporal Convolutional Network (TCN) base model for price and vol.'
        )
        model_xgb = Model(
            name='XGBoost',
            description='XGBoost tabular models (price and vol) on engineered features.'
        )
        model_ensemble = Model(
            name='HybridEnsemble',
            description='TFT+TCN+XGB blended ensemble for final predictions.'
        )
        
        db.add_all([model_tft, model_tcn, model_xgb, model_ensemble])
        
        # We must commit here so the foreign keys for ModelVersion work
        db.commit()

        # --- 2. Add Model Versions ---
        version_tft = ModelVersion(
            model_name='TFT',
            version=1,
            artifact_path='model_artifacts/tft_best_model.pth',
            hyperparameters='{"d_model": 128, "nhead": 4, "num_layers": 3, "seq_len": 60}',
            is_active=True
        )
        version_tcn = ModelVersion(
            model_name='TCN',
            version=1,
            artifact_path='model_artifacts/tcn_best_model.pth',
            hyperparameters='{"channels": [64, 128, 128], "kernel": 3}',
            is_active=True
        )
        version_xgb = ModelVersion(
            model_name='XGBoost',
            version=1,
            artifact_path='model_artifacts/xgb_price_model.joblib',
            hyperparameters='{"note": "Also requires xgb_vol_model.joblib"}',
            is_active=True
        )
        version_ensemble = ModelVersion(
            model_name='HybridEnsemble',
            version=1,
            artifact_path='model_artifacts/hybrid_ensemble_best.pth',
            hyperparameters='{"models": ["TFT", "TCN", "XGBoost"], "note": "This artifact contains the blender weights only."}',
            is_active=True
        )

        db.add_all([version_tft, version_tcn, version_xgb, version_ensemble])
        
        # Final commit
        db.commit()
        logger.info("Successfully seeded database with new models.")

    except Exception as e:
        logger.error(f"Error while seeding database: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()