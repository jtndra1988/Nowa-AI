import json
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sqlalchemy import func

# Internal imports
from app.db.database import SessionLocal
from app.db import models
from app.ml.adv.models_tft import TemporalFusionTransformer
from app.ml.dataset import MultiModalTS
from app.ml.losses import multitask_transformer_loss
from app.ml.adv.feature_engineering import FEATURE_CONFIG, process_market_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Config ---
ARTIFACT_DIR = Path("model_artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 60
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4
MODEL_VERSION = "v1.0"


def load_training_data(days: int = 90) -> pd.DataFrame:
    """
    Load OHLCV training data from the DB for the last `days` days.
    """
    session = SessionLocal()
    try:
        logger.info("Fetching training data from DB...")
        last_ts = session.query(func.max(models.MarketData.timestamp)).scalar()
        if last_ts is None:
            raise RuntimeError("No MarketData found in DB.")

        start_ts = last_ts - timedelta(days=days)
        rows = (
            session.query(models.MarketData)
            .filter(models.MarketData.timestamp >= start_ts)
            .all()
        )

        data = [
            {
                "symbol": r.symbol,
                "timestamp": r.timestamp,
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            }
            for r in rows
        ]

        df = pd.DataFrame(data).sort_values(["symbol", "timestamp"])
        logger.info(f"Loaded {len(df)} rows for training TFT.")
        return df
    finally:
        session.close()


def train():
    # 1. Data prep
    df = load_training_data(days=90)

    # Apply shared feature engineering
    df_processed = df.groupby("symbol", group_keys=False).apply(process_market_data)

    # Targets: next bar return & vol proxy
    df_processed["target_price"] = (
        df_processed.groupby("symbol")["close"].shift(-1) / df_processed["close"] - 1
    )
    df_processed["target_vol"] = df_processed.groupby("symbol")["roll_vol_6h"].shift(-1)
    df_processed = df_processed.dropna()

    # 2. Dataset & loader
    ds = MultiModalTS(
        df=df_processed,
        feature_blocks=FEATURE_CONFIG,
        label_col="target_price",
        vol_label_col="target_vol",
        tabular_feature_cols=[],
        roll_windows=[],
        seq_len=SEQ_LEN,
    )

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Model init
    feature_dims = ds.get_feature_dims()  # e.g. {"price_block": 7, ...}
    logger.info(f"[TFT] Initializing TFT with feature dims: {feature_dims}")

    model = TemporalFusionTransformer(
        feature_dims=feature_dims,
        seq_len=SEQ_LEN,
        d_model=128,
        nhead=4,
        num_layers=3,
        dropout=0.1,
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)

    # 4. Training loop
    model.train()
    last_epoch_loss = None

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for x_blocks, _, y_dict in loader:
            x_blocks = {k: v.to(DEVICE) for k, v in x_blocks.items()}
            y_dict = {k: v.to(DEVICE) for k, v in y_dict.items()}

            optimizer.zero_grad()
            pred = model(x_blocks)
            loss_dict = multitask_transformer_loss(pred, y_dict)
            loss = loss_dict["total_loss"]

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(loader)
        last_epoch_loss = epoch_loss
        logger.info(f"[TFT] Epoch {epoch + 1}/{EPOCHS} | Loss: {epoch_loss:.4f}")

    # 5. Legacy full-model save (for TFTPredictor)
    ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)
    legacy_model_path = ARTIFACT_DIR / "tft_model.pth"
    torch.save(model, legacy_model_path)
    logger.info(f"[TFT] Saved full TFT model to: {legacy_model_path}")

    # 6. Versioned artifact (state_dict + metadata.json)
    val_loss = float(last_epoch_loss) if last_epoch_loss is not None else float("nan")

    # For TFT, using block names as the "feature list" for interpretability.
    feature_list = sorted(FEATURE_CONFIG.keys())

    hyperparams = {
        "num_epochs": EPOCHS,
        "lr": optimizer.param_groups[0]["lr"],
        "batch_size": loader.batch_size,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "dropout": 0.1,
    }

    version = MODEL_VERSION
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = os.path.join("models", "tft", f"{version}_{timestamp}")
    os.makedirs(artifact_dir, exist_ok=True)

    # Save state_dict
    model_filename = f"tft_{version}_{timestamp}.pt"
    model_path = os.path.join(artifact_dir, model_filename)
    torch.save(model.state_dict(), model_path)

    # Save metadata compatible with TemporalFusionTransformer.load_from_artifact()
    metadata = {
        "model_name": "tft",
        "version": version,
        "train_date_utc": timestamp,
        "feature_list": feature_list,         # high-level block list
        "feature_dims": feature_dims,         # exact dims used in model init
        "seq_len": SEQ_LEN,
        "preprocessing": {
            "scaler": "scaler_filename_or_classname",
            "encoder": "encoder_filename_or_classname",
        },
        "model_architecture": "TemporalFusionTransformer",
        "hyperparameters": hyperparams,
        "validation_loss": val_loss,
    }

    metadata_path = os.path.join(artifact_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"[TFT] Saved TFT state_dict to: {model_path}")
    logger.info(f"[TFT] Saved TFT metadata to: {metadata_path}")


if __name__ == "__main__":
    train()
