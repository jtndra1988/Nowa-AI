import logging
import os
import json
from datetime import datetime
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW

from app.ml.adv.models_tcn import TemporalConvolutionalNetwork
from app.ml.train_tft import load_training_data
from app.ml.adv.feature_engineering import FEATURE_CONFIG, apply_price_feature_config
from app.ml.dataset import MultiModalTS
from app.ml.losses import multitask_transformer_loss

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 60
MODEL_VERSION = "v1.0"


def train():
    # 1. Load & preprocess data
    df = load_training_data(days=90)
    df_processed = df.groupby("symbol", group_keys=False).apply(apply_price_feature_config)

    # Targets: next-step return & vol
    df_processed["target_price"] = (
        df_processed.groupby("symbol")["close"].shift(-1) / df_processed["close"] - 1
    )
    df_processed["target_vol"] = df_processed.groupby("symbol")["roll_vol_6h"].shift(-1)
    df_processed = df_processed.dropna()

    ds = MultiModalTS(
        df=df_processed,
        feature_blocks=FEATURE_CONFIG,
        label_col="target_price",
        vol_label_col="target_vol",
        tabular_feature_cols=[],
        roll_windows=[],
        seq_len=SEQ_LEN,
    )

    loader = DataLoader(ds, batch_size=64, shuffle=True)

    # 2. Derive model input dims from dataset
    #    MultiModalTS.get_feature_dims() returns per-block dims; sum → total num_inputs
    feature_dims = ds.get_feature_dims()  # e.g. {"price_block": 4, "orderbook": 3, ...}
    num_inputs = int(sum(feature_dims.values()))

    # Define TCN architecture
    num_channels = [64, 128, 128]  # you can tune this

    model = TemporalConvolutionalNetwork(
        num_inputs=num_inputs,
        num_channels=num_channels,
        kernel_size=3,
        dropout=0.2,
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    # 3. Training loop
    num_epochs = 10
    model.train()
    last_epoch_loss = None

    for epoch in range(num_epochs):
        total_loss = 0.0

        for x_blocks, _, y_dict in loader:
            # Move to device
            x_blocks = {k: v.to(DEVICE) for k, v in x_blocks.items()}
            y_dict = {k: v.to(DEVICE) for k, v in y_dict.items()}

            # Concatenate feature blocks → [B, L, F_total]
            # Keep a deterministic order of blocks
            block_names = sorted(x_blocks.keys())
            x_seq = torch.cat([x_blocks[name] for name in block_names], dim=-1)

            optimizer.zero_grad()
            pred = model(x_seq)  # TemporalConvolutionalNetwork expects [B, L, F_total]

            # pred is {"price": ..., "vol": ...}
            loss = multitask_transformer_loss(pred, y_dict)["total_loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(loader)
        last_epoch_loss = epoch_loss
        logger.info(f"[TCN] Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

    # 4. Legacy full-model save (optional, for backward compatibility)
    os.makedirs("model_artifacts", exist_ok=True)
    legacy_path = "model_artifacts/tcn_model.pth"
    torch.save(model, legacy_path)
    logger.info(f"[TCN] Saved legacy full TCN model to: {legacy_path}")

    # 5. Versioned artifact (state_dict + metadata.json) for TemporalConvolutionalNetwork.load_from_artifact

    # crude "validation" proxy: last epoch loss
    val_loss = float(last_epoch_loss) if last_epoch_loss is not None else float("nan")

    # Feature list for metadata:
    # NOTE: This must match what FeatureEngineering.transform() produces.
    # For now we approximate by using block names; you may refine this to true column names.
    feature_list = sorted(FEATURE_CONFIG.keys())

    hyperparams = {
        "num_epochs": num_epochs,
        "lr": optimizer.param_groups[0]["lr"],
        "batch_size": loader.batch_size,
        "num_inputs": num_inputs,
        "num_channels": num_channels,
        "kernel_size": 3,
        "dropout": 0.2,
    }

    version = MODEL_VERSION
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = os.path.join("models", "tcn", f"{version}_{timestamp}")
    os.makedirs(artifact_dir, exist_ok=True)

    model_filename = f"tcn_{version}_{timestamp}.pt"
    model_path = os.path.join(artifact_dir, model_filename)
    torch.save(model.state_dict(), model_path)

    metadata = {
        "model_name": "tcn",
        "version": version,
        "train_date_utc": timestamp,
        "feature_list": feature_list,
        "model_architecture": "TemporalConvolutionalNetwork",
        "preprocessing": {
            "scaler": "scaler_filename_or_classname",
            "encoder": "encoder_filename_or_classname",
        },
        "hyperparameters": hyperparams,
        "validation_loss": val_loss,
    }

    metadata_path = os.path.join(artifact_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"[TCN] Saved TCN state_dict to: {model_path}")
    logger.info(f"[TCN] Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    train()
