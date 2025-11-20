# app/ml/train_decision_net.py

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from app.ml.adv.decision_net import (
    DecisionNet,
    DECISION_NET_INPUT_KEYS,
    DECISION_NET_PATH,
    DEVICE,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Where to read training data from
TRAIN_CSV_PATH = Path("data/decision_net_training.csv")

# Versioned artifact location
MODEL_ROOT_DIR = Path("models") / "decision_net"
MODEL_VERSION = "v1.0"

# Training hyperparameters
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
SEED = 42


def load_dataset(csv_path: Path) -> Tuple[TensorDataset, TensorDataset]:
    """
    Load decision net training data from CSV.

    Expected columns:
        DECISION_NET_INPUT_KEYS (float)
        'label' in {-1, 0, 1}  (short, flat, long)

    We map label:
        -1 -> 0 (short)
         0 -> 1 (flat)
         1 -> 2 (long)
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"[DecisionNet] Training CSV not found at {csv_path}. "
            f"Create a dataset with columns {DECISION_NET_INPUT_KEYS + ['label']}."
        )

    df = pd.read_csv(csv_path)

    missing_cols = [c for c in DECISION_NET_INPUT_KEYS + ["label"] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[DecisionNet] Training CSV missing columns: {missing_cols}. "
            f"Expected: {DECISION_NET_INPUT_KEYS + ['label']}"
        )

    # Features
    X = df[DECISION_NET_INPUT_KEYS].values.astype(np.float32)

    # Labels
    raw_labels = df["label"].values
    label_map = {-1: 0, 0: 1, 1: 2}
    y = np.array([label_map.get(int(v), 1) for v in raw_labels], dtype=np.int64)  # default to flat

    # Train/val split
    np.random.seed(SEED)
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    split = int(len(X) * (1.0 - VAL_SPLIT))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    return train_ds, val_ds


def train_decision_net() -> None:
    logger.info("[DecisionNet] Starting training...")

    train_ds, val_ds = load_dataset(TRAIN_CSV_PATH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    in_dim = len(DECISION_NET_INPUT_KEYS)
    model = DecisionNet(in_dim=in_dim, hidden=64).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_loss /= max(len(val_loader), 1)
        val_acc = correct / max(total, 1)

        logger.info(
            "[DecisionNet] Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
            epoch,
            EPOCHS,
            train_loss,
            val_loss,
            val_acc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()

    if best_state_dict is None:
        logger.error("[DecisionNet] Training failed to produce a valid model.")
        return

    # ----------------------------------------------------
    # Save runtime artifact (used by decision_net_score)
    # ----------------------------------------------------
    DECISION_NET_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state_dict, DECISION_NET_PATH)
    logger.info("[DecisionNet] Saved runtime model to %s", DECISION_NET_PATH)

    # ----------------------------------------------------
    # Save versioned artifact + metadata
    # ----------------------------------------------------
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = MODEL_ROOT_DIR / f"{MODEL_VERSION}_{timestamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    weights_path = artifact_dir / f"decision_net_{MODEL_VERSION}_{timestamp}.pt"
    torch.save(best_state_dict, weights_path)

    metadata = {
        "model_name": "decision_net",
        "version": MODEL_VERSION,
        "train_date_utc": timestamp,
        "input_keys": DECISION_NET_INPUT_KEYS,
        "in_dim": len(DECISION_NET_INPUT_KEYS),
        "num_classes": 3,
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "val_split": VAL_SPLIT,
            "hidden": 64,
        },
        "validation": {
            "best_val_loss": float(best_val_loss),
        },
    }

    metadata_path = artifact_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info("[DecisionNet] Saved versioned weights to %s", weights_path)
    logger.info("[DecisionNet] Saved metadata to %s", metadata_path)


if __name__ == "__main__":
    train_decision_net()
