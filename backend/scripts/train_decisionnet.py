"""
Train DecisionNet on the dataset built from HybridSignal logs.

Assumes:
  - You have already run:
        python -m backend.scripts.build_decisionnet_dataset
    which produced:
        decisionnet_dataset_60m.csv

  - decision_net_score() at inference time:
        - builds feature vector from a dict
        - uses sorted(feature_keys) as the order

This script:
  - Loads the CSV
  - Selects a consistent set of numeric features
  - Trains a small neural net to predict label (good/bad trade)
  - Saves weights to model_artifacts/decision_net.pth
"""

import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# So we can import app.*
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.ml.adv.decision_net import DecisionNet  # uses same architecture as inference


# --------------------
# Config
# --------------------

DATA_PATH = Path("decisionnet_dataset_60m.csv")  # adjust if needed
OUT_DIR = Path("./model_artifacts")
OUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 15
VAL_SPLIT = 0.2
MIN_SAMPLES = 500  # don't bother training if we have fewer


# --------------------
# Dataset
# --------------------

class DecisionNetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols):
        self.df = df.reset_index(drop=True)
        self.feature_cols = sorted(feature_cols)  # sorted to match inference logic

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = [float(row[c]) for c in self.feature_cols]
        y = int(row["label"])  # 0 or 1
        # We train as 3-class: [short, flat, long] style probabilities,
        # but our label is "correct vs incorrect" for current direction.
        # To align with DecisionNet's 3 logits:
        #   - we map: correct -> class 2 (long side), incorrect -> class 0 (short side)
        # This lets the net learn that "long" logit up = good, "short" logit up = bad.
        target = 2 if y == 1 else 0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(target, dtype=torch.long)


def select_feature_columns(df: pd.DataFrame):
    """
    Choose which columns to feed into DecisionNet.
    We exclude:
      - symbol, instrument_type, strategy_tag, direction, fwd_ret, label (targets/meta)
    We include:
      - expert/meta/specialist numeric features we created earlier.
    """
    exclude = {
        "symbol",
        "instrument_type",
        "strategy_tag",
        "direction",
        "fwd_ret",
        "label",
    }
    cols = [c for c in df.columns if c not in exclude]
    # keep only numeric
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols


# --------------------
# Training loop
# --------------------

def train():
    if not DATA_PATH.exists():
        print(f"[ERROR] Dataset not found at {DATA_PATH}. Run build_decisionnet_dataset first.")
        return

    df = pd.read_csv(DATA_PATH)

    if df.empty or len(df) < MIN_SAMPLES:
        print(f"[WARN] Not enough samples to train. Have {len(df)}, need at least {MIN_SAMPLES}.")
        return

    feature_cols = select_feature_columns(df)
    if not feature_cols:
        print("[ERROR] No usable feature columns found.")
        return

    print(f"Using {len(feature_cols)} features: {feature_cols}")

    dataset = DecisionNetDataset(df, feature_cols)

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = len(feature_cols)

    model = DecisionNet(in_dim=in_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()  # 3 logits: [short, flat, long]

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_train_loss = total_loss / len(train_ds)

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)

                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        avg_val_loss = val_loss / len(val_ds)
        val_acc = correct / total if total > 0 else 0.0

        print(
            f"Epoch {epoch:02d} "
            f"| train_loss={avg_train_loss:.4f} "
            f"| val_loss={avg_val_loss:.4f} "
            f"| val_acc={val_acc:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()

    if best_state is None:
        print("[ERROR] Training did not produce a valid model state.")
        return

    # Save best model
    out_path = OUT_DIR / "decision_net.pth"
    torch.save(best_state, out_path)
    print(f"[OK] Saved DecisionNet weights to {out_path}")


if __name__ == "__main__":
    train()
