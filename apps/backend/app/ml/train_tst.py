import logging
import os
import json
from datetime import datetime
from pyexpat import model
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from app.ml.adv.models_tst import TimeSeriesTransformer
from app.ml.train_tft import load_training_data
from app.ml.adv.feature_engineering import FEATURE_CONFIG, process_market_data
from app.ml.dataset import MultiModalTS
from app.ml.losses import multitask_transformer_loss

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
    df = load_training_data(days=90)
    df_processed = df.groupby("symbol", group_keys=False).apply(process_market_data)
    
    df_processed["target_price"] = df_processed.groupby("symbol")["close"].shift(-1) / df_processed["close"] - 1
    df_processed["target_vol"] = df_processed.groupby("symbol")["roll_vol_6h"].shift(-1)
    df_processed = df_processed.dropna()

    ds = MultiModalTS(
        df=df_processed,
        feature_blocks=FEATURE_CONFIG,
        label_col="target_price",
        vol_label_col="target_vol",
        tabular_feature_cols=[],
        roll_windows=[],
        seq_len=60
    )
    
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    in_feat = sum(ds.get_feature_dims().values())
    
    context_length = 60          # same as seq_len in your dataset
    prediction_length = 1        # or >1 if you want multi-step
    target_size = 2              # price + vol (multi-task)

    model = TimeSeriesTransformer(
    context_length=context_length,
    num_features=in_feat,
    prediction_length=prediction_length,
    target_size=target_size,
    d_model=128,
    nhead=4,
    num_layers=3,
    dim_feedforward=512,
    dropout=0.1,
    ).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    num_epochs = 10 
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in loader:
            x_input = batch["x"].to(DEVICE)
            y_dict = {
                "price": batch["y_price"].to(DEVICE),
                "vol": batch["y_vol"].to(DEVICE),
            }

            optimizer.zero_grad()
            seq_out = model(x_input)
            last_step = seq_out[:, -1, :]  # [B, target_size]
            pred_dict = {
            "price": last_step[:, 0],  # assume index 0 = price
            "vol":   last_step[:, 1],  # assume index 1 = vol
            }

            loss_dict = multitask_transformer_loss(pred_dict, y_dict)
            loss = loss_dict["total_loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(loader)
        logger.info(f"[TST] Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

    # ---- New: versioned save + metadata ----
    from datetime import datetime
    import os
    import json
    import torch

    # crude "validation" loss for metadata – using last epoch loss
    val_loss = epoch_loss

    # If you have a real feature list from your dataset, use that here.
    # For now we approximate using FEATURE_CONFIG keys.
    try:
        feature_list = list(FEATURE_CONFIG.keys())
    except Exception:
        feature_list = []

    # Hyperparameters – keep in sync with what you actually used above
    hyperparams = {
    "num_epochs": num_epochs,
    "lr": optimizer.param_groups[0]["lr"],
    "batch_size": loader.batch_size,
    "context_length": context_length,
    "num_features": in_feat,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 3,
    "dim_feedforward": 512,
    "dropout": 0.1,
    }

    version = "v1.0"
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = os.path.join("models", "tst", f"{version}_{timestamp}")
    os.makedirs(artifact_dir, exist_ok=True)

    model_filename = f"tst_{version}_{timestamp}.pt"
    model_path = os.path.join(artifact_dir, model_filename)
    torch.save(model.state_dict(), model_path)

    metadata = {
    "model_name": "tst",
    "version": version,
    "train_date_utc": timestamp,
    "feature_list": feature_list,
    "prediction_length": prediction_length,
    "target_size": target_size,
    "model_architecture": "TimeSeriesTransformer",
    "preprocessing": {
        "scaler": "scaler_filename_or_classname",
        "encoder": "encoder_filename_or_classname",
    },
    "hyperparameters": hyperparams,
    "validation_loss": float(val_loss),
    }

    metadata_path = os.path.join(artifact_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"[TST] Saved TST model state_dict to: {model_path}")
    logger.info(f"[TST] Saved metadata to: {metadata_path}")
    # ----------------------------------------


if __name__ == "__main__":
    train()