import logging
import torch
import joblib
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from pathlib import Path
from sqlalchemy import func, desc
from datetime import timedelta

# Import internal modules
from app.db.database import SessionLocal
from app.db import models
from app.ml.adv.models_tft import TemporalFusionTransformer
from app.ml.dataset import MultiModalTS
from app.ml.losses import multitask_transformer_loss
# Import shared feature config
from app.ml.adv.feature_engineering import FEATURE_CONFIG, process_market_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Config ---
ARTIFACT_DIR = Path("model_artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparams
SEQ_LEN = 60
BATCH_SIZE = 64
EPOCHS = 20  # Kept low for quick testing
LR = 1e-4

def load_training_data(days=90):
    session = SessionLocal()
    try:
        logger.info("Fetching training data from DB...")
        # Fetch generic large chunk
        last_ts = session.query(func.max(models.MarketData.timestamp)).scalar()
        start_ts = last_ts - timedelta(days=days)
        
        rows = session.query(models.MarketData).filter(models.MarketData.timestamp >= start_ts).all()
        data = [{
            "symbol": r.symbol, "timestamp": r.timestamp, "open": float(r.open),
            "high": float(r.high), "low": float(r.low), "close": float(r.close),
            "volume": float(r.volume)
        } for r in rows]
        
        return pd.DataFrame(data).sort_values(["symbol", "timestamp"])
    finally:
        session.close()

def train():
    # 1. Data Prep
    df = load_training_data()
    
    # Apply Shared Feature Engineering
    df_processed = df.groupby("symbol", group_keys=False).apply(process_market_data)
    
    # Create Targets (Next 15m return and vol)
    df_processed["target_price"] = df_processed.groupby("symbol")["close"].shift(-1) / df_processed["close"] - 1
    df_processed["target_vol"] = df_processed.groupby("symbol")["roll_vol_6h"].shift(-1) # Proxy target
    df_processed = df_processed.dropna()

    # 2. Dataset
    ds = MultiModalTS(
        df=df_processed,
        feature_blocks=FEATURE_CONFIG, # Uses the shared ["close", "volume", "ret_1h"...] list
        label_col="target_price",
        vol_label_col="target_vol",
        tabular_feature_cols=[], # Empty for TFT pure sequence
        roll_windows=[],
        seq_len=SEQ_LEN
    )
    
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Model Init
    feature_dims = ds.get_feature_dims() # {"price": 7}
    logger.info(f"Initializing TFT with dims: {feature_dims}")
    
    model = TemporalFusionTransformer(
        feature_dims=feature_dims,
        seq_len=SEQ_LEN,
        d_model=128
    ).to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=LR)
    
    # 4. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x_blocks, _, y_dict in loader:
            x_blocks = {k: v.to(DEVICE) for k, v in x_blocks.items()}
            y_dict = {k: v.to(DEVICE) for k, v in y_dict.items()}
            
            optimizer.zero_grad()
            pred = model(x_blocks)
            
            # Simple loss for now
            loss_dict = multitask_transformer_loss(pred, y_dict)
            loss = loss_dict["total_loss"]
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        logger.info(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    # 5. Save
    save_path = ARTIFACT_DIR / "tft_model.pth"
    torch.save(model, save_path) # Saving entire model for simplicity in loading
    logger.info(f"Saved PyTorch TFT model to {save_path}")

if __name__ == "__main__":
    train()