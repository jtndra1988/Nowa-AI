import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- Import our new TFT Model ---
from .models_tft import TemporalFusionTransformer
from .dataset import MultiModalTS
from .losses import multitask_transformer_loss

# --- Configuration ---

# TODO: Define your data and feature configuration here
DATA_PATH = "path/to/your/features.parquet" # Or .csv
ARTIFACT_DIR = Path("./model_artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# TODO: Define your feature columns
# This MUST match the features in your data and the keys
# in your dataset's get_feature_dims()
FEATURE_BLOCKS = {
    "price": ['open', 'high', 'low', 'close', 'volume'],
    "ob": ['ob_imbalance_1s', 'ob_spread', 'ob_depth_ask_1', 'ob_depth_bid_1'],
    "sent": ['sent_score_1m', 'sent_score_15m'],
    # "onch": ['onch_metric_1', 'onch_metric_2'] # Example
}
# Save feature block names for interpretability logging
FEATURE_NAMES = list(FEATURE_BLOCKS.keys())

# TODO: Define your multi-task label columns
PRICE_LABEL_COL = "next_return_15m"
VOL_LABEL_COL = "next_vol_15m" #

# --- Model & Training Hyperparameters ---
SEQ_LEN = 60
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model parameters
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3
DROPOUT = 0.1

# [cite_start]Loss parameters [cite: 174-175, 180-181]
PRICE_LOSS_PARAMS = {'alpha': 0.5, 'beta': 0.2, 'lam': 0.1}
VOL_WEIGHT = 0.2 # Weight for the volatility task in the total loss

class EarlyStopping:
    """Utility to stop training when validation loss stops improving."""
    def __init__(self, patience=5, min_delta=0, checkpoint_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            print(f"New best model found with val_loss {self.best_loss:.4f}. Saving...")
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping.")
                self.early_stop = True

def train_one_epoch(model, loader, optimizer, device, price_loss_params, vol_weight):
    model.train()
    total_loss_epoch = 0
    price_loss_epoch = 0
    vol_loss_epoch = 0

    for x_blocks, y_dict in loader:
        # --- KEY CHANGE for TFT ---
        # x_blocks is now a dict, move each tensor to device
        x_blocks = {k: v.to(device) for k, v in x_blocks.items()}
        y_dict = {k: v.to(device) for k, v in y_dict.items()}
        # ------------------------

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        pred_dict = model(x_blocks) # Pass the dictionary

        # Calculate loss
        # The loss function will automatically pick out 'price' and 'vol'
        loss_components = multitask_transformer_loss(
            pred_dict,
            y_dict,
            price_loss_params,
            vol_weight
        )
        
        loss = loss_components['total_loss']
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss_epoch += loss.item()
        price_loss_epoch += loss_components['price_loss'].item()
        vol_loss_epoch += loss_components['vol_loss'].item()

    avg_total_loss = total_loss_epoch / len(loader)
    avg_price_loss = price_loss_epoch / len(loader)
    avg_vol_loss = vol_loss_epoch / len(loader)
    return avg_total_loss, avg_price_loss, avg_vol_loss

def validate(model, loader, device, price_loss_params, vol_weight):
    model.eval()
    total_loss_epoch = 0
    price_loss_epoch = 0
    vol_loss_epoch = 0
    
    # Store feature weights for interpretability
    all_feature_weights = []

    with torch.no_grad():
        for x_blocks, y_dict in loader:
            # --- KEY CHANGE for TFT ---
            x_blocks = {k: v.to(device) for k, v in x_blocks.items()}
            y_dict = {k: v.to(device) for k, v in y_dict.items()}
            # ------------------------

            # Forward pass
            pred_dict = model(x_blocks) # Pass the dictionary

            # Calculate loss
            loss_components = multitask_transformer_loss(
                pred_dict,
                y_dict,
                price_loss_params,
                vol_weight
            )
            
            total_loss_epoch += loss_components['total_loss'].item()
            price_loss_epoch += loss_components['price_loss'].item()
            vol_loss_epoch += loss_components['vol_loss'].item()
            
            # Store interpretability weights
            all_feature_weights.append(pred_dict['feature_weights'].cpu().numpy())

    avg_total_loss = total_loss_epoch / len(loader)
    avg_price_loss = price_loss_epoch / len(loader)
    avg_vol_loss = vol_loss_epoch / len(loader)
    
    # Aggregate and log feature weights
    avg_feature_weights = np.mean(np.concatenate(all_feature_weights, axis=0), axis=(0, 1))
    
    return avg_total_loss, avg_price_loss, avg_vol_loss, avg_feature_weights

def run_training():
    print(f"Using device: {DEVICE}")

    # --- 1. Load Data ---
    # TODO: Load your data into a pandas DataFrame
    # This is just a placeholder, replace with your data loading
    if not os.path.exists(DATA_PATH):
        print(f"Warning: Data file not found at {DATA_PATH}. Using placeholder data.")
        # Create a mock dataframe
        T = 5000 # 5000 time steps
        all_features = [col for cols in FEATURE_BLOCKS.values() for col in cols]
        data = pd.DataFrame(
            np.random.randn(T, len(all_features)),
            columns=all_features
        )
        data[PRICE_LABEL_COL] = np.random.randn(T)
        data[VOL_LABEL_COL] = np.random.rand(T)
    else:
        print(f"Loading data from {DATA_PATH}...")
        # Assuming parquet, use read_csv if needed
        data = pd.read_parquet(DATA_PATH) 

    # --- 2. Split and Create Datasets ---
    # Simple time-series split
    val_split = int(len(data) * 0.8)
    train_df = data.iloc[:val_split]
    val_df = data.iloc[val_split:]

    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

    train_ds = MultiModalTS(
        df=train_df,
        feature_blocks=FEATURE_BLOCKS,
        label_col=PRICE_LABEL_COL,
        vol_label_col=VOL_LABEL_COL,
        seq_len=SEQ_LEN
    )

    val_ds = MultiModalTS(
        df=val_df,
        feature_blocks=FEATURE_BLOCKS,
        label_col=PRICE_LABEL_COL,
        vol_label_col=VOL_LABEL_COL,
        seq_len=SEQ_LEN
    )

    # [cite_start]--- 3. Apply Normalization --- [cite: 172-173]
    print("Applying feature scaler...")
    mean, std = train_ds.fit_scaler()
    train_ds.apply_scaler(mean, std)
    val_ds.apply_scaler(mean, std)
    
    # Save the scaler
    scaler_path = ARTIFACT_DIR / "tft_scaler.npz"
    np.savez(scaler_path, mean=mean, std=std)
    print(f"Scaler saved to {scaler_path}")


    # --- 4. Create DataLoaders ---
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- 5. Initialize Model ---
    # --- KEY CHANGE for TFT ---
    # Get feature dimension dictionary from the dataset
    feature_dims = train_ds.get_feature_dims()
    print(f"Initializing TFT model with feature dims: {feature_dims}")
    
    model = TemporalFusionTransformer(
        feature_dims=feature_dims,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    
    early_stopper = EarlyStopping(
        patience=10, 
        checkpoint_path=ARTIFACT_DIR / "tft_best_model.pth"
    )

    # --- 6. Training Loop ---
    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_price_loss, train_vol_loss = train_one_epoch(
            model, train_loader, optimizer, DEVICE, PRICE_LOSS_PARAMS, VOL_WEIGHT
        )
        
        val_loss, val_price_loss, val_vol_loss, val_feat_weights = validate(
            model, val_loader, DEVICE, PRICE_LOSS_PARAMS, VOL_WEIGHT
        )

        print(f"--- Epoch {epoch}/{EPOCHS} ---")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train Price Loss: {train_price_loss:.4f} | Val Price Loss: {val_price_loss:.4f}")
        print(f"  Train Vol Loss:   {train_vol_loss:.4f} | Val Vol Loss:   {val_vol_loss:.4f}")
        
        # --- Log Interpretability Weights ---
        print("  Validation Feature Weights (Avg):")
        for name, weight in zip(FEATURE_NAMES, val_feat_weights):
            print(f"    - {name}: {weight:.4f}")
        print("-" * (17 + len(str(epoch)) + len(str(EPOCHS))))


        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            break

    print("Training complete.")
    print(f"Best model saved to {early_stopper.checkpoint_path}")

if __name__ == "__main__":
    run_training()