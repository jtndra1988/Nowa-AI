import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- Import our Ensemble Model ---
from .ensemble import MultiTaskEnsemble
from .dataset import MultiModalTS
from .losses import multitask_transformer_loss

# --- Configuration ---
# All configurations must match the base models
# ------------------------------------------------

DATA_PATH = "path/to/your/features.parquet" # Or .csv
ARTIFACT_DIR = Path("./model_artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# Feature blocks
FEATURE_BLOCKS = {
    "price": ['open', 'high', 'low', 'close', 'volume'],
    "ob": ['ob_imbalance_1s', 'ob_spread', 'ob_depth_ask_1', 'ob_depth_bid_1'],
    "sent": ['sent_score_1m', 'sent_score_15m'],
    # "onch": ['onch_metric_1', 'onch_metric_2'] # Example
}
FEATURE_NAMES = list(FEATURE_BLOCKS.keys())

# Label columns
PRICE_LABEL_COL = "next_return_15m"
VOL_LABEL_COL = "next_vol_15m" #

# --- Model & Training Hyperparameters ---
SEQ_LEN = 60
BATCH_SIZE = 64
EPOCHS = 20 # Blending trains very quickly
LR = 1e-3 # Can use a higher LR for just the blender
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. TFT Model Configuration (must match train_adv.py) ---
TFT_CONFIG = {
    # feature_dims will be added later
    "seq_len": SEQ_LEN,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 3,
    "dropout": 0.1
}
TFT_WEIGHTS_PATH = ARTIFACT_DIR / "tft_best_model.pth"
TFT_SCALER_PATH = ARTIFACT_DIR / "tft_scaler.npz"

# --- 2. TCN Model Configuration (must match train_tcn.py) ---
TCN_CONFIG = {
    # in_feat will be added later
    "channels": (64, 128, 128),
    "kernel": 3,
    "dropout": 0.1
}
TCN_WEIGHTS_PATH = ARTIFACT_DIR / "tcn_best_model.pth"

# --- 3. Ensemble Config ---
ENSEMBLE_CHECKPOINT_PATH = ARTIFACT_DIR / "ensemble_best_model.pth"

# Loss parameters
PRICE_LOSS_PARAMS = {'alpha': 0.5, 'beta': 0.2, 'lam': 0.1}
VOL_WEIGHT = 0.2

# --- (Training/Validation loops are identical to other scripts) ---

def train_one_epoch(model, loader, optimizer, device, price_loss_params, vol_weight):
    model.train()
    total_loss_epoch = 0
    price_loss_epoch = 0

    for x_blocks, y_dict in loader:
        x_blocks = {k: v.to(device) for k, v in x_blocks.items()}
        y_dict = {k: v.to(device) for k, v in y_dict.items()}

        optimizer.zero_grad()
        
        # --- KEY CHANGE ---
        # The model.forward() call is now running both sub-models
        # and blending the results. But the autograd will only
        # track the (very few) trainable parameters in the blenders.
        pred_dict = model(x_blocks) 

        loss_components = multitask_transformer_loss(
            pred_dict,
            y_dict,
            price_loss_params,
            vol_weight
        )
        
        loss = loss_components['total_loss']
        loss.backward()
        optimizer.step()

        total_loss_epoch += loss.item()
        price_loss_epoch += loss_components['price_loss'].item()

    avg_total_loss = total_loss_epoch / len(loader)
    avg_price_loss = price_loss_epoch / len(loader)
    return avg_total_loss, avg_price_loss

def validate(model, loader, device, price_loss_params, vol_weight):
    model.eval()
    total_loss_epoch = 0
    price_loss_epoch = 0
    
    with torch.no_grad():
        for x_blocks, y_dict in loader:
            x_blocks = {k: v.to(device) for k, v in x_blocks.items()}
            y_dict = {k: v.to(device) for k, v in y_dict.items()}

            pred_dict = model(x_blocks) 

            loss_components = multitask_transformer_loss(
                pred_dict,
                y_dict,
                price_loss_params,
                vol_weight
            )
            
            total_loss_epoch += loss_components['total_loss'].item()
            price_loss_epoch += loss_components['price_loss'].item()

    avg_total_loss = total_loss_epoch / len(loader)
    avg_price_loss = price_loss_epoch / len(loader)
    return avg_total_loss, avg_price_loss

def run_ensemble_training():
    print(f"Using device: {DEVICE}")

    # --- 1. Load Data ---
    if not os.path.exists(DATA_PATH):
        print(f"Warning: Data file not found at {DATA_PATH}. Using placeholder data.")
        T = 5000 
        all_features = [col for cols in FEATURE_BLOCKS.values() for col in cols]
        data = pd.DataFrame(
            np.random.randn(T, len(all_features)),
            columns=all_features
        )
        data[PRICE_LABEL_COL] = np.random.randn(T)
        data[VOL_LABEL_COL] = np.random.rand(T)
    else:
        print(f"Loading data from {DATA_PATH}...")
        data = pd.read_parquet(DATA_PATH) 

    # --- 2. Split and Create Datasets ---
    # We train the ensemble on the validation data
    val_split_idx = int(len(data) * 0.8)
    # We will use the *validation* set to train the blender
    val_df = data.iloc[val_split_idx:]
    
    # We split this val_df into a "train" and "val" for the ensemble
    ensemble_train_split_idx = int(len(val_df) * 0.7)
    train_ensemble_df = val_df.iloc[:ensemble_train_split_idx]
    val_ensemble_df = val_df.iloc[ensemble_train_split_idx:]

    print(f"Ensemble training samples: {len(train_ensemble_df)}, Ensemble validation samples: {len(val_ensemble_df)}")

    train_ds = MultiModalTS(
        df=train_ensemble_df,
        feature_blocks=FEATURE_BLOCKS,
        label_col=PRICE_LABEL_COL,
        vol_label_col=VOL_LABEL_COL,
        seq_len=SEQ_LEN
    )

    val_ds = MultiModalTS(
        df=val_ensemble_df,
        feature_blocks=FEATURE_BLOCKS,
        label_col=PRICE_LABEL_COL,
        vol_label_col=VOL_LABEL_COL,
        seq_len=SEQ_LEN
    )

    # --- 3. Apply Normalization ---
    # IMPORTANT: We load the scaler from the *original* training
    print("Loading and applying feature scaler...")
    scaler = np.load(TFT_SCALER_PATH)
    mean, std = scaler['mean'], scaler['std']
    
    train_ds.apply_scaler(mean, std)
    val_ds.apply_scaler(mean, std)

    # --- 4. Create DataLoaders ---
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- 5. Initialize Model ---
    # Get feature dimensions and add to configs
    feature_dims = train_ds.get_feature_dims()
    TFT_CONFIG['feature_dims'] = feature_dims
    
    total_in_feat = train_ds.X.shape[1]
    TCN_CONFIG['in_feat'] = total_in_feat

    print("Initializing MultiTaskEnsemble...")
    
    model = MultiTaskEnsemble(
        tft_config=TFT_CONFIG,
        tcn_config=TCN_CONFIG,
        tft_weights_path=TFT_WEIGHTS_PATH,
        tcn_weights_path=TCN_WEIGHTS_PATH
    ).to(DEVICE)

    # We are ONLY training the blender weights
    optimizer = AdamW(model.parameters(), lr=LR)
    
    best_loss = np.inf

    # --- 6. Training Loop ---
    print("Starting Ensemble training (training blender weights)...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_price_loss = train_one_epoch(
            model, train_loader, optimizer, DEVICE, PRICE_LOSS_PARAMS, VOL_WEIGHT
        )
        
        val_loss, val_price_loss = validate(
            model, val_loader, DEVICE, PRICE_LOSS_PARAMS, VOL_WEIGHT
        )

        print(f"--- Epoch {epoch}/{EPOCHS} ---")
        print(f"  Ensemble Train Loss: {train_loss:.4f} | Ensemble Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"New best ensemble found. Saving to {ENSEMBLE_CHECKPOINT_PATH}...")
            # Save the whole model, which includes the blenders
            torch.save(model.state_dict(), ENSEMBLE_CHECKPOINT_PATH)


    print("Ensemble Training complete.")
    
    # --- 7. Log Final Blender Weights (Interpretability) ---
    print("--- Final Ensemble Weights (Softmax) ---")
    price_w = torch.softmax(model.price_blender.w, dim=0).detach().cpu().numpy()
    vol_w = torch.softmax(model.vol_blender.w, dim=0).detach().cpu().numpy()
    
    print(f"  Price Blend: {price_w[0]:.2f} (TFT) + {price_w[1]:.2f} (TCN)")
    print(f"  Vol Blend:   {vol_w[0]:.2f} (TFT) + {vol_w[1]:.2f} (TCN)")
    print("-" * 40)


if __name__ == "__main__":
    run_ensemble_training()