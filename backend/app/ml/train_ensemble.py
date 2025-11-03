import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path

# --- Import All Models and Helpers ---
from .models_tft import TemporalFusionTransformer
from .models_tcn import TemporalConvNet
from .ensemble import StackingEnsemble
from .dataset import MultiModalTS
from .losses import multitask_transformer_loss

# --- Configuration ---
# All configurations must match the base models
# ------------------------------------------------

# TODO: Define your data and feature configuration here
DATA_PATH = "path/to/your/features.parquet" # Or .csv
ARTIFACT_DIR = Path("./model_artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# 1. Sequential Features (for TFT/TCN)
FEATURE_BLOCKS = {
    "price": ['open', 'high', 'low', 'close', 'volume'],
    "ob": ['ob_imbalance_1s', 'ob_spread', 'ob_depth_ask_1', 'ob_depth_bid_1'],
    "sent": ['sent_score_1m', 'sent_score_15m'],
}

# 2. Tabular Features (for XGBoost)
TABULAR_FEATURE_COLS = ['close', 'volume', 'open', 'high', 'low']
ROLL_WINDOWS = [5, 10, 20] # Windows for rolling stats

# 3. Label columns
PRICE_LABEL_COL = "next_return_15m"
VOL_LABEL_COL = "next_vol_15m"

# --- Model & Training Hyperparameters ---
SEQ_LEN = 60
BATCH_SIZE = 64
EPOCHS = 20 # Blending trains very quickly
LR = 1e-3 # Can use a higher LR for just the blenders
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
SCALER_PATH = ARTIFACT_DIR / "tft_scaler.npz" # Use the main scaler

# --- 2. TCN Model Configuration (must match train_tcn.py) ---
TCN_CONFIG = {
    # in_feat will be added later
    "channels": (64, 128, 128),
    "kernel": 3,
    "dropout": 0.1
}
TCN_WEIGHTS_PATH = ARTIFACT_DIR / "tcn_best_model.pth"

# --- 3. XGBoost Model Configuration (must match train_xgb.py) ---
XGB_PRICE_PATH = ARTIFACT_DIR / "xgb_price_model.joblib"
XGB_VOL_PATH = ARTIFACT_DIR / "xgb_vol_model.joblib"

# --- 4. Ensemble Config ---
ENSEMBLE_CHECKPOINT_PATH = ARTIFACT_DIR / "hybrid_ensemble_best.pth"
N_MODELS = 3 # TFT, TCN, XGBoost

# Loss parameters
PRICE_LOSS_PARAMS = {'alpha': 0.5, 'beta': 0.2, 'lam': 0.1}
VOL_WEIGHT = 0.2

# --- Helper function for XGBoost prediction ---
def xgb_predict(xgb_model, x_tabular_tensor: torch.Tensor) -> torch.Tensor:
    """ Helper to run numpy-based XGB model """
    x_tab_np = x_tabular_tensor.cpu().numpy()
    
    # Handle potential NaNs in input just in case
    if np.isnan(x_tab_np).any():
        print("Warning: NaNs found in tabular data for XGB, filling with 0.")
        x_tab_np = np.nan_to_num(x_tab_np)
        
    preds_np = xgb_model.predict(x_tab_np)
    return torch.from_numpy(preds_np).float().to(x_tabular_tensor.device)


def train_one_epoch(models, blenders, loader, optimizer, device, price_loss_params, vol_weight):
    # Set blenders to train mode
    blenders['price'].train()
    blenders['vol'].train()
    
    total_loss_epoch = 0
    price_loss_epoch = 0

    for x_blocks, x_tabular, y_dict in loader:
        # Move all data to device
        x_blocks = {k: v.to(device) for k, v in x_blocks.items()}
        x_tabular = x_tabular.to(device)
        y_dict = {k: v.to(device) for k, v in y_dict.items()}

        optimizer.zero_grad()
        
        # --- 1. Get predictions from all 3 models ---
        with torch.no_grad():
            # PyTorch models
            preds_tft = models['tft'](x_blocks)
            preds_tcn = models['tcn'](x_blocks)
            
            # XGBoost models (requires numpy)
            preds_xgb_price = xgb_predict(models['xgb_price'], x_tabular)
            preds_xgb_vol = xgb_predict(models['xgb_vol'], x_tabular)

        # --- 2. Stack predictions for blending ---
        # Stack price predictions: [B, 3]
        stacked_price = torch.stack([
            preds_tft['price'],
            preds_tcn['price'],
            preds_xgb_price
        ], dim=1)
        
        # Stack volatility predictions: [B, 3]
        stacked_vol = torch.stack([
            preds_tft['vol'],
            preds_tcn['vol'],
            preds_xgb_vol
        ], dim=1)

        # --- 3. Blend the predictions (This part is trainable) ---
        blended_price = blenders['price'](stacked_price)
        blended_vol = blenders['vol'](stacked_vol)
        
        pred_dict = {
            'price': blended_price,
            'vol': blended_vol
        }
        
        # --- 4. Calculate Loss ---
        loss_components = multitask_transformer_loss(
            pred_dict,
            y_dict,
            price_loss_params,
            vol_weight
        )
        
        loss = loss_components['total_loss']
        
        # --- 5. Backpropagate (updates blender weights) ---
        loss.backward()
        optimizer.step()

        total_loss_epoch += loss.item()
        price_loss_epoch += loss_components['price_loss'].item()

    avg_total_loss = total_loss_epoch / len(loader)
    avg_price_loss = price_loss_epoch / len(loader)
    return avg_total_loss, avg_price_loss

def validate(models, blenders, loader, device, price_loss_params, vol_weight):
    # Set blenders to eval mode
    blenders['price'].eval()
    blenders['vol'].eval()
    
    total_loss_epoch = 0
    price_loss_epoch = 0
    
    with torch.no_grad():
        for x_blocks, x_tabular, y_dict in loader:
            x_blocks = {k: v.to(device) for k, v in x_blocks.items()}
            x_tabular = x_tabular.to(device)
            y_dict = {k: v.to(device) for k, v in y_dict.items()}

            # --- 1. Get predictions from all 3 models ---
            preds_tft = models['tft'](x_blocks)
            preds_tcn = models['tcn'](x_blocks)
            preds_xgb_price = xgb_predict(models['xgb_price'], x_tabular)
            preds_xgb_vol = xgb_predict(models['xgb_vol'], x_tabular)

            # --- 2. Stack predictions ---
            stacked_price = torch.stack([preds_tft['price'], preds_tcn['price'], preds_xgb_price], dim=1)
            stacked_vol = torch.stack([preds_tft['vol'], preds_tcn['vol'], preds_xgb_vol], dim=1)

            # --- 3. Blend predictions ---
            blended_price = blenders['price'](stacked_price)
            blended_vol = blenders['vol'](stacked_vol)
            
            pred_dict = {'price': blended_price, 'vol': blended_vol}
            
            # --- 4. Calculate Loss ---
            loss_components = multitask_transformer_loss(
                pred_dict, y_dict, price_loss_params, vol_weight
            )
            
            total_loss_epoch += loss_components['total_loss'].item()
            price_loss_epoch += loss_components['price_loss'].item()

    avg_total_loss = total_loss_epoch / len(loader)
    avg_price_loss = price_loss_epoch / len(loader)
    return avg_total_loss, avg_price_loss

def run_hybrid_ensemble_training():
    print(f"Using device: {DEVICE}")

    # --- 1. Load Data ---
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Warning: Data file not found at {DATA_PATH}. Using placeholder data.")
        T = 5000 
        all_features = list(set([col for cols in FEATURE_BLOCKS.values() for col in cols] + TABULAR_FEATURE_COLS))
        data = pd.DataFrame(np.random.randn(T, len(all_features)), columns=all_features)
        data[PRICE_LABEL_COL] = np.random.randn(T)
        data[VOL_LABEL_COL] = np.random.rand(T)
    else:
        data = pd.read_parquet(DATA_PATH) 

    # --- 2. Split and Create Datasets ---
    # We train the ensemble on the validation data
    val_split_idx = int(len(data) * 0.8)
    val_df = data.iloc[val_split_idx:]
    
    ensemble_train_split_idx = int(len(val_df) * 0.7)
    train_ensemble_df = val_df.iloc[:ensemble_train_split_idx]
    val_ensemble_df = val_df.iloc[ensemble_train_split_idx:]

    print(f"Ensemble training samples: {len(train_ensemble_df)}, Ensemble validation samples: {len(val_ensemble_df)}")

    # Initialize the new dataset
    train_ds = MultiModalTS(
        df=train_ensemble_df,
        feature_blocks=FEATURE_BLOCKS,
        label_col=PRICE_LABEL_COL,
        vol_label_col=VOL_LABEL_COL,
        tabular_feature_cols=TABULAR_FEATURE_COLS,
        roll_windows=ROLL_WINDOWS,
        seq_len=SEQ_LEN
    )

    val_ds = MultiModalTS(
        df=val_ensemble_df,
        feature_blocks=FEATURE_BLOCKS,
        label_col=PRICE_LABEL_COL,
        vol_label_col=VOL_LABEL_COL,
        tabular_feature_cols=TABULAR_FEATURE_COLS,
        roll_windows=ROLL_WINDOWS,
        seq_len=SEQ_LEN
    )

    # --- 3. Apply Normalization ---
    print("Loading and applying sequential feature scaler...")
    scaler = np.load(SCALER_PATH)
    mean, std = scaler['mean'], scaler['std']
    train_ds.apply_scaler(mean, std)
    val_ds.apply_scaler(mean, std)

    # --- 4. Create DataLoaders ---
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- 5. Initialize All Models ---
    
    # Get feature dimensions
    feature_dims = train_ds.get_feature_dims()
    TFT_CONFIG['feature_dims'] = feature_dims
    total_in_feat = train_ds.X.shape[1] # For TCN
    TCN_CONFIG['in_feat'] = total_in_feat

    # Load PyTorch Models
    print("Loading pre-trained TFT model...")
    tft_model = TemporalFusionTransformer(**TFT_CONFIG).to(DEVICE)
    tft_model.load_state_dict(torch.load(TFT_WEIGHTS_PATH))
    tft_model.eval()
    for param in tft_model.parameters():
        param.requires_grad = False
        
    print("Loading pre-trained TCN model...")
    tcn_model = TemporalConvNet(**TCN_CONFIG).to(DEVICE)
    tcn_model.load_state_dict(torch.load(TCN_WEIGHTS_PATH))
    tcn_model.eval()
    for param in tcn_model.parameters():
        param.requires_grad = False

    # Load XGBoost Models
    print("Loading pre-trained XGBoost models...")
    xgb_price_model = joblib.load(XGB_PRICE_PATH)
    xgb_vol_model = joblib.load(XGB_VOL_PATH)
    
    models = {
        'tft': tft_model,
        'tcn': tcn_model,
        'xgb_price': xgb_price_model,
        'xgb_vol': xgb_vol_model
    }
    
    # Initialize Trainable Blenders
    print("Initializing trainable blenders...")
    price_blender = StackingEnsemble(n_models=N_MODELS).to(DEVICE)
    vol_blender = StackingEnsemble(n_models=N_MODELS).to(DEVICE)
    
    blenders = {
        'price': price_blender,
        'vol': vol_blender
    }

    # Optimizer tracks ONLY the blender weights
    optimizer = AdamW(
        list(price_blender.parameters()) + list(vol_blender.parameters()), 
        lr=LR
    )
    
    best_loss = np.inf

    # --- 6. Training Loop ---
    print("Starting Hybrid Ensemble training...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_price_loss = train_one_epoch(
            models, blenders, train_loader, optimizer, DEVICE, PRICE_LOSS_PARAMS, VOL_WEIGHT
        )
        val_loss, val_price_loss = validate(
            models, blenders, val_loader, DEVICE, PRICE_LOSS_PARAMS, VOL_WEIGHT
        )

        print(f"--- Epoch {epoch}/{EPOCHS} ---")
        print(f"  Ensemble Train Loss: {train_loss:.4f} | Ensemble Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"New best ensemble found. Saving blenders to {ENSEMBLE_CHECKPOINT_PATH}...")
            # Save the blenders' weights
            torch.save({
                'price_blender_state_dict': price_blender.state_dict(),
                'vol_blender_state_dict': vol_blender.state_dict(),
            }, ENSEMBLE_CHECKPOINT_PATH)

    print("Hybrid Ensemble Training complete.")
    
    # --- 7. Log Final Blender Weights (Interpretability) ---
    print("--- Final Ensemble Weights (Softmax) ---")
    price_w = torch.softmax(price_blender.w, dim=0).detach().cpu().numpy()
    vol_w = torch.softmax(vol_blender.w, dim=0).detach().cpu().numpy()
    
    print(f"  Price Blend: {price_w[0]:.2f} (TFT) + {price_w[1]:.2f} (TCN) + {price_w[2]:.2f} (XGB)")
    print(f"  Vol Blend:   {vol_w[0]:.2f} (TFT) + {vol_w[1]:.2f} (TCN) + {vol_w[2]:.2f} (XGB)")
    print("-" * 40)

if __name__ == "__main__":
    run_hybrid_ensemble_training()