import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from pathlib import Path

# Import our new feature engineering function
from .feature_engineering import create_tabular_features

# --- Configuration ---

# TODO: Define your data and feature configuration here
DATA_PATH = "path/to/your/features.parquet" # Or .csv
ARTIFACT_DIR = Path("./model_artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# Columns to use for feature engineering
# These should be your base price/volume columns
TABULAR_FEATURE_COLS = ['close', 'volume', 'open', 'high', 'low']
ROLL_WINDOWS = [5, 10, 20] # Windows for rolling stats

# TODO: Define your multi-task label columns
PRICE_LABEL_COL = "next_return_15m"
VOL_LABEL_COL = "next_vol_15m"

# --- XGBoost Hyperparameters ---
# (These can be tuned later)
XGB_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'random_state': 42,
}

def run_xgb_training():
    
    # --- 1. Load Base Data ---
    if not os.path.exists(DATA_PATH):
        print(f"Warning: Data file not found at {DATA_PATH}. Using placeholder data.")
        # Create a mock dataframe
        T = 5000 # 5000 time steps
        all_features = list(set(TABULAR_FEATURE_COLS + [PRICE_LABEL_COL, VOL_LABEL_COL]))
        data = pd.DataFrame(
            np.random.randn(T, len(all_features)),
            columns=all_features
        )
    else:
        print(f"Loading data from {DATA_PATH}...")
        data = pd.read_parquet(DATA_PATH) 

    # --- 2. Create Tabular Features ---
    print("Creating tabular features...")
    tabular_df = create_tabular_features(data, TABULAR_FEATURE_COLS, ROLL_WINDOWS)
    
    # --- 3. Align Features (X) and Labels (y) ---
    # This is critical. Features have NaNs at the start from rolling windows.
    # We must slice both X and y to start at the first valid feature row.
    first_valid_idx = tabular_df.first_valid_index()
    if first_valid_idx is None:
        raise ValueError("Feature engineering resulted in an all-NaN DataFrame.")
        
    print(f"Aligning data: dropping first {data.index.get_loc(first_valid_idx)} rows.")
    
    X_tabular = tabular_df.loc[first_valid_idx:]
    
    # Get corresponding labels from the *original* dataframe
    y_price = data.loc[first_valid_idx:, PRICE_LABEL_COL]
    y_vol = data.loc[first_valid_idx:, VOL_LABEL_COL]
    
    # --- 4. Split Data ---
    # Simple time-series split
    val_split_idx = int(len(X_tabular) * 0.8)
    
    X_train = X_tabular.iloc[:val_split_idx]
    y_train_price = y_price.iloc[:val_split_idx]
    y_train_vol = y_vol.iloc[:val_split_idx]
    
    X_val = X_tabular.iloc[val_split_idx:]
    y_val_price = y_price.iloc[val_split_idx:]
    y_val_vol = y_vol.iloc[val_split_idx:]

    print(f"XGB Train samples: {len(X_train)}, XGB Validation samples: {len(X_val)}")

    # --- 5. Train Price Model ---
    print("\n--- Training XGBoost Price Model ---")
    xgb_price_model = xgb.XGBRegressor(**XGB_PARAMS)
    
    xgb_price_model.fit(
        X_train, y_train_price,
        eval_set=[(X_val, y_val_price)],
        early_stopping_rounds=20,
        verbose=100
    )
    
    # Save the price model
    price_model_path = ARTIFACT_DIR / "xgb_price_model.joblib"
    joblib.dump(xgb_price_model, price_model_path)
    print(f"Price model saved to {price_model_path}")

    # --- 6. Train Volatility Model ---
    print("\n--- Training XGBoost Volatility Model ---")
    xgb_vol_model = xgb.XGBRegressor(**XGB_PARAMS)
    
    xgb_vol_model.fit(
        X_train, y_train_vol,
        eval_set=[(X_val, y_val_vol)],
        early_stopping_rounds=20,
        verbose=100
    )
    
    # Save the volatility model
    vol_model_path = ARTIFACT_DIR / "xgb_vol_model.joblib"
    joblib.dump(xgb_vol_model, vol_model_path)
    print(f"Volatility model saved to {vol_model_path}")

    print("\n--- XGBoost Training Complete ---")

if __name__ == "__main__":
    run_xgb_training()