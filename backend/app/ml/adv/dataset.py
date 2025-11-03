import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

class MultiModalTS(Dataset):
    """
    Windowed time-series dataset, updated for multi-task learning and TFT.
    
    Now expects a single aligned DataFrame with columns:
      - price features: open, high, low, close, volume, etc.
      - ob_* for order-book derived features
      - sent_* for sentiment/signal fusion features
      - onch_* for on-chain metrics
    Labels:
      - label_col (e.g., next_return_{h}): The main price-related target.
      - vol_label_col (e.g., next_vol_{h}): The volatility target.
    
    NOTE: This dataset stores raw, un-normalized data.
    Use the `fit_scaler` and `apply_scaler` methods in your
    training script to handle normalization correctly without leakage.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_blocks: Dict[str, List[str]],
        label_col: str,
        vol_label_col: str, # Column name for the volatility target
        seq_len: int = 60,
        target_scaler: float = 1.0,
        dropna: bool = True,
    ):
        self.seq_len = seq_len
        self.label_col = label_col
        self.vol_label_col = vol_label_col
        self.target_scaler = target_scaler

        if dropna:
            df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # Build feature matrix in blocks (keeps channel groups explicit)
        X_blocks = []
        self.block_slices: Dict[str, slice] = {}
        self.feature_block_names = list(feature_blocks.keys())
        
        start = 0
        for bname, cols in feature_blocks.items():
            # Ensure all columns exist, fill with 0 if they don't
            existing_cols = [c for c in cols if c in df.columns]
            missing_cols = [c for c in cols if c not in df.columns]
            if missing_cols:
                print(f"Warning: Columns missing for block '{bname}', will be filled with 0: {missing_cols}")
                for c in missing_cols:
                    df[c] = 0.0 # Add placeholder column
            
            Xb = df[cols].astype(np.float32).values
            end = start + Xb.shape[1]
            self.block_slices[bname] = slice(start, end)
            X_blocks.append(Xb)
            start = end

        self.X = np.concatenate(X_blocks, axis=1)  # [T, F_total]
        
        # --- Multi-Task Labels ---
        # 1. Price target
        self.y_price = df[label_col].astype(np.float32).values * target_scaler
        # 2. Volatility target
        self.y_vol = df[vol_label_col].astype(np.float32).values
        # -------------------------

        # Store scaler state
        self.mean = None
        self.std = None
        self.is_normalized = False

    def __len__(self):
        return max(0, self.X.shape[0] - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Returns a tuple of (feature_block_dictionary, target_dictionary).
        This matches the multi-task TFT model input and loss function.
        """
        sl = slice(idx, idx + self.seq_len)
        x_seq_full = self.X[sl] # Numpy array: [L, F_total]
        
        # --- Create feature dictionary (x_blocks) ---
        # This is the key change for the TFT model
        x_blocks = {}
        for bname, bslice in self.block_slices.items():
            x_blocks[bname] = torch.from_numpy(x_seq_full[:, bslice]).float()
        # -------------------------------------------
        
        # Get the label index (at the end of the sequence)
        label_idx = idx + self.seq_len - 1
        
        # --- Create target dictionary ---
        y_dict = {
            'price': torch.tensor(self.y_price[label_idx]),
            'vol': torch.tensor(self.y_vol[label_idx])
        }
        # --------------------------------
        
        return x_blocks, y_dict

    def fit_scaler(self):
        """
        Calculates and stores the mean/std of the dataset's features (self.X).
        Call this *only* on the training dataset object.
        """
        print("Calculating scaler...")
        self.mean = np.nanmean(self.X, axis=0, keepdims=True)
        self.std = np.nanstd(self.X, axis=0, keepdims=True) + 1e-6
        print("Scaler calculated.")
        
        # Check for any issues
        if np.any(np.isnan(self.mean)) or np.any(np.isnan(self.std)):
            print("Warning: NaN found in scaler. Check input data.")
            
        return self.mean, self.std

    def apply_scaler(self, mean: np.ndarray, std: np.ndarray):
        """
        Applies a pre-computed mean/std scaler to the dataset's features.
        """
        if self.is_normalized:
            print("Warning: Dataset is already normalized.")
            return

        self.mean = mean
        self.std = std
        
        # Apply the scaling
        self.X = (self.X - self.mean) / self.std
        self.is_normalized = True
        print("Scaler applied.")

    def get_feature_dims(self) -> Dict[str, int]:
        """
        Helper function to return the feature dimensions for each block.
        Needed to initialize the TFT model.
        """
        return {bname: s.stop - s.start for bname, s in self.block_slices.items()}