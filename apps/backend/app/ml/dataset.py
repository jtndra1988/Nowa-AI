import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

# Import our new feature engineering function
from app.ml.adv.feature_engineering import create_tabular_features

class MultiModalTS(Dataset):
    """
    Windowed time-series dataset, updated for the full ensemble.
    
    It now creates and serves:
    1.  `x_blocks`: A dictionary of sequential features for TFT/TCN.
    2.  `x_tabular`: A tensor of tabular features for XGBoost.
    3.  `y_dict`: A dictionary of multi-task targets (price, vol).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_blocks: Dict[str, List[str]], # For TFT/TCN
        label_col: str,
        vol_label_col: str,
        # Configs for XGBoost features
        tabular_feature_cols: List[str], 
        roll_windows: List[int],
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
            
        # --- 1. Create Tabular Features for XGBoost ---
        print("Creating tabular features for dataset...")
        tabular_df = create_tabular_features(df, tabular_feature_cols, roll_windows)
        
        # --- 2. Align all data ---
        # We must drop initial rows where tabular features are NaN
        first_valid_idx = tabular_df.first_valid_index()
        if first_valid_idx is None:
            raise ValueError("Feature engineering resulted in an all-NaN DataFrame.")
        
        print(f"Aligning all data sources to first valid index: {first_valid_idx}")
        df_aligned = df.loc[first_valid_idx:]
        tabular_df_aligned = tabular_df.loc[first_valid_idx:]
        
        self.X_tabular = tabular_df_aligned.astype(np.float32).values

        # --- 3. Build Sequential Feature Matrix (X) ---
        X_blocks = []
        self.block_slices: Dict[str, slice] = {}
        self.feature_block_names = list(feature_blocks.keys())
        
        start = 0
        for bname, cols in feature_blocks.items():
            # Use the *aligned* dataframe
            Xb = df_aligned[cols].astype(np.float32).values 
            end = start + Xb.shape[1]
            self.block_slices[bname] = slice(start, end)
            X_blocks.append(Xb)
            start = end

        self.X = np.concatenate(X_blocks, axis=1)  # [T, F_total]
        
        # --- 4. Build Multi-Task Labels ---
        # Use the *aligned* dataframe
        self.y_price = df_aligned[label_col].astype(np.float32).values * target_scaler
        self.y_vol = df_aligned[vol_label_col].astype(np.float32).values
        # -------------------------

        # Store scaler state
        self.mean = None
        self.std = None
        self.is_normalized = False
        
        # Ensure all data has the same length
        assert len(self.X) == len(self.X_tabular) == len(self.y_price), "Aligned data lengths mismatch!"

    def __len__(self):
        return max(0, self.X.shape[0] - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns a tuple of:
        (feature_block_dictionary, tabular_feature_row, target_dictionary).
        """
        sl = slice(idx, idx + self.seq_len)
        x_seq_full = self.X[sl] # Numpy array: [L, F_total]
        
        # 1. Create feature dictionary (x_blocks) for TFT/TCN
        x_blocks = {}
        for bname, bslice in self.block_slices.items():
            x_blocks[bname] = torch.from_numpy(x_seq_full[:, bslice]).float()
        
        # Get the label index (at the end of the sequence)
        label_idx = idx + self.seq_len - 1
        
        # 2. Get tabular feature row for XGBoost
        # We use the label_idx, which corresponds to the *last*
        # time step in the sequence, which is what we want.
        x_tabular_row = torch.from_numpy(self.X_tabular[label_idx]).float()
        
        # 3. Create target dictionary
        y_dict = {
            'price': torch.tensor(self.y_price[label_idx]),
            'vol': torch.tensor(self.y_vol[label_idx])
        }
        
        return x_blocks, x_tabular_row, y_dict

    def fit_scaler(self):
        """ Calculates and stores the mean/std of *sequential* features (self.X). """
        print("Calculating sequential scaler...")
        self.mean = np.nanmean(self.X, axis=0, keepdims=True)
        self.std = np.nanstd(self.X, axis=0, keepdims=True) + 1e-6
        print("Sequential scaler calculated.")
        
        # Note: We don't scale the tabular features here, as XGBoost
        # is less sensitive, and they are already engineered.
        # If scaling is needed, it should be done separately.
            
        return self.mean, self.std

    def apply_scaler(self, mean: np.ndarray, std: np.ndarray):
        """ Applies a scaler to the *sequential* features (self.X). """
        if self.is_normalized:
            print("Warning: Dataset is already normalized.")
            return

        self.mean = mean
        self.std = std
        self.X = (self.X - self.mean) / self.std
        self.is_normalized = True
        print("Sequential scaler applied.")

    def get_feature_dims(self) -> Dict[str, int]:
        """ Helper for TFT model initialization. """
        return {bname: s.stop - s.start for bname, s in self.block_slices.items()}