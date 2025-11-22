import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Import our shared feature engineering helpers
from app.ml.adv.feature_engineering import (
    apply_price_feature_config,
    create_tabular_features,
)

logger = logging.getLogger(__name__)


class MultiModalTS(Dataset):
    """
    Windowed time-series dataset for the full ensemble.

    It creates and serves:
      1. x_blocks: dict of sequential feature blocks for TFT/TCN/TST
         e.g. {"price": [L, F_price], "sentiment": [L, F_sentiment], ...}
      2. x_tabular: 1D tensor of tabular features for XGBoost/DecisionNet
         (or an empty tensor if tabular is disabled).
      3. y_dict: {"price": scalar tensor, "vol": scalar tensor}

    The caller is responsible for:
      - Joining sentiment (aggregated_sentiment) into `df` before init.
      - Providing `feature_blocks` that match the columns in `df`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_blocks: Dict[str, List[str]],  # For TFT/TCN/TST (e.g. FEATURE_CONFIG)
        label_col: str,
        vol_label_col: str,
        # Configs for XGBoost-style tabular features
        tabular_feature_cols: Optional[List[str]],
        roll_windows: Optional[List[int]],
        seq_len: int = 60,
        target_scaler: float = 1.0,
        dropna: bool = True,
    ) -> None:
        self.seq_len = seq_len
        self.label_col = label_col
        self.vol_label_col = vol_label_col
        self.target_scaler = target_scaler

        # Defensive copy
        df = df.copy()

        # 0. Basic NaN/inf cleanup (mostly for labels/raw inputs)
        if dropna:
            df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # 1. Apply SHARED price preprocessing (returns & rolling stats)
        df = apply_price_feature_config(df)

        # 2. Optional Tabular Features for XGBoost / DecisionNet
        self.tabular_feature_cols = list(tabular_feature_cols or [])
        self.roll_windows = list(roll_windows or [])
        self._tabular_df: Optional[pd.DataFrame] = None
        self.X_tabular: Optional[np.ndarray] = None

        if self.tabular_feature_cols and self.roll_windows:
            logger.info(
                "[MultiModalTS] Creating tabular features (cols=%s, windows=%s)",
                self.tabular_feature_cols,
                self.roll_windows,
            )
            tabular_df = create_tabular_features(
                df,
                base_cols=self.tabular_feature_cols,
                roll_windows=self.roll_windows,
            )

            if tabular_df.empty or tabular_df.shape[1] == 0:
                logger.warning(
                    "[MultiModalTS] Tabular DataFrame is empty – "
                    "disabling tabular block."
                )
                tabular_df = None
            elif tabular_df.isna().all().all():
                logger.warning(
                    "[MultiModalTS] Tabular DataFrame is all-NaN – "
                    "disabling tabular block."
                )
                tabular_df = None
            else:
                tabular_df = tabular_df.astype("float32")
                self._tabular_df = tabular_df
        else:
            logger.info(
                "[MultiModalTS] No tabular_feature_cols/roll_windows provided – "
                "training sequence-only blocks (price/sentiment)."
            )
            self._tabular_df = None

        # 3. Align all data if we *do* have a tabular block
        if self._tabular_df is not None:
            first_valid_idx = self._tabular_df.first_valid_index()
            if first_valid_idx is None:
                logger.warning(
                    "[MultiModalTS] Tabular DataFrame has no valid rows – "
                    "disabling tabular block."
                )
                self._tabular_df = None
                df_aligned = df
                self.X_tabular = None
            else:
                logger.info(
                    "[MultiModalTS] Aligning all data sources to first valid "
                    "tabular index: %s",
                    first_valid_idx,
                )
                df_aligned = df.loc[first_valid_idx:]
                tabular_df_aligned = self._tabular_df.loc[first_valid_idx:]
                self.X_tabular = tabular_df_aligned.astype(np.float32).values
        else:
            # No tabular block -> use full df as-is
            df_aligned = df
            self.X_tabular = None

        # 4. Build Sequential Feature Matrix (X) from feature_blocks
        X_blocks: List[np.ndarray] = []
        self.block_slices: Dict[str, slice] = {}
        self.feature_block_names = list(feature_blocks.keys())

        start = 0
        for bname, cols in feature_blocks.items():
            cols = list(cols or [])
            if not cols:
                continue

            missing = [c for c in cols if c not in df_aligned.columns]
            if missing:
                raise ValueError(
                    f"[MultiModalTS] Missing columns for block '{bname}': {missing}"
                )

            Xb = df_aligned[cols].astype(np.float32).values  # [T, F_block]
            end = start + Xb.shape[1]
            self.block_slices[bname] = slice(start, end)
            X_blocks.append(Xb)
            start = end

        if not X_blocks:
            raise ValueError(
                "[MultiModalTS] No feature blocks produced any data. "
                "Check feature_blocks and df columns."
            )

        # Concatenate all blocks into [T, F_total]
        self.X = np.concatenate(X_blocks, axis=1)

        # 5. Multi-task labels (aligned with df_aligned)
        if label_col not in df_aligned.columns or vol_label_col not in df_aligned.columns:
            raise ValueError(
                f"[MultiModalTS] label_col='{label_col}' or vol_label_col='{vol_label_col}' "
                "not found in DataFrame."
            )

        self.y_price = (
            df_aligned[label_col].astype(np.float32).values * target_scaler
        )
        self.y_vol = df_aligned[vol_label_col].astype(np.float32).values

        # Sanity checks
        if self.X_tabular is not None:
            assert len(self.X) == len(self.X_tabular) == len(self.y_price), (
                "Aligned lengths mismatch between X, X_tabular and y."
            )
        else:
            assert len(self.X) == len(self.y_price), (
                "Aligned lengths mismatch between X and y."
            )

        # 6. Number of valid windows
        self.n_samples = len(self.X) - self.seq_len + 1
        if self.n_samples <= 0:
            raise ValueError(
                f"[MultiModalTS] Not enough rows ({len(self.X)}) "
                f"for seq_len={self.seq_len}."
            )

        # Scaler state for sequential features
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_normalized: bool = False

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
          x_blocks: dict {block_name: [seq_len, n_features_block] tensor}
          x_tabular_row: [n_tabular_features] tensor (or empty tensor if disabled)
          y_dict: {"price": scalar tensor, "vol": scalar tensor}
        """
        # Sequence slice [idx : idx + L]
        sl = slice(idx, idx + self.seq_len)
        x_seq_full = self.X[sl]  # [L, F_total]

        # 1) Build feature blocks for TFT/TCN/TST (price + sentiment + ...)
        x_blocks: Dict[str, torch.Tensor] = {}
        for bname, bslice in self.block_slices.items():
            x_blocks[bname] = torch.from_numpy(
                x_seq_full[:, bslice]
            ).float()  # [L, F_block]

        # Label index is the last timestep in the sequence
        label_idx = idx + self.seq_len - 1

        # 2) Optional tabular row for XGBoost / meta-models
        x_tabular_src = getattr(self, "X_tabular", None)
        if x_tabular_src is not None:
            x_tabular_row = torch.from_numpy(
                x_tabular_src[label_idx]
            ).float()  # [F_tab]
        else:
            # Keep DataLoader happy: always return a tensor
            x_tabular_row = torch.empty(0, dtype=torch.float32)

        # 3) Targets
        y_dict: Dict[str, torch.Tensor] = {
            "price": torch.tensor(self.y_price[label_idx], dtype=torch.float32),
            "vol": torch.tensor(self.y_vol[label_idx], dtype=torch.float32),
        }

        return x_blocks, x_tabular_row, y_dict

    def fit_scaler(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates and stores the mean/std of *sequential* features (self.X)."""
        print("Calculating sequential scaler...")
        self.mean = np.nanmean(self.X, axis=0, keepdims=True)
        self.std = np.nanstd(self.X, axis=0, keepdims=True) + 1e-6
        print("Sequential scaler calculated.")
        self.is_normalized = False
        return self.mean, self.std

    def apply_scaler(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Applies a scaler to the *sequential* features (self.X)."""
        if self.is_normalized:
            print("Warning: Dataset is already normalized.")
            return

        self.mean = mean
        self.std = std
        self.X = (self.X - self.mean) / self.std
        self.is_normalized = True
        print("Sequential scaler applied.")

    def get_feature_dims(self) -> Dict[str, int]:
        """Helper for TFT/TCN model initialization."""
        return {bname: s.stop - s.start for bname, s in self.block_slices.items()}
