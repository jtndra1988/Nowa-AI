import numpy as np
import pandas as pd
from typing import Callable, Dict, List

def purged_splits(df: pd.DataFrame, n_splits=5, embargo=0.02):
    """
    Time-ordered splits with purge + embargo to avoid leakage.
    df must be time-ordered. Returns list of (train_idx, val_idx).
    """
    T = len(df)
    fold = T // (n_splits+1)
    splits = []
    for i in range(n_splits):
        train_end = fold*(i+1)
        val_start = int(train_end*(1 - embargo))
        val_end = fold*(i+2)
        train_idx = np.arange(0, val_start)
        val_idx   = np.arange(train_end, min(val_end, T))
        splits.append((train_idx, val_idx))
    return splits

def walk_forward_indices(T: int, win_train: int, win_val: int, step: int):
    i = 0
    while i + win_train + win_val <= T:
        tr = np.arange(i, i+win_train)
        va = np.arange(i+win_train, i+win_train+win_val)
        yield tr, va
        i += step
