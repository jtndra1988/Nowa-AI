import pandas as pd
import numpy as np
from typing import List

def create_tabular_features(df: pd.DataFrame, price_cols: List[str], roll_windows: List[int]) -> pd.DataFrame:
    """
    Creates a DataFrame of tabular features (lags, rolling stats)
    for use with XGBoost.
    
    Args:
        df: The input DataFrame containing at least 'close' and 'volume'.
        price_cols: Columns to generate features for (e.g., 'close', 'volume').
        roll_windows: A list of window sizes for rolling statistics (e.g., [5, 10, 20]).
    
    Returns:
        A new DataFrame with tabular features.
    """
    print(f"Original df shape: {df.shape}")
    
    # Ensure 'close' is in price_cols if available
    if 'close' in df.columns and 'close' not in price_cols:
        price_cols = ['close'] + price_cols
        
    tabular_df = pd.DataFrame(index=df.index)
    
    for col in price_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not in DataFrame. Skipping.")
            continue
            
        # 1. Create Lag Features
        for lag in [1, 2, 3, 5, 10]:
            tabular_df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
        # 2. Create Rolling Statistics
        for window in roll_windows:
            # Rolling Mean
            tabular_df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
            
            # Rolling Std Dev
            tabular_df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
            
            # Rolling Min/Max
            tabular_df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window).min()
            tabular_df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window).max()
            
        # 3. Create Rate of Change (ROC)
        for period in [1, 5, 10]:
            tabular_df[f'{col}_roc_{period}'] = df[col].pct_change(periods=period)

    # 4. Create Time-based Features
    if isinstance(df.index, pd.DatetimeIndex):
        tabular_df['time_hour'] = df.index.hour
        tabular_df['time_dayofweek'] = df.index.dayofweek
        tabular_df['time_month'] = df.index.month

    # Handle NaNs created by lags/rolling windows
    # We will fill with 0 after the first valid index to keep data alignment.
    # A more robust method would be to fill with the mean, but this is
    # simpler for now and XGBoost can handle 0s well.
    first_valid_idx = tabular_df.first_valid_index()
    if first_valid_idx is not None:
        tabular_df.loc[first_valid_idx:] = tabular_df.loc[first_valid_idx:].fillna(0)
    
    # Replace any inf/-inf from pct_change
    tabular_df.replace([np.inf, -np.inf], 0, inplace=True)
    
    print(f"Tabular features df shape: {tabular_df.shape}")
    
    return tabular_df