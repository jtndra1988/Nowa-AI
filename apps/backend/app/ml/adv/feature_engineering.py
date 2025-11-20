import pandas as pd
import numpy as np
from typing import Any, Dict, List

class FeatureBuilder:
    def __init__(self):
        pass
        
    async def build_features(self, symbol: str) -> Dict[str, Any]:
        # 1. Fetch OHLCV data from DB or Cache
        # 2. Run create_tabular_features(df, ...)
        # 3. Return dictionary of blocks {'price': ..., 'macro': ...}
        return {}
def create_tabular_features(
    df: pd.DataFrame,
    price_cols: List[str],
    roll_windows: List[int],
) -> pd.DataFrame:
    """
    Creates lag, rolling stats, ROC, and time features for XGBoost.
    """
    if "close" in df.columns and "close" not in price_cols:
        price_cols = ["close"] + price_cols

    tabular_df = pd.DataFrame(index=df.index)

    for col in price_cols:
        if col not in df.columns:
            continue

        # Lags
        for lag in [1, 2, 3, 5, 10]:
            tabular_df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        # Rolling features
        for window in roll_windows:
            tabular_df[f"{col}_roll_mean_{window}"] = df[col].rolling(window).mean()
            tabular_df[f"{col}_roll_std_{window}"] = df[col].rolling(window).std()
            tabular_df[f"{col}_roll_min_{window}"] = df[col].rolling(window).min()
            tabular_df[f"{col}_roll_max_{window}"] = df[col].rolling(window).max()

        # ROC
        for period in [1, 5, 10]:
            tabular_df[f"{col}_roc_{period}"] = df[col].pct_change(periods=period)

    # Time features
    if isinstance(df.index, pd.DatetimeIndex):
        tabular_df["time_hour"] = df.index.hour
        tabular_df["time_dayofweek"] = df.index.dayofweek
        tabular_df["time_month"] = df.index.month

    # Clean NaN/inf
    first_valid = tabular_df.first_valid_index()
    if first_valid is not None:
        tabular_df.loc[first_valid:] = tabular_df.loc[first_valid:].fillna(0)

    tabular_df.replace([np.inf, -np.inf], 0, inplace=True)

    return tabular_df
