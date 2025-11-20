import logging
import pandas as pd
import numpy as np
import torch
from typing import Any, Dict, List
from sqlalchemy import desc
from sqlalchemy.orm import Session

# Database imports
from app.db.database import SessionLocal
from app.db.models import MarketData

logger = logging.getLogger(__name__)

def create_tabular_features(
    df: pd.DataFrame,
    price_cols: List[str],
    roll_windows: List[int],
) -> pd.DataFrame:
    """
    Creates lag, rolling stats, ROC, and time features.
    Used by both training pipelines and live inference.
    """
    # Ensure we work on a copy to avoid SettingWithCopy warnings
    df = df.copy()
    
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

        # ROC (Rate of Change)
        for period in [1, 5, 10]:
            tabular_df[f"{col}_roc_{period}"] = df[col].pct_change(periods=period)

    # Time features
    if pd.api.types.is_datetime64_any_dtype(df.index):
        tabular_df["time_hour"] = df.index.hour
        tabular_df["time_dayofweek"] = df.index.dayofweek
        tabular_df["time_month"] = df.index.month

    # Handle NaNs created by shifting/rolling
    tabular_df = tabular_df.fillna(0)
    tabular_df.replace([np.inf, -np.inf], 0, inplace=True)

    return tabular_df


class FeatureBuilder:
    """
    Connects the 'Brain' to the Database. 
    Fetches raw candles and converts them into Tensor blocks for ModelEngine.
    """
    def __init__(self, seq_len: int = 60):
        self.seq_len = seq_len
        # We fetch more data than seq_len to account for lookback/rolling windows
        self.fetch_limit = seq_len + 100 

    async def build_features(self, symbol: str) -> Dict[str, Any]:
        """
        Orchestrates the data loading and feature generation.
        """
        session: Session = SessionLocal()
        try:
            # 1. Fetch Historical Data from DB
            # We fetch in descending order (newest first) to get the latest data, then reverse it.
            rows = (
                session.query(MarketData)
                .filter(MarketData.symbol == symbol.upper())
                .order_by(desc(MarketData.timestamp))
                .limit(self.fetch_limit)
                .all()
            )

            if not rows or len(rows) < self.seq_len:
                logger.warning(f"[FeatureBuilder] Insufficient data for {symbol}. Got {len(rows)} rows.")
                return {}

            # Convert to DataFrame and sort chronological (oldest -> newest)
            data = [
                {
                    "timestamp": r.timestamp,
                    "open": float(r.open),
                    "high": float(r.high),
                    "low": float(r.low),
                    "close": float(r.close),
                    "volume": float(r.volume),
                }
                for r in rows
            ]
            df = pd.DataFrame(data).sort_values("timestamp").reset_index(drop=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)

            # 2. Generate Specific Features (Matching train_tft.py logic)
            # This ensures the inference data distribution matches training.
            df["ret_1h"] = df["close"].pct_change()
            df["log_ret"] = np.log(df["close"]).diff()
            
            # Standard training features from your pipeline
            df["roll_vol_24h"] = df["log_ret"].rolling(24).std().fillna(0)
            df["roll_mean_24h"] = df["log_ret"].rolling(24).mean().fillna(0)
            df["roll_vol_6h"] = df["log_ret"].rolling(6).std().fillna(0)
            df["roll_vol_12h"] = df["log_ret"].rolling(12).std().fillna(0)

            # 3. Generate Generic Tabular Features (XGBoost style)
            # These might be used by DecisionNet or TCN
            tabular_feats = create_tabular_features(
                df, 
                price_cols=["close", "volume"], 
                roll_windows=[6, 12, 24]
            )
            
            # Combine all features
            full_df = pd.concat([df, tabular_feats], axis=1)
            
            # Drop initial rows that have NaNs from rolling windows
            full_df = full_df.fillna(0.0)
            
            # 4. Slice the exact Sequence Length needed for the model
            # We take the *last* 'seq_len' rows to represent the current state
            if len(full_df) < self.seq_len:
                return {}
                
            seq_df = full_df.iloc[-self.seq_len:]

            # 5. Select Columns for the 'Price' Block
            # This list MUST match the input size of your PyTorch models.
            # Based on your train_tft.py, these are the core features:
            tft_cols = [
                "close", "volume", "ret_1h", 
                "roll_vol_6h", "roll_vol_12h", "roll_vol_24h", "roll_mean_24h"
            ]
            
            # Ensure all columns exist (fill missing with 0)
            for c in tft_cols:
                if c not in seq_df.columns:
                    seq_df[c] = 0.0

            # 6. Convert to Tensors
            # We assume 'price' block covers the market data features.
            price_data = seq_df[tft_cols].values.astype(np.float32)
            price_tensor = torch.from_numpy(price_data) # Shape: [seq_len, num_features]

            # 7. Return the Feature Dictionary
            return {
                # This key 'price' matches what TCN/TFT wrappers will look for
                "price": price_tensor, 
                
                # Pass raw values for other specialized experts if needed
                "options_features": {},
                "macro_onchain_features": {}
            }

        except Exception as e:
            logger.error(f"[FeatureBuilder] Failed to build features for {symbol}: {e}")
            return {}
        finally:
            session.close()