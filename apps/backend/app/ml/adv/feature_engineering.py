import logging
import pandas as pd
import numpy as np
import torch
from typing import Any, Dict, List
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import MarketData

logger = logging.getLogger(__name__)

# --- SHARED CONFIGURATION ---
# This is the "Truth" for feature columns. Both Training and Inference MUST use this.
FEATURE_CONFIG = {
    "price": [
        "close", 
        "volume", 
        "ret_1h", 
        "roll_vol_6h", 
        "roll_vol_12h", 
        "roll_vol_24h", 
        "roll_mean_24h"
    ]
}

def create_tabular_features(df: pd.DataFrame, price_cols: List[str], roll_windows: List[int]) -> pd.DataFrame:
    """Shared logic for XGBoost-style tabular features."""
    df = df.copy()
    tabular_df = pd.DataFrame(index=df.index)

    # Add basic lags and rolling stats logic here if needed for XGB
    # For TCN/TFT, we primarily use the sequence data generated below
    return tabular_df

def process_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core feature engineering logic.
    Applies transformations: Returns, Log Returns, Rolling Volatility.
    """
    df = df.copy()
    if "close" not in df.columns:
        return df
        
    # 1. Basic Returns
    df["ret_1h"] = df["close"].pct_change().fillna(0)
    df["log_ret"] = np.log(df["close"]).diff().fillna(0)

    # 2. Rolling Stats (Standardization)
    df["roll_vol_24h"] = df["log_ret"].rolling(24).std().fillna(0)
    df["roll_mean_24h"] = df["log_ret"].rolling(24).mean().fillna(0)
    df["roll_vol_6h"] = df["log_ret"].rolling(6).std().fillna(0)
    df["roll_vol_12h"] = df["log_ret"].rolling(12).std().fillna(0)
    
    return df

class FeatureBuilder:
    def __init__(self, seq_len: int = 60):
        self.seq_len = seq_len
        self.fetch_limit = seq_len + 100 

    async def build_features(self, symbol: str) -> Dict[str, Any]:
        session: Session = SessionLocal()
        try:
            # 1. Fetch Data
            rows = (
                session.query(MarketData)
                .filter(MarketData.symbol == symbol.upper())
                .order_by(desc(MarketData.timestamp))
                .limit(self.fetch_limit)
                .all()
            )

            if not rows or len(rows) < self.seq_len:
                return {}

            # 2. Create DataFrame
            data = [{
                "timestamp": r.timestamp, "open": float(r.open), "high": float(r.high),
                "low": float(r.low), "close": float(r.close), "volume": float(r.volume)
            } for r in rows]
            
            df = pd.DataFrame(data).sort_values("timestamp").reset_index(drop=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)

            # 3. Apply Engineering
            df = process_market_data(df)
            
            # 4. Slice Sequence
            seq_df = df.iloc[-self.seq_len:].copy()

            # 5. Extract "Price" Block using Shared Config
            price_cols = FEATURE_CONFIG["price"]
            for c in price_cols:
                if c not in seq_df.columns:
                    seq_df[c] = 0.0

            price_data = seq_df[price_cols].values.astype(np.float32)
            price_tensor = torch.from_numpy(price_data) # [seq_len, features]

            return {
                "price": price_tensor,
                "options_features": {},
                "macro_onchain_features": {}
            }

        except Exception as e:
            logger.error(f"[FeatureBuilder] Error: {e}")
            return {}
        finally:
            session.close()