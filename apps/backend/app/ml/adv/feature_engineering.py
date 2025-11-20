import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import MarketData

logger = logging.getLogger(__name__)

# --- SHARED CONFIGURATION ---
# This is the single source of truth for feature columns.
# Both training (MultiModalTS) and inference (FeatureBuilder) MUST use this.
FEATURE_CONFIG: Dict[str, List[str]] = {
    # Sequential "price" block for TFT / TCN / TST
    "price": [
        "close",
        "volume",
        "ret_1h",
        "roll_vol_6h",
        "roll_vol_12h",
        "roll_vol_24h",
        "roll_mean_24h",
    ],
}


# ---------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------

def process_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core feature engineering logic for price/volume time series.

    Applies:
      - 1h returns
      - Log returns
      - Rolling volatility/mean on multiple windows

    This *must* be used for:
      - Inference (FeatureBuilder)
      - Training (MultiModalTS and any training scripts)
    """
    df = df.copy()

    if "close" not in df.columns:
        # Nothing to do – caller must ensure 'close' exists
        return df

    # 1. Basic Returns
    df["ret_1h"] = df["close"].pct_change().fillna(0.0)
    df["log_ret"] = np.log(df["close"]).diff().fillna(0.0)

    # 2. Rolling Stats (volatility & mean on log returns)
    df["roll_vol_24h"] = df["log_ret"].rolling(24).std().fillna(0.0)
    df["roll_mean_24h"] = df["log_ret"].rolling(24).mean().fillna(0.0)
    df["roll_vol_6h"] = df["log_ret"].rolling(6).std().fillna(0.0)
    df["roll_vol_12h"] = df["log_ret"].rolling(12).std().fillna(0.0)

    return df


def apply_price_feature_config(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard price features and guarantee that all FEATURE_CONFIG['price']
    columns exist (filling missing ones with 0.0).

    This is the *canonical* helper for both training and inference.
    """
    df = process_market_data(df)

    price_cols = FEATURE_CONFIG["price"]
    for col in price_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df


def build_price_sequence_block(df: pd.DataFrame, seq_len: int) -> torch.Tensor:
    """
    Build a [seq_len, num_price_features] tensor from a time-indexed DataFrame
    using the shared FEATURE_CONFIG.

    This is used by:
      - FeatureBuilder (inference)
      - Training code/tests if needed.
    """
    df = apply_price_feature_config(df)

    # Take the last seq_len rows
    seq_df = df.iloc[-seq_len:].copy()
    price_cols = FEATURE_CONFIG["price"]

    price_data = seq_df[price_cols].astype(np.float32).values  # [L, F]
    return torch.from_numpy(price_data)


# ---------------------------------------------------------------------
# Tabular features for XGBoost
# ---------------------------------------------------------------------

def create_tabular_features(
    df: pd.DataFrame,
    base_cols: List[str],
    roll_windows: List[int],
) -> pd.DataFrame:
    """
    Shared logic for XGBoost-style tabular features.

    For each base_col in `base_cols`, we create:
      - <col>_raw
      - <col>_roll_mean_<w>
      - <col>_roll_std_<w>

    Both training (MultiModalTS) and any future inference code MUST call this.
    """
    df = df.copy()
    tabular_df = pd.DataFrame(index=df.index)

    for col in base_cols:
        if col not in df.columns:
            # Be forgiving: skip missing columns
            continue

        # Raw signal
        tabular_df[f"{col}_raw"] = df[col].astype(np.float32)

        # Rolling stats
        for w in roll_windows:
            if w <= 1:
                continue
            roll = df[col].rolling(w)
            tabular_df[f"{col}_roll_mean_{w}"] = roll.mean().astype(np.float32)
            tabular_df[f"{col}_roll_std_{w}"] = roll.std().astype(np.float32)

    # Drop all-NaN columns (e.g. if base col was missing)
    tabular_df = tabular_df.dropna(axis=1, how="all")

    return tabular_df


# ---------------------------------------------------------------------
# Inference-side feature builder (reusing shared preprocessing)
# ---------------------------------------------------------------------

class FeatureBuilder:
    def __init__(self, seq_len: int = 60):
        self.seq_len = seq_len
        # Fetch slightly more rows to have enough history for rolling windows
        self.fetch_limit = seq_len + 100

    async def build_features(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch recent MarketData from DB, apply shared preprocessing and
        return a dictionary of feature blocks used by the hybrid brain.

        Returns (example):
            {
              "price": torch.Tensor [L, F_price],
              "options_features": dict,
              "macro_onchain_features": dict,
            }
        """
        session: Session = SessionLocal()
        try:
            # 1. Fetch raw OHLCV
            rows = (
                session.query(MarketData)
                .filter(MarketData.symbol == symbol.upper())
                .order_by(desc(MarketData.timestamp))
                .limit(self.fetch_limit)
                .all()
            )

            if not rows or len(rows) < self.seq_len:
                logger.warning(
                    "[FeatureBuilder] Not enough rows for %s (have=%s, need=%s).",
                    symbol,
                    len(rows) if rows else 0,
                    self.seq_len,
                )
                return {}

            # 2. Create DataFrame (oldest -> newest)
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

            # 3. Build the shared price sequence block
            price_tensor = build_price_sequence_block(df, self.seq_len)

            return {
                "price": price_tensor,
                "options_features": {},       # filled by options ingestion
                "macro_onchain_features": {}, # filled by macro ingestion
            }

        except Exception as e:
            logger.error("[FeatureBuilder] Error building features for %s: %s", symbol, e, exc_info=True)
            return {}
        finally:
            session.close()
