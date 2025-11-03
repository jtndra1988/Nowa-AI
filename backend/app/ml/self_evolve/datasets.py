from __future__ import annotations
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

try:
    from app.core.config import settings
    DB_URL = settings.SQLALCHEMY_DATABASE_URI
except Exception:
    import os
    DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@db:5432/postgres")

# Reuse rich preprocessor
from app.ml.data_preprocessor import preprocess_raw_chunk


def load_raw_df(symbol: str, table: str = "futures_market_data", limit: int = 200_000) -> pd.DataFrame:
    engine = create_engine(DB_URL)
    if table == "futures_market_data":
        cols = "timestamp, open, high, low, close, volume, symbol"
        sym_param = symbol
    elif table == "market_data":
        cols = "timestamp, open, high, low, last_price AS close, volume, symbol"
        sym_param = symbol.split("/")[0]
    else:
        raise ValueError(f"Unknown table: {table}")
    q = text(f"""
        SELECT {cols}
        FROM {table}
        WHERE symbol = :s
        ORDER BY timestamp ASC
        LIMIT :lim
    """)
    df = pd.read_sql(q, engine, params={"s": sym_param, "lim": limit}, parse_dates=["timestamp"])\
           .dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
    for c in ["open","high","low","volume"]:
        if c not in df.columns: df[c] = df["close"] if c != "volume" else 0.0
    return df


def build_features(symbol: str, table: str = "futures_market_data") -> Tuple[pd.DataFrame, List[str]]:
    engine = create_engine(DB_URL)
    raw = load_raw_df(symbol, table)
    df, feat_cols = preprocess_raw_chunk(raw, symbol, engine=engine, include_mtf=True)
    if "close" not in df.columns and "last_price" in df.columns:
        df = df.rename(columns={"last_price": "close"})
    return df, feat_cols

