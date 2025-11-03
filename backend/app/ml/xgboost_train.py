# app/ml/xgboost_train.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Project settings (DB URL)
try:
    from app.core.config import settings  # type: ignore
    DEFAULT_DB_URL = getattr(settings, "SQLALCHEMY_DATABASE_URI", None) or getattr(settings, "DATABASE_URL", None)
except Exception:
    DEFAULT_DB_URL = None

# FE & artifacts
from app.ml.data_preprocessor import preprocess_raw_chunk
from app.ml.artifacts import ckpt_dir, fpath, write_features

VALIDATION_SPLIT = 0.1


def build_tabular(df: pd.DataFrame, feat_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build tabular X and y using rich features produced by preprocess_raw_chunk.
    Target is next close (set upstream).
    """
    use_cols = list(feat_cols)

    # Optionally fold in Phase-2/3 columns if present
    extra_cols = [
        "vol_5", "vol_20", "vol_60", "hurst_100", "trend_strength",
        "REGIME_SIDE", "REGIME_UP", "REGIME_DOWN", "final_sentiment",
        "bid_ask_imb", "vw_price_skew", "cdv_1m",
    ]
    for c in extra_cols:
        if c in df.columns and c not in use_cols:
            use_cols.append(c)

    X = df[use_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    return X, y, use_cols


def _engine(db_url: Optional[str]) -> any:
    url = db_url or DEFAULT_DB_URL or os.environ.get("DATABASE_URL") or os.environ.get("SQLALCHEMY_DATABASE_URI")
    if not url:
        raise RuntimeError("No database URL found. Provide --db-url or set DATABASE_URL / SQLALCHEMY_DATABASE_URI.")
    return create_engine(url)


def _load_base(engine, table: str, symbol: str, start: Optional[str], end: Optional[str], limit: Optional[int]) -> pd.DataFrame:
    where = ["symbol = :symbol"]
    params = {"symbol": symbol if table == "futures_market_data" else symbol.split("/")[0]}
    if start:
        where.append("timestamp >= :start"); params["start"] = start
    if end:
        where.append("timestamp <= :end"); params["end"] = end

    if table == "futures_market_data":
        sql = f"""
            SELECT timestamp, open, high, low, close, volume, symbol
            FROM futures_market_data
            WHERE {" AND ".join(where)}
            ORDER BY timestamp ASC
        """
    else:
        sql = f"""
            SELECT timestamp, last_price AS close, volume, symbol
            FROM market_data
            WHERE {" AND ".join(where)}
            ORDER BY timestamp ASC
        """

    df = pd.read_sql(text(sql), engine, params=params, parse_dates=["timestamp"])
    if df.empty:
        raise RuntimeError("No data returned for the given filters.")
    if limit and len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser("Train and export XGBoost regression model for next-close.")
    ap.add_argument("--symbol", required=True, help="e.g., BTC/USDT")
    ap.add_argument("--base-table", default="futures_market_data", choices=["futures_market_data", "market_data"])
    ap.add_argument("--start", default=None, help="inclusive ISO8601 timestamp filter")
    ap.add_argument("--end", default=None, help="inclusive ISO8601 timestamp filter")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of rows used (tail)")
    ap.add_argument("--db-url", type=str, default=None, help="override DB URL")

    # XGBoost params
    ap.add_argument("--n-estimators", type=int, default=768)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--eta", type=float, default=0.07)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample", type=float, default=0.8)

    # Artifacts
    ap.add_argument("--output-dir", type=str, default=None, help="If set, write artifacts here; else use checkpoints/<SYMBOL>")

    args = ap.parse_args()

    symbol = args.symbol.upper()
    engine = _engine(args.db_url)

    # 1) Load base data
    df = _load_base(engine, args.base_table, symbol, args.start, args.end, args.limit)

    # 2) Rich features
    df_enriched, feat_cols = preprocess_raw_chunk(df, symbol, engine=engine, include_mtf=True)
    if "close" not in df_enriched.columns:
        if "last_price" in df_enriched.columns:
            df_enriched = df_enriched.rename(columns={"last_price": "close"})
        else:
            raise ValueError("No 'close' or 'last_price' column present after preprocessing.")

    # 3) Target = next close (regression)
    df_enriched["target"] = df_enriched["close"].shift(-1)
    df_enriched = df_enriched.dropna(subset=["target"]).reset_index(drop=True)

    # 4) Build tabular X/y
    X, y, used_cols = build_tabular(df_enriched, feat_cols)

    # 5) Time-based split
    N = len(X)
    split = max(1, int(N * (1.0 - VALIDATION_SPLIT)))
    Xtr, Xva = X[:split], X[split:]
    ytr, yva = y[:split], y[split:]

    # 6) Train XGB
    model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.eta,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        tree_method="hist",
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    # 7) Validate
    if len(Xva) > 0:
        pred = model.predict(Xva)
        rmse = float(np.sqrt(mean_squared_error(yva, pred)))
        print(f"[XGB:{symbol}] val_RMSE={rmse:.6f} (N={len(yva)})")
    else:
        print(f"[XGB:{symbol}] no validation split (dataset too small)")

    # 8) Save artifacts (model + features) — standardized
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_dir(symbol)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model file (canonical)
    xgb_path = fpath(symbol, "xgb") if args.output_dir is None else (out_dir / "xgb_model.json")
    model.save_model(str(xgb_path))

    # Features file (legacy + unified keys)
    xgb_feats_file = out_dir / "xgb_features.json"
    with xgb_feats_file.open("w", encoding="utf-8") as f:
        json.dump({"xgb_features": used_cols, "features": used_cols}, f, indent=2, ensure_ascii=False)

    # Unified features.json (append/merge)
    write_features(out_dir, xgb_features=used_cols)

    print(f"[XGB:{symbol}] Saved model -> {xgb_path}")
    print(f"[XGB:{symbol}] Saved features -> {xgb_feats_file}")
    print(f"[XGB:{symbol}] Updated unified features.json")

if __name__ == "__main__":
    main()
