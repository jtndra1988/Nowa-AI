# app/ml/train_macro_onchain_model.py

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import func

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from app.core.config import settings
from app.db import models
from app.db.database import SessionLocal
from app.ml.feature_builder import join_sentiment_features

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ARTIFACTS_DIR = Path(getattr(settings, "MODEL_ARTIFACTS_DIR", "model_artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MACRO_MODEL_PATH = ARTIFACTS_DIR / "macro_onchain_bias.joblib"

MACRO_MODEL_VERSION = "v1.0"
MACRO_MODEL_ROOT = Path("models") / "macro_onchain"

SENTIMENT_COLS = ["news_score", "social_score", "global_score", "composite_score"]

FEATURE_COLS = [
    "stablecoin_netflow",
    "btc_exchange_reserves_change",
    "dxy_trend",
    "spx_trend",
] + SENTIMENT_COLS


def _load_macro_onchain_dataset(
    session,
    lookback_days: int = 180,
) -> pd.DataFrame:
    """
    Load macro/on-chain metrics, join with MarketData and hourly sentiment
    to build a regression dataset with target_ret.
    """
    logger.info("[MacroOnchainTrain] Loading macro/on-chain metrics from DB...")

    last_ts = session.query(
        func.max(models.MacroOnchainMetrics.timestamp)
    ).scalar()
    if not last_ts:
        raise RuntimeError(
            "[MacroOnchainTrain] No MacroOnchainMetrics rows available; "
            "run your macro/on-chain collector first."
        )

    start_ts = last_ts - timedelta(days=lookback_days)

    macro_rows = (
        session.query(models.MacroOnchainMetrics)
        .filter(models.MacroOnchainMetrics.timestamp >= start_ts)
        .order_by(models.MacroOnchainMetrics.symbol, models.MacroOnchainMetrics.timestamp)
        .all()
    )
    if not macro_rows:
        raise RuntimeError(
            "[MacroOnchainTrain] No macro/on-chain rows found in the selected window."
        )

    macro_records = []
    for r in macro_rows:
        macro_records.append(
            {
                "symbol": r.symbol,
                "timestamp": r.timestamp,
                "stablecoin_netflow": getattr(r, "stablecoin_netflow", None),
                "btc_exchange_reserves_change": getattr(
                    r, "btc_exchange_reserves_change", None
                ),
                "dxy_trend": getattr(r, "dxy_trend", None),
                "spx_trend": getattr(r, "spx_trend", None),
            }
        )

    df_macro = pd.DataFrame(macro_records)
    df_macro["timestamp"] = pd.to_datetime(df_macro["timestamp"], utc=True)

    logger.info(
        "[MacroOnchainTrain] Loaded %d macro/on-chain rows for %d symbols",
        len(df_macro),
        df_macro["symbol"].nunique(),
    )

    # Load MarketData for forward returns
    logger.info("[MacroOnchainTrain] Loading MarketData for return targets...")
    mkt_rows = (
        session.query(models.MarketData)
        .filter(models.MarketData.timestamp >= start_ts)
        .order_by(models.MarketData.symbol, models.MarketData.timestamp)
        .all()
    )
    if not mkt_rows:
        raise RuntimeError("[MacroOnchainTrain] No MarketData rows available for join.")

    mkt_records = []
    for r in mkt_rows:
        mkt_records.append(
            {
                "symbol": r.symbol,
                "timestamp": r.timestamp,
                "close": float(r.close),
            }
        )

    df_mkt = pd.DataFrame(mkt_records)
    df_mkt["timestamp"] = pd.to_datetime(df_mkt["timestamp"], utc=True)

    # Join macro + prices
    df = pd.merge(
        df_macro,
        df_mkt,
        on=["symbol", "timestamp"],
        how="inner",
    )
    if df.empty:
        raise RuntimeError(
            "[MacroOnchainTrain] Join between MacroOnchainMetrics and MarketData is empty."
        )

    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    df["target_ret"] = df.groupby("symbol")["close"].shift(-1) / df["close"] - 1.0

    # === Attach sentiment features ===
    df = join_sentiment_features(
        session=session,
        df=df,
        symbol_col="symbol",
        timestamp_col="timestamp",
        fill_value=0.0,
    )

    # Clean up & drop rows with missing features/targets
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["target_ret"] + FEATURE_COLS).reset_index(drop=True)

    if df.empty:
        raise RuntimeError(
            "[MacroOnchainTrain] No rows with valid target_ret and feature set."
        )

    logger.info(
        "[MacroOnchainTrain] Final training dataset: %d rows, %d symbols",
        len(df),
        df["symbol"].nunique(),
    )

    return df


def _train_val_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological train/val split to respect time ordering.
    """
    X = df[FEATURE_COLS].values.astype(float)
    y = df["target_ret"].values.astype(float)

    n = len(df)
    split = int(n * train_frac)
    if split <= 0 or split >= n:
        raise RuntimeError("[MacroOnchainTrain] Invalid train/val split; not enough data.")

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    return X_train, X_val, y_train, y_val


def main():
    logger.info("[MacroOnchainTrain] ==== Training macro/on-chain bias model (with sentiment) ====")

    session = SessionLocal()
    try:
        df = _load_macro_onchain_dataset(session, lookback_days=180)
    finally:
        session.close()

    X_train, X_val, y_train, y_val = _train_val_split(df, train_frac=0.8)

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        random_state=42,
    )

    logger.info("[MacroOnchainTrain] Fitting model on %d train rows...", len(X_train))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    rmse = float(mean_squared_error(y_val, y_pred, squared=False))
    logger.info("[MacroOnchainTrain] Validation RMSE: %.6f", rmse)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    metadata = {
        "model_name": "macro_onchain_bias",
        "version": MACRO_MODEL_VERSION,
        "trained_at_utc": timestamp,
        "feature_names": FEATURE_COLS,
        "lookback_days": 180,
        "train_val_split": 0.8,
        "model_type": "GradientBoostingRegressor",
        "hyperparameters": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.9,
            "random_state": 42,
        },
        "metrics": {
            "rmse": rmse,
        },
    }

    artifact = {
        "model": model,
        "metadata": metadata,
    }

    joblib.dump(artifact, MACRO_MODEL_PATH)
    logger.info("[MacroOnchainTrain] Saved runtime artifact to %s", MACRO_MODEL_PATH)

    version_dir = MACRO_MODEL_ROOT / f"{MACRO_MODEL_VERSION}_{timestamp}"
    version_dir.mkdir(parents=True, exist_ok=True)

    versioned_model_path = version_dir / f"macro_onchain_bias_{MACRO_MODEL_VERSION}_{timestamp}.joblib"
    joblib.dump(artifact, versioned_model_path)

    metadata_path = version_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info("[MacroOnchainTrain] Saved versioned artifact to %s", versioned_model_path)
    logger.info("[MacroOnchainTrain] Saved metadata to %s", metadata_path)
    logger.info("[MacroOnchainTrain] Training complete.")


if __name__ == "__main__":
    main()
