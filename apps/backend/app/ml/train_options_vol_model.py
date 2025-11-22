# app/ml/train_options_vol_model.py

import json
import logging
from datetime import timedelta, datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sqlalchemy import func
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib

from app.db.database import SessionLocal
from app.db import models
from app.core.config import settings
from app.ml.feature_builder import join_sentiment_features

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ARTIFACTS_DIR = Path(getattr(settings, "MODEL_ARTIFACTS_DIR", "model_artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

OPTIONS_MODEL_PATH = ARTIFACTS_DIR / "options_vol_edge.joblib"

OPTIONS_MODEL_VERSION = "v1.0"
OPTIONS_MODEL_ROOT = Path("models") / "options_vol"

SENTIMENT_COLS = ["news_score", "social_score", "global_score", "composite_score"]

BASE_FEATURE_COLS = ["iv_rank", "risk_reversal_25d", "term_structure_slope"]
FEATURE_COLS = BASE_FEATURE_COLS + SENTIMENT_COLS


def _load_joined_dataset(session, lookback_days: int = 60) -> pd.DataFrame:
    """
    Load OptionsDerivedMetrics and join with MarketData and sentiment
    to build a training dataset for the options vol model.
    """
    logger.info("[OptionsVolTrain] Loading derived metrics from DB...")

    last_ts = session.query(func.max(models.OptionsDerivedMetrics.timestamp)).scalar()
    if not last_ts:
        raise RuntimeError("[OptionsVolTrain] No OptionsDerivedMetrics rows available.")

    start_ts = last_ts - timedelta(days=lookback_days)

    derived_rows = (
        session.query(models.OptionsDerivedMetrics)
        .filter(models.OptionsDerivedMetrics.timestamp >= start_ts)
        .order_by(models.OptionsDerivedMetrics.symbol, models.OptionsDerivedMetrics.timestamp)
        .all()
    )
    if not derived_rows:
        raise RuntimeError("[OptionsVolTrain] No derived metrics in selected window.")

    derived_records = []
    for r in derived_rows:
        derived_records.append(
            {
                "symbol": r.symbol,
                "timestamp": r.timestamp,
                "avg_iv_near_term": getattr(r, "avg_iv_near_term", None),
                "iv_skew_25d": getattr(r, "iv_skew_25d", None),
                "iv_term_slope_near_far": getattr(r, "iv_term_slope_near_far", None),
            }
        )

    df_derived = pd.DataFrame(derived_records)
    df_derived["timestamp"] = pd.to_datetime(df_derived["timestamp"], utc=True)

    logger.info(
        "[OptionsVolTrain] Loaded %d derived rows for %d symbols",
        len(df_derived),
        df_derived["symbol"].nunique(),
    )

    # MarketData for returns
    logger.info("[OptionsVolTrain] Loading market data for return targets...")
    mkt_rows = (
        session.query(models.MarketData)
        .filter(models.MarketData.timestamp >= start_ts)
        .order_by(models.MarketData.symbol, models.MarketData.timestamp)
        .all()
    )
    if not mkt_rows:
        raise RuntimeError("[OptionsVolTrain] No MarketData rows available for join.")

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

    # Join derived metrics with price
    df = pd.merge(
        df_derived,
        df_mkt,
        on=["symbol", "timestamp"],
        how="inner",
    )
    if df.empty:
        raise RuntimeError("[OptionsVolTrain] Join between derived metrics and MarketData is empty.")

    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    df["target_ret"] = df.groupby("symbol")["close"].shift(-1) / df["close"] - 1.0

    df = df.dropna(subset=["target_ret"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("[OptionsVolTrain] No rows with valid forward return target.")

    # Feature engineering to match options_vol_edge() expectations
    iv_near = df["avg_iv_near_term"].astype(float)
    iv_near = iv_near.replace([np.inf, -np.inf], np.nan)
    median_iv = iv_near.median() if not np.isnan(iv_near.median()) else 0.5
    iv_near = iv_near.fillna(median_iv)
    df["iv_rank"] = np.clip(iv_near / 2.0, 0.0, 1.0)

    skew = df["iv_skew_25d"].astype(float)
    skew = skew.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["risk_reversal_25d"] = skew

    term_slope = df["iv_term_slope_near_far"].astype(float)
    term_slope = term_slope.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["term_structure_slope"] = term_slope

    df = df[
        [
            "symbol",
            "timestamp",
            "iv_rank",
            "risk_reversal_25d",
            "term_structure_slope",
            "target_ret",
        ]
    ]

    # === Attach sentiment features ===
    df = join_sentiment_features(
        session=session,
        df=df,
        symbol_col="symbol",
        timestamp_col="timestamp",
        fill_value=0.0,
    )

    logger.info(
        "[OptionsVolTrain] Final training dataset: %d rows, %d symbols",
        len(df),
        df["symbol"].nunique(),
    )

    return df


def _train_val_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple chronological train/val split.
    """
    X = df[FEATURE_COLS].values.astype(float)
    y = df["target_ret"].values.astype(float)

    n = len(df)
    split = int(n * train_frac)
    if split <= 0 or split >= n:
        raise RuntimeError("[OptionsVolTrain] Invalid split; not enough data.")

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    return X_train, X_val, y_train, y_val


def main():
    logger.info("[OptionsVolTrain] ==== Training options volatility model (with sentiment) ====")
    session = SessionLocal()
    try:
        df = _load_joined_dataset(session, lookback_days=90)
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

    logger.info("[OptionsVolTrain] Fitting model on %d train rows...", len(X_train))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    rmse = float(mean_squared_error(y_val, y_pred, squared=False))
    logger.info("[OptionsVolTrain] Validation RMSE: %.6f", rmse)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    metadata = {
        "model_name": "options_vol_edge",
        "version": OPTIONS_MODEL_VERSION,
        "trained_at_utc": timestamp,
        "feature_names": FEATURE_COLS,
        "lookback_days": 90,
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

    joblib.dump(artifact, OPTIONS_MODEL_PATH)
    logger.info("[OptionsVolTrain] Saved runtime artifact to %s", OPTIONS_MODEL_PATH)

    version_dir = OPTIONS_MODEL_ROOT / f"{OPTIONS_MODEL_VERSION}_{timestamp}"
    version_dir.mkdir(parents=True, exist_ok=True)

    versioned_model_path = version_dir / f"options_vol_edge_{OPTIONS_MODEL_VERSION}_{timestamp}.joblib"
    joblib.dump(artifact, versioned_model_path)

    metadata_path = version_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info("[OptionsVolTrain] Saved versioned artifact to %s", versioned_model_path)
    logger.info("[OptionsVolTrain] Saved metadata to %s", metadata_path)
    logger.info("[OptionsVolTrain] Training complete.")


if __name__ == "__main__":
    main()
