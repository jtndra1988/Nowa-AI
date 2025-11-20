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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------
# Paths & versioning (aligned with macro_onchain_model.py)
# ---------------------------------------------------------------------

ARTIFACTS_DIR = Path(getattr(settings, "MODEL_ARTIFACTS_DIR", "model_artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MACRO_MODEL_PATH = ARTIFACTS_DIR / "macro_onchain_bias.joblib"

MACRO_MODEL_VERSION = "v1.0"
MACRO_MODEL_ROOT = Path("models") / "macro_onchain"

FEATURE_COLS = [
    "stablecoin_netflow",
    "btc_exchange_reserves_change",
    "dxy_trend",
    "spx_trend",
]


# ---------------------------------------------------------------------
# Data loading & dataset construction
# ---------------------------------------------------------------------

def _load_macro_onchain_dataset(
    session,
    lookback_days: int = 180,
) -> pd.DataFrame:
    """
    Load macro/on-chain features and join with MarketData to build
    a regression dataset.

    Assumptions (adjust to your actual ORM names):
      - There is a table models.MacroOnchainMetrics (or similar) with fields:
          symbol: str
          timestamp: datetime
          stablecoin_netflow: float
          btc_exchange_reserves_change: float
          dxy_trend: float
          spx_trend: float

      - MarketData has:
          symbol: str
          timestamp: datetime
          close: numeric

    Target:
      - 1-step-ahead return: close_{t+1} / close_t - 1
    """
    logger.info("[MacroOnchainTrain] Loading macro/on-chain metrics from DB...")

    # Latest macro timestamp
    last_ts = session.query(
        func.max(models.MacroOnchainMetrics.timestamp)  # TODO: adjust model name if different
    ).scalar()
    if not last_ts:
        raise RuntimeError(
            "[MacroOnchainTrain] No MacroOnchainMetrics rows available; "
            "run your macro/on-chain collector first."
        )

    start_ts = last_ts - timedelta(days=lookback_days)

    # ---- Load macro/on-chain metrics ----
    macro_rows = (
        session.query(models.MacroOnchainMetrics)  # TODO: adjust model name if different
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

    # ---- Load MarketData for forward returns ----
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

    # ---- Join macro + prices on (symbol, timestamp) ----
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

    # ---- Compute forward 1-step return per symbol ----
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    df["target_ret"] = df.groupby("symbol")["close"].shift(-1) / df["close"] - 1.0

    # Drop rows with missing targets or features
    df = df.dropna(
        subset=["target_ret"] + FEATURE_COLS
    ).reset_index(drop=True)

    if df.empty:
        raise RuntimeError(
            "[MacroOnchainTrain] No rows with valid target_ret and feature set."
        )

    # Replace inf / -inf and clamp extremes if needed
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=FEATURE_COLS + ["target_ret"]).reset_index(drop=True)

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


# ---------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------

def main():
    logger.info("[MacroOnchainTrain] ==== Training macro/on-chain bias model ====")

    session = SessionLocal()
    try:
        df = _load_macro_onchain_dataset(session, lookback_days=180)
    finally:
        session.close()

    X_train, X_val, y_train, y_val = _train_val_split(df, train_frac=0.8)

    # You can tune these hyperparameters; this is a solid starting point.
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

    # ---- Save runtime artifact (used by macro_onchain_model.py) ----
    joblib.dump(artifact, MACRO_MODEL_PATH)
    logger.info("[MacroOnchainTrain] Saved runtime artifact to %s", MACRO_MODEL_PATH)

    # ---- Save versioned artifact + metadata JSON ----
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
