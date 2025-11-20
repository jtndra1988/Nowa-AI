# app/ml/train_tft.py

import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import func
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

from app.db.database import SessionLocal
from app.db import models
from app.core.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


ARTIFACTS_DIR = Path(getattr(settings, "MODEL_ARTIFACTS_DIR", "model_artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_PATH = ARTIFACTS_DIR / "tft_model.pkl"


def _load_market_data(session, lookback_days: int = 60) -> pd.DataFrame:
    """
    Load recent OHLCV data from MarketData for all symbols.
    """
    logger.info("[TFT] Loading market data from DB...")
    last_ts = session.query(func.max(models.MarketData.timestamp)).scalar()
    if not last_ts:
        raise RuntimeError("[TFT] No MarketData available in DB")

    start_ts = last_ts - timedelta(days=lookback_days)

    rows = (
        session.query(models.MarketData)
        .filter(models.MarketData.timestamp >= start_ts)
        .order_by(models.MarketData.symbol, models.MarketData.timestamp)
        .all()
    )

    if not rows:
        raise RuntimeError("[TFT] No rows found in MarketData for given window")

    data = [
        {
            "symbol": r.symbol,
            "timestamp": r.timestamp,
            "open": float(r.open),
            "high": float(r.high),
            "low": float(r.low),
            "close": float(r.close),
            "volume": float(r.volume),
        }
        for r in rows
    ]

    df = pd.DataFrame(data)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    logger.info("[TFT] Loaded %d OHLCV rows for %d symbols", len(df), df["symbol"].nunique())
    return df


def _build_features(df: pd.DataFrame, horizon: int = 1) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build time-series features and target:
      - features from past OHLCV stats
      - target = next-period return (close_{t+1} / close_t - 1)
    """
    logger.info("[TFT] Building features...")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    feats = []
    groups = df.groupby("symbol", group_keys=False)

    for sym, g in groups:
        g = g.sort_values("timestamp").copy()
        g["ret_1h"] = g["close"].pct_change()
        g["log_ret"] = np.log(g["close"]).diff()

        g["roll_vol_24h"] = g["log_ret"].rolling(24).std()
        g["roll_mean_24h"] = g["log_ret"].rolling(24).mean()
        g["roll_vol_6h"] = g["log_ret"].rolling(6).std()
        g["roll_vol_12h"] = g["log_ret"].rolling(12).std()

        g["roll_vol_24h"] = g["roll_vol_24h"].fillna(g["roll_vol_24h"].median())
        g["roll_mean_24h"] = g["roll_mean_24h"].fillna(0.0)
        g["roll_vol_6h"] = g["roll_vol_6h"].fillna(g["roll_vol_6h"].median())
        g["roll_vol_12h"] = g["roll_vol_12h"].fillna(g["roll_vol_12h"].median())

        g["target_ret"] = g["close"].shift(-horizon) / g["close"] - 1.0

        feats.append(g)

    df_feat = pd.concat(feats, axis=0).reset_index(drop=True)

    feature_cols = [
        "close",
        "volume",
        "ret_1h",
        "roll_vol_6h",
        "roll_vol_12h",
        "roll_vol_24h",
        "roll_mean_24h",
    ]

    df_feat = df_feat.dropna(subset=feature_cols + ["target_ret"])
    X = df_feat[feature_cols].values.astype(float)
    y = df_feat["target_ret"].values.astype(float)

    logger.info("[TFT] Final training rows: %d", len(df_feat))
    return X, y, feature_cols


def _train_tft_like_model(X: np.ndarray, y: np.ndarray):
    """
    A TFT-like regressor: RandomForest on engineered features.
    """
    if len(X) < 500:
        raise RuntimeError(f"[TFT] Not enough rows to train (got {len(X)}, need >= 500)")

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    logger.info("[TFT] Training RandomForestRegressor as TFT-like model...")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info("[TFT] Validation RMSE: %.6f", rmse)

    return model, {"rmse": float(rmse)}


def main():
    logger.info("[TFT] ==== Training TFT core model ====")
    session = SessionLocal()
    try:
        df = _load_market_data(session, lookback_days=90)
        X, y, feature_cols = _build_features(df)

        model, metrics = _train_tft_like_model(X, y)

        artifact = {
            "model": model,
            "feature_cols": feature_cols,
            "metrics": metrics,
        }
        joblib.dump(artifact, ARTIFACT_PATH)
        logger.info("[TFT] Saved model artifact to %s", ARTIFACT_PATH)
    finally:
        session.close()
    logger.info("[TFT] Training complete.")


if __name__ == "__main__":
    main()
