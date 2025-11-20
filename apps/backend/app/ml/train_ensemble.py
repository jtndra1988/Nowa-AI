# app/ml/train_ensemble.py

import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import func
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

from app.db.database import SessionLocal
from app.db import models
from app.core.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


ARTIFACTS_DIR = Path(getattr(settings, "MODEL_ARTIFACTS_DIR", "model_artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TFT_PATH = ARTIFACTS_DIR / "tft_model.pkl"
TCN_PATH = ARTIFACTS_DIR / "tcn_model.pkl"
XGB_PATH = ARTIFACTS_DIR / "xgb_model.pkl"
ENSEMBLE_PATH = ARTIFACTS_DIR / "ensemble_model.pkl"


def _load_market_data(session, lookback_days: int = 60) -> pd.DataFrame:
    logger.info("[ENSEMBLE] Loading market data from DB...")
    last_ts = session.query(func.max(models.MarketData.timestamp)).scalar()
    if not last_ts:
        raise RuntimeError("[ENSEMBLE] No MarketData available in DB")

    start_ts = last_ts - timedelta(days=lookback_days)
    rows = (
        session.query(models.MarketData)
        .filter(models.MarketData.timestamp >= start_ts)
        .order_by(models.MarketData.symbol, models.MarketData.timestamp)
        .all()
    )
    if not rows:
        raise RuntimeError("[ENSEMBLE] No rows found in MarketData for given window")

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

    df = pd.DataFrame(data).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    logger.info("[ENSEMBLE] Loaded %d OHLCV rows for %d symbols", len(df), df["symbol"].nunique())
    return df


def _build_base_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    feats = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("timestamp").copy()
        g["ret_1h"] = g["close"].pct_change()
        g["log_ret"] = np.log(g["close"]).diff()
        g["roll_vol_24h"] = g["log_ret"].rolling(24).std()
        g["roll_vol_24h"] = g["roll_vol_24h"].fillna(g["roll_vol_24h"].median())
        g["target_ret"] = g["close"].shift(-horizon) / g["close"] - 1.0
        feats.append(g)

    df_feat = pd.concat(feats, axis=0).reset_index(drop=True)
    df_feat = df_feat.dropna(subset=["close", "volume", "ret_1h", "roll_vol_24h", "target_ret"])
    return df_feat


def main():
    logger.info("[ENSEMBLE] ==== Training ensemble model ====")
    # Ensure base models exist
    if not TFT_PATH.exists() or not TCN_PATH.exists() or not XGB_PATH.exists():
        raise RuntimeError(
            "[ENSEMBLE] Base model artifacts missing. Train TFT/TCN/XGB first."
        )

    tft_art = joblib.load(TFT_PATH)
    tcn_art = joblib.load(TCN_PATH)
    xgb_art = joblib.load(XGB_PATH)

    tft_model = tft_art["model"]
    tcn_model = tcn_art["model"]
    xgb_model = xgb_art["model"]

    tft_cols = tft_art["feature_cols"]
    xgb_cols = xgb_art["feature_cols"]
    # TCN uses sequence features; here we'll reuse tft/xgb style features for predictions

    session = SessionLocal()
    try:
        df = _load_market_data(session, lookback_days=90)
        df_feat = _build_base_features(df)

        # Build feature matrices for TFT/XGB from same rows
        X_tft = df_feat[tft_cols].values.astype(float)
        X_xgb = df_feat[xgb_cols].values.astype(float)

        # For TCN, we fallback to TFT feature space for ensemble-level predictions
        X_tcn = X_tft

        y = df_feat["target_ret"].values.astype(float)

        logger.info("[ENSEMBLE] Generating base model predictions...")
        p_tft = tft_model.predict(X_tft)
        p_tcn = tcn_model.predict(X_tcn)
        p_xgb = xgb_model.predict(X_xgb)

        # Stack predictions as ensemble features
        X_ens = np.vstack([p_tft, p_tcn, p_xgb]).T

        if len(X_ens) < 500:
            raise RuntimeError(
                f"[ENSEMBLE] Not enough rows to train (got {len(X_ens)}, need >= 500)"
            )

        split = int(len(X_ens) * 0.8)
        X_train, X_val = X_ens[:split], X_ens[split:]
        y_train, y_val = y[:split], y[split:]

        logger.info("[ENSEMBLE] Training LinearRegression stacker...")
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        logger.info("[ENSEMBLE] Validation RMSE: %.6f", rmse)

        artifact = {
            "model": model,
            "base_models": ["tft", "tcn", "xgb"],
            "metrics": {"rmse": float(rmse)},
        }
        joblib.dump(artifact, ENSEMBLE_PATH)
        logger.info("[ENSEMBLE] Saved ensemble artifact to %s", ENSEMBLE_PATH)
    finally:
        session.close()

    logger.info("[ENSEMBLE] Training complete.")


if __name__ == "__main__":
    main()
