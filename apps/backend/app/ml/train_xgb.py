import logging
from datetime import timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import func
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

from app.db.database import SessionLocal
from app.db import models
from app.core.config import settings
from app.ml.adv.feature_engineering import apply_price_feature_config
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ARTIFACTS_DIR = Path(getattr(settings, "MODEL_ARTIFACTS_DIR", "model_artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_PATH = ARTIFACTS_DIR / "xgb_model.pkl"

# Try real XGBoost if available
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # noqa: BLE001
    XGBRegressor = None


def _load_market_data(session, lookback_days: int = 60) -> pd.DataFrame:
    logger.info("[XGB] Loading market data from DB...")
    last_ts = session.query(func.max(models.MarketData.timestamp)).scalar()
    if not last_ts:
        raise RuntimeError("[XGB] No MarketData available in DB")

    start_ts = last_ts - timedelta(days=lookback_days)
    rows = (
        session.query(models.MarketData)
        .filter(models.MarketData.timestamp >= start_ts)
        .order_by(models.MarketData.symbol, models.MarketData.timestamp)
        .all()
    )
    if not rows:
        raise RuntimeError("[XGB] No rows found in MarketData for given window")

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
    logger.info("[XGB] Loaded %d OHLCV rows for %d symbols", len(df), df["symbol"].nunique())
    return df


def _build_features(
    df: pd.DataFrame,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Build features + target for training.

    Returns:
        X:           np.ndarray [N, F]
        y:           np.ndarray [N]
        feature_cols: list of feature column names
        df_feat:     full feature DataFrame (for debugging / potential reuse)
    """
    logger.info("[XGB] Building features (using apply_price_feature_config)...")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    feats: List[pd.DataFrame] = []
    groups = df.groupby("symbol", group_keys=False)

    for sym, g in groups:
        g = g.sort_values("timestamp").copy()

        # ✅ Shared preprocessing: returns + rolling vols, same as rest of the system
        g = apply_price_feature_config(g)

        # XGB-specific extra returns
        g["ret_4h"] = g["close"].pct_change(4)
        g["ret_12h"] = g["close"].pct_change(12)

        # Ensure rolling vols are finite (they were created by apply_price_feature_config)
        if "roll_vol_12h" in g.columns:
            g["roll_vol_12h"] = g["roll_vol_12h"].fillna(g["roll_vol_12h"].median())
        if "roll_vol_24h" in g.columns:
            g["roll_vol_24h"] = g["roll_vol_24h"].fillna(g["roll_vol_24h"].median())

        # Target: forward return over `horizon`
        g["target_ret"] = g["close"].shift(-horizon) / g["close"] - 1.0

        feats.append(g)

    df_feat = pd.concat(feats, axis=0).reset_index(drop=True)

    feature_cols = [
        "close",
        "volume",
        "ret_1h",
        "ret_4h",
        "ret_12h",
        "roll_vol_12h",
        "roll_vol_24h",
    ]

    # Drop rows that don't have all features or target
    df_feat = df_feat.dropna(subset=feature_cols + ["target_ret"])
    X = df_feat[feature_cols].values.astype(float)
    y = df_feat["target_ret"].values.astype(float)

    logger.info("[XGB] Final training rows: %d", len(df_feat))
    return X, y, feature_cols, df_feat

def _train_xgb_model(X: np.ndarray, y: np.ndarray):
    if len(X) < 500:
        raise RuntimeError(f"[XGB] Not enough rows to train (got {len(X)}, need >= 500)")

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # --- New: tabular scaler for training ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if XGBRegressor is not None:
        logger.info("[XGB] Training real XGBRegressor...")
        model = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
    else:
        logger.info("[XGB] xgboost not installed, falling back to GradientBoostingRegressor...")
        model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_val_scaled)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info("[XGB] Validation RMSE: %.6f", rmse)

    return model, scaler, {"rmse": float(rmse)}


# =========================
# Inference utilities
# =========================

def load_xgb_artifact(artifact_path: Path = ARTIFACT_PATH) -> Dict[str, Any]:
    """
    Load the trained XGB artifact from disk.

    Returns:
        {
          "model":       trained regressor,
          "scaler":      StandardScaler,
          "feature_cols": [...],
          "metrics":     {...}
        }
    """
    if not artifact_path.exists():
        raise FileNotFoundError(f"[XGB] Artifact path not found: {artifact_path}")
    artifact = joblib.load(artifact_path)
    required_keys = {"model", "scaler", "feature_cols"}
    if not required_keys.issubset(artifact.keys()):
        raise RuntimeError(f"[XGB] Artifact missing keys: {required_keys - set(artifact.keys())}")
    return artifact


def build_features_for_inference(
    df_raw: pd.DataFrame,
    feature_cols: List[str],
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Build the SAME tabular features for inference as in training.

    df_raw must contain at least:
        ["symbol", "timestamp", "open", "high", "low", "close", "volume"]

    Returns:
        df_feat: DataFrame with feature_cols and "target_ret"
                 (target_ret will typically be NaN at inference)
    """
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    feats: List[pd.DataFrame] = []
    groups = df.groupby("symbol", group_keys=False)

    for sym, g in groups:
        g = g.sort_values("timestamp").copy()

        # ✅ Same shared preprocessing as training
        g = apply_price_feature_config(g)

        # XGB-specific extra returns (same as training)
        g["ret_4h"] = g["close"].pct_change(4)
        g["ret_12h"] = g["close"].pct_change(12)

        # Reuse existing rolling vols, just ensure finite values
        if "roll_vol_12h" in g.columns:
            g["roll_vol_12h"] = g["roll_vol_12h"].fillna(g["roll_vol_12h"].median())
        if "roll_vol_24h" in g.columns:
            g["roll_vol_24h"] = g["roll_vol_24h"].fillna(g["roll_vol_24h"].median())

        # At inference we usually don't have future prices, but keep the column
        g["target_ret"] = g["close"].shift(-horizon) / g["close"] - 1.0

        feats.append(g)

    df_feat = pd.concat(feats, axis=0).reset_index(drop=True)

    # Don't drop on target_ret; but ensure all required feature_cols are valid.
    df_feat = df_feat.dropna(subset=feature_cols)

    return df_feat


def predict_xgb_from_raw(
    df_raw: pd.DataFrame,
    artifact_path: Path = ARTIFACT_PATH,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    High-level inference entrypoint.

    1. Loads artifact (model + scaler + feature_cols).
    2. Rebuilds tabular features from raw OHLCV (same logic as training).
    3. Applies scaler and runs model.predict().
    4. Returns predictions + the feature DataFrame (aligned by row).

    Returns:
        preds:    np.ndarray [N]
        df_feat:  DataFrame with feature columns and original metadata (symbol, timestamp, etc.)
    """
    artifact = load_xgb_artifact(artifact_path)
    model = artifact["model"]
    scaler: StandardScaler = artifact["scaler"]
    feature_cols: List[str] = artifact["feature_cols"]

    df_feat = build_features_for_inference(df_raw, feature_cols)
    if df_feat.empty:
        raise RuntimeError("[XGB] No valid rows after feature engineering in inference.")

    X = df_feat[feature_cols].values.astype(float)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    return preds, df_feat


# =========================
# Training entrypoint
# =========================

def main():
    logger.info("[XGB] ==== Training XGB core model ====")
    session = SessionLocal()
    try:
        df = _load_market_data(session, lookback_days=90)
        X, y, feature_cols, _ = _build_features(df)

        model, scaler, metrics = _train_xgb_model(X, y)

        artifact = {
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "metrics": metrics,
        }
        joblib.dump(artifact, ARTIFACT_PATH)
        logger.info("[XGB] Saved model artifact to %s", ARTIFACT_PATH)
    finally:
        session.close()
    logger.info("[XGB] Training complete.")


if __name__ == "__main__":
    main()
