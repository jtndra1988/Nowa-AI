"""
Build training dataset for DecisionNet from logged HybridSignal + market data.

What it does:
- Loads HybridSignal rows (your hybrid decisions).
- For each executed decision (meta_execute=True, direction != 'flat'):
    - Looks ahead a fixed horizon.
    - Computes realized return.
    - Labels the decision as good (1) or bad (0).
- Extracts features from the debug_payload (experts + context).
- Writes a CSV file suitable for training a direction classifier
  (DecisionNet) that learns when a signal is likely to be correct.

Run:
    python -m backend.scripts.build_decisionnet_dataset

Adjust DB_URL and horizon as needed.
"""

import os
from datetime import timedelta

from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker

import pandas as pd

# --- CONFIG --- #

# Prefer reading from env; fallback to typical local URL
DB_URL = os.getenv("DATABASE_URL", "postgresql://appuser:yoursecurepassword@db:5432/appdb")

# Horizon to evaluate outcome (in minutes)
HORIZON_MINUTES = 60  # e.g. 1h forward return

# Minimum abs(return) to consider meaningful (to avoid noise)
MIN_RETURN_THRESHOLD = 0.0  # set like 0.0005 (0.05%) if you want


# --- DB MODELS IMPORT --- #

from app.db.models import Base, HybridSignal, FuturesMarketData  # adjust if your table name differs

# If you use a different price source for some instruments, you can extend later.


# --- SETUP --- #

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)


def get_forward_return(db, symbol, timestamp, horizon_minutes):
    """
    Compute forward return over [t, t + horizon] using FuturesMarketData.close.

    return = (price_future - price_now) / price_now

    If insufficient data: return None.
    """
    # current price at or just before timestamp
    now_row = (
        db.query(FuturesMarketData)
        .filter(
            FuturesMarketData.symbol == symbol,
            FuturesMarketData.timestamp <= timestamp,
        )
        .order_by(FuturesMarketData.timestamp.desc())
        .first()
    )

    if not now_row or not now_row.close:
        return None

    horizon_ts = timestamp + timedelta(minutes=horizon_minutes)

    fut_row = (
        db.query(FuturesMarketData)
        .filter(
            FuturesMarketData.symbol == symbol,
            FuturesMarketData.timestamp >= horizon_ts,
        )
        .order_by(FuturesMarketData.timestamp.asc())
        .first()
    )

    if not fut_row or not fut_row.close:
        return None

    p0 = float(now_row.close)
    p1 = float(fut_row.close)
    if p0 <= 0:
        return None

    return (p1 - p0) / p0


def label_decision(direction: str, fwd_ret: float) -> int:
    """
    Turn a decision + realized forward return into a binary label.

    - If we went long, it's a hit if fwd_ret > 0.
    - If we went short, it's a hit if fwd_ret < 0.
    - Flat decisions are ignored earlier.
    """
    if direction == "long":
        return int(fwd_ret > 0)
    elif direction == "short":
        return int(fwd_ret < 0)
    else:
        return 0


def extract_features_from_debug(debug: dict) -> dict:
    """
    Extract stable numeric features from HybridDecision.debug payload.

    We avoid using DecisionNet's own score to prevent circular training.
    We focus on:
      - expert predictions
      - meta outputs
      - simple specialists signals
    """
    features = {}

    if not isinstance(debug, dict):
        return features

    expert = debug.get("expert", {})
    meta = debug.get("meta", {})
    specialists = debug.get("specialists", {})

    # Core expert features
    for key in [
        "tft_price",
        "tcn_price",
        "xgb_price",
        "tft_vol",
        "tcn_vol",
        "xgb_vol",
    ]:
        val = expert.get(key)
        if val is not None:
            features[key] = float(val)

    # Meta-ensemble features
    if "p_edge" in meta:
        features["meta_p_edge"] = float(meta["p_edge"])
    if "confidence" in meta:
        features["meta_confidence"] = float(meta["confidence"])

    # Specialists (we can include them as signals; they’re not the model itself)
    if "options_vol_edge" in specialists:
        features["options_vol_edge"] = float(specialists["options_vol_edge"])
    if "macro_onchain_bias" in specialists:
        features["macro_onchain_bias"] = float(specialists["macro_onchain_bias"])

    # You can add more from debug["weights"], etc., if useful

    return features


def build_dataset():
    db = SessionLocal()
    try:
        qs = (
            db.query(HybridSignal)
            .filter(
                HybridSignal.meta_execute == True,
                HybridSignal.direction.in_(("long", "short")),
            )
            .order_by(HybridSignal.created_at.asc())
        )

        rows = []
        total = 0
        used = 0
        hits = 0

        for sig in qs:
            total += 1
            debug = sig.debug_payload or {}

            fwd_ret = get_forward_return(
                db=db,
                symbol=sig.symbol,
                timestamp=sig.created_at,
                horizon_minutes=HORIZON_MINUTES,
            )
            if fwd_ret is None:
                continue

            if abs(fwd_ret) < MIN_RETURN_THRESHOLD:
                # optional: skip tiny, noise-level moves
                continue

            y = label_decision(sig.direction, fwd_ret)

            feats = extract_features_from_debug(debug)
            if not feats:
                continue

            # Add some context fields too (no leakage)
            feats["symbol"] = sig.symbol
            feats["instrument_type"] = sig.instrument_type
            feats["strategy_tag"] = sig.strategy_tag
            feats["direction"] = sig.direction
            feats["fwd_ret"] = float(fwd_ret)
            feats["label"] = int(y)

            rows.append(feats)
            used += 1
            hits += int(y)

        if not rows:
            print("No usable samples found. Check signals, prices, and horizon.")
            return

        df = pd.DataFrame(rows)

        # Basic stats
        overall_hit_rate = df["label"].mean()
        print(f"Total signals scanned: {total}")
        print(f"Samples used for training: {used}")
        print(f"Hit rate (on used samples): {overall_hit_rate:.4f}")

        # Save dataset
        out_path = f"decisionnet_dataset_{HORIZON_MINUTES}m.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved training dataset to: {out_path}")

    finally:
        db.close()


if __name__ == "__main__":
    build_dataset()
