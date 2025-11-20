# app/ml/etl_options_derived.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional, List

import math
import pandas as pd

from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db import models


# -----------------------------
# Helpers
# -----------------------------

def _parse_ts(value: str | datetime) -> datetime:
    """
    Parse ISO8601 string or datetime into a timezone-aware datetime (UTC).
    """
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_mean(series: pd.Series) -> Optional[float]:
    if series is None or len(series) == 0:
        return None
    s = series.dropna()
    if s.empty:
        return None
    return float(s.mean())


def _safe_ratio(num: float | None, den: float | None) -> Optional[float]:
    if num is None or den is None:
        return None
    if den == 0:
        return None
    return float(num / den)


def _select_spot_price(
    db: Session,
    symbol: str,
    end_ts: datetime,
) -> Optional[float]:
    """
    Try to get a spot/underlying price from FuturesMarketData close to end_ts.
    If none found, return None (GEX metrics will be None).
    """
    q = (
        db.query(models.FuturesMarketData.close)
        .filter(
            models.FuturesMarketData.symbol == symbol,
            models.FuturesMarketData.timestamp <= end_ts,
        )
        .order_by(models.FuturesMarketData.timestamp.desc())
        .limit(1)
    )
    price = q.scalar()
    if price is None:
        return None
    try:
        return float(price)
    except Exception:
        return None


# -----------------------------
# Dataclass for derived metrics
# -----------------------------

@dataclass
class DerivedRow:
    symbol: str
    timestamp: datetime

    total_put_volume: Optional[float] = None
    total_call_volume: Optional[float] = None
    put_call_volume_ratio: Optional[float] = None

    total_put_oi: Optional[float] = None
    total_call_oi: Optional[float] = None
    put_call_oi_ratio: Optional[float] = None

    avg_iv_near_term: Optional[float] = None
    avg_iv_mid_term: Optional[float] = None
    iv_skew_25d: Optional[float] = None

    iv_atm_near: Optional[float] = None
    iv_atm_mid: Optional[float] = None

    iv_term_slope_near_far: Optional[float] = None
    iv_term_slope_reg_logT: Optional[float] = None

    smile_25d_put_iv: Optional[float] = None
    smile_atm_iv: Optional[float] = None
    smile_25d_call_iv: Optional[float] = None
    smile_curvature: Optional[float] = None

    gex_total: Optional[float] = None
    gex_calls: Optional[float] = None
    gex_puts: Optional[float] = None


# -----------------------------
# Core ETL logic
# -----------------------------

def _load_options_chain(
    db: Session,
    symbol: str,
    start_ts: datetime,
    end_ts: datetime,
):
    """
    Load raw options chain rows for `symbol` between start_ts and end_ts.
    """
    return (
        db.query(models.OptionsChain)
        .filter(
            models.OptionsChain.symbol == symbol,
            models.OptionsChain.timestamp >= start_ts,
            models.OptionsChain.timestamp <= end_ts,
        )
        .all()
    )


def _to_dataframe(rows: List[models.OptionsChain]) -> pd.DataFrame:
    """
    Convert list of OptionsChain rows to pandas DataFrame.
    """
    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        records.append(
            {
                "symbol": r.symbol,
                "expiry": r.expiry,
                "strike": r.strike,
                "option_type": (r.option_type or "").upper(),  # 'C' / 'P'
                "timestamp": r.timestamp,
                "bid": r.bid,
                "ask": r.ask,
                "last_price": r.last_price,
                "mark_price": r.mark_price,
                "volume": r.volume,
                "open_interest": r.open_interest,
                "iv": r.iv,
                "delta": r.delta,
                "gamma": r.gamma,
                "vega": r.vega,
                "theta": r.theta,
            }
        )
    df = pd.DataFrame.from_records(records)

    # Ensure datetime types
    for col in ["expiry", "timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Time to expiry in days from "as of" timestamp (we'll use end_ts for that).
    # We'll fill "t_expiry_days" in the main function once we know end_ts.
    return df


def _compute_derived_metrics(
    df: pd.DataFrame,
    symbol: str,
    spot_price: Optional[float],
    end_ts: datetime,
    contract_multiplier: float,
) -> Optional[DerivedRow]:
    """
    Given an options chain DataFrame and context, compute a single DerivedRow.
    """
    if df.empty:
        return None

    # Time to expiry from end_ts (used for bucket classification: near vs mid)
    df = df.copy()
    df["t_expiry_days"] = (df["expiry"] - end_ts).dt.total_seconds() / 86400.0

    # Basic masks
    is_put = df["option_type"] == "P"
    is_call = df["option_type"] == "C"

    # Vol / OI aggregates
    total_put_volume = _safe_mean(df.loc[is_put, "volume"].sum()) if not df.loc[is_put].empty else float(df.loc[is_put, "volume"].sum()) if "volume" in df else None
    total_call_volume = _safe_mean(df.loc[is_call, "volume"].sum()) if not df.loc[is_call].empty else float(df.loc[is_call, "volume"].sum()) if "volume" in df else None

    # But using sum directly is better – override above:
    if "volume" in df.columns:
        total_put_volume = float(df.loc[is_put, "volume"].sum()) if not df.loc[is_put].empty else None
        total_call_volume = float(df.loc[is_call, "volume"].sum()) if not df.loc[is_call].empty else None

    if "open_interest" in df.columns:
        total_put_oi = float(df.loc[is_put, "open_interest"].sum()) if not df.loc[is_put].empty else None
        total_call_oi = float(df.loc[is_call, "open_interest"].sum()) if not df.loc[is_call].empty else None
    else:
        total_put_oi = None
        total_call_oi = None

    put_call_volume_ratio = _safe_ratio(total_put_volume, total_call_volume)
    put_call_oi_ratio = _safe_ratio(total_put_oi, total_call_oi)

    # Term buckets
    near_mask = df["t_expiry_days"].between(0, 7, inclusive="both")
    mid_mask = df["t_expiry_days"].between(7, 30, inclusive="right")

    iv_col = "iv"
    if iv_col not in df.columns:
        df[iv_col] = None

    avg_iv_near_term = _safe_mean(df.loc[near_mask, iv_col])
    avg_iv_mid_term = _safe_mean(df.loc[mid_mask, iv_col])

    # 25d skew: call 25d iv - put 25d iv
    # We approximate 25d as |delta| in [0.15, 0.35].
    if "delta" in df.columns:
        near_25_put = df[
            (df["option_type"] == "P")
            & df["delta"].between(-0.35, -0.15)
        ]
        near_25_call = df[
            (df["option_type"] == "C")
            & df["delta"].between(0.15, 0.35)
        ]
        iv_25d_put = _safe_mean(near_25_put[iv_col])
        iv_25d_call = _safe_mean(near_25_call[iv_col])
        if iv_25d_put is not None and iv_25d_call is not None:
            iv_skew_25d = float(iv_25d_call - iv_25d_put)
        else:
            iv_skew_25d = None
    else:
        iv_skew_25d = None
        iv_25d_put = None
        iv_25d_call = None

    # ATM IV: |delta| <= 0.1
    if "delta" in df.columns:
        atm_mask = df["delta"].between(-0.1, 0.1)
        smile_atm_iv = _safe_mean(df.loc[atm_mask, iv_col])
    else:
        smile_atm_iv = None

    smile_25d_put_iv = iv_25d_put
    smile_25d_call_iv = iv_25d_call

    if smile_25d_put_iv is not None and smile_25d_call_iv is not None and smile_atm_iv is not None:
        smile_curvature = float(
            smile_25d_put_iv + smile_25d_call_iv - 2.0 * smile_atm_iv
        )
    else:
        smile_curvature = None

    # Term structure slopes
    if avg_iv_near_term is not None and avg_iv_mid_term is not None:
        iv_term_slope_near_far = float(avg_iv_mid_term - avg_iv_near_term)
    else:
        iv_term_slope_near_far = None

    # A crude reg on log(T) with just near/mid points.
    if avg_iv_near_term is not None and avg_iv_mid_term is not None:
        # Use midpoints of buckets for T
        T_near = 3.5  # days
        T_mid = 18.5  # days
        try:
            logT_near = math.log(max(T_near, 1e-6))
            logT_mid = math.log(max(T_mid, 1e-6))
            slope = (avg_iv_mid_term - avg_iv_near_term) / (logT_mid - logT_near)
            iv_term_slope_reg_logT = float(slope)
        except Exception:
            iv_term_slope_reg_logT = None
    else:
        iv_term_slope_reg_logT = None

    # GEX: gamma exposure, if spot price is available
    # A very approximate formula: GEX ≈ gamma * S^2 * OI * contract_multiplier
    if spot_price is not None and "gamma" in df.columns and "open_interest" in df.columns:
        df["gex_unit"] = (
            df["gamma"].fillna(0.0)
            * (spot_price ** 2)
            * df["open_interest"].fillna(0.0)
            * float(contract_multiplier)
        )
        gex_calls = float(df.loc[is_call, "gex_unit"].sum())
        gex_puts = float(df.loc[is_put, "gex_unit"].sum())
        gex_total = float(gex_calls + gex_puts)
    else:
        gex_calls = None
        gex_puts = None
        gex_total = None

    # ATM by bucket (optional)
    if "delta" in df.columns:
        atm_near = df[near_mask & df["delta"].between(-0.1, 0.1)]
        atm_mid = df[mid_mask & df["delta"].between(-0.1, 0.1)]
        iv_atm_near = _safe_mean(atm_near[iv_col])
        iv_atm_mid = _safe_mean(atm_mid[iv_col])
    else:
        iv_atm_near = None
        iv_atm_mid = None

    return DerivedRow(
        symbol=symbol,
        timestamp=end_ts,
        total_put_volume=total_put_volume,
        total_call_volume=total_call_volume,
        put_call_volume_ratio=put_call_volume_ratio,
        total_put_oi=total_put_oi,
        total_call_oi=total_call_oi,
        put_call_oi_ratio=put_call_oi_ratio,
        avg_iv_near_term=avg_iv_near_term,
        avg_iv_mid_term=avg_iv_mid_term,
        iv_skew_25d=iv_skew_25d,
        iv_atm_near=iv_atm_near,
        iv_atm_mid=iv_atm_mid,
        iv_term_slope_near_far=iv_term_slope_near_far,
        iv_term_slope_reg_logT=iv_term_slope_reg_logT,
        smile_25d_put_iv=smile_25d_put_iv,
        smile_atm_iv=smile_atm_iv,
        smile_25d_call_iv=smile_25d_call_iv,
        smile_curvature=smile_curvature,
        gex_total=gex_total,
        gex_calls=gex_calls,
        gex_puts=gex_puts,
    )


def _upsert_options_derived_metrics(
    db: Session,
    row: DerivedRow,
) -> None:
    """
    Upsert into options_derived_metrics with (symbol, timestamp) uniqueness.
    If a row exists at that timestamp, update; otherwise, insert.
    """
    existing = (
        db.query(models.OptionsDerivedMetrics)
        .filter(
            models.OptionsDerivedMetrics.symbol == row.symbol,
            models.OptionsDerivedMetrics.timestamp == row.timestamp,
        )
        .one_or_none()
    )

    payload = asdict(row)

    if existing is None:
        obj = models.OptionsDerivedMetrics(**payload)
        db.add(obj)
    else:
        for k, v in payload.items():
            setattr(existing, k, v)


# -----------------------------
# Public entrypoint (used by Celery task)
# -----------------------------

def run(
    symbol: str,
    base_table_for_spot: str = "futures_market_data",
    contract_multiplier: float = 1.0,
    start: str | datetime = None,
    end: str | datetime = None,
) -> int:
    """
    Entry point called by tasks.calculate_options_derived_metrics.

    Parameters:
      symbol:
        Base symbol, e.g. 'BTC', 'ETH'. Must match OptionsChain.symbol.
      base_table_for_spot:
        Currently only "futures_market_data" is used if we compute GEX.
      contract_multiplier:
        Used in GEX calculation: gamma * S^2 * OI * multiplier.
      start, end:
        ISO8601 or datetime bounds for the options window. We aggregate
        a *single* derived row at `timestamp = end`.

    Returns:
      Number of rows written (0 or 1).
    """
    if start is None or end is None:
        raise ValueError("etl_options_derived.run requires both start and end timestamps.")

    start_ts = _parse_ts(start)
    end_ts = _parse_ts(end)

    db: Session = SessionLocal()
    try:
        print(
            f"[Options ETL] Running options-derived ETL for {symbol} "
            f"from {start_ts} to {end_ts}"
        )

        # 1) Load raw options chain
        rows = _load_options_chain(db, symbol=symbol, start_ts=start_ts, end_ts=end_ts)
        if not rows:
            print(f"[Options ETL] No options_chain rows found for {symbol} in window.")
            return 0

        df = _to_dataframe(rows)

        # 2) Spot price (for GEX)
        if base_table_for_spot == "futures_market_data":
            spot_price = _select_spot_price(db, symbol=symbol, end_ts=end_ts)
        else:
            spot_price = None

        # 3) Aggregate metrics
        derived = _compute_derived_metrics(
            df=df,
            symbol=symbol,
            spot_price=spot_price,
            end_ts=end_ts,
            contract_multiplier=contract_multiplier,
        )

        if derived is None:
            print(f"[Options ETL] Dataframe empty after processing for {symbol}.")
            return 0

        # 4) Upsert into options_derived_metrics
        _upsert_options_derived_metrics(db, derived)
        db.commit()

        print(
            f"[Options ETL] Upserted options_derived_metrics row for "
            f"{symbol} @ {derived.timestamp}."
        )
        return 1

    except Exception as e:
        db.rollback()
        print(f"[Options ETL] ERROR while processing {symbol}: {e}")
        raise
    finally:
        db.close()
