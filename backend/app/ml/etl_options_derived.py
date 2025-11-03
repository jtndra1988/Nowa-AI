# app/ml/etl_options_derived.py
from __future__ import annotations

import os
import math
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# --- Project config (DB URL) ---
try:
    from app.core.config import settings
    DEFAULT_DB_URL = getattr(settings, "SQLALCHEMY_DATABASE_URI", None) or getattr(settings, "DATABASE_URL", None)
except Exception:
    DEFAULT_DB_URL = None


# ------------------------------
# Utility: DB engine
# ------------------------------
def _engine(db_url: Optional[str]):
    url = db_url or DEFAULT_DB_URL or os.environ.get("DATABASE_URL") or os.environ.get("SQLALCHEMY_DATABASE_URI")
    if not url:
        raise RuntimeError("No database URL found. Provide --db-url or set DATABASE_URL / SQLALCHEMY_DATABASE_URI.")
    return create_engine(url)


# ------------------------------
# Black-Scholes gamma (if gamma missing)
# Assumptions:
#  - iv in decimal (e.g., 0.6 for 60%)
#  - r ~ 0 (crypto), dividend q ~ 0
#  - T in years
#  - Returns gamma per unit of underlying (not per % move)
# ------------------------------
def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)

def _bs_gamma(S: np.ndarray, K: np.ndarray, T: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    # numerical safety
    eps = 1e-12
    S = np.maximum(S, eps); K = np.maximum(K, eps); T = np.maximum(T, eps); sigma = np.maximum(sigma, eps)
    d1 = (np.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * np.sqrt(T))
    return _norm_pdf(d1) / (S * sigma * np.sqrt(T))


# ------------------------------
# Bucketing helpers
# ------------------------------
def _expiry_days(now_ts: pd.Timestamp, expiry_ts: pd.Timestamp) -> float:
    return max(0.0, (expiry_ts - now_ts).total_seconds() / 86400.0)

def _bucket_by_ttm_days(ttm_days: float) -> str:
    # near ~7D, mid ~30D, far ~90D
    if ttm_days <= 14: return "near"
    if ttm_days <= 60: return "mid"
    return "far"

def _pick_atm_iv_for_bucket(df_bucket: pd.DataFrame, spot: float) -> Optional[float]:
    # choose option row whose strike is closest to spot (ATM proxy)
    if df_bucket.empty: return None
    df_bucket = df_bucket.dropna(subset=["iv", "strike"])
    if df_bucket.empty: return None
    ix = (df_bucket["strike"] - spot).abs().idxmin()
    return float(df_bucket.loc[ix, "iv"]) if pd.notna(ix) else None

def _pick_delta_iv(df_expiry: pd.DataFrame, target_delta: float, call: bool) -> Optional[float]:
    # If delta column exists, find closest to +0.25 for calls or -0.25 for puts
    if "delta" not in df_expiry.columns or df_expiry["delta"].isna().all():
        return None
    target = target_delta if call else -target_delta
    ix = (df_expiry["delta"] - target).abs().idxmin()
    v = df_expiry.loc[ix, "iv"]
    return float(v) if pd.notna(v) else None

def _pick_moneyness_iv(df_expiry: pd.DataFrame, spot: float, m: float, call: bool) -> Optional[float]:
    # Delta fallback: pick strike closest to moneyness target (m*S)
    if df_expiry.empty: return None
    k_target = m * spot
    ix = (df_expiry["strike"] - k_target).abs().idxmin()
    v = df_expiry.loc[ix, "iv"]
    return float(v) if pd.notna(v) else None


# ------------------------------
# GEX aggregation
#  GEX ≈ sum(gamma_i * OI_i * (S^2) * contract_multiplier)
#  - If gamma missing, compute BS gamma from IV, S, K, T.
#  - Use contract_multiplier=1 by default (override via CLI/env if needed).
#  - Calls/puts summed separately too.
# ------------------------------
def _aggregate_gex(df_ts: pd.DataFrame, spot: float, contract_multiplier: float = 1.0) -> Tuple[float, float, float]:
    if df_ts.empty: return 0.0, 0.0, 0.0

    work = df_ts.copy()
    work["open_interest"] = work["open_interest"].fillna(0.0).astype(float)
    work["iv"] = work["iv"].clip(lower=1e-6).astype(float)

    # time to expiry in years
    T_days = (work["expiry"] - work["timestamp"]).dt.total_seconds() / 86400.0
    work["T_years"] = np.maximum(T_days / 365.25, 1e-6)

    # gamma: from column if available; else compute BS gamma
    if "gamma" in work.columns and not work["gamma"].isna().all():
        gamma = work["gamma"].fillna(0.0).to_numpy(dtype=float)
    else:
        gamma = _bs_gamma(
            S=np.full(len(work), spot, dtype=float),
            K=work["strike"].to_numpy(dtype=float),
            T=work["T_years"].to_numpy(dtype=float),
            sigma=work["iv"].to_numpy(dtype=float),
        )

    # Exposure per contract ~ gamma * OI * S^2 * multiplier
    gex_per = gamma * work["open_interest"].to_numpy(dtype=float) * (spot ** 2) * float(contract_multiplier)

    typ = work["option_type"].astype(str).str.upper().str[0]  # 'C'/'P'
    gex_calls = float(gex_per[typ == "C"].sum())
    gex_puts  = float(gex_per[typ == "P"].sum())
    gex_total = float(gex_per.sum())
    return gex_total, gex_calls, gex_puts


# ------------------------------
# Main ETL
# ------------------------------
def run(symbol: str,
        db_url: Optional[str] = None,
        base_table_for_spot: str = "futures_market_data",
        contract_multiplier: float = 1.0,
        start: Optional[str] = None,
        end: Optional[str] = None):

    eng = _engine(db_url)

    # 1) Load spot series (for ATM/moneyness and GEX S)
    spot_sql = f"""
        SELECT timestamp, close
        FROM {base_table_for_spot}
        WHERE symbol = :sym
        { "AND timestamp >= :start" if start else "" }
        { "AND timestamp <= :end"   if end   else "" }
        ORDER BY timestamp ASC
    """
    params = {"sym": symbol}
    if start: params["start"] = start
    if end:   params["end"]   = end
    spot = pd.read_sql(text(spot_sql), eng, params=params, parse_dates=["timestamp"])
    if spot.empty:
        print(f"[!] No spot/futures close for {symbol} in {base_table_for_spot}.")
        return

    # 2) Load options_chain rows in the same window
    chain_sql = """
        SELECT timestamp, symbol, expiry, strike, option_type,
               bid, ask, last_price, mark_price,
               volume, open_interest,
               iv, delta, gamma
        FROM options_chain
        WHERE symbol = :sym
        {start_clause}
        {end_clause}
        ORDER BY timestamp ASC
    """.format(
        start_clause=("AND timestamp >= :start" if start else ""),
        end_clause=("AND timestamp <= :end" if end else "")
    )
    chain = pd.read_sql(text(chain_sql), eng, params=params, parse_dates=["timestamp","expiry"])
    if chain.empty:
        print(f"[!] No options_chain rows for {symbol} in window.")
        return

    # 3) Align to spot timestamps (left-join nearest or exact)
    #    Here we do an exact join on timestamp; adjust if you want nearest-previous alignment.
    chain_grouped = chain.groupby("timestamp", sort=True)

    rows = []
    for ts, df_ts in chain_grouped:
        # spot at ts
        srow = spot[spot["timestamp"] == ts]
        if srow.empty:
            # you can choose nearest-previous if desired; we skip if no exact match
            continue
        S = float(srow["close"].iloc[0])

        # ---- Buckets for term structure ----
        df_ts = df_ts.copy()
        df_ts["ttm_days"] = (df_ts["expiry"] - df_ts["timestamp"]).dt.total_seconds() / 86400.0
        df_ts = df_ts[df_ts["ttm_days"] > 0].dropna(subset=["iv", "strike"])

        if df_ts.empty:
            continue

        # Split into near/mid/far buckets
        df_ts["bucket"] = df_ts["ttm_days"].map(_bucket_by_ttm_days)

        iv_atm_near = _pick_atm_iv_for_bucket(df_ts[df_ts["bucket"] == "near"], S)
        iv_atm_mid  = _pick_atm_iv_for_bucket(df_ts[df_ts["bucket"] == "mid"],  S)
        iv_atm_far  = _pick_atm_iv_for_bucket(df_ts[df_ts["bucket"] == "far"],  S)

        # Term structure slopes
        slope_nm = (iv_atm_mid - iv_atm_near) if (iv_atm_mid is not None and iv_atm_near is not None) else None
        slope_nf = (iv_atm_far - iv_atm_near) if (iv_atm_far is not None and iv_atm_near is not None) else None

        # Regression slope IV ~ a + b * log(T_days) using (near/mid/far) ATM IVs
        pairs = []
        if iv_atm_near is not None:
            t_near = df_ts.loc[df_ts["bucket"] == "near", "ttm_days"].median()
            if pd.notna(t_near): pairs.append((t_near, iv_atm_near))
        if iv_atm_mid is not None:
            t_mid = df_ts.loc[df_ts["bucket"] == "mid", "ttm_days"].median()
            if pd.notna(t_mid): pairs.append((t_mid, iv_atm_mid))
        if iv_atm_far is not None:
            t_far = df_ts.loc[df_ts["bucket"] == "far", "ttm_days"].median()
            if pd.notna(t_far): pairs.append((t_far, iv_atm_far))

        slope_reg = None
        if len(pairs) >= 2:
            T = np.log(np.array([p[0] for p in pairs], dtype=float))
            Y = np.array([p[1] for p in pairs], dtype=float)
            A = np.vstack([T, np.ones_like(T)]).T
            try:
                b, a = np.linalg.lstsq(A, Y, rcond=None)[0]  # IV = a + b*log(T)
                slope_reg = float(b)
            except Exception:
                slope_reg = None

        # ---- Smile curvature (25d put/atm/25d call) at a "mid" expiry snapshot ----
        # pick the expiry closest to ~30D bucket as the canonical smile expiry
        df_mid = df_ts[df_ts["bucket"] == "mid"]
        smile_p25 = smile_atm = smile_c25 = smile_curv = None
        if not df_mid.empty:
            # first try delta-based; fallback to moneyness 0.8 / 1.0 / 1.2
            p25 = _pick_delta_iv(df_mid, target_delta=0.25, call=False)
            c25 = _pick_delta_iv(df_mid, target_delta=0.25, call=True)
            atm = _pick_atm_iv_for_bucket(df_mid, S)

            if atm is None:
                atm = _pick_moneyness_iv(df_mid, S, m=1.0, call=True)
            if p25 is None:
                p25 = _pick_moneyness_iv(df_mid[df_mid["option_type"].str.upper().str[0] == "P"], S, m=0.8, call=False)
            if c25 is None:
                c25 = _pick_moneyness_iv(df_mid[df_mid["option_type"].str.upper().str[0] == "C"], S, m=1.2, call=True)

            smile_p25 = p25
            smile_atm = atm
            smile_c25 = c25
            if (p25 is not None) and (c25 is not None) and (atm is not None):
                smile_curv = 0.5 * (p25 + c25) - atm  # convexity measure

        # ---- GEX aggregation across all expiries at ts ----
        g_total, g_calls, g_puts = _aggregate_gex(df_ts, spot=S, contract_multiplier=contract_multiplier)

        rows.append({
            "symbol": symbol,
            "timestamp": ts.to_pydatetime(),
            "iv_atm_near": iv_atm_near,
            "iv_atm_mid":  iv_atm_mid,
            "iv_atm_far":  iv_atm_far,
            "iv_term_slope_near_mid": slope_nm,
            "iv_term_slope_near_far": slope_nf,
            "iv_term_slope_reg_logT": slope_reg,
            "smile_25d_put_iv":  smile_p25,
            "smile_atm_iv":      smile_atm,
            "smile_25d_call_iv": smile_c25,
            "smile_curvature":   smile_curv,
            "gex_total": g_total,
            "gex_calls": g_calls,
            "gex_puts":  g_puts,
        })

    if not rows:
        print("[!] Nothing to write.")
        return

    out = pd.DataFrame(rows)

    # 4) Upsert into options_derived_metrics
    #    Use your DB's dialect-specific upsert; here’s a simple delete+insert per batch for portability.
    with eng.begin() as conn:
        # optional: delete existing rows in window for (symbol) to avoid duplicates
        del_sql = """
            DELETE FROM options_derived_metrics
            WHERE symbol = :sym
              AND timestamp >= :start
              AND timestamp <= :end
        """
        if start and end:
            conn.execute(text(del_sql), {"sym": symbol, "start": start, "end": end})

        # bulk insert
        out.to_sql("options_derived_metrics", con=conn, index=False, if_exists="append")
    print(f"[OK] Wrote {len(out)} rows into options_derived_metrics for {symbol}.")
    

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Compute IV term-structure, smile curvature and GEX; upsert into options_derived_metrics.")
    ap.add_argument("--symbol", required=True, help="Underlying symbol (e.g., BTC or BTC/USDT)")
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--base-table-for-spot", default="futures_market_data", choices=["futures_market_data", "market_data"])
    ap.add_argument("--contract-multiplier", type=float, default=1.0)
    ap.add_argument("--start", default=None, help="ISO8601 start time filter")
    ap.add_argument("--end", default=None, help="ISO8601 end time filter")
    args = ap.parse_args()

    run(
        symbol=args.symbol,
        db_url=args.db_url,
        base_table_for_spot=args.base_table_for_spot,
        contract_multiplier=args.contract_multiplier,
        start=args.start, end=args.end
    )
