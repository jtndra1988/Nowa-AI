from dataclasses import dataclass
from typing import Literal, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from .event_bus import set_kv, get_kv, publish

Regime = Literal["bull", "bear", "chop"]
VolState = Literal["low_vol", "mid_vol", "high_vol"]

@dataclass
class RegimeSnapshot:
    regime: Regime
    vol: VolState
    atr_pct: float
    trend_slope: float

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _atr_pct(df: pd.DataFrame, lookback=14) -> float:
    # expects columns: high, low, last_price
    high = df["high"].astype(float)
    low  = df["low"].astype(float)
    close = df["last_price"].astype(float)
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr = tr.rolling(lookback).mean().iloc[-1]
    price = float(close.iloc[-1])
    return float(atr / price) if price else 0.0

def _trend_slope(df: pd.DataFrame, span=50) -> float:
    ema = _ema(df["last_price"].astype(float), span=span)
    # simple slope over last N points (normalized)
    window = 20
    tail = ema.tail(window)
    if len(tail) < 2:
        return 0.0
    x = np.arange(len(tail))
    slope = np.polyfit(x, tail.values, 1)[0]
    return float(slope / (tail.iloc[-1] + 1e-9))

def _assign_regime(trend_slope: float, atr_pct: float, slope_up=+2e-3, slope_dn=-2e-3, hv=0.02) -> Tuple[Regime, VolState]:
    # tune these guardrails as needed
    if atr_pct >= hv:
        vol: VolState = "high_vol"
    elif atr_pct <= hv * 0.4:
        vol = "low_vol"
    else:
        vol = "mid_vol"

    if trend_slope >= slope_up:
        r: Regime = "bull"
    elif trend_slope <= slope_dn:
        r = "bear"
    else:
        r = "chop"
    return r, vol

def detect_and_store(engine_url: str, redis_url: str, symbol: str) -> RegimeSnapshot:
    eng = create_engine(engine_url)
    # pull last 500 1m candles (edit table/SQL to your schema)
    q = text("""
        SELECT ts, last_price, high, low, volume
        FROM ohlc_1m
        WHERE symbol = :sym
        ORDER BY ts DESC
        LIMIT 600
    """)
    df = pd.read_sql(q, eng, params={"sym": symbol})
    df = df.sort_values("ts")

    if len(df) < 60:
        snap = RegimeSnapshot("chop","mid_vol",0.0,0.0)
        set_kv(redis_url, f"regime:{symbol}", snap.__dict__, ttl=120)
        return snap

    atrp = _atr_pct(df, lookback=14)
    slope = _trend_slope(df, span=50)
    regime, vol = _assign_regime(slope, atrp)

    prev = get_kv(redis_url, f"regime:{symbol}")
    snap = RegimeSnapshot(regime, vol, atrp, slope)
    set_kv(redis_url, f"regime:{symbol}", snap.__dict__, ttl=120)

    # fire event if changed
    if not prev or prev.get("regime") != regime or prev.get("vol") != vol:
        publish(redis_url, "market.regime_change", {"symbol": symbol, "regime": regime, "vol": vol, "atr_pct": atrp, "trend_slope": slope})
    return snap
