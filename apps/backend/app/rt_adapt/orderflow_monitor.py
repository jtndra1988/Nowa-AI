from typing import Dict, Any
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from .event_bus import set_kv, publish

def _orderbook_imbalance(bids: pd.DataFrame, asks: pd.DataFrame) -> float:
    # expects columns: price, size
    top = 5
    b = (bids.sort_values("price", ascending=False).head(top)["size"]).sum()
    a = (asks.sort_values("price", ascending=True).head(top)["size"]).sum()
    denom = (b + a) + 1e-9
    return float((b - a) / denom)

def _aggression_ratio(trades: pd.DataFrame) -> float:
    # expects columns: side in {"buy","sell"}, size
    buys = trades.loc[trades["side"] == "buy", "size"].sum()
    sells = trades.loc[trades["side"] == "sell", "size"].sum()
    denom = buys + sells + 1e-9
    return float(buys / denom)

def _vpin_proxy(trades: pd.DataFrame, bucket=100):
    # toy VPIN proxy by bucketing by volume
    if trades.empty:
        return 0.0
    t = trades.copy()
    t["signed"] = np.where(t["side"] == "buy", t["size"], -t["size"])
    t["cumv"] = t["size"].cumsum()
    t["bucket"] = (t["cumv"] // bucket).astype(int)
    gb = t.groupby("bucket")["signed"].sum().abs()
    return float(gb.mean() / (bucket + 1e-9))

def scan_orderflow(engine_url: str, redis_url: str, symbol: str) -> Dict[str, Any]:
    eng = create_engine(engine_url)
    # replace with your tables or cache
    bids = pd.read_sql(text("SELECT price,size FROM ob_bids WHERE symbol=:s ORDER BY ts DESC LIMIT 1000"), eng, params={"s": symbol})
    asks = pd.read_sql(text("SELECT price,size FROM ob_asks WHERE symbol=:s ORDER BY ts DESC LIMIT 1000"), eng, params={"s": symbol})
    trades = pd.read_sql(text("SELECT side,size FROM trades WHERE symbol=:s AND ts > NOW() - INTERVAL '5 minutes' ORDER BY ts ASC"), eng, params={"s": symbol})

    obi = _orderbook_imbalance(bids, asks) if len(bids) and len(asks) else 0.0
    ar = _aggression_ratio(trades) if len(trades) else 0.5
    vpin = _vpin_proxy(trades, bucket=200)

    payload = {"symbol": symbol, "obi": obi, "aggr": ar, "vpin": vpin}
    set_kv(redis_url, f"orderflow:{symbol}", payload, ttl=60)

    # fire alerts on extremes
    if abs(obi) > 0.6 or vpin > 0.6:
        publish(redis_url, "market.orderflow_alert", payload)
    return payload
