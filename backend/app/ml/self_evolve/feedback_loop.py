from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from .utils import out_dir, read_manifest, _save_manifest

try:
    from app.core.config import settings
    DB_URL = settings.SQLALCHEMY_DATABASE_URI
except Exception:
    import os
    DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@db:5432/postgres")


def harvest_trade_kpis(symbol: str) -> Dict[str, Any]:
    """Compute rolling KPIs from your Trades/Positions tables if available; fallback to ledger_json."""
    eng = create_engine(DB_URL)
    kpis = {
        "n_trades": 0,
        "win_rate": None,
        "avg_r": None,
        "sharpe_30d": None,
    }
    try:
        q = text("""
            SELECT timestamp, symbol, side, entry_price, exit_price, pnl
            FROM trades
            WHERE symbol = :s
            ORDER BY timestamp ASC
        """)
        df = pd.read_sql(q, eng, params={"s": symbol})
        if df.empty:
            return kpis
        kpis["n_trades"] = len(df)
        wins = (df["pnl"] > 0).mean()
        kpis["win_rate"] = float(wins)
        # simple R multiple if stop/take columns exist — else fallback
        if {"risk_amt","reward_amt"}.issubset(df.columns):
            kpis["avg_r"] = float((df["reward_amt"] / (df["risk_amt"] + 1e-9)).mean())
        # daily equity curve → sharpe
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        eq = df.groupby("date")["pnl"].sum().cumsum()
        ret = eq.diff().fillna(0.0)
        if len(ret) >= 5:
            kpis["sharpe_30d"] = float(ret.tail(30).mean() / (ret.tail(30).std() + 1e-9) * np.sqrt(365))
        return kpis
    except Exception:
        return kpis


def adaptive_threshold_update(symbol: str, base_cutoff: float = 0.55) -> float:
    """Use KPIs to nudge decision threshold (±0.03) per month.
    Increase cutoff if win_rate < 0.5; decrease if sharpe strong.
    """
    m = read_manifest(symbol)
    k = harvest_trade_kpis(symbol)
    cut = float(m.get("decision_params", {}).get("base_confidence_cutoff", base_cutoff))
    if k.get("win_rate") is not None and k["win_rate"] < 0.5:
        cut = min(cut + 0.02, 0.75)
    if k.get("sharpe_30d") is not None and k["sharpe_30d"] > 1.5:
        cut = max(cut - 0.02, 0.52)
    dp = m.get("decision_params", {}) ; dp["base_confidence_cutoff"] = cut
    _save_manifest(out_dir(symbol, "prod"), decision_params=dp)
    return cut

