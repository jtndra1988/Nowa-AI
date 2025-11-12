from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import math

def equity_curve(closed: List[Dict[str, Any]], start_equity: float = 100_000.0):
    eq = [start_equity]
    for tr in closed:
        eq.append(eq[-1] + float(tr["pnl_net_usd"]))
    return np.array(eq)

def sharpe(closed: List[Dict[str, Any]], start_equity: float = 100_000.0, periods_per_year: int = 252):
    eq = equity_curve(closed, start_equity)
    rets = np.diff(eq) / eq[:-1]
    if len(rets) < 2: return 0.0
    mu = np.mean(rets); sd = np.std(rets, ddof=1)
    if sd <= 1e-12: return 0.0
    return (mu / sd) * math.sqrt(periods_per_year)

def sortino(closed: List[Dict[str, Any]], start_equity: float = 100_000.0, periods_per_year: int = 252):
    eq = equity_curve(closed, start_equity)
    rets = np.diff(eq) / eq[:-1]
    if len(rets) < 2: return 0.0
    downside = rets[rets < 0]
    if len(downside) == 0: return 0.0
    mu = np.mean(rets); dd = np.std(downside, ddof=1)
    if dd <= 1e-12: return 0.0
    return (mu / dd) * math.sqrt(periods_per_year)

def max_drawdown(closed: List[Dict[str, Any]], start_equity: float = 100_000.0):
    eq = equity_curve(closed, start_equity)
    peak = -1e18; mdd = 0.0
    for v in eq:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak if peak > 0 else 0.0)
    return mdd

def summary(closed: List[Dict[str, Any]], start_equity: float = 100_000.0) -> Dict[str, Any]:
    wins = sum(1 for t in closed if t["pnl_net_usd"] >= 0)
    losses = len(closed) - wins
    pnl_sum = float(sum(t["pnl_net_usd"] for t in closed))
    wr = (wins / len(closed)) if closed else 0.0
    return {
        "trades": len(closed),
        "win_rate": wr,
        "net_pnl": pnl_sum,
        "sharpe": sharpe(closed, start_equity),
        "sortino": sortino(closed, start_equity),
        "max_drawdown": max_drawdown(closed, start_equity),
        "final_equity": start_equity + pnl_sum,
    }
