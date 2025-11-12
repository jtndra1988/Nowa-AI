import pandas as pd
import numpy as np
from app.backtest.engine import BacktestEngine
from app.backtest.strategies import BaselineATRBreakout

def _series(n=4000, start=100.0):
    rng = np.random.default_rng(0)
    close = np.cumsum(rng.normal(0.02, 0.8, size=n)) + start
    high = close + rng.uniform(0.1, 0.6, size=n)
    low = close - rng.uniform(0.1, 0.6, size=n)
    openp = (close + np.roll(close,1))/2; openp[0]=close[0]
    return pd.DataFrame({"open":openp,"high":high,"low":low,"close":close})

def test_run_once_smoke():
    df = _series(1200)
    eng = BacktestEngine("BTC/USDT", fee_bps=6.0, start_equity=100000.0)
    strat = BaselineATRBreakout()
    res = eng.run_once(df, strat)
    assert "trades" in res and "final_equity" in res

def test_walk_forward_smoke():
    df = _series(5000)
    eng = BacktestEngine("BTC/USDT")
    def factory(train_df): return BaselineATRBreakout()
    res = eng.walk_forward(df, factory, train_bars=2000, test_bars=500)
    assert "segments" in res
