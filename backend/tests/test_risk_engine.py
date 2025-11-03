import pandas as pd
import numpy as np
from app.services.execution import RiskEngine, RiskConfig

def _mk_df(n=100, start=100.0):
    # simple upward drift to ensure BUY signals get nonzero size
    rng = np.random.default_rng(0)
    close = np.cumsum(rng.normal(0.1, 0.5, size=n)) + start
    high = close + rng.uniform(0.1, 0.5, size=n)
    low = close - rng.uniform(0.1, 0.5, size=n)
    openp = (close + np.roll(close,1))/2; openp[0]=close[0]
    return pd.DataFrame({"open":openp,"high":high,"low":low,"close":close})

def test_propose_positive_size_when_confident():
    df = _mk_df(200)
    cfg = RiskConfig(account_equity_usd=100000.0, atr_length=14)
    eng = RiskEngine("BTC/USDT", cfg)
    prop = eng.propose_position(
        dfe=df, side="buy", confidence=0.8, regime=1,
        mark_price=float(df.close.iloc[-1]), open_positions_usd=0.0, portfolio_gross_exposure=0.0
    )
    assert prop["qty_usd"] > 0
    assert prop["sl_price"] < prop["tp_price"]

def test_symbol_exposure_cap_blocks():
    df = _mk_df(200)
    cfg = RiskConfig(account_equity_usd=100000.0, atr_length=14, max_symbol_exposure_pct=0.01)  # 1%
    eng = RiskEngine("BTC/USDT", cfg)
    prop = eng.propose_position(
        dfe=df, side="buy", confidence=0.8, regime=1,
        mark_price=float(df.close.iloc[-1]), open_positions_usd=cfg.account_equity_usd*cfg.max_symbol_exposure_pct,
        portfolio_gross_exposure=0.0
    )
    assert prop["qty_usd"] == 0.0
