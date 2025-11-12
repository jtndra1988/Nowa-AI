# app/backtest/cli.py
from __future__ import annotations
import argparse
import pandas as pd

from app.backtest.engine import BacktestEngine
from app.backtest.strategies import BaselineATRBreakout
from app.backtest.trainer import train_segment, TrainConfig
from app.backtest.strategies_ensemble import (
    EnsemblePaths, EnsembleModelAdapter, EnsembleBacktestStrategy, EnsembleSignalConfig
)

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

def wf_training_factory(symbol: str):
    """
    Returns a factory(train_df) that trains LSTM/XGB on the train window,
    writes artifacts to a temp folder, loads them via the ensemble adapter,
    and returns a ready strategy instance for the test window.
    """
    def factory(train_df: pd.DataFrame):
        out_dir = train_segment(
            train_df,
            out_root=None,  # temp folder
            cfg=TrainConfig(
                seq_len=60,
                lstm_epochs=5,
                lstm_batch=128,
                lstm_lr=1e-3,
                xgb_rounds=150,
                xgb_lr=0.05,
                xgb_max_depth=5,
                buy_threshold=0.58,
                sell_threshold=0.58,
                sl_atr_mult=1.8,
                tp_atr_mult=3.0,
                size_factor=1.0,
                w_lstm=0.5,
                w_xgb=0.5,
            ),
        )
        paths = EnsemblePaths(root=out_dir)
        adapter = EnsembleModelAdapter(paths).load()
        cfg = EnsembleSignalConfig(
            buy_threshold=0.58,
            sell_threshold=0.58,
            sl_atr_mult=1.8,
            tp_atr_mult=3.0,
            size_factor=1.0,
            model_profile_buy="trend",
            model_profile_sell="trend",
        )
        return EnsembleBacktestStrategy(adapter, symbol, cfg)
    return factory

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fee_bps", type=float, default=6.0)
    ap.add_argument("--start_equity", type=float, default=100000.0)
    ap.add_argument("--wf", action="store_true", help="Walk-forward test")
    ap.add_argument("--train_bars", type=int, default=20000)
    ap.add_argument("--test_bars", type=int, default=2000)
    ap.add_argument("--baseline", action="store_true", help="Use simple ATR breakout instead of ensemble")
    args = ap.parse_args()

    df = load_csv(args.csv)
    eng = BacktestEngine(args.symbol, fee_bps=args.fee_bps, start_equity=args.start_equity)

    if not args.wf:
        if args.baseline:
            strat = BaselineATRBreakout(sma=50, atr=14, k=1.0, profile="trend")
            res = eng.run_once(df, strat)
        else:
            # Optional: single-run with pre-trained artifacts if you have them
            # from app.backtest.strategies_ensemble import EnsemblePaths, EnsembleModelAdapter, EnsembleBacktestStrategy
            # adapter = EnsembleModelAdapter(EnsemblePaths(root="/app/models/BTCUSDT_1h")).load()
            # res = eng.run_once(df, EnsembleBacktestStrategy(adapter, args.symbol))
            strat = BaselineATRBreakout(sma=50, atr=14, k=1.0, profile="trend")
            res = eng.run_once(df, strat)
    else:
        if args.baseline:
            # Walk-forward with baseline (no training)
            def baseline_factory(_train_df): return BaselineATRBreakout(sma=50, atr=14, k=1.0, profile="trend")
            res = eng.walk_forward(df, baseline_factory, train_bars=args.train_bars, test_bars=args.test_bars)
        else:
            # Walk-forward WITH training per segment (this is what you asked for)
            factory = wf_training_factory(args.symbol)
            res = eng.walk_forward(df, factory, train_bars=args.train_bars, test_bars=args.test_bars)

    print(res)

if __name__ == "__main__":
    main()
