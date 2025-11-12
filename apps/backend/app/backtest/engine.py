from __future__ import annotations
from typing import Dict, Any, List, Tuple, Callable
import pandas as pd
import numpy as np
import uuid

from app.services.execution import RiskEngine, RiskConfig  # reuse your live risk logic
from .broker import SimBroker
from .metrics import summary

class BacktestEngine:
    """
    Drives a Strategy over OHLCV data using your RiskEngine for sizing + ATR SL/TP.
    - Enters at next bar open (after signal)
    - SL/TP evaluated intra-bar via SimBroker
    """
    def __init__(self, symbol: str, fee_bps: float = 6.0, start_equity: float = 100_000.0):
        self.symbol = symbol
        self.broker = SimBroker(taker_fee_bps=fee_bps)
        self.start_equity = float(start_equity)
        self.equity = float(start_equity)
        self.engine = None  # per-run RiskEngine

    def _risk_engine(self, cfg_overrides: Dict[str, Any] | None = None) -> RiskEngine:
        cfg = RiskConfig()
        if cfg_overrides:
            for k, v in cfg_overrides.items():
                setattr(cfg, k, v)
        eng = RiskEngine(self.symbol, cfg)
        return eng

    def run_once(self, df: pd.DataFrame, strategy, cfg_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
        df = df.copy()
        df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)
        assert {"open","high","low","close"}.issubset(set(df.columns)), "df must have OHLC columns"

        eng = self._risk_engine(cfg_overrides)
        self.engine = eng

        # loop bars
        for i in range(len(df)-1):
            # strategy decides (at bar i) and we fill at bar i+1 open
            sig = strategy.on_bar(df, i)
            if sig:
                sig = dict(sig)
                sig["symbol"] = self.symbol
                price = float(df["open"].iloc[i+1])  # next bar open
                # use risk engine to get size + SL/TP
                prop = eng.propose_position(
                    dfe=df.iloc[:i+1], side=sig["action"].lower(), confidence=float(sig.get("confidence", 0.55)),
                    regime=int(sig.get("regime", 0)), mark_price=price,
                    open_positions_usd=0.0, portfolio_gross_exposure=0.0
                )
                if prop.get("qty_usd", 0) > 0:
                    tid = f"{self.symbol.replace('/','_')}-{uuid.uuid4().hex[:10]}"
                    self.broker.open_trade(
                        trade_id=tid, symbol=self.symbol, side="buy" if sig["action"].upper()=="BUY" else "sell",
                        entry_price=price, sl=prop["sl_price"], tp=prop["tp_price"], qty_usd=prop["qty_usd"]
                    )
                    self.broker.mark_opened_at(tid, i+1)

            # step broker on the *current* bar (i+1)
            o = float(df["open"].iloc[i+1]); h = float(df["high"].iloc[i+1]); l=float(df["low"].iloc[i+1]); c=float(df["close"].iloc[i+1])
            self.broker.step_bar(i+1, o, h, l, c)

        return summary(self.broker.closed, self.start_equity)

    def walk_forward(
        self,
        df: pd.DataFrame,
        strategy_factory: Callable[[pd.DataFrame], Any],
        train_bars: int,
        test_bars: int,
        cfg_overrides: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Expanding (or rolling) walk-forward:
        - For each segment, create a new strategy via factory(train_df) to allow fit/calibration.
        - Evaluate on the next test window.
        """
        df = df.copy()
        df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)

        i = train_bars
        all_closed: List[Dict[str, Any]] = []
        while i + test_bars < len(df) - 1:
            train_df = df.iloc[:i].copy()
            test_df  = df.iloc[i - 1 : i + test_bars].copy()  # include overlap bar for open price alignment

            # (user strategy can "fit" on train_df inside the factory)
            strat = strategy_factory(train_df)

            # new broker per segment
            self.broker = SimBroker(taker_fee_bps=self.broker.taker_fee_bps)
            self.engine = self._risk_engine(cfg_overrides)

            # run over test_df
            for j in range(len(test_df)-1):
                sig = strat.on_bar(test_df, j)
                if sig:
                    sig = dict(sig); sig["symbol"] = self.symbol
                    price = float(test_df["open"].iloc[j+1])
                    prop = self.engine.propose_position(
                        dfe=test_df.iloc[:j+1], side=sig["action"].lower(), confidence=float(sig.get("confidence", 0.55)),
                        regime=int(sig.get("regime", 0)), mark_price=price,
                        open_positions_usd=0.0, portfolio_gross_exposure=0.0
                    )
                    if prop.get("qty_usd", 0) > 0:
                        tid = f"{self.symbol.replace('/','_')}-{uuid.uuid4().hex[:10]}"
                        self.broker.open_trade(
                            trade_id=tid, symbol=self.symbol, side="buy" if sig["action"].upper()=="BUY" else "sell",
                            entry_price=price, sl=prop["sl_price"], tp=prop["tp_price"], qty_usd=prop["qty_usd"]
                        )
                        self.broker.mark_opened_at(tid, j+1)

                o=float(test_df["open"].iloc[j+1]); h=float(test_df["high"].iloc[j+1]); l=float(test_df["low"].iloc[j+1]); c=float(test_df["close"].iloc[j+1])
                self.broker.step_bar(j+1, o, h, l, c)

            all_closed.extend(self.broker.closed)
            i += test_bars  # slide forward

        from .metrics import summary as sumf
        return sumf(all_closed, self.start_equity) | {"segments": len(all_closed)}
