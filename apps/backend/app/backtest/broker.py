from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import math

@dataclass
class Fill:
    id: str
    symbol: str
    side: str
    entry: float
    sl: float
    tp: float
    qty_contracts: float
    opened_idx: int

class SimBroker:
    """
    Discrete-time broker using bar data:
    - Enters at next bar's open.
    - Checks SL/TP with intra-bar Hi/Lo priority: hit SL first if both touched.
    - Exit price = SL/TP level.
    - Fees applied on entry and exit (bps).
    """
    def __init__(self, taker_fee_bps: float = 6.0):
        self.taker_fee_bps = float(taker_fee_bps)
        self.open: Dict[str, Fill] = {}
        self.closed: List[Dict[str, Any]] = []

    def _fee(self, notional_usd: float) -> float:
        return notional_usd * (self.taker_fee_bps / 10000.0)

    def open_trade(self, trade_id: str, symbol: str, side: str, entry_price: float,
                   sl: float, tp: float, qty_usd: float) -> None:
        qty_contracts = qty_usd / max(entry_price, 1e-9)
        self.open[trade_id] = Fill(
            id=trade_id, symbol=symbol, side=side, entry=float(entry_price),
            sl=float(sl), tp=float(tp), qty_contracts=float(qty_contracts), opened_idx=-1
        )

    def mark_opened_at(self, trade_id: str, bar_index: int):
        f = self.open.get(trade_id)
        if f:
            f.opened_idx = bar_index

    def step_bar(self, bar_index: int, open_p: float, high: float, low: float, close: float):
        # evaluate SL/TP on each open trade with intra-bar logic
        to_close = []
        for tid, f in list(self.open.items()):
            side = f.side.lower()
            if side == "buy":
                sl_hit = low <= f.sl <= high
                tp_hit = low <= f.tp <= high
                if sl_hit and tp_hit:
                    # assume SL triggers first (conservative)
                    exit_price = f.sl
                    reason = "SL"
                elif sl_hit:
                    exit_price = f.sl
                    reason = "SL"
                elif tp_hit:
                    exit_price = f.tp
                    reason = "TP"
                else:
                    continue
            else:  # sell
                sl_hit = low <= f.sl <= high
                tp_hit = low <= f.tp <= high
                if sl_hit and tp_hit:
                    exit_price = f.sl
                    reason = "SL"
                elif sl_hit:
                    exit_price = f.sl
                    reason = "SL"
                elif tp_hit:
                    exit_price = f.tp
                    reason = "TP"
                else:
                    continue

            # PnL
            if f.side.lower() == "buy":
                pnl = (exit_price - f.entry) * f.qty_contracts
                entry_notional = f.entry * f.qty_contracts
                exit_notional = exit_price * f.qty_contracts
            else:
                pnl = (f.entry - exit_price) * f.qty_contracts
                entry_notional = f.entry * f.qty_contracts
                exit_notional = exit_price * f.qty_contracts

            fees = self._fee(entry_notional) + self._fee(exit_notional)
            pnl_net = pnl - fees
            to_close.append((tid, f, exit_price, reason, pnl_net))

        for tid, f, exit_price, reason, pnl_net in to_close:
            self.closed.append({
                "id": tid, "symbol": f.symbol, "side": f.side, "entry": f.entry,
                "exit": float(exit_price), "reason": reason, "pnl_net_usd": float(pnl_net)
            })
            self.open.pop(tid, None)
