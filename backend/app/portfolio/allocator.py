from __future__ import annotations
from typing import Tuple
from .state import PortfolioState
from .registry import DEFAULT_TARGETS, BUCKET_MAX, PER_SYMBOL_CAP, MAX_LEVERAGE, CASH_BUFFER

class PortfolioAllocator:
    def __init__(self):
        self.state = PortfolioState.load()

    def refresh(self):
        self.state = PortfolioState.load()
        return self.state

    def _bucket_room_usd(self, bucket: str) -> Tuple[float, float]:
        eq = self.state.equity_usd
        target = DEFAULT_TARGETS.get(bucket, 0.0) * eq
        hardcap = BUCKET_MAX.get(bucket, target) * eq
        used = abs(self.state.by_strategy.get(bucket, 0.0))
        return max(0.0, target - used), max(0.0, hardcap - used)

    def _symbol_room_usd(self, symbol: str) -> float:
        eq = self.state.equity_usd
        used = abs(self.state.by_symbol.get(symbol.upper(), 0.0))
        hard = PER_SYMBOL_CAP * eq
        return max(0.0, hard - used)

    def _portfolio_room_usd(self) -> float:
        eq = self.state.equity_usd
        used = self.state.gross_exposure_usd
        hard = MAX_LEVERAGE * eq
        return max(0.0, hard - used)

    def _cash_room_usd(self) -> float:
        eq = self.state.equity_usd
        min_cash = CASH_BUFFER * eq
        # treat "room" as what we can still deploy without violating buffer
        return max(0.0, max(0.0, self.state.cash_usd - min_cash))

    def approve(self, *, symbol: str, asset_class: str, strategy: str, desired_usd: float) -> float:
        """
        Return approved USD notional given desire, considering:
        - bucket target & hard cap
        - per-symbol cap
        - portfolio leverage cap
        - cash buffer
        """
        self.refresh()
        room_target, room_bucket = self._bucket_room_usd(strategy)
        room_symbol = self._symbol_room_usd(symbol)
        room_port  = self._portfolio_room_usd()
        room_cash  = self._cash_room_usd()

        approved = min(desired_usd, room_bucket, room_symbol, room_port, room_cash)
        # be friendly toward target (if under target, allow desired_usd; else clamp to remaining target)
        if room_target <= 0:
            approved = min(approved, room_bucket)
        else:
            approved = min(approved, max(room_target, 0.0))

        return max(0.0, float(approved))

    def on_open(self, trade_id: str, *, symbol: str, asset_class: str, strategy: str,
                side: str, qty_usd: float, qty_contracts: float, entry_price: float) -> None:
        self.state.record_open(
            trade_id, symbol=symbol, asset_class=asset_class, strategy=strategy,
            side=side, notional_usd=qty_usd, qty_contracts=qty_contracts, entry_price=entry_price
        )

    def on_close(self, trade_id: str, *, exit_value_usd: float, pnl_net_usd: float) -> None:
        self.state.record_close(trade_id, exit_value_usd, pnl_net_usd)
