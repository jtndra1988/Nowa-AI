from __future__ import annotations
import json, math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

STATE_DIR = Path("/app/state/portfolio")
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = STATE_DIR / "portfolio.json"

# Allowed values (feel free to expand)
ASSET_CLASSES = ("futures", "options")
STRATEGIES = ("trend_futures", "meanrev_futures", "options_delta", "options_premium")

@dataclass
class Holding:
    symbol: str
    asset_class: str          # "futures" | "options"
    strategy: str             # one of STRATEGIES
    side: str                 # "buy" | "sell"
    notional_usd: float       # signed exposure (+ long, - short)
    qty_contracts: float
    entry_price: float
    opened_at: str

@dataclass
class PortfolioState:
    cash_usd: float = 100_000.0
    equity_usd: float = 100_000.0
    gross_exposure_usd: float = 0.0
    net_exposure_usd: float = 0.0
    # per buckets & symbols
    by_symbol: Dict[str, float] = field(default_factory=dict)          # net exposure
    by_asset: Dict[str, float] = field(default_factory=dict)           # net exposure by asset class
    by_strategy: Dict[str, float] = field(default_factory=dict)        # net exposure by strategy
    open: Dict[str, Holding] = field(default_factory=dict)             # trade_id -> Holding

    ts: str = datetime.now(timezone.utc).isoformat()

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)

    # ---- persistence ----
    @staticmethod
    def load() -> "PortfolioState":
        if STATE_PATH.exists():
            try:
                js = json.load(open(STATE_PATH))
                ps = PortfolioState(
                    cash_usd=js.get("cash_usd", 0.0),
                    equity_usd=js.get("equity_usd", 0.0),
                    gross_exposure_usd=js.get("gross_exposure_usd", 0.0),
                    net_exposure_usd=js.get("net_exposure_usd", 0.0),
                    by_symbol=js.get("by_symbol", {}),
                    by_asset=js.get("by_asset", {}),
                    by_strategy=js.get("by_strategy", {}),
                    open={k: Holding(**v) for k, v in js.get("open", {}).items()},
                    ts=js.get("ts", datetime.now(timezone.utc).isoformat()),
                )
                return ps
            except Exception:
                pass
        return PortfolioState()

    def save(self) -> None:
        self.ts = datetime.now(timezone.utc).isoformat()
        tmp = STATE_PATH.with_suffix(".tmp")
        json.dump(self.snapshot(), open(tmp, "w"))
        tmp.replace(STATE_PATH)

    # ---- mutation ----
    def _bump(self, d: Dict[str, float], key: str, delta: float) -> None:
        d[key] = float(d.get(key, 0.0) + delta)

    def record_open(self, trade_id: str, *, symbol: str, asset_class: str, strategy: str,
                    side: str, notional_usd: float, qty_contracts: float, entry_price: float) -> None:
        signed = notional_usd if side.lower() == "buy" else -notional_usd
        self.open[trade_id] = Holding(
            symbol=symbol, asset_class=asset_class, strategy=strategy, side=side,
            notional_usd=signed, qty_contracts=qty_contracts, entry_price=entry_price,
            opened_at=datetime.now(timezone.utc).isoformat()
        )
        self.gross_exposure_usd += abs(signed)
        self.net_exposure_usd += signed
        self._bump(self.by_symbol, symbol, signed)
        self._bump(self.by_asset, asset_class, signed)
        self._bump(self.by_strategy, strategy, signed)
        self.save()

    def record_close(self, trade_id: str, exit_value_usd: float, pnl_net_usd: float) -> None:
        h = self.open.pop(trade_id, None)
        if not h: return
        signed = h.notional_usd
        self.gross_exposure_usd -= abs(signed)
        self.net_exposure_usd -= signed
        self._bump(self.by_symbol, h.symbol, -signed)
        self._bump(self.by_asset, h.asset_class, -signed)
        self._bump(self.by_strategy, h.strategy, -signed)
        self.cash_usd += pnl_net_usd
        self.equity_usd += pnl_net_usd
        self.save()

    # ---- analytics ----
    def utilization(self, max_leverage: float) -> float:
        if self.equity_usd <= 0: return 1.0
        return min(1.0, self.gross_exposure_usd / (max_leverage * self.equity_usd))

    def symbol_exposure(self, symbol: str) -> float:
        return float(self.by_symbol.get(symbol.upper(), 0.0))
