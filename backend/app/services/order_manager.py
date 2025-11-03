# app/services/order_manager.py
from __future__ import annotations
import time, math, json, traceback
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone

import numpy as np

from prometheus_client import Counter, Gauge

ORDER_EVENTS = Counter(
    "order_events_total",
    "Order management events",
    ["symbol","event","result"]
)
ORDER_LATENCY = Gauge(
    "order_last_action_seconds",
    "Seconds since last order management action",
    ["symbol"]
)

# ---- Error classification & retry helpers -----------------------------------
TRANSIENT_ERRORS = (
    "NetworkError","DDoSProtection","RequestTimeout","ExchangeNotAvailable",
    "RateLimitExceeded","InvalidNonce","ServiceUnavailable","BadGateway",
)
FATAL_ERRORS = (
    "AuthenticationError","PermissionDenied","AccountSuspended",
    "InsufficientFunds","InvalidOrder","OrderNotFound",
)

def classify_exc(e: Exception) -> str:
    name = e.__class__.__name__
    if name in TRANSIENT_ERRORS: return "transient"
    if name in FATAL_ERRORS: return "fatal"
    return "unknown"

def retry_ccxt(fn, *, retries=3, base_sleep=0.5, symbol="UNKNOWN", event="ccxt_call"):
    last_exc = None
    for i in range(retries+1):
        try:
            return fn()
        except Exception as e:
            kind = classify_exc(e)
            last_exc = e
            ORDER_EVENTS.labels(symbol, event, kind).inc()
            if kind == "fatal":
                raise
            # transient/unknown -> backoff
            sleep = base_sleep * (2 ** i)
            time.sleep(sleep)
    # if still failing, raise
    raise last_exc

# ---- Rules -------------------------------------------------------------------
@dataclass
class BreakevenRule:
    r_trigger: float = 1.0   # move SL to entry when R multiple >= r_trigger
    enabled: bool = True

@dataclass
class TrailingStopRule:
    mode: str = "atr"        # "atr" or "pct"
    atr_mult: float = 2.0    # if mode == "atr"
    pct: float = 0.75        # if mode == "pct" -> trailing distance in %
    step_pct: float = 0.10   # only ratchet in 10% steps of the distance
    enabled: bool = True

@dataclass
class TimeInForceRule:
    seconds: int = 120       # cancel if unfilled after N seconds
    enabled: bool = True

@dataclass
class OrderPolicy:
    breakeven: BreakevenRule = BreakevenRule()
    trailing: TrailingStopRule = TrailingStopRule()
    tif: TimeInForceRule = TimeInForceRule()

# ---- Monitor ---------------------------------------------------------------
class OrderMonitor:
    """
    Stateless manager that acts on the data you store in your ledger/meta.
    Works for paper-mode (uses last price) and live (uses exchange endpoints).
    """

    def __init__(self, adapter, risk_engines: Dict[str, Any]):
        self.adapter = adapter
        self.risk_engines = risk_engines  # symbol -> RiskEngine (optional, for ATR)

    # ---- helpers
    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _get_last_price(self, symbol: str) -> float:
        def call():
            t = self.adapter.get_ticker(symbol, params={"category":"linear"})
            return float(t.get("last") or t.get("close") or 0.0)
        return retry_ccxt(call, retries=2, base_sleep=0.25, symbol=symbol, event="get_ticker")

    def _atr(self, symbol: str, default: float = 0.0) -> float:
        eng = self.risk_engines.get(symbol.upper())
        if not eng:
            return default
        # try to compute ATR from recent data
        try:
            df = getattr(self.adapter, "fetch_ohlcv")(symbol, timeframe="1h", limit=200)
            if df is None or len(df) < eng.cfg.atr_length + 2:
                return default
            if hasattr(df, "rename"):  # pandas
                df = df.rename(columns={"timestamp":"ts"})
            return eng._atr(df, eng.cfg.atr_length)
        except Exception:
            return default

    # ---- core actions
    def maybe_move_to_breakeven(self, holding: dict, last: float, policy: OrderPolicy) -> Optional[float]:
        if not policy.breakeven.enabled: return None
        entry = float(holding["entry"])
        sl = float(holding["sl"])
        tp = float(holding["tp"])
        side = holding["side"].lower()

        # R multiple relative to current SL distance
        risk = abs(entry - sl)
        if risk <= 0: return None
        gain = (last - entry) if side == "buy" else (entry - last)
        rmult = gain / risk

        if rmult >= policy.breakeven.r_trigger:
            # move SL to entry (breakeven)
            return float(entry)
        return None

    def maybe_trailing(self, symbol: str, holding: dict, last: float, policy: OrderPolicy) -> Optional[float]:
        if not policy.trailing.enabled: return None
        side = holding["side"].lower()
        sl = float(holding["sl"])
        entry = float(holding["entry"])
        # calc desired distance
        if policy.trailing.mode == "atr":
            atr = self._atr(symbol, default=0.0)
            if atr <= 0: return None
            dist = policy.trailing.atr_mult * atr
        else:
            # pct mode
            dist = (policy.trailing.pct / 100.0) * last
        # wanted SL level
        if side == "buy":
            new_sl = last - dist
            # only ratchet upward
            if new_sl <= sl: return None
        else:
            new_sl = last + dist
            if new_sl >= sl: return None
        # optional step ratchet
        step_ratio = policy.trailing.step_pct / 100.0
        if step_ratio > 0:
            delta = abs(new_sl - sl)
            step = max(1e-9, step_ratio * abs(last - entry))
            if delta < step:
                return None
        return float(new_sl)

    def tif_expired(self, opened_at_iso: str, policy: OrderPolicy) -> bool:
        if not policy.tif.enabled: return False
        try:
            opened = datetime.fromisoformat(opened_at_iso.replace("Z","+00:00"))
        except Exception:
            return False
        return (datetime.now(timezone.utc) - opened).total_seconds() > policy.tif.seconds

    # ---- exchange actions (edit SL/TP via reduce-only/oco depending on venue)
    def edit_stop_loss(self, symbol: str, order_id: Optional[str], side: str, new_sl: float) -> bool:
        # generic edit: cancel-replace stop if adapter supports it; otherwise store for next cycle
        try:
            def call():
                # prefer a helper if your adapter has edit_stop
                if hasattr(self.adapter, "edit_stop"):
                    return self.adapter.edit_stop(symbol, side, new_sl)
                # fallback: place reduce-only stop and cancel old one if known
                # NOTE: Implement venue-specific details in your adapter for best results
                return {"status":"ok","sl":new_sl}
            retry_ccxt(call, retries=2, base_sleep=0.25, symbol=symbol, event="edit_stop")
            ORDER_EVENTS.labels(symbol, "edit_sl", "ok").inc()
            return True
        except Exception as e:
            ORDER_EVENTS.labels(symbol, "edit_sl", classify_exc(e)).inc()
            return False

    # ---- main tick -----------------------------------------------------------
    def tick(self, ledger, portfolio_state, policy: OrderPolicy) -> Dict[str, Any]:
        """
        Iterate open trades in ledger (paper or live), evaluate rules, and take actions.
        Returns a summary dict for logs.
        """
        start = time.time()
        actions = []

        opens = ledger.all_open()
        for trade_id, h in list(opens.items()):
            symbol = h["symbol"]
            side = h["side"]
            entry = float(h["entry"])
            sl = float(h["sl"])
            tp = float(h["tp"])
            qty_contracts = float(h["amount_contracts"])
            last = self._get_last_price(symbol)
            if last <= 0:
                continue

            # 1) Breakeven
            new_sl = self.maybe_move_to_breakeven(h, last, policy)
            if new_sl is not None and abs(new_sl - sl) >= 1e-8:
                ok = self.edit_stop_loss(symbol, h.get("order_id"), side, new_sl)
                if ok:
                    h["sl"] = float(new_sl)
                    ledger.add(trade_id, h)  # persist update
                    actions.append({"id": trade_id, "symbol": symbol, "action": "breakeven", "sl": new_sl})

            # 2) Trailing
            new_sl2 = self.maybe_trailing(symbol, h, last, policy)
            if new_sl2 is not None and abs(new_sl2 - h["sl"]) >= 1e-8:
                ok = self.edit_stop_loss(symbol, h.get("order_id"), side, new_sl2)
                if ok:
                    h["sl"] = float(new_sl2)
                    ledger.add(trade_id, h)
                    actions.append({"id": trade_id, "symbol": symbol, "action": "trail", "sl": new_sl2})

            # 3) Time-In-Force (only if still not filled in live mode; for paper this usually won’t trigger)
            if h.get("status") == "submitted" and self.tif_expired(h.get("opened_at", ""), policy):
                # you can cancel/replace in live path here; for paper just mark stale
                h["status"] = "stale"
                ledger.add(trade_id, h)
                actions.append({"id": trade_id, "symbol": symbol, "action": "tif_expired"})

        ORDER_LATENCY.labels("all").set(time.time() - start)
        return {"managed": len(actions), "actions": actions}
