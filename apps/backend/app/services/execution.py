# app/services/execution.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram  # metrics

from app.exchange.base import BaseExchangeAdapter
from app.exchange.adapters import BybitAdapter, BinanceDataAdapter
from app.core.config import settings
from app.db.database import get_db
from app.services.settings_service import load_risk_settings, load_learned_state
from app.db.models import RiskLearnedState

# --- Portfolio allocator (NEW) ---
from app.portfolio.allocator import PortfolioAllocator


# ============================================================
# Prometheus (Adaptive) Metrics
# ============================================================
ADAPTED_TRADES = Counter(
    "adaptive_trade_signals_total",
    "Trades executed with/without adaptive params",
    ["symbol", "profile", "used"],  # used in {"yes","no"}
)

ADAPT_CLOSED = Counter(
    "adaptive_closed_trades_total",
    "Closed trades (SL/TP) counted by adaptive usage",
    ["symbol", "profile", "used", "reason", "outcome"],  # outcome in {"win","loss"}
)

ADAPT_PNL = Histogram(
    "adaptive_trade_pnl_usd",
    "PnL distribution (USD) for closed trades",
    buckets=[-2000, -1000, -500, -200, -100, -50, -10, 0, 10, 50, 100, 200, 500, 1000, 2000],
    labelnames=["symbol", "profile", "used"],
)


# ============================================================
# Risk Engine State & Config
# ============================================================

def _state_dir(symbol: str) -> Path:
    p = Path("/app/state/risk") / symbol.replace("/", "_")
    p.mkdir(parents=True, exist_ok=True)
    return p

def _state_path(symbol: str) -> Path:
    return _state_dir(symbol) / "risk_state.json"

@dataclass
class RiskConfig:
    # capital & exposure
    account_equity_usd: float = 100000.0
    max_portfolio_leverage: float = 2.0
    max_symbol_leverage: float = 1.0
    max_symbol_exposure_pct: float = 0.25
    max_concurrent_positions: int = 10

    # volatility targeting
    target_daily_vol: float = 0.01
    min_position_usd: float = 200.0

    # ATR setup
    atr_length: int = 14
    sl_atr_mult_init: float = 1.8
    tp_atr_mult_init: float = 3.0

    # confidence & regime adjustments
    base_confidence_cutoff: float = 0.55
    max_confidence_boost: float = 1.6
    regime_risk_multipliers: Dict[int, float] = field(
        default_factory=lambda: {0: 0.8, 1: 1.1, 2: 1.0}
    )  # {0:chop,1:up,2:down}

    # loss controls
    daily_loss_limit_pct: float = 0.03
    rolling_max_dd_pct: float = 0.15
    per_trade_risk_pct_cap: float = 0.01

    # learning
    ewma_alpha: float = 0.2
    min_trades_to_learn: int = 20
    bandit_lr: float = 0.05

    # guards
    min_tp_sl_ratio: float = 1.3
    min_sl_atr_mult: float = 0.8
    max_sl_atr_mult: float = 4.0
    min_tp_atr_mult: float = 1.2
    max_tp_atr_mult: float = 6.0


@dataclass
class RiskState:
    sl_atr_mult: float
    tp_atr_mult: float

    trades: int = 0
    wins: int = 0
    gross_pnl: float = 0.0
    gross_pnl_after_fees: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    trading_day: str = ""
    day_pnl: float = 0.0
    rolling_equity_peak: float = 100000.0
    rolling_equity_min: float = 100000.0
    last_regime: Optional[int] = None

    def update_after_trade(self, pl: float, fee: float, alpha: float = 0.2):
        self.trades += 1
        pl_net = pl - fee
        self.gross_pnl += pl
        self.gross_pnl_after_fees += pl_net
        if pl_net >= 0:
            self.wins += 1
            self.avg_win = (1 - alpha) * self.avg_win + alpha * pl_net
        else:
            self.avg_loss = (1 - alpha) * self.avg_loss + alpha * abs(pl_net)

    def win_rate(self) -> float:
        return (self.wins / self.trades) if self.trades > 0 else 0.0

    def payoff_ratio(self) -> float:
        if self.avg_loss <= 1e-9:
            return 2.0
        return self.avg_win / self.avg_loss


class RiskEngine:
    """Self-adapting risk engine for a single symbol."""
    def __init__(self, symbol: str, cfg: Optional[RiskConfig] = None):
        self.symbol = symbol.upper()
        self.cfg = cfg or RiskConfig()
        self.state = self._load_state()

    def _load_state(self) -> RiskState:
        p = _state_path(self.symbol)
        if p.exists():
            try:
                js = json.load(open(p))
                js['rolling_equity_peak'] = max(js.get('rolling_equity_peak', 0), self.cfg.account_equity_usd)
                js['rolling_equity_min'] = min(js.get('rolling_equity_min', self.cfg.account_equity_usd), self.cfg.account_equity_usd)
                return RiskState(**js)
            except Exception as e:
                print(f"[!] Failed to load risk state for {self.symbol}: {e}")
        return RiskState(
            sl_atr_mult=self.cfg.sl_atr_mult_init,
            tp_atr_mult=self.cfg.tp_atr_mult_init,
            rolling_equity_peak=self.cfg.account_equity_usd,
            rolling_equity_min=self.cfg.account_equity_usd,
        )

    def _save_state(self):
        try:
            json.dump(asdict(self.state), open(_state_path(self.symbol), "w"))
        except Exception as e:
            print(f"[!] Failed to save risk state for {self.symbol}: {e}")

    @staticmethod
    def _atr(df: pd.DataFrame, length: int) -> float:
        try:
            import pandas_ta as ta
            s = ta.atr(df["high"], df["low"], df["close"], length=length)
            return float(s.iloc[-1])
        except Exception:
            high = df["high"].astype(float).values
            low = df["low"].astype(float).values
            close = df["close"].astype(float).values
            prev_close = np.roll(close, 1); prev_close[0] = close[0]
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)
            tr = np.maximum.reduce([tr1, tr2, tr3])
            atr = pd.Series(tr).rolling(window=length, min_periods=length).mean().iloc[-1]
            return float(atr)

    def _daily_reset_if_needed(self, today: str):
        if self.state.trading_day != today:
            self.state.trading_day = today
            self.state.day_pnl = 0.0

    def check_global_guards(self, current_equity: float) -> Dict[str, Any]:
        self.state.rolling_equity_peak = max(self.state.rolling_equity_peak, current_equity)
        self.state.rolling_equity_min = min(self.state.rolling_equity_min, current_equity)
        dd = 1.0 - (current_equity / max(self.state.rolling_equity_peak, 1e-9))
        if dd >= self.cfg.rolling_max_dd_pct:
            return {"halt": True, "reason": f"Rolling drawdown {dd:.2%} exceeds limit"}
        return {"halt": False}

    def update_day_pnl(self, pl_net: float, today: str) -> Dict[str, Any]:
        self._daily_reset_if_needed(today)
        self.state.day_pnl += pl_net
        if self.state.day_pnl <= -self.cfg.daily_loss_limit_pct * self.cfg.account_equity_usd:
            return {"halt": True, "reason": f"Daily loss limit hit ({self.state.day_pnl:.2f} USD)"}
        return {"halt": False}

    def propose_position(
        self,
        dfe: pd.DataFrame,
        side: str,
        confidence: float,
        regime: int,
        mark_price: float,
        open_positions_usd: float,
        portfolio_gross_exposure: float,
    ) -> Dict[str, Any]:
        side = side.upper()
        if side not in ("BUY", "SELL"):
            return {"qty_usd": 0.0, "reason": "HOLD/no action"}

        max_sym_exposure = self.cfg.max_symbol_exposure_pct * self.cfg.account_equity_usd
        if open_positions_usd >= max_sym_exposure:
            return {"qty_usd": 0.0, "reason": "symbol exposure cap"}
        if portfolio_gross_exposure >= self.cfg.max_portfolio_leverage * self.cfg.account_equity_usd:
            return {"qty_usd": 0.0, "reason": "portfolio leverage cap"}

        atr = self._atr(dfe.tail(max(200, self.cfg.atr_length + 2)), self.cfg.atr_length)
        if atr <= 0 or mark_price <= 0:
            return {"qty_usd": 0.0, "reason": "invalid ATR/price"}

        sl_mult = float(np.clip(self.state.sl_atr_mult, self.cfg.min_sl_atr_mult, self.cfg.max_sl_atr_mult))
        sl_dist = sl_mult * atr
        target_risk_usd = self.cfg.target_daily_vol * self.cfg.account_equity_usd
        base_size_usd = target_risk_usd * mark_price / max(sl_dist, 1e-9)

        max_risk_usd = self.cfg.per_trade_risk_pct_cap * self.cfg.account_equity_usd
        base_size_usd = min(base_size_usd, max_risk_usd * mark_price / max(sl_dist, 1e-9))

        c0 = self.cfg.base_confidence_cutoff
        conf_scale = 1.0 + max(0.0, confidence - c0) * 2.0
        conf_scale = float(np.clip(conf_scale, 0.5, self.cfg.max_confidence_boost))
        reg_scale = float(self.cfg.regime_risk_multipliers.get(regime, 1.0))
        size_usd = base_size_usd * conf_scale * reg_scale

        size_usd = min(size_usd, max_sym_exposure - open_positions_usd)
        size_usd = min(size_usd, self.cfg.max_symbol_leverage * self.cfg.account_equity_usd)
        size_usd = min(size_usd, (self.cfg.max_portfolio_leverage * self.cfg.account_equity_usd) - portfolio_gross_exposure)

        if size_usd < self.cfg.min_position_usd:
            return {"qty_usd": 0.0, "reason": "below min position"}

        tp_mult = float(np.clip(self.state.tp_atr_mult, self.cfg.min_tp_atr_mult, self.cfg.max_tp_atr_mult))
        if side == "BUY":
            sl_price = mark_price - sl_mult * atr
            tp_price = mark_price + tp_mult * atr
        else:
            sl_price = mark_price + sl_mult * atr
            tp_price = mark_price - tp_mult * atr

        rr = (tp_mult / max(sl_mult, 1e-9))
        if rr < self.cfg.min_tp_sl_ratio:
            tp_mult = max(tp_mult, self.cfg.min_tp_sl_ratio * sl_mult)
            if side == "BUY":
                tp_price = mark_price + tp_mult * atr
            else:
                tp_price = mark_price - tp_mult * atr

        return {
            "qty_usd": float(size_usd),
            "sl_price": float(sl_price),
            "tp_price": float(tp_price),
            "atr": float(atr),
            "sl_mult": float(sl_mult),
            "tp_mult": float(tp_mult),
            "conf_scale": float(conf_scale),
            "reg_scale": float(reg_scale),
            "reason": "ok",
        }


class TradeLedger:
    def __init__(self, path: str = "/app/state/risk/ledger.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._open: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._open = json.load(open(self.path))
            except Exception:
                self._open = {}
        else:
            self._open = {}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        json.dump(self._open, open(tmp, "w"))
        tmp.replace(self.path)

    def add(self, trade_id: str, payload: Dict[str, Any]) -> None:
        self._open[trade_id] = payload
        self._save()

    def remove(self, trade_id: str) -> None:
        if trade_id in self._open:
            self._open.pop(trade_id, None)
            self._save()

    def all_open(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._open)


# -------------------------------
# Execution Service (paper mode)
# -------------------------------
class ExecutionService:
    def __init__(self, adapter: Optional[BaseExchangeAdapter] = None, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.ledger = TradeLedger()

        # Load global settings & adapter choice
        db = next(get_db())
        eff = load_risk_settings(db, symbol="GLOBAL")
        adapter_name = (eff.get("default_adapter", "BYBIT") or "BYBIT").upper()
        self.adapter = adapter or (BybitAdapter(paper_mode=True) if adapter_name == "BYBIT" else BinanceDataAdapter())

        self._settings_cache: Dict[str, dict] = {}                 # symbol -> merged settings
        self._learned_cache: Dict[str, RiskLearnedState] = {}      # symbol -> DB learned state
        self.risk_engines: Dict[str, RiskEngine] = {}              # symbol -> engine

        self.account_equity = float(eff.get("account_equity_usd", 100000.0))
        self.portfolio_gross_exposure = 0.0

        # --- Unified portfolio allocator (NEW) ---
        self.portfolio = PortfolioAllocator()

        print(f"[*] ExecutionService: Paper={self.paper_mode} Adapter={adapter_name}")

    # ------- settings/state helpers -------
    def _get_effective_settings(self, symbol: str) -> dict:
        sym = symbol.upper()
        if sym not in self._settings_cache:
            db = next(get_db())
            self._settings_cache[sym] = load_risk_settings(db, sym)
        return self._settings_cache[sym]

    def _get_learned(self, symbol: str) -> RiskLearnedState:
        sym = symbol.upper()
        if sym not in self._learned_cache:
            db = next(get_db())
            self._learned_cache[sym] = load_learned_state(db, sym)
        return self._learned_cache[sym]

    # ------- engine -------
    def _get_risk_engine(self, symbol: str) -> RiskEngine:
        sym = symbol.upper()
        if sym not in self.risk_engines:
            cfg_dict = self._get_effective_settings(sym)
            cfg = RiskConfig(
                account_equity_usd=cfg_dict["account_equity_usd"],
                max_portfolio_leverage=cfg_dict["max_portfolio_leverage"],
                max_symbol_leverage=cfg_dict.get("max_symbol_leverage", 1.0),
                max_symbol_exposure_pct=cfg_dict.get("max_symbol_exposure_pct", 0.25),
                max_concurrent_positions=cfg_dict["max_concurrent_positions"],
                target_daily_vol=cfg_dict["target_daily_vol"],
                min_position_usd=cfg_dict["min_position_usd"],
                atr_length=cfg_dict["atr_length"],
                sl_atr_mult_init=cfg_dict["sl_atr_mult_init"],
                tp_atr_mult_init=cfg_dict["tp_atr_mult_init"],
                base_confidence_cutoff=cfg_dict["base_confidence_cutoff"],
                max_confidence_boost=cfg_dict["max_confidence_boost"],
                regime_risk_multipliers={int(k): float(v) for k, v in cfg_dict["regime_risk_multipliers"].items()},
                daily_loss_limit_pct=cfg_dict["daily_loss_limit_pct"],
                rolling_max_dd_pct=cfg_dict["rolling_max_dd_pct"],
                per_trade_risk_pct_cap=cfg_dict["per_trade_risk_pct_cap"],
                ewma_alpha=cfg_dict["ewma_alpha"],
                min_trades_to_learn=cfg_dict["min_trades_to_learn"],
                bandit_lr=cfg_dict["bandit_lr"],
                min_tp_sl_ratio=cfg_dict["min_tp_sl_ratio"],
                min_sl_atr_mult=cfg_dict["min_sl_atr_mult"],
                max_sl_atr_mult=cfg_dict["max_sl_atr_mult"],
                min_tp_atr_mult=cfg_dict["min_tp_atr_mult"],
                max_tp_atr_mult=cfg_dict["max_tp_atr_mult"],
            )
            eng = RiskEngine(sym, cfg)
            st = self._get_learned(sym)
            # hydrate engine from DB learned state
            eng.state.sl_atr_mult = st.sl_atr_mult
            eng.state.tp_atr_mult = st.tp_atr_mult
            eng.state.trades = st.trades
            eng.state.wins = st.wins
            eng.state.gross_pnl = st.gross_pnl
            eng.state.gross_pnl_after_fees = st.gross_pnl_after_fees
            eng.state.avg_win = st.avg_win
            eng.state.avg_loss = st.avg_loss
            eng.state.trading_day = st.trading_day
            eng.state.day_pnl = st.day_pnl
            eng.state.rolling_equity_peak = st.rolling_equity_peak
            eng.state.rolling_equity_min = st.rolling_equity_min
            eng.state.last_regime = st.last_regime
            self.risk_engines[sym] = eng
        return self.risk_engines[sym]

    # ------- data helpers -------
    def _fetch_ohlcv_for_atr(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> pd.DataFrame:
        """Prefer adapter.fetch_ohlcv; fall back to raw ccxt if needed."""
        if hasattr(self.adapter, "fetch_ohlcv"):
            df = self.adapter.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df.rename(columns={"timestamp": "ts"})
        ex = getattr(self.adapter, "exchange", None)
        if ex is None:
            return pd.DataFrame()
        try:
            raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not raw:
                return pd.DataFrame()
            df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            return df
        except Exception as e:
            print(f"[!] _fetch_ohlcv_for_atr failed for {symbol}: {e}")
            return pd.DataFrame()

    # ------- misc -------
    def _new_trade_id(self, symbol: str) -> str:
        return f"{symbol.replace('/','_')}-{int(time.time()*1000)}"

    def _estimate_fee_usd(self, notional_usd: float, symbol: Optional[str] = None) -> float:
        bps = 6.0
        if symbol:
            bps = float(self._get_effective_settings(symbol).get("taker_fee_bps", 6.0))
        return notional_usd * (bps / 10000.0)

    def _record_open(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        filled_price: float,
        amount_contracts: float,
        amount_usd: float,
        sl: float,
        tp: float,
        meta: Optional[Dict[str, Any]] = None,  # meta kept for adaptive info
    ) -> None:
        self.ledger.add(
            trade_id,
            {
                "symbol": symbol,
                "side": side,
                "entry": float(filled_price),
                "amount_contracts": float(amount_contracts),
                "amount_usd": float(amount_usd),
                "sl": float(sl),
                "tp": float(tp),
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "meta": meta or {},
            },
        )
        self.portfolio_gross_exposure += amount_usd

    # ------- public API -------
    async def execute_trade_signal_async(self, signal: dict) -> Optional[Dict[str, Any]]:
        """Async alias to avoid shadowing the sync method name."""
        return self.execute_trade_signal_sync(signal)

    def execute_trade_signal(self, signal: dict) -> Optional[Dict[str, Any]]:
        return self.execute_trade_signal_sync(signal)

    def execute_trade_signal_sync(self, signal: dict) -> Optional[Dict[str, Any]]:
        symbol = signal.get("symbol")
        action = (signal.get("action") or signal.get("side") or "").lower()
        price = signal.get("price")
        confidence = float(signal.get("confidence", 0.55))
        regime = int(signal.get("regime", 0))

        if not symbol or action not in ("buy", "sell"):
            print("[!] ExecutionService: Invalid signal fields.")
            return None

        if price is None or price <= 0:
            try:
                ticker = self.adapter.get_ticker(symbol, params={"category": "linear"})
                price = float(ticker.get("last") or ticker.get("close") or 0.0)
            except Exception:
                price = 0.0
        if price <= 0:
            print(f"[!] ExecutionService: Could not resolve price for {symbol}; skipping.")
            return None

        eng = self._get_risk_engine(symbol)
        guards = eng.check_global_guards(self.account_equity)
        if guards.get("halt"):
            print(f"[!] HALT: {guards.get('reason')}")
            return None

        dfe = self._fetch_ohlcv_for_atr(symbol, "1h", 200)
        if dfe.empty or len(dfe) < eng.cfg.atr_length + 2:
            print(f"[!] Not enough data for ATR calc on {symbol}; skipping.")
            return None

        open_pos_usd = sum(
            t.get("amount_usd", 0.0)
            for t in self.ledger.all_open().values()
            if t.get("symbol") == symbol
        )

        # --- Adaptive parameter overrides (per-trade, temporary) ---
        dyn_size_factor = float(signal.get("size_factor", 1.0))
        dyn_sl_mult = float(signal.get("sl_atr_mult", 0.0))
        dyn_tp_mult = float(signal.get("tp_atr_mult", 0.0))
        model_profile = signal.get("model_profile", "default")

        _prev_sl, _prev_tp = eng.state.sl_atr_mult, eng.state.tp_atr_mult
        try:
            if dyn_sl_mult > 0:
                eng.state.sl_atr_mult = float(np.clip(
                    dyn_sl_mult, eng.cfg.min_sl_atr_mult, eng.cfg.max_sl_atr_mult
                ))
            if dyn_tp_mult > 0:
                eng.state.tp_atr_mult = float(np.clip(
                    dyn_tp_mult, eng.cfg.min_tp_atr_mult, eng.cfg.max_tp_atr_mult
                ))

            # --- Compute position proposal ---
            proposal = eng.propose_position(
                dfe=dfe.rename(columns={"ts": "timestamp"}) if "ts" in dfe.columns else dfe,
                side=action,
                confidence=confidence,
                regime=regime,
                mark_price=float(price),
                open_positions_usd=open_pos_usd,
                portfolio_gross_exposure=self.portfolio_gross_exposure,
            )
        finally:
            # Restore engine state so global learning baseline is not polluted
            eng.state.sl_atr_mult, eng.state.tp_atr_mult = _prev_sl, _prev_tp

        # --- Apply dynamic size factor safely ---
        if proposal.get("qty_usd", 0.0) > 0:
            proposal["qty_usd"] *= max(0.1, min(dyn_size_factor, 2.5))
            base_reason = proposal.get("reason", "ok")
            proposal["reason"] = f"{base_reason} | dyn_size_factor={dyn_size_factor:.2f}"
        else:
            print(f"[*] RiskEngine: HOLD for {symbol}. Reason: {proposal.get('reason')}")
            return None

        # --- Portfolio allocation clamp (strategy buckets, per-symbol cap, leverage & cash buffer) ---
        asset_class = "futures"  # current path is futures; for options path set "options" accordingly

        # Map model_profile to portfolio strategy bucket
        profile = model_profile or "default"
        if profile == "trend":
            strategy = "trend_futures"
        elif profile == "meanrev":
            strategy = "meanrev_futures"
        else:
            strategy = "trend_futures"  # default bucket, adjust if you have more profiles

        desired_usd = float(proposal["qty_usd"])
        approved_usd = self.portfolio.approve(
            symbol=symbol,
            asset_class=asset_class,
            strategy=strategy,
            desired_usd=desired_usd,
        )

        if approved_usd <= 0:
            print(f"[*] PortfolioAllocator: HOLD for {symbol}. No capital available for {strategy}.")
            return None

        # If allocator granted less than desired, scale down notional here.
        if approved_usd < desired_usd:
            scale = approved_usd / max(desired_usd, 1e-9)
            proposal["qty_usd"] = approved_usd

        qty_usd = proposal["qty_usd"]
        sl_price = proposal["sl_price"]
        tp_price = proposal["tp_price"]
        amount_contracts = qty_usd / float(price)

        # Was adaptive used?
        adaptive_used = (
            (abs(dyn_size_factor - 1.0) > 1e-9) or (dyn_sl_mult > 0) or (dyn_tp_mult > 0) or (model_profile != "default")
        )
        used_label = "yes" if adaptive_used else "no"
        try:
            ADAPTED_TRADES.labels(symbol, model_profile, used_label).inc()
        except Exception:
            pass

        trade_id = self._new_trade_id(symbol)
        order = {
            "symbol": symbol,
            "side": "buy" if action == "buy" else "sell",
            "amount": float(f"{amount_contracts:.8f}"),
            "type": "market",
            "params": {
                "reduceOnly": False,
                "stopLoss": f"{sl_price:.2f}",
                "takeProfit": f"{tp_price:.2f}",
            },
            "meta": {
                **proposal,
                "adaptive_used": adaptive_used,
                "adaptive_params": {
                    "size_factor": dyn_size_factor,
                    "sl_atr_mult": dyn_sl_mult,
                    "tp_atr_mult": dyn_tp_mult,
                    "model_profile": model_profile,
                },
                "portfolio": {
                    "asset_class": asset_class,
                    "strategy": strategy,
                    "approved_usd": float(approved_usd),
                },
            },
        }

        if self.paper_mode:
            # paper fills immediately
            order["id"] = trade_id
            order["status"] = "filled"

            filled = {
                "status": "simulated",
                "id": trade_id,
                "symbol": order["symbol"],
                "side": order["side"],
                "amount": order["amount"],
                "type": order["type"],
                "params": order["params"],
                "filled_price": float(price),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "meta": order.get("meta", {}),
            }
            self._record_open(
                trade_id, symbol, order["side"], float(price), order["amount"], qty_usd, sl_price, tp_price, meta=order.get("meta")
            )

            # Mirror order_id/status into ledger so the monitor can track it
            try:
                rec = self.ledger.all_open().get(trade_id)
                if rec is not None:
                    rec["order_id"] = trade_id
                    rec["status"] = "filled"
                    self.ledger.add(trade_id, rec)
            except Exception as e:
                print(f"[Ledger] order status mirror failed: {e}")

            # Keep unified portfolio state in sync (NEW)
            try:
                self.portfolio.on_open(
                    trade_id,
                    symbol=symbol,
                    asset_class=asset_class,     # "futures" here; set "options" in options path
                    strategy=strategy,           # mapped above from model_profile
                    side=order["side"],          # "buy" / "sell"
                    qty_usd=float(qty_usd),
                    qty_contracts=float(order["amount"]),
                    entry_price=float(price),
                )
            except Exception as e:
                print(f"[Portfolio] on_open error: {e}")

            print(
                f"[PAPER] opened {symbol} {order['side']} qty={order['amount']:.6f} "
                f"({qty_usd:.2f} USD) @ {price:.2f} SL={sl_price:.2f} TP={tp_price:.2f}"
            )
            return filled

        # (future) live trading path can go here
        return None

    def reconcile_and_learn(self) -> Dict[str, Any]:
        """
        Close SL/TP hits, learn, persist learned state per symbol, and update equity.
        """
        closed: List[dict] = []
        still_open: List[str] = []
        today_str = datetime.now(timezone.utc).date().isoformat()

        for trade_id, t in list(self.ledger.all_open().items()):
            symbol = t["symbol"]
            side = t["side"]
            entry = float(t["entry"])
            amount_contracts = float(t["amount_contracts"])
            amount_usd = float(t["amount_usd"])
            sl = float(t["sl"])
            tp = float(t["tp"])
            meta = t.get("meta", {}) or {}
            adaptive_used = bool(meta.get("adaptive_used", False))
            adaptive_params = meta.get("adaptive_params", {}) or {}
            model_profile = adaptive_params.get("model_profile", "default")
            used_label = "yes" if adaptive_used else "no"

            try:
                ticker = self.adapter.get_ticker(symbol, params={"category": "linear"})
                last = float(ticker.get("last") or ticker.get("close") or 0.0)
            except Exception:
                last = 0.0

            if last <= 0:
                still_open.append(trade_id)
                continue

            exit_reason = None
            exit_price = 0.0
            if side == "buy":
                if last <= sl:
                    exit_price = sl
                    exit_reason = "SL"
                elif last >= tp:
                    exit_price = tp
                    exit_reason = "TP"
            else:  # sell
                if last >= sl:
                    exit_price = sl
                    exit_reason = "SL"
                elif last <= tp:
                    exit_price = tp
                    exit_reason = "TP"

            if exit_reason is None:
                still_open.append(trade_id)
                continue

            # PnL
            if side == "buy":
                pnl_usd = (exit_price - entry) * amount_contracts
            else:
                pnl_usd = (entry - exit_price) * amount_contracts

            # Fees (entry + exit legs)
            fee_usd = self._estimate_fee_usd(amount_usd, symbol) + self._estimate_fee_usd(exit_price * amount_contracts, symbol)
            pnl_net_usd = pnl_usd - fee_usd

            # Learn
            eng = self._get_risk_engine(symbol)
            try:
                eng.learn_from_trade(
                    entry_price=entry,
                    exit_price=exit_price,
                    side=side,
                    sl_price=sl,
                    tp_price=tp,
                    fee_usd=fee_usd,
                    today_iso=today_str,
                    adaptive_used=adaptive_used,
                    model_profile=model_profile,
                )
            except TypeError:
                try:
                    eng.learn_from_trade(
                        entry_price=entry,
                        exit_price=exit_price,
                        side=side,
                        sl_price=sl,
                        tp_price=tp,
                        fee_usd=fee_usd,
                        today_iso=today_str,
                    )
                except Exception as e:
                    print(f"[Risk] learn_from_trade error: {e}")
            except Exception as e:
                print(f"[Risk] learn_from_trade error: {e}")

            try:
                eng.update_day_pnl(pnl_net_usd, today_str)
            except Exception as e:
                print(f"[Risk] update_day_pnl error: {e}")

            # Persist learned state (DB)
            st = self._get_learned(symbol)
            st.sl_atr_mult = eng.state.sl_atr_mult
            st.tp_atr_mult = eng.state.tp_atr_mult
            st.trades = eng.state.trades
            st.wins = eng.state.wins
            st.gross_pnl = eng.state.gross_pnl
            st.gross_pnl_after_fees = eng.state.gross_pnl_after_fees
            st.avg_win = eng.state.avg_win
            st.avg_loss = eng.state.avg_loss
            st.trading_day = eng.state.trading_day
            st.day_pnl = eng.state.day_pnl
            st.rolling_equity_peak = eng.state.rolling_equity_peak
            st.rolling_equity_min = eng.state.rolling_equity_min
            st.last_regime = eng.state.last_regime
            db = next(get_db()); db.merge(st); db.commit()

            # Update unified portfolio on close (NEW)
            exit_value_usd = abs(exit_price * amount_contracts)
            try:
                self.portfolio.on_close(
                    trade_id,
                    exit_value_usd=float(exit_value_usd),
                    pnl_net_usd=float(pnl_net_usd),
                )
            except Exception as e:
                print(f"[Portfolio] on_close error: {e}")

            # Portfolio & ledger
            self.portfolio_gross_exposure -= amount_usd
            self.account_equity += pnl_net_usd
            self.ledger.remove(trade_id)

            # Metrics on close
            try:
                outcome = "win" if pnl_net_usd >= 0 else "loss"
                ADAPT_CLOSED.labels(symbol, model_profile, used_label, exit_reason, outcome).inc()
                ADAPT_PNL.labels(symbol, model_profile, used_label).observe(pnl_net_usd)
            except Exception:
                pass

            closed.append(
                {
                    "id": trade_id,
                    "symbol": symbol,
                    "side": side,
                    "entry": entry,
                    "exit": exit_price,
                    "reason": exit_reason,
                    "pnl_usd": pnl_net_usd,
                    "adaptive_used": adaptive_used,
                    "model_profile": model_profile,
                }
            )

            print(
                f"[PAPER] closed {symbol} {side} @ {exit_price:.2f} ({exit_reason}) "
                f"PnL={pnl_net_usd:+.2f} USD | adaptive={adaptive_used} profile={model_profile}"
            )

        return {"closed": closed, "open": still_open}
