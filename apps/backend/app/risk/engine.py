# app/risk/engine.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


# -------------------------------
# Helpers for on-disk risk state
# -------------------------------
def _state_dir(symbol: str) -> Path:
    p = Path("/app/state/risk") / symbol.replace("/", "_")
    p.mkdir(parents=True, exist_ok=True)
    return p

def _state_path(symbol: str) -> Path:
    return _state_dir(symbol) / "risk_state.json"


# -------------------------------
# Config & State
# -------------------------------
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
        default_factory=lambda: {0: 0.8, 1: 1.1, 2: 1.0}  # {0:chop,1:up,2:down}
    )

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

    def update_after_trade(self, pl: float, fee: float, alpha: float = 0.2) -> None:
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


# -------------------------------
# Risk Engine (single source)
# -------------------------------
class RiskEngine:
    """Self-adapting risk engine for a single symbol."""

    def __init__(self, symbol: str, cfg: Optional[RiskConfig] = None) -> None:
        self.symbol = symbol.upper()
        self.cfg = cfg or RiskConfig()
        self.state = self._load_state()

    # -------- persistence (local JSON for redundancy) --------
    def _load_state(self) -> RiskState:
        p = _state_path(self.symbol)
        if p.exists():
            try:
                js = json.load(open(p))
                # Guard against stale zeros: keep peaks around current equity baseline
                js["rolling_equity_peak"] = max(js.get("rolling_equity_peak", 0.0), self.cfg.account_equity_usd)
                js["rolling_equity_min"] = min(js.get("rolling_equity_min", self.cfg.account_equity_usd), self.cfg.account_equity_usd)
                return RiskState(**js)
            except Exception:
                pass
        return RiskState(
            sl_atr_mult=self.cfg.sl_atr_mult_init,
            tp_atr_mult=self.cfg.tp_atr_mult_init,
            rolling_equity_peak=self.cfg.account_equity_usd,
            rolling_equity_min=self.cfg.account_equity_usd,
        )

    def save_state(self) -> None:
        try:
            json.dump(asdict(self.state), open(_state_path(self.symbol), "w"))
        except Exception as e:
            print(f"[!] Failed to save risk state for {self.symbol}: {e}")

    # -------- indicators --------
    @staticmethod
    def _atr(df: pd.DataFrame, length: int) -> float:
        """ATR with pandas_ta when available; otherwise True Range fallback."""
        try:
            import pandas_ta as ta  # type: ignore
            s = ta.atr(df["high"], df["low"], df["close"], length=length)
            return float(s.iloc[-1])
        except Exception:
            high = df["high"].astype(float).values
            low = df["low"].astype(float).values
            close = df["close"].astype(float).values
            prev = np.roll(close, 1); prev[0] = close[0]
            tr1 = high - low
            tr2 = np.abs(high - prev)
            tr3 = np.abs(low - prev)
            tr = np.maximum.reduce([tr1, tr2, tr3])
            atr = pd.Series(tr).rolling(window=length, min_periods=length).mean().iloc[-1]
            return float(atr)

    # -------- day/guard updates (no I/O) --------
    def daily_reset_if_needed(self, today_iso: str) -> None:
        if self.state.trading_day != today_iso:
            self.state.trading_day = today_iso
            self.state.day_pnl = 0.0

    def check_global_guards(self, current_equity: float) -> Dict[str, Any]:
        self.state.rolling_equity_peak = max(self.state.rolling_equity_peak, current_equity)
        self.state.rolling_equity_min = min(self.state.rolling_equity_min, current_equity)
        dd = 1.0 - (current_equity / max(self.state.rolling_equity_peak, 1e-9))
        if dd >= self.cfg.rolling_max_dd_pct:
            return {"halt": True, "reason": f"Rolling drawdown {dd:.2%} exceeds limit"}
        return {"halt": False}

    def update_day_pnl(self, pnl_net_usd: float, today_iso: str) -> Dict[str, Any]:
        self.daily_reset_if_needed(today_iso)
        self.state.day_pnl += pnl_net_usd
        if self.state.day_pnl <= -self.cfg.daily_loss_limit_pct * self.cfg.account_equity_usd:
            return {"halt": True, "reason": f"Daily loss limit hit ({self.state.day_pnl:.2f} USD)"}
        return {"halt": False}

    # -------- position proposal --------
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
        """
        Returns:
          qty_usd, sl_price, tp_price, atr, sl_mult, tp_mult, conf_scale, reg_scale, reason
        """
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

    # -------- learning hook (pure math, no I/O) --------
    def learn_from_trade(
        self,
        entry_price: float,
        exit_price: float,
        side: str,
        sl_price: float,
        tp_price: float,
        fee_usd: float,
        today_iso: str,
    ) -> Dict[str, Any]:
        """
        Update the adaptive multipliers with a simple bandit-like adjustment.
        """
        hit = "tp" if (side == "buy" and exit_price >= tp_price) or (side == "sell" and exit_price <= tp_price) else (
              "sl" if (side == "buy" and exit_price <= sl_price) or (side == "sell" and exit_price >= sl_price) else "mid")

        # PnL in USD per 1 contract notional (directional)
        if side == "buy":
            per_unit = exit_price - entry_price
        else:
            per_unit = entry_price - exit_price

        # Reward signal (net of fees proxy)
        reward = per_unit - (fee_usd or 0.0)

        # Bandit-like adjustment
        lr = self.cfg.bandit_lr
        if hit == "tp":
            self.state.tp_atr_mult = float(np.clip(self.state.tp_atr_mult * (1.0 + lr), self.cfg.min_tp_atr_mult, self.cfg.max_tp_atr_mult))
        elif hit == "sl":
            self.state.sl_atr_mult = float(np.clip(self.state.sl_atr_mult * (1.0 + lr), self.cfg.min_sl_atr_mult, self.cfg.max_sl_atr_mult))

        # Update stats
        self.state.update_after_trade(pl=per_unit, fee=fee_usd, alpha=self.cfg.ewma_alpha)
        self.daily_reset_if_needed(today_iso)

        return {
            "hit": hit,
            "reward": reward,
            "sl_mult": self.state.sl_atr_mult,
            "tp_mult": self.state.tp_atr_mult,
            "win_rate": self.state.win_rate(),
            "payoff": self.state.payoff_ratio(),
        }
