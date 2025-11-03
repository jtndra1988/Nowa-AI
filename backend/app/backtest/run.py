# app/backtest/run.py
from __future__ import annotations

import argparse
import json
import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine, text

# --- Project Imports ---
# Ensure these paths are correct relative to your project structure
# Adjust if your structure is different (e.g., using absolute imports if configured)
try:
    from app.core.config import settings
    DATABASE_URL = settings.SQLALCHEMY_DATABASE_URI
except ImportError:
    # Fallback if config is not available directly (e.g., running script standalone)
    import os
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@db:5432/postgres")

# Core components used in the backtest loop
from app.ml.data_preprocessor import preprocess_raw_chunk # Although run_ensemble calls it, might be needed if run_ensemble changes
from app.ml.inference_ensemble import run as run_ensemble # The primary signal generator
from app.services.execution import ExecutionService       # The consolidated risk/execution service
from app.ml.artifacts import ckpt_dir                     # To find model checkpoints

# --- Constants ---
INITIAL_EQUITY = 100000.0
COMMISSION_BPS = 6 # 6 basis points (0.06%) for round-trip taker fees (3bps per side)
SLIPPAGE_BPS = 2   # 2 basis points (0.02%) slippage simulation per side
MIN_LOOKBACK_BARS = 200 # Need enough history for indicators & models

# --- Financial Metrics ---
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculates the annualized Sharpe ratio."""
    # Assuming daily returns if len(returns) > 252, else use number of trades. Adapt as needed.
    periods_per_year = 252 if len(returns) > 252 else len(returns) if len(returns) > 1 else 1
    excess_returns = returns - risk_free_rate / periods_per_year
    std_dev = excess_returns.std()
    if std_dev == 0 or pd.isna(std_dev): # Handle zero or NaN standard deviation
        return 0.0
    return (excess_returns.mean() / std_dev) * np.sqrt(periods_per_year)

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculates the annualized Sortino ratio."""
    periods_per_year = 252 if len(returns) > 252 else len(returns) if len(returns) > 1 else 1
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) < 2:
        return 0.0 # Not enough data points for std dev
    downside_std = downside_returns.std()
    if downside_std == 0 or pd.isna(downside_std): # Handle zero or NaN standard deviation
        return 0.0
    return (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculates the maximum drawdown percentage."""
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    # If equity never drops below initial, drawdown is 0
    max_dd = drawdown.min()
    return max_dd * 100 if pd.notna(max_dd) else 0.0 # Return as percentage


# --- Backtester Class ---
class Backtester:
    def __init__(self, symbol: str, start_date: str, end_date: str, initial_equity: float = INITIAL_EQUITY):
        self.symbol = symbol.upper() # Ensure consistent format e.g., BTC/USDT
        self.start_date = start_date
        self.end_date = end_date
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.engine = create_engine(DATABASE_URL)
        self.data: pd.DataFrame = None
        self.results: List[Dict[str, Any]] = [] # To store trade records
        self.equity_curve = []
        self.models_loaded = False

        # Initialize Execution Service in paper mode
        # The adapter isn't strictly needed for backtesting fills, but RiskEngine might use it for ATR.
        class DummyAdapter:
            def get_ticker(*args, **kwargs): return {}
            def place_trade_order(*args, **kwargs): return {}
            # Add fetch_ohlcv for RiskEngine's ATR calculation
            def fetch_ohlcv(self_dummy, symbol, timeframe='1h', since=None, limit=200, params={}):
                 # This needs access to the backtester's data
                 # We'll handle this by passing data directly later
                 pass # Placeholder

        self.execution_service = ExecutionService(adapter=DummyAdapter(), paper_mode=True)
        # Set initial equity in the risk engine's config
        self.execution_service.global_config.account_equity_usd = initial_equity
        # Ensure the risk engine for the specific symbol exists and has correct equity
        risk_engine = self.execution_service._get_risk_engine(self.symbol)
        risk_engine.cfg.account_equity_usd = initial_equity
        risk_engine.state.rolling_equity_peak = initial_equity
        risk_engine.state.rolling_equity_min = initial_equity


        # Pre-load models (assuming they exist)
        self.load_models()

    def load_models(self):
        """Loads necessary ML models and artifacts for inference."""
        # Check if essential model files exist for the given symbol
        try:
            model_dir = ckpt_dir(self.symbol)
            required_files = ["lstm_best.pt", "xgb_model.json", "xgb_features.json", "fusion_head.pt", "lstm_scaler.pkl"]
            missing = []
            for f in required_files:
                if not (model_dir / f).exists():
                     missing.append(f)
            if missing:
                raise FileNotFoundError(f"Missing required model artifacts in {model_dir}: {', '.join(missing)}")
            print(f"[*] Backtester: Found all model artifacts for {self.symbol} in {model_dir}")
            self.models_loaded = True
        except Exception as e:
            print(f"[!] Backtester: Failed to find or load all models for {self.symbol}: {e}")
            print("[!] Ensure models are trained and artifacts placed correctly before backtesting.")
            self.models_loaded = False


    def load_data(self):
        """Loads historical futures data from the database."""
        print(f"[*] Loading historical data for {self.symbol} from {self.start_date} to {self.end_date}...")
        q = text(f"""
            SELECT timestamp, open, high, low, close, volume
            FROM futures_market_data
            WHERE symbol = :symbol
            AND timestamp >= :start AND timestamp <= :end
            ORDER BY timestamp ASC
        """)
        self.data = pd.read_sql(q, self.engine, params={
            "symbol": self.symbol,
            "start": self.start_date,
            "end": self.end_date
        }, parse_dates=["timestamp"])

        if self.data.empty or len(self.data) < MIN_LOOKBACK_BARS:
            raise ValueError(f"Not enough historical data found for {self.symbol} in the specified range ({len(self.data)} bars). Need at least {MIN_LOOKBACK_BARS}.")
        print(f"[+] Loaded {len(self.data)} bars.")

    def run(self):
        """Runs the backtest simulation loop."""
        if not self.models_loaded:
             print("[!] Cannot run backtest: Models are not loaded.")
             return

        self.load_data()
        self.equity_curve.append({"timestamp": self.data.iloc[0]['timestamp'], "equity": self.equity})

        # --- Simulation Loop ---
        print(f"[*] Starting backtest loop for {len(self.data) - MIN_LOOKBACK_BARS} steps...")
        for i in range(MIN_LOOKBACK_BARS, len(self.data)):
            current_bar = self.data.iloc[i]
            current_dt = current_bar['timestamp']
            current_price = current_bar['close'] # Use close price for decision making context

            # Update Risk Engine with current equity peak/min *before* making decisions
            risk_engine = self.execution_service._get_risk_engine(self.symbol)
            risk_engine.state.rolling_equity_peak = max(risk_engine.state.rolling_equity_peak, self.equity)
            risk_engine.state.rolling_equity_min = min(risk_engine.state.rolling_equity_min, self.equity)

            # 1. Prepare data window for feature engineering & inference
            # The inference ensemble needs `lookback_rows`
            # Use data UP TO the start of the current bar (index i)
            start_idx = max(0, i - 5000) # Use a sufficiently large window for run_ensemble internal needs
            data_window_for_inference = self.data.iloc[start_idx:i].copy()

            if len(data_window_for_inference) < MIN_LOOKBACK_BARS: # Ensure enough history for features
                self.equity_curve.append({"timestamp": current_dt, "equity": self.equity})
                continue # Skip bar if not enough history

            # 2. Get ML Signal using inference_ensemble
            try:
                # IMPORTANT: We pass the *historical data window* to run_ensemble, which handles preprocessing
                # and model loading internally. It simulates what the live system would see.
                signal_output = run_ensemble(
                    symbol=self.symbol,
                    # Pass the dataframe directly instead of letting it query DB
                    df_override=data_window_for_inference
                 )

                signal = {
                    "symbol": self.symbol,
                    "action": signal_output.get("decision", "HOLD"),
                    "confidence": signal_output.get("confidence", 0.0),
                    "regime": signal_output.get("regime", 0),
                    "price": current_price, # Current close for context
                    "type": "futures",
                }
            except Exception as e:
                # Log occasional inference errors but continue
                # print(f"[!] Inference error at {current_dt} for {self.symbol}: {e}")
                signal = {"action": "HOLD"} # Treat errors as HOLD

            # 3. Pass Signal to Execution Service (Propose Trade)
            if signal.get("action") != "HOLD":
                try:
                    # Provide necessary data to RiskEngine (via ExecutionService)
                    # RiskEngine needs OHLCV data for ATR calculation
                    atr_window_data = data_window_for_inference.tail(risk_engine.cfg.atr_length + 5)

                    # Update account equity for risk assessment
                    self.execution_service.account_equity = self.equity

                    # Get proposal - MODIFIED to pass data directly
                    trade_proposal_dict = self.execution_service.propose_trade_sync(
                         signal=signal,
                         dfe_for_atr=atr_window_data # Pass data needed by risk engine
                    )


                    # Simulate placing the trade IF proposal is valid
                    if trade_proposal_dict and trade_proposal_dict.get("qty_usd", 0.0) > 0:
                         # Simulate fill at the *next* bar's open price with slippage
                         if i + 1 < len(self.data):
                            next_open = self.data.iloc[i+1]['open']
                            side_mult = 1 if trade_proposal_dict['side'] == 'buy' else -1
                            slip_adj = next_open * (SLIPPAGE_BPS / 10000.0) * side_mult
                            fill_price = next_open + slip_adj

                            # Record the trade in the ledger (simulating what ExecutionService does)
                            trade_id = self.execution_service._new_trade_id(self.symbol)
                            amount_contracts = trade_proposal_dict['qty_usd'] / fill_price
                            sl_price = trade_proposal_dict['sl_price']
                            tp_price = trade_proposal_dict['tp_price']

                            # Record open with FILLED price and details
                            self.execution_service._record_open(
                                trade_id, self.symbol, trade_proposal_dict['side'],
                                fill_price, amount_contracts, trade_proposal_dict['qty_usd'],
                                sl_price, tp_price
                            )
                            # Add fill time to ledger data
                            ledger_entry = self.execution_service.ledger.all_open().get(trade_id)
                            if ledger_entry:
                                ledger_entry['filled_at'] = self.data.iloc[i+1]['timestamp']
                                self.execution_service.ledger.add(trade_id, ledger_entry)


                            print(f"[{ledger_entry['filled_at']}] SIM FILL: {trade_proposal_dict['side']} {amount_contracts:.4f} {self.symbol} @ {fill_price:.2f} (SL={sl_price:.2f}, TP={tp_price:.2f})")

                            # Deduct entry commission
                            commission = trade_proposal_dict['qty_usd'] * (COMMISSION_BPS / 2.0 / 10000.0)
                            self.equity -= commission
                         else:
                              # Cannot fill on the last bar
                              print(f"[{current_dt}] Signal generated but cannot fill on last bar.")

                except Exception as e:
                    print(f"[!] Execution Service proposal error at {current_dt} for {self.symbol}: {e}")

            # 4. Reconcile Positions (Check for SL/TP Hits) using CURRENT bar's High/Low
            try:
                # Pass current bar's high/low to simulate intra-bar price movement
                closed_trades_info = self.simulate_reconciliation(
                    current_bar['high'],
                    current_bar['low'],
                    current_bar['timestamp']
                )

                # Update equity based on closed trades PnL
                for trade_info in closed_trades_info.get("closed", []):
                    pnl_usd = trade_info.get("pnl_usd", 0.0) # This PnL already includes exit commission
                    self.equity += pnl_usd

                    # Store trade result for reporting
                    self.results.append({
                        "entry_time": trade_info.get("filled_at"), # Use fill time
                        "exit_time": current_dt,
                        "symbol": trade_info.get("symbol"),
                        "side": trade_info.get("side"),
                        "entry_price": trade_info.get("entry"),
                        "exit_price": trade_info.get("exit"),
                        "amount_contracts": trade_info.get("amount_contracts"),
                        "pnl_usd": pnl_usd,
                        "exit_reason": trade_info.get("reason"),
                    })

            except Exception as e:
                 print(f"[!] Reconciliation error at {current_dt} for {self.symbol}: {e}")

            # 5. Record Equity for the current bar
            self.equity_curve.append({"timestamp": current_dt, "equity": self.equity})
            # --- Safety Break ---
            if self.equity <= 0:
                print(f"[!!!] Equity depleted at {current_dt}. Stopping backtest.")
                break


        # --- End Simulation Loop ---
        self.report_results()


    def simulate_reconciliation(self, current_high: float, current_low: float, current_timestamp: datetime) -> Dict[str, Any]:
        """
        Modified reconciliation for backtesting. Checks SL/TP against H/L prices.
        Calls the advanced learning method of the risk engine.
        """
        closed = []
        still_open = []
        today_str = current_timestamp.strftime("%Y-%m-%d")
        commission_exit_bps = COMMISSION_BPS / 2.0 # Only exit commission here

        # Iterate over a copy, as we modify the ledger
        for trade_id, t in list(self.execution_service.ledger.all_open().items()):
            symbol = t["symbol"]
            side = t["side"]
            entry = float(t["entry"])
            amount_contracts = float(t["amount_contracts"])
            amount_usd = float(t["amount_usd"]) # Notional at entry
            sl = float(t["sl"])
            tp = float(t["tp"])
            filled_at = t.get("filled_at") # Timestamp of entry fill

            # Skip if trade was filled in the future (relevant if using future data slices)
            if filled_at and filled_at > current_timestamp:
                still_open.append(trade_id)
                continue

            exit_reason = None
            exit_price = 0.0

            # Check SL/TP hit based on high/low of the *current* bar
            # Prioritize SL over TP if both could trigger (conservative)
            if side == "buy":
                if current_low <= sl:
                    exit_price = sl  # Assume worst price for SL hit
                    exit_reason = "SL"
                elif current_high >= tp:
                    exit_price = tp  # Assume TP price is hit
                    exit_reason = "TP"
            else: # sell
                if current_high >= sl:
                    exit_price = sl
                    exit_reason = "SL"
                elif current_low <= tp:
                    exit_price = tp
                    exit_reason = "TP"

            if exit_reason is None:
                still_open.append(trade_id)
                continue # Trade remains open

            # --- Trade Closed ---
            # 1. Calculate Gross PnL
            if side == "buy":
                pnl_usd = (exit_price - entry) * amount_contracts
            else:
                pnl_usd = (entry - exit_price) * amount_contracts

            # 2. Calculate Exit Commission
            exit_notional = exit_price * amount_contracts
            exit_commission = abs(exit_notional) * (commission_exit_bps / 10000.0) # Use abs for safety

            # 3. Net PnL (Entry commission was already deducted when opening)
            pnl_net_usd = pnl_usd - exit_commission

            # 4. Get the symbol's engine and Learn
            engine = self.execution_service._get_risk_engine(symbol)
            try:
                # Fee passed should be *total* fee (entry+exit)
                entry_commission = abs(amount_usd) * (commission_exit_bps / 10000.0) # Use abs for safety
                total_fee_usd = entry_commission + exit_commission

                engine.learn_from_trade(
                    entry_price=entry,
                    exit_price=exit_price,
                    side=side,
                    sl_price=sl,
                    tp_price=tp,
                    fee_usd=total_fee_usd, # Pass total fee
                    today=today_str
                )
                # Update daily PnL tracker within the risk engine state
                engine.update_day_pnl(pnl_net_usd, today_str)

            except Exception as e:
                print(f"[!] Risk learn error during backtest reconcile: {e}")

            # 5. Remove from ledger & update portfolio state simulation
            self.execution_service.ledger.remove(trade_id)
            self.execution_service.portfolio_gross_exposure -= abs(amount_usd) # Reduce exposure by entry notional

            closed.append({
                "id": trade_id, "symbol": symbol, "side": side, "entry": entry,
                "exit": exit_price, "reason": exit_reason, "pnl_usd": pnl_net_usd,
                "amount_contracts": amount_contracts, "filled_at": filled_at
            })

            print(f"[{current_timestamp}] SIM CLOSE: {side} {amount_contracts:.4f} {symbol} @ {exit_price:.2f} ({exit_reason}). PnL: {pnl_net_usd:+.2f} USD. Equity: {self.equity + pnl_net_usd:.2f}")

        return {"closed": closed, "open": still_open}


    def report_results(self):
        """Calculates and prints performance metrics."""
        print("\n--- Backtest Results ---")

        equity_df = pd.DataFrame(self.equity_curve)
        if equity_df.empty:
             print("No equity data recorded.")
             return
        equity_df = equity_df.set_index("timestamp")
        equity_df['returns'] = equity_df['equity'].pct_change().fillna(0.0)

        # Ensure equity_df index is timezone-naive or consistent for calculations if needed
        # equity_df.index = equity_df.index.tz_localize(None)


        total_return = (self.equity / self.initial_equity - 1) * 100
        # Use daily returns for annualization if data spans > 1 year, else don't annualize.
        daily_returns = equity_df['returns'].resample('D').sum() # Resample to daily returns

        # Check if we have enough data points for meaningful std dev
        if len(daily_returns) > 1:
            sharpe = calculate_sharpe_ratio(daily_returns)
            sortino = calculate_sortino_ratio(daily_returns)
        else:
             sharpe = 0.0
             sortino = 0.0

        max_dd = calculate_max_drawdown(equity_df['equity'])

        print(f"Period:                 {self.start_date} to {self.end_date}")
        print(f"Initial Equity:         ${self.initial_equity:,.2f}")
        print(f"Final Equity:           ${self.equity:,.2f}")
        print(f"Total Return:           {total_return:.2f}%")
        print("-" * 25)
        print(f"Sharpe Ratio (ann.):    {sharpe:.2f}")
        print(f"Sortino Ratio (ann.):   {sortino:.2f}")
        print(f"Max Drawdown:           {max_dd:.2f}%")
        print("-" * 25)

        if not self.results:
            print("No trades executed.")
            print("-" * 25)
        else:
            trades_df = pd.DataFrame(self.results)
            num_trades = len(trades_df)
            wins = trades_df[trades_df['pnl_usd'] > 0]
            losses = trades_df[trades_df['pnl_usd'] <= 0]
            win_rate = (len(wins) / num_trades) * 100 if num_trades > 0 else 0
            avg_win = wins['pnl_usd'].mean() if len(wins) > 0 else 0
            avg_loss = losses['pnl_usd'].mean() if len(losses) > 0 else 0 # Avg loss is negative
            # Profit factor: Gross Profit / Gross Loss
            gross_profit = wins['pnl_usd'].sum()
            gross_loss = abs(losses['pnl_usd'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')


            print(f"Total Trades:           {num_trades}")
            print(f"Win Rate:               {win_rate:.2f}%")
            print(f"Avg Win / Trade:        ${avg_win:,.2f}")
            print(f"Avg Loss / Trade:       ${avg_loss:,.2f}")
            print(f"Profit Factor:          {profit_factor:.2f}")
            print("-" * 25)

            # Optional: Save equity curve and trades to CSV
            try:
                equity_df.to_csv(f"backtest_equity_{self.symbol.replace('/','_')}.csv")
                trades_df.to_csv(f"backtest_trades_{self.symbol.replace('/','_')}.csv", index=False)
                print(f"[*] Equity curve saved to backtest_equity_{self.symbol.replace('/','_')}.csv")
                print(f"[*] Trade log saved to backtest_trades_{self.symbol.replace('/','_')}.csv")
            except Exception as e:
                print(f"[!] Error saving results to CSV: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtest for a given symbol and date range.")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., BTC/USDT)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--equity", type=float, default=INITIAL_EQUITY, help="Initial equity")
    args = parser.parse_args()

    # --- Add basic input validation ---
    try:
        datetime.strptime(args.start, "%Y-%m-%d")
        datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError:
        print("[!] Invalid date format. Please use YYYY-MM-DD.")
        exit(1)

    if args.equity <= 0:
        print("[!] Initial equity must be positive.")
        exit(1)

    print(f"--- Starting Backtest ---")
    print(f"Symbol: {args.symbol}, Period: {args.start} to {args.end}, Initial Equity: ${args.equity:,.2f}")
    print(f"Commission: {COMMISSION_BPS} bps round-trip, Slippage: {SLIPPAGE_BPS} bps per side")
    print(f"-------------------------")


    try:
        backtester = Backtester(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            initial_equity=args.equity
        )
        backtester.run()
    except ValueError as ve:
        print(f"[!] Backtest Initialization Error: {ve}")
    except Exception as e:
        print(f"[!!!] An unexpected error occurred during the backtest: {e}")
        # Consider adding more detailed error logging here if needed
        import traceback
        traceback.print_exc()


    

