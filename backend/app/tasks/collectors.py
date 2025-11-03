# app/tasks/collectors.py
import asyncio
import requests
import sqlalchemy
import yfinance as yf
from datetime import datetime, timedelta, timezone # Ensure timezone is imported
from typing import Dict, Any, List, Optional
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert # Import postgresql specific insert
from app.celery_app.app import celery_app
from app.core.config import settings
# Use SessionLocal consistently, get_db might not be needed if SessionLocal is used directly
from app.db.database import SessionLocal
from app.db.models import (
    FearAndGreedIndex, MacroData, MarketData, FuturesMarketData,
    FundingRate, OrderbookSnapshot, OptionsChain # Assuming FundingRate is needed elsewhere
)
from app.exchange.adapters import BybitAdapter, BinanceDataAdapter
# Import ALL necessary helpers from utils
from app.utils import _norm_futures_symbol, _safe_float, _utcnow
import logging
# --------------------------------------------------------------------
# Single engine (Optional - can often use SessionLocal().bind instead)
# --------------------------------------------------------------------
# If save_data_hybrid is used elsewhere, keep this. Otherwise, it might be removable.
engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)

# Deprecate save_data_hybrid if possible, favor session-based operations within tasks
# def save_data_hybrid(df, table_name, csv_path): ...

# Helper function (if needed elsewhere, move to utils.py)
def _safe_datetime_from_ms(val: Any) -> Optional[datetime]:
    """Safely convert a millisecond timestamp to a datetime object."""
    if val is None:
        return None
    try:
        timestamp_sec = int(val) / 1000.0
        # Ensure it creates timezone-aware datetime
        return datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
    except (ValueError, TypeError):
        try:
            # Fallback for ISO strings etc.
            dt = pd.to_datetime(val, utc=True).to_pydatetime()
            # Ensure the result is timezone-aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

# app/tasks/collectors.py

# ... (other imports remain the same) ...
from app.db.database import SessionLocal, get_db # Ensure get_db is imported if used elsewhere
from app.db.models import OptionsChain
from app.utils import _norm_futures_symbol, _safe_float, _utcnow # Ensure imports
from app.exchange.adapters import BinanceDataAdapter, BybitAdapter # Ensure imports
from app.core.config import settings # Ensure settings is imported
import logging # Import the logging module

# ... (_safe_datetime_from_ms function and other tasks remain the same) ...

# =============================================================================
#  TASK 1: Collect OPTIONS Data (per-contract) → options_chain
# =============================================================================
@celery_app.task(name="tasks.collect_options", bind=True) # Add bind=True
def collect_options_task(self, symbol: str, adapter_payload: Optional[dict] = None) -> int: # Add self
    """
    Insert per-contract options rows into OptionsChain with full identifiers & greeks.
    Commits data in batches. Includes enhanced error logging.
    """
    task_id = self.request.id # Get Celery task ID for logging
    logger = logging.getLogger(__name__) # Use standard Python logger
    logger.info(f"Task {task_id}: Starting options collection for {symbol}.")

    tz_now = _utcnow()
    rows: List[dict] = []
    BATCH_SIZE = 50 # Keep reduced batch size

    # --- Data fetching ---
    if adapter_payload is None:
        adapter = BinanceDataAdapter() if getattr(settings, "USE_BINANCE_FOR_DATA", True) else BybitAdapter()
        fetcher_names = ["get_options_chain_snapshot", "get_options_chain", "fetch_options_chain"]
        for name in fetcher_names:
            if hasattr(adapter, name):
                try:
                    res = getattr(adapter, name)(symbol)
                    if isinstance(res, dict) and "rows" in res: rows = res["rows"]
                    elif isinstance(res, list): rows = res
                    if rows:
                        logger.info(f"Task {task_id}: Fetched {len(rows)} rows via {name} for {symbol}.")
                        break
                except Exception as e:
                    logger.warning(f"Task {task_id}: Options fetch ({symbol}) via {name} failed: {e}")
        if not rows and adapter_payload is not None: rows = adapter_payload.get("rows", [])
    else: rows = adapter_payload.get("rows", [])

    if not rows:
        logger.info(f"Task {task_id}: No options rows found/provided for {symbol}.")
        return 0

    written_total = 0
    written_batch = 0
    db = SessionLocal()
    logger.info(f"Task {task_id}: Processing {len(rows)} rows for {symbol}...")
    try:
        for i, r in enumerate(rows):
            # --- Per-Row Processing Start ---
            # Use a nested try/except to isolate row processing and commit errors
            try:
                # --- Data cleaning ---
                ts_raw = r.get("timestamp")
                ts = _safe_datetime_from_ms(ts_raw) if ts_raw is not None else tz_now

                ot = str(r.get("option_type", "")).upper()
                if ot.startswith("CALL"): ot = "C"
                elif ot.startswith("PUT"): ot = "P"
                elif ot not in ("C", "P"): continue

                expiry_dt = _safe_datetime_from_ms(r.get("expiry"))
                if not expiry_dt: continue

                strike_price = _safe_float(r.get("strike"))
                if not strike_price: continue

                base_symbol = symbol.split('/')[0]
                # --- ADD LOGGING HERE ---
                logger.debug(f"Task {task_id}: Checking existence for {base_symbol} {expiry_dt} {strike_price} {ot} at {ts} (Row #{i+1})")
                try:
                   # --- Existence check ---
                   exists = db.query(OptionsChain.id).filter(
                   OptionsChain.symbol == base_symbol, OptionsChain.expiry == expiry_dt,
                   OptionsChain.strike == strike_price, OptionsChain.option_type == ot,
                   OptionsChain.timestamp == ts,
                   ).first() # <<< The point where ResourceClosedError occurs
                   logger.debug(f"Task {task_id}: Existence check completed for row #{i+1}.") # Log if successful
                except sqlalchemy.exc.ResourceClosedError as rce:
                 logger.error(f"Task {task_id}: ResourceClosedError occurred during existence check for row #{i+1}! Session likely invalid BEFORE query. Error: {rce}")
                 # Re-raise or handle as appropriate, maybe break the loop
                 raise rce # Re-raise immediately to see this specific error first
                 # --- END ADDED LOGGING ---
                # --- Existence check ---
                # This is where the ResourceClosedError often manifests if the session is bad
                exists = db.query(OptionsChain.id).filter(
                    OptionsChain.symbol == base_symbol, OptionsChain.expiry == expiry_dt,
                    OptionsChain.strike == strike_price, OptionsChain.option_type == ot,
                    OptionsChain.timestamp == ts,
                ).first()
                if exists: continue

                # --- Create object ---
                oc = OptionsChain(
                    symbol=base_symbol, expiry=expiry_dt, strike=strike_price, option_type=ot, timestamp=ts,
                    bid=_safe_float(r.get("bid")), ask=_safe_float(r.get("ask")),
                    last_price=_safe_float(r.get("last_price")), mark_price=_safe_float(r.get("mark_price")),
                    volume=_safe_float(r.get("volume")), open_interest=_safe_float(r.get("open_interest")),
                    iv=_safe_float(r.get("iv")), delta=_safe_float(r.get("delta")),
                    gamma=_safe_float(r.get("gamma")), theta=_safe_float(r.get("theta")),
                    vega=_safe_float(r.get("vega")),
                )
                db.add(oc)
                written_batch += 1
                written_total += 1

                # --- Commit in batches ---
                if written_batch >= BATCH_SIZE:
                    try:
                        db.commit()
                        logger.info(f"Task {task_id}: Committed batch of {written_batch} options rows for {base_symbol}.")
                        written_batch = 0
                    except Exception as commit_err:
                         # Log commit error specifically
                         logger.error(f"Task {task_id}: Commit failed during batch ({base_symbol}): {commit_err}", exc_info=True)
                         # Attempt rollback, but expect it might fail if connection is bad
                         try: db.rollback()
                         except Exception as rb_err: logger.error(f"Task {task_id}: Rollback also failed after commit error: {rb_err}")
                         raise commit_err # Re-raise to fail the task

            # --- Catch Error During Single Row Processing (before commit) ---
            except Exception as e_row:
                 # Log the ORIGINAL error that likely broke the session
                 logger.error(f"Task {task_id}: Error processing options row #{i+1}, skipping: {e_row}. Data: {r}", exc_info=True)
                 # Attempt to rollback this specific failure
                 try: db.rollback()
                 except Exception as rb_err_inner: logger.error(f"Task {task_id}: Rollback failed after row processing error: {rb_err_inner}")
                 # Continue to the next row - session might still be broken

        # --- Commit any remaining rows ---
        if written_batch > 0:
            try:
                db.commit()
                logger.info(f"Task {task_id}: Committed final batch of {written_batch} options rows for {base_symbol}.")
            except Exception as final_commit_err:
                 logger.error(f"Task {task_id}: Commit failed during final batch ({base_symbol}): {final_commit_err}", exc_info=True)
                 try: db.rollback()
                 except Exception as rb_err_final: logger.error(f"Task {task_id}: Rollback also failed after final commit error: {rb_err_final}")
                 raise final_commit_err # Re-raise

        logger.info(f"Task {task_id}: Finished processing options for {base_symbol}. Total written: {written_total}")
        return written_total

    # --- Catch Error During Overall Task Execution (outside loop/commits) ---
    except Exception as e_task:
        logger.error(f"Task {task_id}: CRITICAL error in collect_options_task ({symbol}), attempting rollback: {e_task}", exc_info=True)
        try: db.rollback()
        except Exception as rb_err_outer: logger.error(f"Task {task_id}: Rollback failed after critical task error: {rb_err_outer}")
        raise e_task # Re-raise for Celery failure
    finally:
        # Always try to close the session
        try: db.close()
        except Exception as close_err: logger.error(f"Task {task_id}: Error closing DB session: {close_err}")
# =============================================================================
#  TASK 2: Collect FUTURES Data → futures_market_data
# =============================================================================
@celery_app.task(name="tasks.collect_futures")
def collect_futures_task(underlying_symbol: str): # Accepts base symbol e.g., 'BTC'
    print(f"[*] Starting FUTURES data collection for: {underlying_symbol}")
    PAPER_MODE = bool(getattr(settings, "PAPER_TRADING", True))
    # Assuming Bybit is the primary for futures ticker data based on worker.py EXEC init
    adapter = BybitAdapter(paper_mode=PAPER_MODE)
    futures_symbol = f"{underlying_symbol}/USDT" # Construct the pair symbol

    try:
        # Fetch ticker using the pair symbol
        ticker_data = adapter.get_ticker(futures_symbol, params={'category': 'linear'})
    except Exception as fetch_e:
        print(f"[!] Could not fetch futures ticker for {futures_symbol}: {fetch_e}. Skipping.")
        return # Exit if fetching fails

    if not ticker_data or not ticker_data.get('timestamp'): # Ensure data and timestamp exist
        print(f"[!] Invalid or empty ticker data received for {futures_symbol}. Skipping.")
        return

    # Normalize symbol from ticker response (might be BTCUSDT or BTC/USDT)
    norm_symbol = _norm_futures_symbol(ticker_data.get("symbol") or futures_symbol)

    db_session = SessionLocal()
    try:
        ts_ms = ticker_data.get('timestamp')
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) # Already timezone-aware

        # Explicit check for duplicate using normalized symbol and timestamp
        exists = db_session.query(FuturesMarketData.id).filter(
            FuturesMarketData.symbol == norm_symbol,
            FuturesMarketData.timestamp == ts
        ).first()

        if not exists:
            futures_data_record = FuturesMarketData(
                symbol=norm_symbol, timestamp=ts,
                open=_safe_float(ticker_data.get('openPrice') or ticker_data.get('open')), # Check Bybit keys
                high=_safe_float(ticker_data.get('highPrice') or ticker_data.get('high')),
                low=_safe_float(ticker_data.get('lowPrice') or ticker_data.get('low')),
                close=_safe_float(ticker_data.get('lastPrice') or ticker_data.get('last') or ticker_data.get('close')),
                volume=_safe_float(ticker_data.get('volume24h') or ticker_data.get('baseVolume')), # Check Bybit keys
            )
            db_session.add(futures_data_record)
            db_session.commit()
            print(f"[✔] Successfully saved futures data for {norm_symbol}.")
        else:
            print(f"[i] Skipping duplicate futures data for {norm_symbol} at {ts}.")

    except Exception as e:
        print(f"[!] ERROR saving futures data for {norm_symbol} to DB: {e}")
        db_session.rollback()
        # raise e # Optionally re-raise
    finally:
        db_session.close()

# =============================================================================
#  TASK 3: Collect Fear & Greed Index
# =============================================================================
@celery_app.task(name="tasks.collect_fear_and_greed")
def collect_fear_and_greed_task():
    print("[*] Starting Fear & Greed Index collection...")
    db_session = SessionLocal() # Use SessionLocal
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        response.raise_for_status()
        data = response.json()['data'][0]

        # Convert timestamp and make timezone-aware (UTC)
        ts_unix = int(data['timestamp'])
        ts_dt = datetime.fromtimestamp(ts_unix, tz=timezone.utc)
        value_int = int(data['value'])

        # Check if a record for this timestamp already exists
        exists = db_session.query(FearAndGreedIndex.id).filter(
            FearAndGreedIndex.timestamp == ts_dt
        ).first()

        if not exists:
            new_entry = FearAndGreedIndex(
                timestamp=ts_dt, # Use the timezone-aware datetime
                value=value_int,
                value_classification=data['value_classification']
            )
            db_session.add(new_entry)
            db_session.commit()
            print(f"[✔] Successfully saved Fear & Greed Index value: {value_int} at {ts_dt}")
        else:
             print(f"[i] Skipping duplicate Fear & Greed Index value for {ts_dt}.")

    except Exception as e:
        print(f"[!] ERROR during Fear & Greed collection: {e}")
        db_session.rollback()
        # Optionally re-raise e
    finally:
        db_session.close()

# =============================================================================
#  TASK 4: Collect Macro Data (DXY)
# =============================================================================
@celery_app.task(name="tasks.collect_macro_data")
def collect_macro_data_task():
    print("[*] Starting Macro Data (DXY) collection...")
    db_session = SessionLocal() # Use SessionLocal
    now_utc = _utcnow() # Get current UTC time
    try:
        dxy = yf.Ticker("DX-Y.NYB")
        # Fetch slightly more history to ensure we get the latest close
        hist = dxy.history(period="5d") # Fetch last 5 days
        if not hist.empty:
            # Get the most recent non-NaN closing price
            latest_close = hist['Close'].dropna().iloc[-1]
            # Get the timestamp of that closing price (index is datetime)
            latest_timestamp = hist['Close'].dropna().index[-1].to_pydatetime()
            # Make the timestamp timezone-aware (yfinance usually returns naive)
            if latest_timestamp.tzinfo is None:
                # Assume NY time and convert to UTC, or just assign UTC if index is already UTC-like
                try:
                    # This depends on your system's timezone settings for yfinance
                    # Safer to explicitly handle timezone if possible
                    latest_timestamp = latest_timestamp.tz_localize('America/New_York').tz_convert('UTC')
                except Exception: # Fallback: Assume UTC if localization fails
                    latest_timestamp = latest_timestamp.replace(tzinfo=timezone.utc)

            # Check if a recent entry exists (e.g., within last 12 hours)
            cutoff_time = now_utc - timedelta(hours=12)
            exists = db_session.query(MacroData.id).filter(
                MacroData.indicator == 'DXY',
                MacroData.timestamp >= cutoff_time
            ).first()

            if not exists:
                new_entry = MacroData(
                    timestamp=latest_timestamp, # Use timestamp from yfinance data
                    indicator='DXY',
                    value=float(latest_close) # Ensure it's a float
                )
                db_session.add(new_entry)
                db_session.commit()
                print(f"[✔] Successfully saved DXY value: {latest_close:.2f} at {latest_timestamp}")
            else:
                print(f"[i] Skipping recent DXY value.")
        else:
             print("[!] yfinance returned empty history for DXY.")

    except Exception as e:
        print(f"[!] ERROR during Macro Data collection: {e}")
        db_session.rollback()
        # Optionally re-raise e
    finally:
        db_session.close()

# =============================================================================
#  MASTER TASK: Run all collection jobs
# =============================================================================
@celery_app.task(name="tasks.collect_all_assets")
def collect_all_assets_task():
    print("--- [MASTER TASK] Kicking off all data collection jobs ---")
    try:
        # Determine adapter based on settings (consistent with worker.py)
        PAPER_MODE = bool(getattr(settings, "PAPER_TRADING", True))
        adapter = BinanceDataAdapter() if getattr(settings, "USE_BINANCE_FOR_DATA", True) else BybitAdapter(paper_mode=PAPER_MODE)

        # Get base symbols (e.g., BTC, ETH)
        assets_bases = adapter.get_top_symbols_by_volume(limit=30)
        if not assets_bases:
            print("[!] Could not fetch top symbols. Aborting collection.")
            return

        print(f"[*] Found top {len(assets_bases)} assets. Triggering collection tasks...")

        # --- Trigger individual tasks ---

        # 1. Binance historical (if using Binance adapter)
        if isinstance(adapter, BinanceDataAdapter):
            # Pass base symbols to the task if it expects them, or let it fetch again
            # Assuming the task fetches symbols itself as per its definition
            collect_binance_historical_data.delay()
        else:
            # If using Bybit, maybe trigger a different historical task or skip
            print("[i] Skipping Binance historical task (using Bybit adapter).")

        # 2. Options, Futures per asset
        for asset in assets_bases:
            # Options (pass base symbol)
            # Fetching logic is now inside collect_options_task
            collect_options_task.delay(asset) # Pass base symbol 'BTC'
            # Futures (pass base symbol)
            collect_futures_task.delay(asset) # Pass base symbol 'BTC'

        # 3. Sentiment (pass base symbols)
        from .sentiment_collector import collect_sentiment_all_sources
        collect_sentiment_all_sources.delay(assets_bases) # Pass base symbols ['BTC', 'ETH', ...]

        # 4. Orderbook (task fetches its own symbols)
        from .orderbook_collector import collect_orderbook_snapshot_task
        collect_orderbook_snapshot_task.delay()

        # Add other tasks if needed (e.g., Funding, OnChain, GitHub)
        # from .funding_collector import collect_funding_rates_task
        # collect_funding_rates_task.delay() # Task fetches its own symbols
        # from .onchain_collector import collect_onchain_data_task
        # collect_onchain_data_task.delay() # Task uses hardcoded symbols
        # from .github_collector import collect_github_activity_task
        # collect_github_activity_task.delay() # Task reads from file

        print("[MASTER] All data collection tasks triggered.")
    except Exception as e:
        print(f"[!] An error occurred in the master collection task: {e}")

# =============================================================================
#  Binance Historical Data Collector (Corrected)
# =============================================================================
@celery_app.task(name="tasks.collect_binance_historical_data")
def collect_binance_historical_data():
    # Increased log level for this task temporarily
    logger = logging.getLogger(__name__)
    logger.info("[*] Starting Binance historical data collection...")
    adapter = BinanceDataAdapter()
    symbols_bases = adapter.get_top_symbols_by_volume(limit=30)
    if not symbols_bases:
        logger.warning("[!] No symbols returned from Binance. Aborting.")
        return

    processed_symbols = 0
    failed_symbols = []
    total_rows_inserted = 0

    for sym_base in symbols_bases:
        sym_usdt = f"{sym_base}/USDT"
        db = SessionLocal()
        symbol_success = True
        rows_inserted_for_symbol = 0
        try:
            logger.info(f"Processing symbol: {sym_base}")
            # --- Fetch Spot Data ---
            spot_df = adapter.fetch_ohlcv(sym_usdt, "1h", 1000)
            if not spot_df.empty:
                if 'symbol' not in spot_df.columns: spot_df['symbol'] = sym_usdt
                spot_path = f"/app/data/{sym_base}_SPOT_1h.csv"
                try:
                    spot_df.to_csv(spot_path, index=False)
                    logger.info(f"[💾] Saved SPOT CSV → {spot_path}")
                except Exception as csv_e:
                    logger.warning(f"[!] Failed to save SPOT CSV for {sym_base}: {csv_e}")

                # --- Insert Spot Data with ON CONFLICT ---
                logger.info(f"[*] Inserting/Skipping {len(spot_df)} SPOT rows for {sym_base}...")
                # Convert DataFrame to list of dictionaries for bulk insert
                data_to_insert = spot_df.to_dict(orient='records')
                if data_to_insert:
                    # Define the insert statement with ON CONFLICT
                    stmt = pg_insert(MarketData.__table__).values(data_to_insert)
                    # Specify the conflict target (unique constraint columns) and action
                    # Assuming your constraint/index is on (symbol, timestamp)
                    stmt = stmt.on_conflict_do_nothing(
                        index_elements=['symbol', 'timestamp']
                    )
                    # Execute the statement
                    result = db.execute(stmt)
                    # result.rowcount might not be accurate for ON CONFLICT DO NOTHING
                    # We can't easily tell how many were actually inserted vs skipped here
                    # logger.info(f"[✔] Processed {len(data_to_insert)} SPOT rows for {sym_base} (inserted/skipped).")
                    # rows_inserted_for_symbol += result.rowcount # This might be 0

            # --- Fetch Futures Data ---
            futures_df = adapter.fetch_ohlcv(sym_usdt, "1h", 1000)
            if not futures_df.empty:
                 if 'symbol' not in futures_df.columns: futures_df['symbol'] = sym_usdt
                 fut_path = f"/app/data/{sym_base}_FUTURES_1h.csv"
                 try:
                    futures_df.to_csv(fut_path, index=False)
                    logger.info(f"[💾] Saved FUTURES CSV → {fut_path}")
                 except Exception as csv_e:
                    logger.warning(f"[!] Failed to save FUTURES CSV for {sym_base}: {csv_e}")

                 # --- Insert Futures Data with ON CONFLICT ---
                 logger.info(f"[*] Inserting/Skipping {len(futures_df)} FUTURES rows for {sym_base}...")
                 data_to_insert = futures_df.to_dict(orient='records')
                 if data_to_insert:
                    stmt = pg_insert(FuturesMarketData.__table__).values(data_to_insert)
                    # Assuming unique constraint on (symbol, timestamp)
                    stmt = stmt.on_conflict_do_nothing(
                        index_elements=['symbol', 'timestamp']
                    )
                    result = db.execute(stmt)
                    # logger.info(f"[✔] Processed {len(data_to_insert)} FUTURES rows for {sym_base} (inserted/skipped).")
                    # rows_inserted_for_symbol += result.rowcount # Might be 0

            # Commit after processing both tables for the symbol
            db.commit()
            processed_symbols += 1
            # Note: We don't have an accurate count of *newly* inserted rows with this method easily.

        except Exception as e:
            symbol_success = False
            failed_symbols.append(sym_base)
            logger.error(f"[!!!] Error processing {sym_base}: {e}", exc_info=True) # Log full traceback
            try:
                db.rollback()
            except Exception as rb_err:
                logger.error(f"[!] Rollback failed for {sym_base} after error: {rb_err}")
        finally:
            try:
                db.close()
            except Exception as close_err:
                 logger.error(f"[!] Error closing session for {sym_base}: {close_err}")

    logger.info(f"✅ Completed Binance historical data collection. Symbols attempted: {len(symbols_bases)}, Succeeded: {processed_symbols}, Failed: {len(failed_symbols)}. Failures: {failed_symbols}")
    # Return value might be less useful now without accurate row counts
    # return total_rows_inserted
# NOTE: Removed duplicate _safe_float and _safe_datetime_from_ms definitions
# Ensure they are imported correctly from app.utils