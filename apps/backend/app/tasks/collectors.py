# app/tasks/collectors.py
import asyncio
import requests
import sqlalchemy
import yfinance as yf
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from app.celery_app.app import celery_app
from app.core.config import settings
from app.db.database import SessionLocal
from app.db.models import (
    FearAndGreedIndex, MacroData, MarketData, FuturesMarketData,
    FundingRate, OrderbookSnapshot, OptionsChain
)
from app.exchange.adapters import BybitAdapter, BinanceDataAdapter
from app.utils import _norm_futures_symbol, _safe_float, _utcnow
import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Helper function
# --------------------------------------------------------------------
def _safe_datetime_from_ms(val: Any) -> Optional[datetime]:
    """Safely convert a millisecond timestamp to a datetime object."""
    if val is None:
        return None
    try:
        timestamp_sec = int(val) / 1000.0
        return datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
    except (ValueError, TypeError):
        try:
            # Fallback for ISO strings etc.
            dt = pd.to_datetime(val, utc=True).to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

# =============================================================================
#  TASK 1: Collect OPTIONS Data
# =============================================================================
@celery_app.task(name="tasks.collect_options", bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def collect_options_task(self, symbol: str, adapter_payload: Optional[dict] = None) -> int:
    task_id = self.request.id
    logger.debug(f"Task {task_id}: Starting options collection for {symbol}.")

    tz_now = _utcnow()
    rows: List[dict] = []
    BATCH_SIZE = 50

    # --- Data fetching ---
    if adapter_payload is None:
        use_binance = getattr(settings, "USE_BINANCE_FOR_DATA", True)
        adapter = BinanceDataAdapter() if use_binance else BybitAdapter()
        
        fetcher_names = ["get_options_chain_snapshot", "get_options_chain", "fetch_options_chain"]
        for name in fetcher_names:
            if hasattr(adapter, name):
                try:
                    res = getattr(adapter, name)(symbol)
                    if isinstance(res, dict) and "rows" in res: rows = res["rows"]
                    elif isinstance(res, list): rows = res
                    if rows:
                        break
                except Exception as e:
                    if "404" not in str(e): # Ignore expected 404s
                         logger.warning(f"Task {task_id}: Options fetch ({symbol}) failed: {e}")
        
        if not rows and adapter_payload is not None: rows = adapter_payload.get("rows", [])
    else: rows = adapter_payload.get("rows", [])

    if not rows:
        return 0

    written_total = 0
    written_batch = 0
    db = SessionLocal()
    
    try:
        for i, r in enumerate(rows):
            try:
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

                # Existence check
                exists = db.query(OptionsChain.id).filter(
                    OptionsChain.symbol == base_symbol, OptionsChain.expiry == expiry_dt,
                    OptionsChain.strike == strike_price, OptionsChain.option_type == ot,
                    OptionsChain.timestamp == ts,
                ).first()
                if exists: continue

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

                if written_batch >= BATCH_SIZE:
                    db.commit()
                    written_batch = 0
            
            except Exception as e_row:
                 continue

        if written_batch > 0:
            db.commit()

        logger.info(f"Task {task_id}: Options {symbol}: Written {written_total} rows.")
        return written_total

    except Exception as e_task:
        logger.error(f"Task {task_id}: CRITICAL error in collect_options_task ({symbol}): {e_task}")
        db.rollback()
        raise e_task
    finally:
        db.close()

# =============================================================================
#  TASK 2: Collect FUTURES Data
# =============================================================================
@celery_app.task(name="tasks.collect_futures", autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def collect_futures_task(underlying_symbol: str):
    # Note: Removed print() to reduce noise, rely on logger if needed
    use_binance = getattr(settings, "USE_BINANCE_FOR_DATA", True)
    
    if use_binance:
        adapter = BinanceDataAdapter()
        futures_symbol = f"{underlying_symbol}/USDT"
        ticker_params = {}
    else:
        PAPER_MODE = bool(getattr(settings, "PAPER_TRADING", True))
        adapter = BybitAdapter(paper_mode=PAPER_MODE)
        futures_symbol = f"{underlying_symbol}/USDT"
        ticker_params = {'category': 'linear'}

    try:
        ticker_data = adapter.get_ticker(futures_symbol, params=ticker_params)
    except Exception as fetch_e:
        return

    if not ticker_data or not ticker_data.get('timestamp'):
        return

    norm_symbol = _norm_futures_symbol(ticker_data.get("symbol") or futures_symbol)

    db_session = SessionLocal()
    try:
        ts_ms = ticker_data.get('timestamp')
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

        exists = db_session.query(FuturesMarketData.id).filter(
            FuturesMarketData.symbol == norm_symbol,
            FuturesMarketData.timestamp == ts
        ).first()

        if not exists:
            futures_data_record = FuturesMarketData(
                symbol=norm_symbol, timestamp=ts,
                open=_safe_float(ticker_data.get('openPrice') or ticker_data.get('open')),
                high=_safe_float(ticker_data.get('highPrice') or ticker_data.get('high')),
                low=_safe_float(ticker_data.get('lowPrice') or ticker_data.get('low')),
                close=_safe_float(ticker_data.get('lastPrice') or ticker_data.get('last') or ticker_data.get('close')),
                volume=_safe_float(ticker_data.get('volume24h') or ticker_data.get('baseVolume')),
            )
            db_session.add(futures_data_record)
            db_session.commit()
            logger.info(f"[✔] Saved Futures: {norm_symbol}")
    except Exception as e:
        db_session.rollback()
        raise e
    finally:
        db_session.close()

# =============================================================================
#  TASK 3 & 4: Fear/Greed & Macro
# =============================================================================
@celery_app.task(name="tasks.collect_fear_and_greed", autoretry_for=(Exception,), retry_backoff=True)
def collect_fear_and_greed_task():
    db_session = SessionLocal()
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        data = response.json()['data'][0]
        ts_dt = datetime.fromtimestamp(int(data['timestamp']), tz=timezone.utc)
        exists = db_session.query(FearAndGreedIndex.id).filter(FearAndGreedIndex.timestamp == ts_dt).first()
        if not exists:
            db_session.add(FearAndGreedIndex(timestamp=ts_dt, value=int(data['value']), value_classification=data['value_classification']))
            db_session.commit()
            logger.info(f"[✔] Saved F&G: {data['value']}")
    except Exception:
        db_session.rollback()
    finally:
        db_session.close()

@celery_app.task(name="tasks.collect_macro_data", autoretry_for=(Exception,), retry_backoff=True)
def collect_macro_data_task():
    db_session = SessionLocal()
    now_utc = _utcnow()
    try:
        dxy = yf.Ticker("DX-Y.NYB")
        hist = dxy.history(period="5d")
        if not hist.empty:
            latest = hist.iloc[-1]
            ts = latest.name.to_pydatetime()
            if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
            cutoff = now_utc - timedelta(hours=12)
            if not db_session.query(MacroData.id).filter(MacroData.indicator == 'DXY', MacroData.timestamp >= cutoff).first():
                db_session.add(MacroData(timestamp=ts, indicator='DXY', value=float(latest['Close'])))
                db_session.commit()
                logger.info(f"[✔] Saved DXY: {latest['Close']:.2f}")
    except Exception:
        db_session.rollback()
    finally:
        db_session.close()

# =============================================================================
#  MASTER TASK: Run all collection jobs (DEV MODE: TOP 10)
# =============================================================================
@celery_app.task(name="tasks.collect_all_assets")
def collect_all_assets_task():
    print("--- [MASTER TASK] Kicking off all data collection jobs ---")
    try:
        use_binance = getattr(settings, "USE_BINANCE_FOR_DATA", True)
        if use_binance:
             adapter = BinanceDataAdapter() 
        else:
             PAPER_MODE = bool(getattr(settings, "PAPER_TRADING", True))
             adapter = BybitAdapter(paper_mode=PAPER_MODE)

        # ✅ UPDATED: Fetch only TOP 10 assets for DEV mode
        assets_bases = adapter.get_top_symbols_by_volume(limit=10)
        
        if not assets_bases:
            print("[!] No symbols found. Aborting.")
            return

        print(f"[*] Found top {len(assets_bases)} assets. Triggering tasks with 5.0s delay...")

        # 1. Historical Data (Runs independently)
        if isinstance(adapter, BinanceDataAdapter):
            collect_binance_historical_data.delay()

        # 2. Options & Futures per asset
        for i, asset in enumerate(assets_bases):
            # Keep the delay to be safe, even with 10 assets
            time.sleep(5.0) 
            
            collect_options_task.delay(asset)
            collect_futures_task.delay(asset)
            
            if i % 5 == 0:
                print(f"[Master] Dispatched {i}/{len(assets_bases)} assets...")

        # 3. Batch Tasks
        from .sentiment_collector import collect_sentiment_all_sources
        collect_sentiment_all_sources.delay(assets_bases)
        
        from .orderbook_collector import collect_orderbook_snapshot_task
        collect_orderbook_snapshot_task.delay()
        
        from .funding_collector import collect_funding_rates_task
        collect_funding_rates_task.delay()
        
        from .onchain_collector import collect_onchain_data_task
        collect_onchain_data_task.delay()
        
        from .github_collector import collect_github_activity_task
        collect_github_activity_task.delay()
        
        from .cross_asset_corre_collector import run_cross_asset_corre
        run_cross_asset_corre.delay()

        print("[MASTER] All data collection tasks dispatched.")
    except Exception as e:
        print(f"[!] An error occurred in the master collection task: {e}")

# =============================================================================
#  Binance Historical Data Collector (DEV MODE: TOP 10)
# =============================================================================
@celery_app.task(name="tasks.collect_binance_historical_data")
def collect_binance_historical_data():
    logger.info("[*] Starting Binance historical data collection...")
    adapter = BinanceDataAdapter()
    
    # ✅ UPDATED: Fetch only TOP 10 assets for DEV mode
    symbols_bases = adapter.get_top_symbols_by_volume(limit=10)
    
    processed = 0
    for sym_base in symbols_bases:
        time.sleep(2.0)
        
        sym_usdt = f"{sym_base}/USDT"
        db = SessionLocal()
        try:
            # Spot
            spot_df = adapter.fetch_ohlcv(sym_usdt, "1h", 1000)
            if not spot_df.empty:
                if 'symbol' not in spot_df.columns: spot_df['symbol'] = sym_usdt
                data = spot_df.to_dict(orient='records')
                stmt = pg_insert(MarketData.__table__).values(data)
                stmt = stmt.on_conflict_do_nothing(index_elements=['symbol', 'timestamp'])
                db.execute(stmt)

            # Futures
            futures_df = adapter.fetch_ohlcv(sym_usdt, "1h", 1000)
            if not futures_df.empty:
                if 'symbol' not in futures_df.columns: futures_df['symbol'] = sym_usdt
                data = futures_df.to_dict(orient='records')
                stmt = pg_insert(FuturesMarketData.__table__).values(data)
                stmt = stmt.on_conflict_do_nothing(index_elements=['symbol', 'timestamp'])
                db.execute(stmt)

            db.commit()
            processed += 1
        except Exception:
            db.rollback()
        finally:
            db.close()

    logger.info(f"✅ Backfill complete. Processed: {processed}")