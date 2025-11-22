import datetime as dt
import time
from typing import List, Tuple
import numpy as np
import ccxt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.celery_app.app import celery_app
from app.db.models import OrderbookSnapshot
from app.exchange.adapters import BybitAdapter
from app.db.database import SessionLocal
from app.utils import _norm_futures_symbol

TOP_LEVELS = 50
CDV_LOOKBACK_SEC = 60  # compute CDV from last 60s trades


def vwap(levels: List[Tuple[float, float]]) -> float:
    """Compute volume-weighted average price."""
    if not levels:
        return 0.0
    arr = np.array(levels, dtype=float)
    pv = np.sum(arr[:, 0] * arr[:, 1])
    v = np.sum(arr[:, 1]) + 1e-9
    return float(pv / v)


def collect_orderbook_and_trades(symbol: str) -> dict:
    """Collect orderbook + trades data for one symbol."""
    ex = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

    ob = ex.fetch_order_book(symbol, limit=TOP_LEVELS)
    ts = dt.datetime.utcfromtimestamp(ob["timestamp"] / 1000.0).replace(tzinfo=dt.timezone.utc)

    bids = ob.get("bids", [])[:TOP_LEVELS]
    asks = ob.get("asks", [])[:TOP_LEVELS]
    bid_vol = float(np.sum([s for _, s in bids])) if bids else 0.0
    ask_vol = float(np.sum([s for _, s in asks])) if asks else 0.0
    mid = (bids[0][0] + asks[0][0]) / 2.0 if bids and asks else ob.get("nonce", 0.0)

    vwap_bids = vwap(bids)
    vwap_asks = vwap(asks)
    vw_skew = (vwap_asks - vwap_bids) / (mid + 1e-9)

    # Cumulative Delta Volume (CDV)
    since_ms = int((ts - dt.timedelta(seconds=CDV_LOOKBACK_SEC)).timestamp() * 1000)
    trades = []
    try:
        trades = ex.fetch_trades(symbol, since=since_ms, limit=1000)
    except Exception:
        trades = []

    buy_vol = sum(t["amount"] for t in trades if t.get("side") == "buy")
    sell_vol = sum(t["amount"] for t in trades if t.get("side") == "sell")
    cdv_1m = float(buy_vol - sell_vol)
    bid_ask_imb = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)

    return dict(
        timestamp=ts,
        symbol=symbol,
        bid_volume=bid_vol,
        ask_volume=ask_vol,
        mid_price=mid,
        bid_ask_imb=bid_ask_imb,
        vw_price_skew=vw_skew,
        cdv_1m=cdv_1m,
    )


def run_orderbook_snapshot(symbols: List[str]):
    """Run snapshot collection for multiple symbols and store in DB."""
    # engine = create_engine(settings.SQLALCHEMY_DATABASE_URI) # Don't need separate engine
    # Session = sessionmaker(bind=engine) # Use SessionLocal from database module
    s = SessionLocal() # Use SessionLocal

    added_count = 0
    skipped_count = 0
    try:
        for sym_raw in symbols:
            time.sleep(3.0)
            norm_symbol = _norm_futures_symbol(sym_raw) # Normalize symbol
            try:
                data = collect_orderbook_and_trades(norm_symbol) # Use normalized symbol
            except Exception as fetch_e:
                 print(f"[!] Failed to fetch orderbook for {norm_symbol}: {fetch_e}")
                 continue # Skip this symbol if fetch fails

            # Check if record already exists for this exact timestamp and symbol
            exists = s.query(OrderbookSnapshot.id).filter(
                OrderbookSnapshot.symbol == data['symbol'],
                OrderbookSnapshot.timestamp == data['timestamp']
            ).first()

            if not exists:
                rec = OrderbookSnapshot(**data)
                s.add(rec)
                added_count += 1
            else:
                skipped_count += 1

        s.commit()
        print(f"[i] Orderbook snapshots: Added {added_count}, Skipped {skipped_count} duplicates.")
    except Exception as e:
        s.rollback()
        print(f"[!] Error saving orderbook snapshots: {e}")
        # Consider re-raising e if you want the Celery task to show as failed
        # raise e
    finally:
        s.close()

# ✅ Celery wrapper task
@celery_app.task(name="tasks.collect_orderbook_snapshot")
def collect_orderbook_snapshot_task():
    """Periodic task: collect orderbook + trades for top symbols."""
    # Use paper_mode setting consistent with worker.py
    PAPER_MODE = bool(getattr(settings, "PAPER_TRADING", True))
    adapter = BybitAdapter(paper_mode=PAPER_MODE)
    # Ensure symbols are normalized if get_top_symbols_by_volume doesn't return /USDT format
    top_bases = adapter.get_top_symbols_by_volume(limit=10) # Get base symbols like BTC, ETH
    top_symbols = [_norm_futures_symbol(base) for base in top_bases] # Convert to BTC/USDT etc.

    if not top_symbols:
        print("[!] No top symbols found for orderbook collection.")
        return

    print(f"📊 Collecting orderbook snapshots for {len(top_symbols)} symbols...")
    try:
        run_orderbook_snapshot(top_symbols) # Pass normalized symbols
        # Removed print success message from here as it's now in run_orderbook_snapshot
    except Exception as e:
        # The error is already logged inside run_orderbook_snapshot's finally block
        print(f"[!] Orderbook snapshot task failed.") # Add a task-level log message