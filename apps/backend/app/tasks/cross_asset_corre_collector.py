# app/tasks/cross_asset_corre_collector.py
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import text

from app.celery_app.app import celery_app
from app.core.config import settings
from app.exchange.adapters import BybitAdapter, BinanceDataAdapter
from app.db.database import SessionLocal, engine
from app.db.models import CrossAssetCorr

WINDOWS = [60, 240, 1440]  # minutes (1h, 4h, 1d)


def _fetch_df(session, sql: str, params=None) -> pd.DataFrame:
    return pd.read_sql(text(sql), session.bind, params or {}, parse_dates=["timestamp"])


def _load_series(db_engine, table: str, symbol: str, limit: int) -> pd.DataFrame:
    """
    Load latest `limit` rows of close prices for `symbol` from `table`.
    """
    sql = text(
        f"""
        SELECT timestamp, close
        FROM {table}
        WHERE symbol = :sym
        ORDER BY timestamp DESC
        LIMIT :lim
        """
    )
    return pd.read_sql(sql, db_engine, params={"sym": symbol, "lim": limit}, parse_dates=["timestamp"])




def _corre(a: pd.DataFrame, b: pd.DataFrame, window_mins: int) -> float:
    """Compute correlation on returns over a rolling window (using resampled closes)."""
    if a.empty or b.empty:
        return 0.0
    # Resample to align timestamps
    a = a.set_index("timestamp").resample(f"{window_mins}min").last()
    b = b.set_index("timestamp").resample(f"{window_mins}min").last()
    
    # --- FIX START: Normalize timezones ---
    # Convert both to timezone-naive to avoid "tz-naive vs tz-aware" join errors
    if a.index.tz is not None:
        a.index = a.index.tz_localize(None)
    if b.index.tz is not None:
        b.index = b.index.tz_localize(None)
    # --- FIX END ---
    
    # Join on index
    merged = a.join(b, how="inner", lsuffix="_a", rsuffix="_b").dropna()
    
    if len(merged) < 10:
        return 0.0
        
    merged["ret_a"] = merged["close_a"].pct_change()
    merged["ret_b"] = merged["close_b"].pct_change()
    
    return float(merged["ret_a"].corr(merged["ret_b"]) or 0.0)

def _load_macro(indicator: str, limit: int = 3000) -> pd.DataFrame:
    """
    Load macro series by indicator (e.g., 'DXY', 'NDX', 'GOLD') from macro_data.
    """
    sql = text(
        """
        SELECT timestamp, value AS close
        FROM macro_data
        WHERE indicator = :ind
        ORDER BY timestamp DESC
        LIMIT :lim
        """
    )
    return pd.read_sql(sql, engine, params={"ind": indicator, "lim": limit}, parse_dates=["timestamp"])


# ✅ FIXED: Renamed function to match the import in collectors.py
@celery_app.task(name="tasks.run_cross_asset_corre")
def run_cross_asset_corre(symbol: Optional[str] = None, window: int = 500) -> int:
    """
    Compute cross-asset correlations for dynamic top-10 symbols (plus optional `symbol`)
    vs ETH, DXY, NDX, GOLD and store in CrossAssetCorr.
    Returns the number of correlation rows inserted.
    """
    print("[*] Starting cross-asset correlation collection ...")

    # ✅ FIXED: Use the correct adapter based on settings
    use_binance = getattr(settings, "USE_BINANCE_FOR_DATA", True)
    if use_binance:
        adapter = BinanceDataAdapter()
    else:
        PAPER_MODE = bool(getattr(settings, "PAPER_TRADING", True))
        adapter = BybitAdapter(paper_mode=PAPER_MODE)

    # ✅ FIXED: Limit to Top 10 to match other collectors
    top_syms: List[str] = adapter.get_top_symbols_by_volume(limit=10) or []
    
    if symbol and symbol not in top_syms:
        top_syms.insert(0, symbol)

    # Load common series once to save DB hits
    eth_df = _load_series(engine, table="futures_market_data", symbol="ETH/USDT", limit=max(1000, window * 3))
    dxy_df = _load_macro("DXY", limit=max(1000, window * 3))
    ndx_df = _load_macro("NDX", limit=max(1000, window * 3))
    gold_df = _load_macro("GOLD", limit=max(1000, window * 3))

    written = 0
    session = SessionLocal()
    
    try:
        for base_sym in top_syms:
            # Skip ETH vs ETH correlation (redundant)
            if base_sym == "ETH": 
                continue
                
            # print(f"[→] Calculating correlations for {base_sym}...")
            
            base_df = _load_series(
                engine, table="futures_market_data", symbol=f"{base_sym}/USDT", limit=max(1000, window * 3)
            )
            
            if base_df.empty:
                continue

            for w in WINDOWS:
                row = CrossAssetCorr(
                    timestamp=datetime.now(timezone.utc),
                    base_symbol=base_sym,
                    window_minutes=w,
                    corr_btc_eth=_corre(base_df, eth_df, w),
                    corr_btc_dxy=_corre(base_df, dxy_df, w),
                    corr_btc_ndx=_corre(base_df, ndx_df, w),
                    corr_btc_gold=_corre(base_df, gold_df, w),
                )
                session.add(row)
                written += 1
            session.commit()
            
    except Exception as e:
        print(f"[!] Cross-asset correlation error: {e}")
        session.rollback()
    finally:
        session.close()
        print(f"[✔] Cross-asset correlations updated. Rows written: {written}")
        
    return written