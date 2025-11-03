# app/tasks/options_metrics_collector.py
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional
from sqlalchemy import text
from sqlalchemy.orm import Session # Import Session type hint

from app.celery_app.app import celery_app
from app.db.database import SessionLocal # Use SessionLocal for db connection
from app.db.models import OptionsDerivedMetrics # Ensure model is imported
from app.core.config import settings # Import settings if db_url fallback is needed

# Import the ETL function - adjust path if needed
try:
    from app.ml.etl_options_derived import run as run_options_derived_etl
except ImportError:
    print("[!!!] Failed to import run_options_derived_etl from app.ml.etl_options_derived. Task will fail.")
    # Define a dummy function to allow worker to start, but task will crash when called
    def run_options_derived_etl(*args, **kwargs):
        raise ImportError("etl_options_derived.run could not be imported")

@celery_app.task(name="tasks.calculate_options_derived_metrics")
def calculate_options_derived_metrics_task(
    symbol: str, # Expects base symbol like 'BTC'
    lookback_hours: int = 24,
    # db_url is likely not needed if ETL uses SessionLocal or engine directly
    # db_url: Optional[str] = None,
    base_table_for_spot: str = "futures_market_data", # Default table for spot price
    contract_multiplier: float = 1.0, # Example: 1.0 for BTC/ETH on Bybit
) -> int:
    """
    Compute options-derived metrics for `symbol` using etl_options_derived.run
    and store results. Returns count of new rows in the lookback window.
    """
    end_ts = datetime.now(timezone.utc)
    start_ts = end_ts - timedelta(hours=lookback_hours)
    count = 0
    # Define db variable for the finally block
    db: Optional[Session] = None

    print(f"[*] Calculating derived options metrics for {symbol} ({start_ts} to {end_ts})")

    try:
        # --- Run the ETL ---
        # Assuming run_options_derived_etl handles its own DB session internally
        # It should accept start/end times and the base symbol
        # If it needs db_url, pass settings.SQLALCHEMY_DATABASE_URI explicitly
        run_options_derived_etl(
            symbol=symbol,
            # db_url=settings.SQLALCHEMY_DATABASE_URI, # Pass if needed by ETL func
            base_table_for_spot=base_table_for_spot,
            contract_multiplier=contract_multiplier,
            start=start_ts.isoformat(),
            end=end_ts.isoformat(),
        )
        print(f"[+] ETL run completed for {symbol}.")

        # --- Count rows written (using a separate, clean session) ---
        print(f"[*] Counting new rows for {symbol} in the window...")
        db = SessionLocal() # Create a new session JUST for this count query
        count_sql = text(
            """
            SELECT COUNT(*) FROM options_derived_metrics
            WHERE symbol = :sym AND timestamp BETWEEN :s AND :e
            """
        )
        res = db.execute(count_sql, {"sym": symbol, "s": start_ts, "e": end_ts}).scalar()
        count = int(res or 0)
        print(f"[i] Found {count} new derived options metrics rows for {symbol}.")
        # No commit needed for SELECT

    except ImportError as imp_err:
         # Log if the ETL function couldn't be imported
         print(f"[!!!] ETL function import error prevented execution: {imp_err}")
         raise # Re-raise to fail the task clearly
    except Exception as e:
        # Catch errors from ETL run or the count query
        print(f"[!] ERROR during calculate_options_derived_metrics for {symbol}: {e}")
        # Rollback the session used for COUNT query (though it likely didn't change anything)
        if db:
            try: db.rollback()
            except: pass
        raise e # Re-raise the exception to mark task as failed
    finally:
        # Ensure the session used for COUNT is always closed
        if db:
            try: db.close()
            except: pass

    return count