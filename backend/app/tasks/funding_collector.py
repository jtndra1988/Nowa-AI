# In app/tasks/funding_collector.py
from datetime import datetime, timedelta, timezone # Add timezone
from app.celery_app.app import celery_app
from app.db.database import SessionLocal # Use SessionLocal
from app.db.models import FundingRate
from app.exchange.adapters import BybitAdapter
from app.utils import _norm_futures_symbol, _safe_float
from backend.app.api.v1 import settings # Import helper

@celery_app.task(name="tasks.collect_funding_rates")
def collect_funding_rates_task():
    print("--- Kicking off funding rate collection job ---")
    # Use paper_mode setting consistent with worker.py
    PAPER_MODE = bool(getattr(settings, "PAPER_TRADING", True))
    adapter = BybitAdapter(paper_mode=PAPER_MODE)
    db_session = SessionLocal()

    added_count = 0
    skipped_count = 0
    now_utc = datetime.now(timezone.utc) # Use timezone-aware timestamp

    try:
        assets = adapter.get_top_symbols_by_volume(limit=30)
        if not assets:
            print("[!] No top symbols found for funding rate collection.")
            return

        perp_symbols = [f"{asset}/USDT" for asset in assets] # Ensure /USDT format

        # Assuming adapter.get_funding_rates handles fetching correctly
        # The method might need implementation or adjustments in adapters.py
        # For now, let's assume it returns a dict like {'BTC/USDT': 0.0001, ...}
        funding_rates = {} # Placeholder - Implement get_funding_rates in BybitAdapter
        # Example using fetch_funding_rates (might need adjustments based on ccxt)
        try:
             rates_data = adapter.exchange.fetch_funding_rates(symbols=perp_symbols)
             for sym, data in rates_data.items():
                 # Extract the rate - check ccxt unified structure ('fundingRate', 'info')
                 rate = data.get('fundingRate')
                 if rate is not None:
                     funding_rates[sym] = rate
                 # Fallback: Check 'info' field if necessary
                 # elif 'info' in data and 'fundingRate' in data['info']:
                 #    funding_rates[sym] = _safe_float(data['info']['fundingRate'])
        except Exception as fetch_e:
             print(f"[!] Failed to fetch funding rates via ccxt: {fetch_e}")
             # Decide whether to proceed or return

        if not funding_rates:
            print("[!] No funding rates returned from adapter.")
            return

        for symbol, rate in funding_rates.items():
            norm_symbol = _norm_futures_symbol(symbol) # Normalize symbol just in case
            safe_rate = _safe_float(rate)

            if safe_rate is None:
                print(f"[!] Skipping invalid funding rate for {symbol}: {rate}")
                continue

            # Check if a recent record exists (e.g., within last 8 hours) to avoid duplicates
            cutoff_time = now_utc - timedelta(hours=8)
            exists = db_session.query(FundingRate.id).filter(
                FundingRate.symbol == norm_symbol,
                FundingRate.timestamp >= cutoff_time # Avoid adding too frequently if rate hasn't changed
            ).first()

            if not exists:
                new_entry = FundingRate(
                    symbol=norm_symbol,
                    timestamp=now_utc,
                    funding_rate=safe_rate
                )
                db_session.add(new_entry)
                added_count += 1
            else:
                 skipped_count += 1

        db_session.commit()
        print(f"[✔] Funding rates: Added {added_count}, Skipped {skipped_count} recent duplicates.")

    except Exception as e:
        print(f"[!] ERROR during funding rate collection: {e}")
        db_session.rollback()
    finally:
        db_session.close()