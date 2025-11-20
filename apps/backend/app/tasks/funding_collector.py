from datetime import datetime, timedelta, timezone

from app.celery_app.app import celery_app
from app.db.database import SessionLocal
from app.db.models import FundingRate
from app.exchange.adapters import BybitAdapter
from app.utils import _norm_futures_symbol, _safe_float
from app.core.config import settings  # ✅ fixed import


@celery_app.task(name="tasks.collect_funding_rates")
def collect_funding_rates_task():
    print("--- Kicking off funding rate collection job ---")

    # Use the same PAPER_TRADING flag as the rest of the system
    PAPER_MODE = bool(getattr(settings, "PAPER_TRADING", True))
    adapter = BybitAdapter(paper_mode=PAPER_MODE)
    db_session = SessionLocal()

    added_count = 0
    skipped_count = 0
    now_utc = datetime.now(timezone.utc)  # timezone-aware timestamp

    try:
        # ✅ top-100 universe instead of 30
        assets = adapter.get_top_symbols_by_volume(limit=100)
        if not assets:
            print("[!] No top symbols found for funding rate collection.")
            return

        # Perp symbols in CCXT format
        perp_symbols = [f"{asset}/USDT" for asset in assets]

        funding_rates = {}

        try:
            # NOTE: ccxt.fetch_funding_rates usually returns a list; if your
            # adapter normalizes to dict, this will work as-is. Otherwise,
            # we may need to adapt this part later.
            rates_data = adapter.exchange.fetch_funding_rates(symbols=perp_symbols)

            # If rates_data is a dict-like structure
            if isinstance(rates_data, dict):
                items = rates_data.items()
            else:
                # Otherwise assume iterable of dicts with "symbol" key
                items = [(d.get("symbol"), d) for d in rates_data or []]

            for sym, data in items:
                if not sym or not isinstance(data, dict):
                    continue
                rate = data.get("fundingRate")
                if rate is not None:
                    funding_rates[sym] = rate

        except Exception as fetch_e:
            print(f"[!] Failed to fetch funding rates via ccxt: {fetch_e}")
            # We don't hard-fail the task here, just log; if nothing is fetched,
            # the "no funding rates" check below will exit cleanly.

        if not funding_rates:
            print("[!] No funding rates returned from adapter.")
            return

        cutoff_time = now_utc - timedelta(hours=8)

        for symbol, rate in funding_rates.items():
            norm_symbol = _norm_futures_symbol(symbol)  # Normalize symbol
            safe_rate = _safe_float(rate)

            if safe_rate is None:
                print(f"[!] Skipping invalid funding rate for {symbol}: {rate}")
                continue

            # Avoid duplicates in an 8h window
            exists = (
                db_session.query(FundingRate.id)
                .filter(
                    FundingRate.symbol == norm_symbol,
                    FundingRate.timestamp >= cutoff_time,
                )
                .first()
            )

            if not exists:
                new_entry = FundingRate(
                    symbol=norm_symbol,
                    timestamp=now_utc,
                    funding_rate=safe_rate,
                )
                db_session.add(new_entry)
                added_count += 1
            else:
                skipped_count += 1

        db_session.commit()
        print(
            f"[✔] Funding rates: Added {added_count}, "
            f"Skipped {skipped_count} recent duplicates."
        )

    except Exception as e:
        print(f"[!] ERROR during funding rate collection: {e}")
        db_session.rollback()
    finally:
        db_session.close()
