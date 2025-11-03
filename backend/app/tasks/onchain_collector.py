# In app/tasks/onchain_collector.py
from datetime import datetime, timezone, timedelta # Add timezone, timedelta
from app.celery_app.app import celery_app
from app.db.database import SessionLocal # Use SessionLocal
from app.db.models import OnchainMetrics
from app.exchange.onchain_adapter import OnChainAdapter
from app.utils import _safe_float # Import helper

@celery_app.task(name="tasks.collect_onchain_data")
def collect_onchain_data_task():
    print("--- Kicking off on-chain data collection job ---")
    adapter = OnChainAdapter()
    db_session = SessionLocal()

    symbols_to_track = ["btc", "eth"]
    now_utc = datetime.now(timezone.utc) # Use timezone-aware timestamp
    added_count = 0
    skipped_count = 0

    try:
        for symbol in symbols_to_track:
            # Check if a recent record exists (e.g., within last 6 hours)
            cutoff_time = now_utc - timedelta(hours=6)
            exists = db_session.query(OnchainMetrics.id).filter(
                OnchainMetrics.symbol == symbol,
                OnchainMetrics.timestamp >= cutoff_time
            ).first()

            if exists:
                print(f"[i] Skipping recent on-chain data for {symbol}.")
                skipped_count += 1
                continue

            # Fetch data only if needed
            whale_count, whale_volume = adapter.get_whale_alert_data(symbol)
            net_flow, active_addr, tx_count = adapter.get_coinmetrics_data(symbol)

            new_entry = OnchainMetrics(
                symbol=symbol,
                timestamp=now_utc,
                # Use safe_float for robustness
                whale_tx_count=_safe_float(whale_count),
                whale_volume_usd=_safe_float(whale_volume),
                exchange_net_flow_usd=_safe_float(net_flow),
                active_addresses=_safe_float(active_addr),
                total_tx=_safe_float(tx_count),
                source="CoinMetrics+WhaleAlert" # Assuming source is constant
            )
            db_session.add(new_entry)
            added_count += 1

        db_session.commit()
        print(f"[✔] On-chain metrics: Added {added_count}, Skipped {skipped_count} recent symbols.")
    except Exception as e:
        print(f"[!] ERROR during on-chain data collection: {e}")
        db_session.rollback()
    finally:
        db_session.close()