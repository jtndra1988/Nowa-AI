import sys
import logging
from datetime import datetime, timedelta, timezone
from sqlalchemy import select, func

# Adjust imports to match your project structure
try:
    from app.db.database import SessionLocal
    # Import models - assuming they are available under app.db.models
    from app.db.models import (
        SentimentFusion, 
        OnchainMetrics, 
        OptionsDerivedMetrics, 
        FundingRate, 
        OrderbookSnapshot, 
        CrossAssetCorr
    )
except ImportError as e:
    print(f"Error importing app modules: {e}")
    print("Make sure you run this from the project root.")
    sys.exit(1)

# Configuration
SYMBOL = "BTC"  # Change this to the symbol you are testing
LOOKBACK_HOURS = 24

def check_table(session, model, name, candidates, since):
    try:
        # Check count of ALL rows for symbol
        stmt_all = select(func.count()).select_from(model).where(
            model.symbol.in_(candidates)
        )
        total_count = session.execute(stmt_all).scalar()

        # Check count of RECENT rows (last 24h)
        stmt_recent = select(func.count()).select_from(model).where(
            model.symbol.in_(candidates),
            model.timestamp >= since
        )
        recent_count = session.execute(stmt_recent).scalar()
        
        print(f"[{name:<25}] Total Rows: {total_count:<6} | Recent (24h): {recent_count:<6} | Status: {'✅ OK' if recent_count > 0 else '❌ EMPTY'}")
    except Exception as e:
        print(f"[{name:<25}] Error querying: {e}")

def main():
    db = SessionLocal()
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=LOOKBACK_HOURS)
    
    # Replicate the candidate logic from your code
    candidates = [SYMBOL, SYMBOL + "USDT", SYMBOL + "USD", SYMBOL + "-PERP"]
    
    print(f"--- Checking Data for {SYMBOL} (Candidates: {candidates}) ---")
    print(f"--- Looking for data since: {since.isoformat()} ---")
    print("-" * 80)

    check_table(db, SentimentFusion, "SentimentFusion", candidates, since)
    check_table(db, OnchainMetrics, "OnchainMetrics", candidates, since)
    check_table(db, OptionsDerivedMetrics, "OptionsDerivedMetrics", candidates, since)
    check_table(db, FundingRate, "FundingRate", candidates, since)
    check_table(db, OrderbookSnapshot, "OrderbookSnapshot", candidates, since)
    
    # CrossAssetCorr usually has 'base_symbol' instead of 'symbol'
    try:
        stmt = select(func.count()).select_from(CrossAssetCorr).where(CrossAssetCorr.base_symbol.in_(candidates))
        count = db.execute(stmt).scalar()
        print(f"[{'CrossAssetCorr':<25}] Total Rows: {count:<6} | Status: {'✅ OK' if count > 0 else '❌ EMPTY'}")
    except Exception as e:
         print(f"[{'CrossAssetCorr':<25}] Error: {e}")

    db.close()

if __name__ == "__main__":
    main()