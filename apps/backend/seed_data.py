import sys
import random
import ccxt
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session

# Try to import app modules
try:
    from app.db.database import SessionLocal
    from app.db.models import (
        SentimentFusion,
        OnchainMetrics,
        FundingRate,
        OrderbookSnapshot,
        OptionsDerivedMetrics,
        CrossAssetCorr
    )
    print("✅ App modules imported successfully.")
except ImportError as e:
    print(f"❌ Error importing app modules: {e}")
    print("Run this script inside the backend container!")
    sys.exit(1)

# Initialize Binance (No API keys needed for public data)
exchange = ccxt.binance({'enableRateLimit': True})

def generate_synthetic_history(session: Session, symbol: str):
    print(f"--- Seeding data for {symbol} ---")
    
    # 1. Fetch Real Price History (OHLCV) for the last 24 hours
    # We use this to make the synthetic sentiment/metrics look correlated and realistic
    try:
        ohlcv = exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe='1h', limit=24)
    except Exception as e:
        print(f"⚠️ Could not fetch real prices for {symbol}, using mock trend. Error: {e}")
        ohlcv = []
        now = datetime.now(timezone.utc)
        for i in range(24):
            ts = int((now - timedelta(hours=24-i)).timestamp() * 1000)
            ohlcv.append([ts, 60000 + i*10, 60000 + i*10 + 5, 60000 + i*10 - 5, 60000 + i*10, 100])

    # Clear existing data for this symbol to avoid duplicates
    # (Optional: simplistic approach for seeding)
    
    for candle in ohlcv:
        ts_ms = candle[0]
        close_price = candle[4]
        volume = candle[5]
        
        # Convert ms timestamp to datetime
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

        # --- A. Sentiment Fusion (Synthetic) ---
        # Logic: If price is high/rising, sentiment is higher
        trend_bias = (close_price % 100) / 100.0  # Random-ish noise based on price
        sentiment_val = -1.0 + (trend_bias * 2.0) # Map to -1..1
        
        sent_obj = SentimentFusion(
            symbol=symbol,
            timestamp=dt,
            reddit_score=sentiment_val * 0.8,
            twitter_score=sentiment_val,
            news_score=sentiment_val * 0.5,
            final_sentiment=sentiment_val
        )
        session.add(sent_obj)

        # --- B. Funding Rates (Real-ish) ---
        # Most funding rates are around 0.01% (0.0001)
        fund_obj = FundingRate(
            symbol=symbol,
            timestamp=dt,
            funding_rate=0.0001 + (random.uniform(-0.00005, 0.00005)),
            predicted_rate=0.0001
        )
        session.add(fund_obj)

        # --- C. On-Chain Metrics (Synthetic) ---
        # Correlate active addresses with Volume
        active_addr = volume * random.uniform(0.5, 1.5)
        if active_addr < 1000: active_addr = 5000 # Floor
        
        chain_obj = OnchainMetrics(
            symbol=symbol,
            timestamp=dt,
            active_addresses=active_addr,
            transaction_volume=volume * 1000,
            whale_transaction_count=int(volume / 100)
        )
        session.add(chain_obj)

        # --- D. Orderbook / CVD (Synthetic) ---
        cvd_val = (random.random() - 0.5) * 1000000
        book_obj = OrderbookSnapshot(
            symbol=symbol,
            timestamp=dt,
            bid_depth_1m=1000000,
            ask_depth_1m=1000000,
            cdv_1m=cvd_val,
            spread_pct=0.01
        )
        session.add(book_obj)
        
        # --- E. Options Metrics (Synthetic) ---
        opt_obj = OptionsDerivedMetrics(
            symbol=symbol,
            timestamp=dt,
            total_oi=volume * 50,
            put_call_oi_ratio=0.6 + random.uniform(-0.1, 0.1),
            avg_iv_near_term=45.0 + random.uniform(-5, 5),
            gamma_exposure=100000
        )
        session.add(opt_obj)

    # --- F. Cross Asset Correlation (Single Entry) ---
    corr_obj = CrossAssetCorr(
        base_symbol=symbol,
        timestamp=datetime.now(timezone.utc),
        corr_btc_eth=0.85,
        corr_btc_dxy=-0.45,
        corr_btc_spx=0.30
    )
    session.add(corr_obj)

    session.commit()
    print(f"✅ Successfully seeded 24h data for {symbol}")

def main():
    db = SessionLocal()
    try:
        generate_synthetic_history(db, "BTC")
        generate_synthetic_history(db, "ETH")
        generate_synthetic_history(db, "SOL")
    except Exception as e:
        print(f"❌ Error during seeding: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()