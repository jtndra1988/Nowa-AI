# app/tasks/sentiment_fusion_collector.py
import pandas as pd
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
from app.celery_app.app import celery_app
from app.db.database import SessionLocal # Use SessionLocal
from app.db.models import SentimentData, SentimentFusion
from app.utils import _safe_float # Import helper

@celery_app.task(name="tasks.run_sentiment_fusion")
def run_sentiment_fusion(hours_window: int = 6):
    print(f"[*] Running Sentiment Fusion (last {hours_window} hours)...")
    db = SessionLocal() # Use SessionLocal explicitly
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(hours=hours_window)

    try:
        # --- Load recent sentiment data ---
        query = text("""
            SELECT symbol, source, sentiment_score, timestamp
            FROM sentiment_data
            WHERE timestamp >= :start_time AND sentiment_score IS NOT NULL
        """)
        # Use session directly, not db.bind
        df = pd.read_sql(query, db.connection(), params={"start_time": start_time}, parse_dates=["timestamp"])
        if df.empty:
            print("[!] No sentiment data found for fusion window.")
            return

        df["sentiment_score"] = df["sentiment_score"].apply(_safe_float).clip(-1, 1) # Clean score
        df = df.dropna(subset=['sentiment_score']) # Drop rows where score couldn't be cleaned
        df["source_lower"] = df["source"].str.lower()

        symbols = df["symbol"].unique()
        fusion_records_to_add = []
        fusion_records_to_update = []

        for sym in symbols:
            subset = df[df["symbol"] == sym]
            # ... (Aggregations and Weighted final sentiment logic - unchanged) ...
            avg_news = subset[
                subset["source_lower"].str.contains("newsapi|cryptopanic|news|headline")
            ]["sentiment_score"].mean()

            whale = subset[
                subset["source_lower"].str.contains("santiment|whale")
            ]["sentiment_score"].mean()

            social = subset[
                subset["source_lower"].str.contains("lunarcrush|social")
            ]["sentiment_score"].mean()

            fear_greed = subset[
                subset["source_lower"].str.contains(
                    "feargreedindex|fear & greed|feargreed"
                )
            ]["sentiment_score"].mean()

            market_ctx = subset[
                subset["source_lower"].str.contains("cmc/global_metrics|coinmarketcap|cmc")
            ]["sentiment_score"].mean()


            weights = { "avg_news_sentiment": 1.0, "whale_emotion": 1.3, "social_score": 1.2, "fear_greed_index": 0.8, "market_ctx": 1.0, }
            components = { "avg_news_sentiment": avg_news, "whale_emotion": whale, "social_score": social, "fear_greed_index": fear_greed, "market_ctx": market_ctx, }
            valid = {k: v for k, v in components.items() if pd.notna(v)}
            final_sentiment = sum(v * weights[k] for k, v in valid.items()) / sum(weights[k] for k in valid) if valid else 0.0

            # Prepare data record
            record_data = {
                "symbol": sym, "timestamp": now,
                "avg_news_sentiment": round(avg_news if pd.notna(avg_news) else 0.0, 4),
                "whale_emotion": round(whale if pd.notna(whale) else 0.0, 4),
                "social_score": round(social if pd.notna(social) else 0.0, 4),
                "fear_greed_index": round(fear_greed if pd.notna(fear_greed) else 0.0, 4),
                "final_sentiment": round(final_sentiment, 4),
            }

            # Check if a recent record exists to update, otherwise add
            # (Define 'recent' - e.g., within the same hour?)
            cutoff_time = now - timedelta(minutes=70) # Example: Update if within last 70 mins
            existing = db.query(SentimentFusion).filter(
                SentimentFusion.symbol == sym,
                SentimentFusion.timestamp >= cutoff_time
            ).order_by(SentimentFusion.timestamp.desc()).first()

            if existing:
                # Update existing record
                for key, value in record_data.items():
                    setattr(existing, key, value)
                fusion_records_to_update.append(existing) # Add to list (though session tracks changes)
            else:
                # Create new record
                fusion_records_to_add.append(SentimentFusion(**record_data))

        if fusion_records_to_add:
            db.add_all(fusion_records_to_add)

        # No explicit add needed for updated records, session tracks them

        db.commit()
        print(f"[✔] Sentiment fusion complete: Added {len(fusion_records_to_add)}, Updated {len(fusion_records_to_update)} symbols.")

    except Exception as e:
        db.rollback()
        print(f"[!] Sentiment fusion error: {e}")
    finally:
        db.close()
        print("[ℹ] Fusion process finished.")