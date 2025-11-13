# In app/tasks/sentiment_scorer.py

from app.celery_app.app import celery_app
from app.db.database import SessionLocal, get_db
from app.db.models import SentimentData
from app.sentiment.analyzer import get_sentiment_score

@celery_app.task(name="tasks.score_headlines")
def score_headlines_task():
    """
    Finds a batch of unscored headlines, analyzes them, and
    updates them with a sentiment score.
    """
    print("[*] Starting headline scoring task...")
    db_session = next(get_db())
    
    try:
        # --- FIX: Process in smaller batches (e.g., 100 at a time) ---
        unscored_headlines = db_session.query(SentimentData)\
            .filter(SentimentData.sentiment_score == None)\
            .limit(100)\
            .all()
        # -----------------------------------------------------------
        
        if not unscored_headlines:
            print("[+] No new headlines to score.")
            return

        print(f"[*] Found {len(unscored_headlines)} unscored headlines to process in this batch...")
        
        for item in unscored_headlines:
            score = get_sentiment_score(item.headline)
            item.sentiment_score = score
            
        db_session.commit()
        print(f"[✔] Successfully scored and updated {len(unscored_headlines)} headlines.")
        
    except Exception as e:
        print(f"[!] ERROR during headline scoring: {e}")
        db_session.rollback()
    finally:
        db_session.close()
@celery_app.task(name="app.tasks.sentiment_scorer.run_sentiment_scorer")
def run_sentiment_scorer(batch_size: int = 100) -> int:
    """
    Periodic task:
    - Finds a batch of unscored headlines in sentiment_data
    - Scores them using get_sentiment_score(...)
    - Saves sentiment_score back to DB

    Returns: number of rows scored in this run.
    """
    print("[*] Starting headline scoring task...")
    db_session = SessionLocal()
    processed = 0

    try:
        # Fetch a limited batch of unscored headlines
        unscored_headlines = (
            db_session.query(SentimentData)
            .filter(SentimentData.sentiment_score == None)  # noqa: E711
            .limit(batch_size)
            .all()
        )

        if not unscored_headlines:
            print("[+] No new headlines to score.")
            return 0

        print(f"[*] Found {len(unscored_headlines)} unscored headlines to process...")

        for item in unscored_headlines:
            try:
                score = get_sentiment_score(item.headline or "")
            except Exception as e:
                print(f"[!] Error scoring headline id={item.id}: {e}")
                score = 0.0  # fall back to neutral
            item.sentiment_score = score
            processed += 1

        db_session.commit()
        print(f"[✔] Successfully scored and updated {processed} headlines.")
        return processed

    except Exception as e:
        print(f"[!] ERROR during headline scoring: {e}")
        db_session.rollback()
        raise
    finally:
        db_session.close()
        print("[ℹ] Sentiment scoring run finished.")        