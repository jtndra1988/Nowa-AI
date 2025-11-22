# app/tasks/sentiment_scorer.py

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple

from app.celery_app.app import celery_app
from app.db.database import SessionLocal
from app.db.models import SentimentData, AggregatedSentiment
from app.utils import build_top100_slug_map


@dataclass
class BucketAccumulator:
    news_sum: float = 0.0
    news_cnt: int = 0
    social_sum: float = 0.0
    social_cnt: int = 0
    global_sum: float = 0.0
    global_cnt: int = 0


def _classify_source(symbol: str, source: str) -> str:
    """
    Classify a SentimentData.source into one of: 'news', 'social', 'global'.
    Adjust rules as needed.
    """
    s = (source or "").lower()
    sym = (symbol or "").upper()

    if sym == "GLOBAL" or "feargreed" in s or "fng" in s:
        return "global"
    if "lunarcrush" in s:
        return "social"
    if "cryptopanic" in s:
        # news + social, but we treat as 'news' for now
        return "news"
    if "newsapi" in s:
        return "news"

    # Fallback: treat as news
    return "news"


@celery_app.task(name="app.tasks.sentiment_scorer.run_sentiment_scorer")
def run_sentiment_scorer() -> int:
    """
    Aggregate latest SentimentData rows into per-symbol, per-hour sentiment buckets.

    This is your "market sentiment" layer for the AI:
      - news_score: average news sentiment in the hour
      - social_score: average social sentiment in the hour
      - global_score: e.g. Fear & Greed
      - composite_score: weighted blend of the above

    Returns: number of (symbol, hour) buckets updated.
    """
    db = SessionLocal()
    now = datetime.now(timezone.utc)

    # We'll look back a bit (2 hours) to ensure we cover last complete hour(s).
    lookback_hours = 2
    cutoff = now - timedelta(hours=lookback_hours)

    # Limit to top-100 + GLOBAL
    top100_map = build_top100_slug_map()
    valid_symbols = set(top100_map.keys()) | {"GLOBAL"}

    try:
        rows = (
            db.query(SentimentData)
            .filter(SentimentData.timestamp >= cutoff)
            .all()
        )
    except Exception as e:
        print(f"[SentimentScorer] Error querying SentimentData: {e}")
        db.close()
        return 0

    buckets: Dict[Tuple[str, datetime], BucketAccumulator] = defaultdict(
        BucketAccumulator
    )

    for row in rows:
        sym = (row.symbol or "").upper()
        if sym not in valid_symbols:
            # ignore symbols outside our universe
            continue

        if row.sentiment_score is None:
            continue

        ts = row.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # truncate to hour
        bucket_start = ts.replace(minute=0, second=0, microsecond=0)

        key = (sym, bucket_start)
        acc = buckets[key]

        kind = _classify_source(sym, row.source or "")

        if kind == "news":
            acc.news_sum += float(row.sentiment_score)
            acc.news_cnt += 1
        elif kind == "social":
            acc.social_sum += float(row.sentiment_score)
            acc.social_cnt += 1
        elif kind == "global":
            acc.global_sum += float(row.sentiment_score)
            acc.global_cnt += 1
        else:
            # default to news
            acc.news_sum += float(row.sentiment_score)
            acc.news_cnt += 1

    updated = 0

    for (sym, bucket_start), acc in buckets.items():
        news_score = (
            acc.news_sum / acc.news_cnt if acc.news_cnt > 0 else None
        )
        social_score = (
            acc.social_sum / acc.social_cnt if acc.social_cnt > 0 else None
        )
        global_score = (
            acc.global_sum / acc.global_cnt if acc.global_cnt > 0 else None
        )

        # Composite sentiment:
        #   - news & social get full weight
        #   - global is softer macro context
        components = []
        weights = []

        if news_score is not None:
            components.append(news_score)
            weights.append(1.0)
        if social_score is not None:
            components.append(social_score)
            weights.append(1.0)
        if global_score is not None:
            components.append(global_score)
            weights.append(0.5)

        if components and weights:
            composite = sum(c * w for c, w in zip(components, weights)) / sum(
                weights
            )
        else:
            composite = None

        try:
            existing = (
                db.query(AggregatedSentiment)
                .filter(
                    AggregatedSentiment.symbol == sym,
                    AggregatedSentiment.bucket_start == bucket_start,
                )
                .one_or_none()
            )
            if not existing:
                existing = AggregatedSentiment(
                    symbol=sym,
                    bucket_start=bucket_start,
                )

            existing.news_score = news_score
            existing.social_score = social_score
            existing.global_score = global_score
            existing.composite_score = composite
            existing.source_count = (
                acc.news_cnt + acc.social_cnt + acc.global_cnt
            )

            db.add(existing)
            updated += 1
        except Exception as e:
            print(
                f"[SentimentScorer] Error upserting AggregatedSentiment for "
                f"{sym} @ {bucket_start}: {e}"
            )
            db.rollback()
            continue

    try:
        db.commit()
    except Exception as e:
        print(f"[SentimentScorer] Commit error: {e}")
        db.rollback()
        db.close()
        return 0

    db.close()
    print(
        f"[SentimentScorer] Aggregation complete. Buckets updated={updated}"
    )
    return updated
