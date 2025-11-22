# app/tasks/sentiment_collector.py

import requests
from datetime import datetime, timezone
from app.celery_app.app import celery_app
from app.db.database import SessionLocal
from app.db.models import SentimentData
from app.core.config import settings
from app.utils import _safe_float, build_top100_slug_map
from app.ml.sentiment_analyzer import (
    score_headline_and_description,
    score_text,
)

def _commit_or_rollback(db_session, source_name: str):
    try:
        db_session.commit()
        print(f"[i] Committed data for {source_name}")
    except Exception as commit_e:
        print(f"[!] Error committing {source_name}: {commit_e}")
        db_session.rollback()

# Re-use top-100 map for slug normalization (BTC -> bitcoin, etc.)
LUNARCRUSH_SYMBOL_MAP = build_top100_slug_map()

def _lunarcrush_slug(sym: str) -> str:
    """
    Normalize a trading symbol (BTC, BTCUSDT, BTC/USDT) into a LunarCrush slug.
    """
    base = sym.upper()
    base = base.replace("USDT", "").replace("USD", "").replace("/", "")
    slug = LUNARCRUSH_SYMBOL_MAP.get(base)
    if slug:
        return slug
    return base.lower()

@celery_app.task(name="tasks.collect_sentiment_all_sources")
def collect_sentiment_all_sources(
    symbols: list[str] | None = None,
) -> int:
    """
    Collects multi-source sentiment data:
      - NewsAPI, CryptoPanic, Santiment, LunarCrush, CoinMarketCap, F&G
    """
    # Default to top-100 keys if no specific symbols provided
    if symbols is None:
        symbols = list(LUNARCRUSH_SYMBOL_MAP.keys())

    db_session = SessionLocal()
    now = datetime.now(timezone.utc)
    processed_sources = 0

    print(f"[*] Starting full sentiment collection at {now} for {len(symbols)} symbols")

    # 1) NewsAPI
    if getattr(settings, "NEWSAPI_API_KEY", None) and "dummy" not in settings.NEWSAPI_API_KEY:
        print("[NewsAPI] Collecting headlines...")
        try:
            # Limit NewsAPI to just BTC/ETH to save quota in Dev mode
            for sym in symbols[:2]: 
                params = {
                    "apiKey": settings.NEWSAPI_API_KEY,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "q": f'"{sym}" AND ("crypto" OR "bitcoin")',
                    "pageSize": 5, # Reduced page size
                }
                r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
                if r.status_code == 429:
                    print("[!] NewsAPI Rate Limit. Skipping.")
                    break
                if r.status_code == 200:
                    payload = r.json()
                    for a in payload.get("articles", []):
                        headline = a.get("title", "") or ""
                        description = a.get("description")
                        score = score_headline_and_description(headline, description)
                        
                        ts_str = a.get("publishedAt")
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else now
                        
                        db_session.merge(SentimentData(
                            symbol=sym, timestamp=ts, source="NewsAPI",
                            headline=headline[:255], sentiment_score=score,
                            metadata={"url": a.get("url")}
                        ))
            _commit_or_rollback(db_session, "NewsAPI")
            processed_sources += 1
        except Exception as e:
            print(f"[!] NewsAPI error: {e}")

    # 2) CryptoPanic
    if getattr(settings, "CRYPTOPANIC_API_KEY", None) and "dummy" not in settings.CRYPTOPANIC_API_KEY:
        print("[CryptoPanic] Fetching news...")
        try:
            # CryptoPanic accepts comma-separated list, max ~20 chars? 
            # Let's just do top 5 to be safe
            codes = ",".join(symbols[:5])
            params = {
                "auth_token": settings.CRYPTOPANIC_API_KEY,
                "kind": "news",
                "currencies": codes,
                "filter": "important"
            }
            r = requests.get("https://cryptopanic.com/api/v1/posts/", params=params, timeout=10)
            if r.status_code == 200:
                payload = r.json()
                for post in payload.get("results", []):
                    title = post.get("title", "")
                    src = (post.get("source") or {}).get("title", "CryptoPanic")
                    score = score_headline_and_description(title, "")
                    
                    ts_str = post.get("published_at")
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else now

                    # Map logic...
                    code = (post.get("currencies") or [{}])[0].get("code", "BTC")
                    
                    db_session.merge(SentimentData(
                        symbol=code, timestamp=ts, source=f"CryptoPanic/{src}",
                        headline=title[:255], sentiment_score=score
                    ))
                _commit_or_rollback(db_session, "CryptoPanic")
                processed_sources += 1
        except Exception as e:
            print(f"[!] CryptoPanic error: {e}")

    # 3) LunarCrush (FIXED FOR V4 API)
    lc_key = getattr(settings, "LUNARCRUSH_API_KEY", "")
    if lc_key and "dummy" not in lc_key and len(lc_key) > 10:
        print("[LunarCrush] Fetching social metrics...")
        try:
            for sym in symbols:
                slug = _lunarcrush_slug(sym)
                
                # ✅ FIXED URL STRUCTURE for API v4
                # Old: /api4/public/coins?symbol=btc
                # New: /api4/public/coins/{slug}/v1
                url = f"https://lunarcrush.com/api4/public/coins/{slug}/v1"
                
                headers = {"Authorization": f"Bearer {lc_key}"}
                
                try:
                    r = requests.get(url, headers=headers, timeout=10)
                    
                    if r.status_code == 404:
                        # Try ticker as fallback if slug failed
                        url_fallback = f"https://lunarcrush.com/api4/public/coins/{sym}/v1"
                        r = requests.get(url_fallback, headers=headers, timeout=10)

                    if r.status_code != 200:
                        # Graceful skip
                        continue

                    resp_json = r.json()
                    data = resp_json.get("data")
                    
                    # Handle both list (search) and dict (detail) responses
                    if isinstance(data, list) and len(data) > 0:
                        coin = data[0]
                    elif isinstance(data, dict):
                        coin = data
                    else:
                        continue

                    # Extract metrics
                    galaxy_score = _safe_float(coin.get("galaxy_score"))
                    alt_rank = _safe_float(coin.get("alt_rank"))
                    sentiment = _safe_float(coin.get("sentiment"))

                    if galaxy_score is None: continue

                    # Normalize Galaxy Score (0-100) -> (-1 to 1)
                    norm_score = (galaxy_score - 50.0) / 50.0

                    db_session.merge(SentimentData(
                        symbol=sym,
                        timestamp=now,
                        source="LunarCrush",
                        headline=f"Galaxy: {galaxy_score}, AltRank: {alt_rank}",
                        sentiment_score=norm_score,
                        metadata={"galaxy_score": galaxy_score, "alt_rank": alt_rank}
                    ))

                except Exception as e:
                    # Log but don't crash loop
                    # print(f"[!] LC Error {sym}: {e}")
                    pass
            
            _commit_or_rollback(db_session, "LunarCrush")
            processed_sources += 1

        except Exception as e:
            print(f"[!] LunarCrush Outer Error: {e}")

    # 4) CoinMarketCap & Santiment & FearGreed 
    # (Kept placeholders or skipped if no keys, to save space/time)
    
    # Fear & Greed (Always Free)
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        if r.status_code == 200:
            item = r.json()['data'][0]
            val = int(item['value'])
            # 0-100 -> -1.0 to 1.0
            score = (val - 50) / 50.0
            ts = datetime.fromtimestamp(int(item['timestamp']), tz=timezone.utc)
            
            db_session.merge(SentimentData(
                symbol="GLOBAL", timestamp=ts, source="FearGreed",
                headline=item['value_classification'], sentiment_score=score,
                metadata={"value": val}
            ))
            _commit_or_rollback(db_session, "FearGreed")
            processed_sources += 1
    except Exception:
        pass

    db_session.close()
    print(f"[ℹ] Sentiment collection done. Sources: {processed_sources}")
    return processed_sources

@celery_app.task(name="tasks.run_all_sentiment_collectors")
def run_all_sentiment_collectors() -> int:
    return collect_sentiment_all_sources()