# app/tasks/sentiment_collector.py
import requests
from datetime import datetime, timezone, timedelta
from app.celery_app.app import celery_app
from app.db.database import SessionLocal # Use SessionLocal
from app.db.models import SentimentData
from app.core.config import settings
from app.utils import _safe_float

# Helper function to commit and handle potential errors during commit
def _commit_or_rollback(db_session, source_name: str):
    try:
        db_session.commit()
        print(f"[i] Committed data for {source_name}")
    except Exception as commit_e:
        print(f"[!] Commit failed after {source_name}: {commit_e}")
        db_session.rollback()
# -----------------------------
# Master Sentiment Collector
# -----------------------------

@celery_app.task(name="tasks.collect_sentiment_all_sources")
def collect_sentiment_all_sources(symbols: list[str] = ["BTC", "ETH", "SOL", "BNB", "DOGE"]):
    """
    Collects multi-source sentiment data:
    NewsAPI, CryptoPanic, Santiment, LunarCrush, CoinMarketCap, and Fear & Greed.
    Each module gracefully skips if API key is missing.
    """
    db_session = SessionLocal()
    now = datetime.now(timezone.utc)
    processed_sources = 0
    print(f"[*] Starting full sentiment collection at {now}")

    # =========================
    # 1️⃣  NewsAPI
    # =========================
    if getattr(settings, "NEWSAPI_API_KEY", None):
        print("[NewsAPI] Collecting headlines...")
        try:
            for sym in symbols:
                params = {
                    "apiKey": settings.NEWSAPI_API_KEY,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "q": f'"{sym}" AND ("crypto" OR "bitcoin" OR "blockchain")',
                    "pageSize": 50,
                }
                r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
                r.raise_for_status()
                for a in r.json().get("articles", []):
                    ts_str = a.get("publishedAt")
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else now # Handle missing ts
                    db_session.merge(
                        SentimentData(
                            symbol=sym, timestamp=ts,
                            source=a.get("source", {}).get("name") or "NewsAPI",
                            headline=a.get("title", ""),
                            sentiment_score=None, # Scored later
                        )
                    )
            print("[✔] NewsAPI collection done.")
            _commit_or_rollback(db_session, "NewsAPI") # Commit after this source
            processed_sources += 1
        except Exception as e:
            print(f"[!] NewsAPI error: {e}")
            db_session.rollback()

    # =========================
    # 2️⃣  CryptoPanic
    # =========================
    if getattr(settings, "CRYPTOPANIC_API_KEY", None):
        print("[CryptoPanic] Fetching tagged crypto news...")
        try:
            params = {
                "auth_token": settings.CRYPTOPANIC_API_KEY,
                "public": "true",
                "filter": "hot",
                "kind": "news",
                "currencies": ",".join(symbols),
                "regions": "en",
            }
            r = requests.get("https://cryptopanic.com/api/developer/v2/posts/", params=params, timeout=10)
            r.raise_for_status()
            for post in r.json().get("results", []):
                ts_str = post.get("published_at")
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else now
                src = post.get("source", {}).get("title", "CryptoPanic")
                headline = post.get("title", "")
                s = post.get("sentiment", "")
                score = 0.0
                if s == "positive":
                    score = 0.6
                elif s == "negative":
                    score = -0.6
                elif s == "bullish":
                    score = 0.9
                elif s == "bearish":
                    score = -0.9
                elif s == "important":
                    score = 0.3
                db_session.merge(
                     SentimentData(
                         symbol=(post.get("currencies", [{}])[0].get("code", "ALL")),
                         timestamp=ts, source=f"CryptoPanic/{src}",
                         headline=headline, sentiment_score=score,
                     )
                 )
            print("[✔] CryptoPanic done.")
            _commit_or_rollback(db_session, "CryptoPanic") # Commit after this source
            processed_sources += 1
        except Exception as e:
            print(f"[!] CryptoPanic error: {e}")
            db_session.rollback()

    # =========================
    # 3️⃣  Santiment Whale Tx
    # =========================
    if getattr(settings, "SANTIMENT_API_KEY", None):
        print("[Santiment] Fetching whale metrics...")
        try:
            for sym in symbols:
                q = {
                    "query": f"""
                    {{
                      getMetric(metric: "whale_transactions_count") {{
                        timeseriesData(
                          slug: "{sym.lower()}",
                          from: "utc_now-1h",
                          to: "utc_now",
                          interval: "1h"
                        ) {{
                          datetime
                          value
                        }}
                      }}
                    }}
                    """
                }
                r = requests.post(
                    "https://api.santiment.net/graphql",
                    json=q,
                    headers={"Authorization": f"Apikey {settings.SANTIMENT_API_KEY}"},
                    timeout=10,
                )
                r.raise_for_status()
                data = r.json().get("data", {}).get("getMetric", {}).get("timeseriesData", [])
                for d in data:
                    ts_str = d.get("datetime")
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else now
                    val = d["value"]
                    score = min(val / 50.0, 1.0)
                    db_session.merge(
                         SentimentData(
                             symbol=sym, timestamp=ts, source="Santiment",
                             headline=f"Whale txn count {val:.0f} for {sym}",
                             sentiment_score=score,
                         )
                     )
            print("[✔] Santiment done.")
            _commit_or_rollback(db_session, "Santiment") # Commit after this source
            processed_sources += 1
        except Exception as e:
            print(f"[!] Santiment error: {e}")
            db_session.rollback()

    # =========================
    # 4️⃣  LunarCrush
    # =========================
    if getattr(settings, "LUNARCRUSH_API_KEY", None):
        print("[LunarCrush] Fetching social sentiment...")
        try:
            for sym in symbols:
                url = f"https://lunarcrush.com/api3/assets"
                params = {
                    "symbol": sym,
                    "interval": "1h",
                    "data": "assets",
                    "time_series_indicators": "social_score,social_volume",
                }
                headers = {"Authorization": f"Bearer {settings.LUNARCRUSH_API_KEY}"}
                r = requests.get(url, params=params, headers=headers, timeout=10)
                r.raise_for_status()
                data = r.json().get("data", [])
                for d in data:
                    ts_str = d.get("timeSeries", [{}])[0].get("timestamp")
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else now
                    score = d["timeSeries"][0].get("social_score", 0)
                    db_session.merge(
                         SentimentData(
                             symbol=sym, timestamp=ts, source="LunarCrush",
                             headline=f"Social score={score}",
                             sentiment_score=min(score / 100.0, 1.0), # Use safe_float? score should be numeric
                         )
                     )
            print("[✔] LunarCrush done.")
            _commit_or_rollback(db_session, "LunarCrush") # Commit after this source
            processed_sources += 1
        except Exception as e:
            print(f"[!] LunarCrush error: {e}")
            db_session.rollback()

    # =========================
    # 5️⃣  CoinMarketCap (Market Context)
    # =========================
    if getattr(settings, "COINMARKETCAP_API_KEY", None):
        print("[CoinMarketCap] Fetching global metrics...")
        try:
            headers = {"X-CMC_PRO_API_KEY": settings.COINMARKETCAP_API_KEY}
            g = requests.get("https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest", headers=headers, timeout=10)
            g.raise_for_status()
            data = g.json()["data"]
            btc_dom = data["btc_dominance"]
            mcap = data["quote"]["USD"]["total_market_cap"]
            vol = data["quote"]["USD"]["total_volume_24h"]
            db_session.merge(
                SentimentData(
                    symbol="GLOBAL", timestamp=now, source="CoinMarketCap",
                    headline=f"Global cap={mcap/1e9:.2f}B, BTC dom={btc_dom:.1f}%, vol={vol/1e9:.2f}B",
                    sentiment_score=0.3 if btc_dom < 45 else -0.3, # Example logic
                )
            )
            print("[✔] CoinMarketCap done.")
            _commit_or_rollback(db_session, "CoinMarketCap") # Commit after this source
            processed_sources += 1
        except Exception as e:
            print(f"[!] CoinMarketCap error: {e}")
            db_session.rollback()

    # =========================
    # 6️⃣  Fear & Greed Index (no key)
    # =========================
    print("[Alt.me] Fetching Fear & Greed index...")
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=10)
        fg = _safe_float(r.json()["data"][0]["value"]) # Use safe_float
        score = (fg - 50) / 50 if fg is not None else 0.0 # Calculate score safely
        db_session.merge(
            SentimentData(
                symbol="GLOBAL", timestamp=now, source="Alternative.me",
                headline="Fear & Greed Index",
                sentiment_score=score,
            )
        )
        print("[✔] Fear & Greed done.")
        _commit_or_rollback(db_session, "Fear & Greed") # Commit after this source
        processed_sources += 1
    except Exception as e:
        print(f"[!] Fear & Greed error: {e}")
        db_session.rollback()

    # --------------------------
    # Commit all
    # --------------------------
    try:
        db_session.commit()
        print("[✅] All sentiment data committed successfully.")
    except Exception as e:
        db_session.rollback()
        print(f"[!] Commit failed: {e}")
    finally:
        db_session.close()
        print("[ℹ] Sentiment collection completed.")
