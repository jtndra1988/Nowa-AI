# app/tasks/sentiment_collector.py

import requests
from datetime import datetime, timezone
from app.celery_app.app import celery_app
from app.db.database import SessionLocal
from app.db.models import SentimentData
from app.core.config import settings
from app.utils import _safe_float


def _commit_or_rollback(db_session, source_name: str):
    try:
        db_session.commit()
        print(f"[i] Committed data for {source_name}")
    except Exception as commit_e:
        print(f"[!] Commit failed after {source_name}: {commit_e}")
        db_session.rollback()


@celery_app.task(name="tasks.collect_sentiment_all_sources")
def collect_sentiment_all_sources(
    symbols: list[str] | None = None,
) -> int:
    """
    Collects multi-source sentiment data:
      - NewsAPI
      - CryptoPanic
      - Santiment
      - LunarCrush
      - CoinMarketCap
      - Fear & Greed Index

    Returns: number of 'sources' successfully processed.
    """
    if symbols is None:
        symbols = ["BTC", "ETH", "SOL", "BNB", "DOGE"]

    db_session = SessionLocal()
    now = datetime.now(timezone.utc)
    processed_sources = 0

    print(f"[*] Starting full sentiment collection at {now} for {symbols}")

    # 1) NewsAPI
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
                r = requests.get(
                    "https://newsapi.org/v2/everything",
                    params=params,
                    timeout=10,
                )
                r.raise_for_status()
                for a in r.json().get("articles", []):
                    ts_str = a.get("publishedAt")
                    ts = (
                        datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts_str else now
                    )
                    db_session.merge(
                        SentimentData(
                            symbol=sym,
                            timestamp=ts,
                            source=a.get("source", {}).get("name") or "NewsAPI",
                            headline=a.get("title", "") or "",
                            sentiment_score=None,  # scored later
                        )
                    )
            print("[✔] NewsAPI collection done.")
            _commit_or_rollback(db_session, "NewsAPI")
            processed_sources += 1
        except Exception as e:
            print(f"[!] NewsAPI error: {e}")
            db_session.rollback()

    # 2) CryptoPanic
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
            r = requests.get(
                "https://cryptopanic.com/api/developer/v2/posts/",
                params=params,
                timeout=10,
            )
            r.raise_for_status()
            for post in r.json().get("results", []):
                ts_str = post.get("published_at")
                ts = (
                    datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts_str else now
                )
                src = post.get("source", {}).get("title", "CryptoPanic")
                headline = post.get("title", "") or ""
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

                sym_code = (
                    (post.get("currencies") or [{}])[0].get("code", "ALL")
                )

                db_session.merge(
                    SentimentData(
                        symbol=sym_code,
                        timestamp=ts,
                        source=f"CryptoPanic/{src}",
                        headline=headline,
                        sentiment_score=score,
                    )
                )
            print("[✔] CryptoPanic done.")
            _commit_or_rollback(db_session, "CryptoPanic")
            processed_sources += 1
        except Exception as e:
            print(f"[!] CryptoPanic error: {e}")
            db_session.rollback()

    # 3) Santiment Whale Tx
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
                data = (
                    r.json()
                    .get("data", {})
                    .get("getMetric", {})
                    .get("timeseriesData", [])
                )
                for d in data:
                    ts_str = d.get("datetime")
                    ts = (
                        datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts_str else now
                    )
                    val = d.get("value", 0.0)
                    score = min(float(val) / 50.0, 1.0)
                    db_session.merge(
                        SentimentData(
                            symbol=sym,
                            timestamp=ts,
                            source="Santiment",
                            headline=f"Whale txn count {val:.0f} for {sym}",
                            sentiment_score=score,
                        )
                    )
            print("[✔] Santiment done.")
            _commit_or_rollback(db_session, "Santiment")
            processed_sources += 1
        except Exception as e:
            print(f"[!] Santiment error: {e}")
            db_session.rollback()

    # 4) LunarCrush
    if getattr(settings, "LUNARCRUSH_API_KEY", None):
        print("[LunarCrush] Fetching social sentiment...")
        try:
            for sym in symbols:
                url = "https://lunarcrush.com/api3/assets"
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
                    ts_str = (
                        (d.get("timeSeries") or [{}])[0].get("timestamp")
                    )
                    ts = (
                        datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts_str else now
                    )
                    score_raw = (d.get("timeSeries") or [{}])[0].get(
                        "social_score", 0
                    )
                    score = _safe_float(score_raw) or 0.0
                    db_session.merge(
                        SentimentData(
                            symbol=sym,
                            timestamp=ts,
                            source="LunarCrush",
                            headline=f"Social score={score}",
                            sentiment_score=min(float(score) / 100.0, 1.0),
                        )
                    )
            print("[✔] LunarCrush done.")
            _commit_or_rollback(db_session, "LunarCrush")
            processed_sources += 1
        except Exception as e:
            print(f"[!] LunarCrush error: {e}")
            db_session.rollback()

    # 5) CoinMarketCap (Global Context)
    if getattr(settings, "COINMARKETCAP_API_KEY", None):
        print("[CoinMarketCap] Fetching global metrics...")
        try:
            headers = {"X-CMC_PRO_API_KEY": settings.COINMARKETCAP_API_KEY}
            g = requests.get(
                "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest",
                headers=headers,
                timeout=10,
            )
            g.raise_for_status()
            data = g.json()["data"]
            btc_dom = float(data.get("btc_dominance", 0.0))
            quote = data.get("quote", {}).get("USD", {})
            mcap = float(quote.get("total_market_cap", 0.0))
            vol = float(quote.get("total_volume_24h", 0.0))
            db_session.merge(
                SentimentData(
                    symbol="GLOBAL",
                    timestamp=now,
                    source="CoinMarketCap",
                    headline=(
                        f"Global cap={mcap/1e9:.2f}B, "
                        f"BTC dom={btc_dom:.1f}%, "
                        f"vol={vol/1e9:.2f}B"
                    ),
                    sentiment_score=0.3 if btc_dom < 45 else -0.3,
                )
            )
            print("[✔] CoinMarketCap done.")
            _commit_or_rollback(db_session, "CoinMarketCap")
            processed_sources += 1
        except Exception as e:
            print(f"[!] CoinMarketCap error: {e}")
            db_session.rollback()

    # 6) Fear & Greed Index
    print("[Alt.me] Fetching Fear & Greed index...")
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=10)
        r.raise_for_status()
        fg = _safe_float(r.json()["data"][0]["value"])
        score = ((fg - 50) / 50) if fg is not None else 0.0
        db_session.merge(
            SentimentData(
                symbol="GLOBAL",
                timestamp=now,
                source="Alternative.me",
                headline="Fear & Greed Index",
                sentiment_score=score,
            )
        )
        print("[✔] Fear & Greed done.")
        _commit_or_rollback(db_session, "Fear & Greed")
        processed_sources += 1
    except Exception as e:
        print(f"[!] Fear & Greed error: {e}")
        db_session.rollback()

    db_session.close()
    print(f"[ℹ] Sentiment collection completed. Sources processed={processed_sources}")
    return processed_sources


@celery_app.task(name="tasks.run_all_sentiment_collectors")
def run_all_sentiment_collectors() -> int:
    """
    Thin wrapper so worker.py can schedule a single task.
    """
    return collect_sentiment_all_sources()
