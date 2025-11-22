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
    # Strip common suffixes & separators
    base = base.replace("USDT", "").replace("USD", "")
    base = base.replace("/", "")

    slug = LUNARCRUSH_SYMBOL_MAP.get(base)
    if slug:
        return slug

    # Fallback: just lowercase the base
    return base.lower()


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
    # ✅ No hardcoded symbols: default to our dynamic top-100 universe
    if symbols is None:
        # Default to top-100 symbols based on our slug map (keys)
        symbols = list(LUNARCRUSH_SYMBOL_MAP.keys())

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
                    "pageSize": 20,
                }
                r = requests.get(
                    "https://newsapi.org/v2/everything",
                    params=params,
                    timeout=30,
                )
                r.raise_for_status()
                try:
                    payload = r.json()
                except ValueError:
                    print(f"[!] NewsAPI: Non-JSON response for {sym}: {r.text[:200]}")
                    continue

                for a in payload.get("articles", []):
                    ts_str = a.get("publishedAt")
                    ts = (
                        datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts_str
                        else now
                    )
                    headline = a.get("title", "") or ""
                    description = a.get("description")
                    score = score_headline_and_description(headline, description)
                    db_session.merge(
                        SentimentData(
                        symbol=sym,
                        timestamp=ts,
                        source=a.get("source", {}).get("name") or "NewsAPI",
                        headline=headline,
                        sentiment_score=score,
                        metadata={
                        "url": a.get("url"),
                        "description": description,
                    },
                )
            )

            print("[✔] NewsAPI done.")
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
                "kind": "news",
                "currencies": ",".join(symbols),
            }
            r = requests.get(
                "https://cryptopanic.com/api/v1/posts/",
                params=params,
                timeout=30,
            )
            r.raise_for_status()
            try:
                payload = r.json()
            except ValueError:
                print(f"[!] CryptoPanic: Non-JSON response: {r.text[:200]}")
                payload = {}

            for post in payload.get("results", []):
                ts_str = post.get("published_at")
                ts = (
                datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts_str
                else now
            )
            title = post.get("title") or ""
            desc = post.get("description") or ""
            src = (post.get("source") or {}).get("title", "CryptoPanic")
            headline = f"{src}: {title}"

            # 1) Lexical sentiment from title/description (free, local)
            lex_score = score_headline_and_description(title, desc)

            # 2) Map CryptoPanic 'sentiment' tag to numeric
            s = (post.get("sentiment") or "").lower()
            tag_score = 0.0
            if s == "positive":
               tag_score = 0.6
            elif s == "negative":
               tag_score = -0.6
            elif s == "bullish":
               tag_score = 0.9
            elif s == "bearish":
               tag_score = -0.9
            elif s == "important":
               tag_score = 0.3

            # 3) Combine both if tag is present, else only lex
            if s:
                score = 0.5 * lex_score + 0.5 * tag_score
            else:
                score = lex_score

            sym_code = ((post.get("currencies") or [{}])[0].get("code", "ALL"))

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

    # 3) Santiment Whale Tx (HARDENED, free-plan friendly window)
    if getattr(settings, "SANTIMENT_API_KEY", None):
        print("[Santiment] Fetching whale metrics...")
        try:
            for sym in symbols:
                slug = LUNARCRUSH_SYMBOL_MAP.get(sym.upper(), sym.lower())
                q = {
                    "query": f"""
                    {{
                      getMetric(metric: "whale_transaction_count_1m_usd_to_inf") {{
                        timeseriesData(
                          slug: "{slug}",
                          from: "utc_now-60d",
                          to: "utc_now-30d",
                          interval: "1h"
                        ) {{
                          datetime
                          value
                        }}
                      }}
                    }}
                    """
                }
                try:
                    r = requests.post(
                        "https://api.santiment.net/graphql",
                        json=q,
                        headers={"Authorization": f"Apikey {settings.SANTIMENT_API_KEY}"},
                        timeout=30,
                    )
                    r.raise_for_status()
                except requests.RequestException as re:
                    print(f"[!] Santiment HTTP error for {sym}: {re}")
                    continue

                try:
                    resp_json = r.json()
                except ValueError:
                    print(
                        f"[!] Santiment: Non-JSON response for {sym}: {r.text[:200]}"
                    )
                    continue

                root = (resp_json or {}).get("data") or {}
                metric = root.get("getMetric") or {}
                series = metric.get("timeseriesData") or []

                if not isinstance(series, list) or not series:
                    print(
                        f"[!] Santiment: No timeseriesData for {sym}. Raw={resp_json}"
                    )
                    continue

                for d in series:
                    ts_str = d.get("datetime")
                    ts = (
                        datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts_str
                        else now
                    )
                    raw_value = _safe_float(d.get("value"))
                    if raw_value is None:
                        continue

                    # Scale whale tx count into a rough sentiment score
                    # (you can adjust this heuristic later)
                    score = min(max(raw_value / 100.0, -1.0), 1.0)

                    db_session.merge(
                        SentimentData(
                            symbol=sym,
                            timestamp=ts,
                            source="Santiment/whale_tx_1m_usd",
                            headline="Whale transaction count (USD > 1m)",
                            sentiment_score=score,
                            metadata={"raw_value": raw_value},
                        )
                    )
            print("[✔] Santiment done.")
            _commit_or_rollback(db_session, "Santiment")
            processed_sources += 1
        except Exception as e:
            print(f"[!] Santiment error: {e}")
            db_session.rollback()

    # 4) LunarCrush Social (HARDENED)
    if getattr(settings, "LUNARCRUSH_API_KEY", None):
        print("[LunarCrush] Fetching social sentiment...")
        try:
            for sym in symbols:
                slug = _lunarcrush_slug(sym)
                url = "https://lunarcrush.com/api4/public/coins"
                params = {
                    "symbol": slug,
                    "data_points": 24,  # last 24h
                }
                headers = {"Authorization": f"Bearer {settings.LUNARCRUSH_API_KEY}"}

                try:
                    r = requests.get(url, params=params, headers=headers, timeout=30)
                    if r.status_code == 404:
                        print(f"[!] LunarCrush: Coin {sym} (slug={slug}) not found.")
                        continue
                    r.raise_for_status()
                except requests.RequestException as re:
                    print(f"[!] LunarCrush HTTP error for {sym} (slug={slug}): {re}")
                    continue

                try:
                    resp_json = r.json()
                except ValueError:
                    print(
                        f"[!] LunarCrush: Non-JSON response for {sym} (slug={slug}): "
                        f"{r.text[:200]}"
                    )
                    continue

                # Very simple aggregation of social metrics -> score
                data = (resp_json or {}).get("data") or []
                if not data:
                    print(f"[!] LunarCrush: Empty data for {sym} (slug={slug}).")
                    continue

                coin = data[0]
                social_score = _safe_float(coin.get("galaxy_score"))
                social_volume = _safe_float(coin.get("social_volume"))
                if social_score is None:
                    continue

                # Normalize galaxy_score (0-100) to [-1, 1]
                norm_score = (social_score - 50.0) / 50.0
                norm_score = max(min(norm_score, 1.0), -1.0)

                ts = now  # using "now" since LC often returns aggregate snapshot
                db_session.merge(
                    SentimentData(
                        symbol=sym,
                        timestamp=ts,
                        source="LunarCrush",
                        headline="LunarCrush social sentiment",
                        sentiment_score=norm_score,
                        metadata={
                            "raw_galaxy_score": social_score,
                            "social_volume": social_volume,
                        },
                    )
                )
            print("[✔] LunarCrush done.")
            _commit_or_rollback(db_session, "LunarCrush")
            processed_sources += 1
        except Exception as e:
            print(f"[!] LunarCrush error: {e}")
            db_session.rollback()

    # 5) CoinMarketCap Global Metrics
    if getattr(settings, "CMC_API_KEY", None):
        print("[CoinMarketCap] Fetching global metrics...")
        try:
            headers = {"X-CMC_PRO_API_KEY": settings.CMC_API_KEY}
            r = requests.get(
                "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest",
                headers=headers,
                timeout=30,
            )
            r.raise_for_status()
            try:
                payload = r.json()
            except ValueError:
                print(f"[!] CoinMarketCap: Non-JSON response: {r.text[:200]}")
                payload = {}

            data = (payload.get("data") or {})
            btc_dominance = _safe_float(data.get("btc_dominance"))
            total_mcap = _safe_float((data.get("quote") or {}).get("USD", {}).get("total_market_cap"))
            total_vol = _safe_float((data.get("quote") or {}).get("USD", {}).get("total_volume_24h"))

            # We'll store this under a special symbol, e.g. "GLOBAL"
            ts = now
            db_session.merge(
                SentimentData(
                    symbol="GLOBAL",
                    timestamp=ts,
                    source="CMC/global_metrics",
                    headline="Global crypto market metrics",
                    sentiment_score=None,
                    metadata={
                        "btc_dominance": btc_dominance,
                        "total_market_cap_usd": total_mcap,
                        "total_volume_24h_usd": total_vol,
                    },
                )
            )
            print("[✔] CoinMarketCap done.")
            _commit_or_rollback(db_session, "CoinMarketCap")
            processed_sources += 1
        except Exception as e:
            print(f"[!] CoinMarketCap error: {e}")
            db_session.rollback()

    # 6) Alternative.me Fear & Greed Index
    print("[Alt.me] Fetching Fear & Greed index...")
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=30)
        r.raise_for_status()
        try:
            payload = r.json()
        except ValueError:
            print(f"[!] Fear & Greed: Non-JSON response: {r.text[:200]}")
            payload = {}

        data_list = payload.get("data") or []
        if data_list:
            item = data_list[0]
            ts_str = item.get("timestamp")
            ts = (
                datetime.fromtimestamp(int(ts_str), tz=timezone.utc)
                if ts_str
                else now
            )
            value = _safe_float(item.get("value"))
            classification = item.get("value_classification") or ""

            # Map 0-100 index to [-1, 1]
            score = None
            if value is not None:
                score = (value - 50.0) / 50.0
                score = max(min(score, 1.0), -1.0)

            db_session.merge(
                SentimentData(
                    symbol="GLOBAL",
                    timestamp=ts,
                    source="FearGreedIndex",
                    headline=f"Fear & Greed Index ({classification})",
                    sentiment_score=score,
                    metadata={"raw_index": value},
                )
            )
            print("[✔] Fear & Greed done.")
            _commit_or_rollback(db_session, "FearGreedIndex")
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
