# app/ml/feature_builder.py

"""
Feature builder utilities for Nowa AI models.

This centralises how we join auxiliary features such as multi-source
sentiment onto the core OHLCV / futures / options time-series.

join_sentiment_features(df) expects a DataFrame with at least:
    ['symbol', 'timestamp']

and returns the same rows with extra columns:
    composite_score, news_score, social_score, global_score
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db import models

logger = logging.getLogger(__name__)

BUCKET_FREQ = "1H"  # hourly sentiment grid


def _ensure_dt(series: pd.Series) -> pd.Series:
    """Force a pandas Series to UTC datetimes, ignoring invalid rows."""
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, utc=True, errors="coerce")
    return series


def _load_sentiment_frame(
    db: Session,
    symbols: Iterable[str],
    start_ts,
    end_ts,
) -> pd.DataFrame:
    """
    Load SentimentData rows for given symbols/time window into a DataFrame.

    Also pulls 'GLOBAL' rows for macro sentiment (Fear & Greed, CMC, etc.).
    """
    sym_list = sorted({s for s in symbols if s})
    if not sym_list:
        return pd.DataFrame(columns=["symbol", "timestamp", "source", "sentiment_score"])

    sym_list_with_global = list(sym_list) + ["GLOBAL"]

    rows = (
        db.query(models.SentimentData)
        .filter(models.SentimentData.symbol.in_(sym_list_with_global))
        .filter(models.SentimentData.timestamp >= start_ts - timedelta(hours=1))
        .filter(models.SentimentData.timestamp <= end_ts + timedelta(hours=1))
        .all()
    )

    if not rows:
        logger.warning(
            "[FeatureBuilder] No SentimentData rows found between %s and %s for %s",
            start_ts,
            end_ts,
            sym_list_with_global,
        )
        return pd.DataFrame(columns=["symbol", "timestamp", "source", "sentiment_score"])

    data = [
        {
            "symbol": r.symbol,
            "timestamp": r.timestamp,
            "source": r.source,
            "sentiment_score": r.sentiment_score,
        }
        for r in rows
    ]

    s_df = pd.DataFrame(data)
    s_df["timestamp"] = _ensure_dt(s_df["timestamp"])
    return s_df


def _classify_kind(source: Optional[str]) -> str:
    """Classify a sentiment source into 'news', 'social', 'global', or 'other'."""
    if not source:
        return "other"
    s = source.lower()

    # News-like sources (NewsAPI, CryptoPanic headlines, etc.)
    if "newsapi" in s or "news" in s or "crypto" in s or "headline" in s or "coindesk" in s:
        return "news"

    # Social / crowd sentiment (LunarCrush, social volume, etc.)
    if "lunarcrush" in s or "social" in s or "twitter" in s or "x.com" in s:
        return "social"

    # Global / market-wide sentiment (Fear & Greed, CMC global metrics)
    if "fear" in s or "greed" in s or "coinmarketcap" in s or "global" in s:
        return "global"

    return "other"


def join_sentiment_features(
    df: pd.DataFrame,
    db_session: Session | None = None,
) -> pd.DataFrame:
    """
    Left-join hourly sentiment aggregates onto a symbol/timestamp DataFrame.

    Input:
        df: DataFrame with at least ['symbol', 'timestamp'].

    Output:
        Same rows +:
            composite_score, news_score, social_score, global_score

    • Each row at time t gets sentiment aggregated in its 1-hour bucket.
    • Build your targets (t+1h return/vol) *after* calling this to avoid leakage.
    """
    if df.empty:
        logger.warning("[FeatureBuilder] Received empty df; attaching zero sentiment columns.")
        df = df.copy()
        for col in ["composite_score", "news_score", "social_score", "global_score"]:
            df[col] = 0.0
        return df

    if "symbol" not in df.columns or "timestamp" not in df.columns:
        logger.warning(
            "[FeatureBuilder] df missing required columns 'symbol'/'timestamp'; "
            "skipping sentiment join."
        )
        df = df.copy()
        for col in ["composite_score", "news_score", "social_score", "global_score"]:
            df[col] = 0.0
        return df

    df = df.copy()
    df["timestamp"] = _ensure_dt(df["timestamp"])

    symbols = df["symbol"].dropna().astype(str).unique().tolist()
    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()

    close_session = False
    if db_session is None:
        db_session = SessionLocal()
        close_session = True

    try:
        s_df = _load_sentiment_frame(db_session, symbols, start_ts, end_ts)
    finally:
        if close_session:
            db_session.close()

    if s_df.empty:
        for col in ["composite_score", "news_score", "social_score", "global_score"]:
            df[col] = 0.0
        return df

    # Bucket both price & sentiment on the same hourly grid
    df["bucket_ts"] = df["timestamp"].dt.floor(BUCKET_FREQ)
    s_df["bucket_ts"] = s_df["timestamp"].dt.floor(BUCKET_FREQ)

    # Global vs local
    s_global = s_df[s_df["symbol"] == "GLOBAL"].copy()
    s_local = s_df[s_df["symbol"] != "GLOBAL"].copy()

    # Global sentiment per hour
    if not s_global.empty:
        s_global = s_global.dropna(subset=["sentiment_score"])
        global_agg = (
            s_global.groupby("bucket_ts")["sentiment_score"]
            .mean()
            .rename("global_score")
            .reset_index()
        )
    else:
        global_agg = pd.DataFrame(columns=["bucket_ts", "global_score"])

    # Local per-symbol sentiment
    if not s_local.empty:
        s_local = s_local.dropna(subset=["sentiment_score"]).copy()
        s_local["kind"] = s_local["source"].apply(_classify_kind)

        comp = (
            s_local.groupby(["symbol", "bucket_ts"])["sentiment_score"]
            .mean()
            .rename("composite_score")
        )

        news = (
            s_local[s_local["kind"] == "news"]
            .groupby(["symbol", "bucket_ts"])["sentiment_score"]
            .mean()
            .rename("news_score")
        )

        social = (
            s_local[s_local["kind"] == "social"]
            .groupby(["symbol", "bucket_ts"])["sentiment_score"]
            .mean()
            .rename("social_score")
        )

        sym_agg = (
            pd.concat([comp, news, social], axis=1)
            .reset_index()
        )
    else:
        sym_agg = pd.DataFrame(
            columns=["symbol", "bucket_ts", "composite_score", "news_score", "social_score"]
        )

    # Join onto main df
    out = df.merge(
        sym_agg,
        how="left",
        on=["symbol", "bucket_ts"],
    )

    out = out.merge(
        global_agg,
        how="left",
        on="bucket_ts",
    )

    for col in ["composite_score", "news_score", "social_score", "global_score"]:
        if col not in out.columns:
            out[col] = 0.0
        else:
            out[col] = out[col].fillna(0.0)

    out = out.drop(columns=["bucket_ts"])

    logger.info(
        "[FeatureBuilder] Joined sentiment onto df: %d rows, columns now: %s",
        len(out),
        list(out.columns),
    )

    return out
