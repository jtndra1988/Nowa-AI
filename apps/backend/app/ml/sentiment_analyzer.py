# app/ml/sentiment_analyzer.py

from __future__ import annotations

from functools import lru_cache
from typing import Optional, List, Union

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@lru_cache(maxsize=1)
def _get_analyzer() -> SentimentIntensityAnalyzer:
    """
    Lazily initialize a single VADER analyzer instance.
    """
    return SentimentIntensityAnalyzer()


def score_text(text: Optional[Union[str, float]]) -> float:
    """
    Return sentiment score in [-1.0, 1.0] using VADER's compound score.
    -1 = very negative, 0 = neutral, +1 = very positive
    
    Handles None, NaN, and empty strings gracefully by returning 0.0.
    """
    # Guard against NaN (which is a float) or None
    if not isinstance(text, str):
        return 0.0
    
    # Guard against empty or whitespace-only strings
    if not text.strip():
        return 0.0

    try:
        analyzer = _get_analyzer()
        scores = analyzer.polarity_scores(text)
        # 'compound' is already in [-1, 1]
        return float(max(min(scores["compound"], 1.0), -1.0))
    except Exception:
        # Fallback for any unexpected tokenizer errors
        return 0.0


def score_headline_and_description(
    headline: Optional[str],
    description: Optional[str],
) -> float:
    """
    Convenience helper: combine headline + description.
    """
    parts: list[str] = []
    if isinstance(headline, str) and headline.strip():
        parts.append(headline)
    if isinstance(description, str) and description.strip():
        parts.append(description)
        
    if not parts:
        return 0.0
        
    return score_text(" ".join(parts))


def aggregate_sentiment_scores(scores: List[Optional[float]]) -> float:
    """
    Safely calculates the average sentiment for a list of scores (e.g., a daily batch).
    
    - Handles empty lists -> Returns 0.0
    - Handles lists with None/NaN -> Filters them out
    - Returns 0.0 (Neutral) if no valid scores exist
    """
    if not scores:
        return 0.0
    
    # Filter out None, NaNs, and infinite values
    valid_scores = [
        s for s in scores 
        if s is not None and isinstance(s, (int, float)) and not np.isnan(s) and not np.isinf(s)
    ]
    
    if not valid_scores:
        return 0.0
        
    return float(sum(valid_scores) / len(valid_scores))