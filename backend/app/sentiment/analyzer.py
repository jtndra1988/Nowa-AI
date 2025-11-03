# app/nlp/analyzer.py  (or app/analyzer.py if that's your layout)
from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

try:
    import torch
except Exception:
    torch = None  # CPU fallback if torch isn't installed

from transformers import pipeline

# Prefer your app config, but fall back gracefully
try:
    from app.core.config import settings  # type: ignore
except Exception:
    class _Fallback:
        FINBERT_MODEL_NAME: str = os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")
        USE_GPU: bool = os.getenv("USE_GPU", "1") not in ("0", "false", "False", "")
    settings = _Fallback()


# ---------- Internals ----------

def _choose_device() -> int:
    """Return 0 for CUDA/GPU if allowed & available, else -1 for CPU."""
    use_gpu = getattr(settings, "USE_GPU", True)
    if isinstance(use_gpu, str):
        use_gpu = use_gpu.lower() not in ("0", "false", "no")
    if use_gpu and torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        return 0
    return -1


@lru_cache(maxsize=1)
def _get_pipeline():
    """
    Lazily load and cache the FinBERT pipeline once per process.
    Uses model name from settings.FINBERT_MODEL_NAME (or env), CPU by default, GPU if available.
    """
    model_name = getattr(settings, "FINBERT_MODEL_NAME", "ProsusAI/finbert") or "ProsusAI/finbert"
    device = _choose_device()

    # Note: we request 'sentiment-analysis'; we'll pass return_all_scores=True at call time.
    nlp = pipeline(
        task="sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=device,
    )
    return nlp


def _dist_to_score(label_dist: List[dict]) -> float:
    """
    Convert a label distribution into a scalar sentiment score in [-1, 1].
    Uses: score = P(positive) - P(negative).
    Handles labels case-insensitively and ignores unknown labels safely.
    """
    p_pos = 0.0
    p_neg = 0.0
    for item in label_dist:
        # Expected keys: {'label': 'POSITIVE'/'NEGATIVE'/'NEUTRAL', 'score': float}
        label = str(item.get("label", "")).lower()
        prob = float(item.get("score", 0.0))
        if "pos" in label:
            p_pos += prob
        elif "neg" in label:
            p_neg += prob
        # 'neutral' contributes to neither side; it shrinks |p_pos - p_neg|
    # Clamp defensively
    score = max(-1.0, min(1.0, p_pos - p_neg))
    return score


# ---------- Public API ----------

def get_sentiment_score(text: Optional[str]) -> float:
    """
    Score a single text. Returns a float in [-1, 1]:
      +1 → strongly positive, -1 → strongly negative, ~0 → neutral/mixed.
    """
    if not text or not str(text).strip():
        return 0.0

    nlp = _get_pipeline()
    # Ask pipeline for full distribution so we can compute a stable scalar
    # Pass truncation=True to avoid long-text errors, and batch size of 1 implicitly
    result = nlp(text, return_all_scores=True, truncation=True)
    # HF returns a list[ list[ {label, score}, ... ] ] when return_all_scores=True
    if isinstance(result, list) and result and isinstance(result[0], list):
        return _dist_to_score(result[0])
    # Fallback: if it's already a flat list of dicts
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return _dist_to_score(result)
    return 0.0


def get_sentiment_scores(texts: List[str], batch_size: int = 32) -> List[float]:
    """
    Vectorized version for throughput. Accepts a list of texts and returns
    a list of scores in [-1, 1], using efficient batching under the hood.

    Notes:
      - Empty/None strings are returned as 0.0.
      - Uses p(Positive) - p(Negative) mapping for each text.
    """
    if not texts:
        return []

    # Short-circuit for all-empty inputs
    cleaned = [t if (t is not None and str(t).strip()) else "" for t in texts]
    if all(t == "" for t in cleaned):
        return [0.0] * len(cleaned)

    nlp = _get_pipeline()
    # HF pipelines support batch input: returns list of distributions (one per text)
    # Use truncation to avoid tokenizer overflow; rely on default padding strategy
    distributions = nlp(cleaned, return_all_scores=True, truncation=True, batch_size=batch_size)

    scores: List[float] = []
    for text_input, dist in zip(cleaned, distributions):
        if not text_input:
            scores.append(0.0)
            continue
        # dist should be list[dict] for this item
        if isinstance(dist, list) and dist and isinstance(dist[0], dict):
            scores.append(_dist_to_score(dist))
        else:
            # In rare shapes, attempt one more normalization
            if isinstance(dist, list) and dist and isinstance(dist[0], list):
                scores.append(_dist_to_score(dist[0]))
            else:
                scores.append(0.0)
    return scores


__all__ = [
    "get_sentiment_score",
    "get_sentiment_scores",
]
