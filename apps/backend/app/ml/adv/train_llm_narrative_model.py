# app/ml/adv/train_llm_narrative_model.py

import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from app.db.database import SessionLocal
from app.db import models
from app.core.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ARTIFACTS_DIR = Path(getattr(settings, "MODEL_ARTIFACTS_DIR", "model_artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_PATH = ARTIFACTS_DIR / "llm_narrative_model.pkl"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_labeled_text_from_db() -> Tuple[List[str], List[int]]:
    """
    Try to load training data from the database.

    EXPECTED (you can adjust this to your schema):
      - A table models.NarrativeTrainingSample with fields:
          - text: str  (headline / news / tweet / description)
          - label: int (e.g. -1=bearish, 0=neutral, 1=bullish)
    If that table is not present or has too few rows, this will raise RuntimeError
    and we will fall back to a synthetic dataset.
    """
    session = SessionLocal()
    try:
        if not hasattr(models, "NarrativeTrainingSample"):
            raise RuntimeError(
                "models.NarrativeTrainingSample not found. "
                "Please implement it or adjust this loader."
            )

        Sample = getattr(models, "NarrativeTrainingSample")

        rows = session.query(Sample).all()
        if not rows or len(rows) < 50:
            raise RuntimeError(
                f"Found {len(rows) if rows else 0} NarrativeTrainingSample rows; "
                "need at least 50 for a meaningful model."
            )

        texts: List[str] = []
        labels: List[int] = []

        for r in rows:
            text = getattr(r, "text", None)
            label = getattr(r, "label", None)
            if not text or label is None:
                continue
            texts.append(text)
            labels.append(int(label))

        if len(texts) < 50:
            raise RuntimeError(
                f"After filtering, only {len(texts)} usable samples; "
                "need at least 50."
            )

        logger.info(
            "[LLM-NARR] Loaded %d labeled narrative samples from DB.", len(texts)
        )
        return texts, labels

    finally:
        session.close()


def _synthetic_training_data() -> Tuple[List[str], List[int]]:
    """
    Fallback: small synthetic dataset if DB has no labeled narrative samples.

    -1 = bearish, 0 = neutral, 1 = bullish
    """
    logger.warning(
        "[LLM-NARR] Falling back to synthetic training data. "
        "You should later replace this with real labeled text from DB."
    )
    texts = [
        "Bitcoin crashes as market panics over regulation fears",
        "Ethereum price drops after network congestion worries",
        "Crypto markets tumble, traders expect further downside",
        "Market stays flat amid lack of major crypto news",
        "Bitcoin trades sideways with low volatility",
        "Investors wait for key macro data before making moves",
        "Bitcoin surges to new monthly high on strong demand",
        "Ethereum rallies as DeFi activity picks up",
        "Altcoins pop as risk appetite returns to market",
    ]
    labels = [
        -1, -1, -1,  # bearish
         0,  0,  0,  # neutral
         1,  1,  1,  # bullish
    ]
    return texts, labels


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def _train_narrative_model(texts: List[str], labels: List[int]):
    """
    Train a simple narrative classifier: TF-IDF + LogisticRegression.
    """
    logger.info("[LLM-NARR] Building TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        lowercase=True,
        stop_words="english",
    )

    X = vectorizer.fit_transform(texts)
    y = np.array(labels, dtype=int)

    # Simple train/validation split
    n = len(y)
    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    logger.info("[LLM-NARR] Training LogisticRegression classifier...")
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        multi_class="auto",
    )
    clf.fit(X_train, y_train)

    if X_val.shape[0] > 0:
        y_pred = clf.predict(X_val)
        report = classification_report(
            y_val, y_pred,
            target_names=["bearish (-1)", "neutral (0)", "bullish (1)"],
            zero_division=0,
        )
        logger.info("[LLM-NARR] Validation report:\n%s", report)

    return vectorizer, clf


def main():
    logger.info("[LLM-NARR] ==== Training LLM narrative specialist model ====")

    # Try DB first, then synthetic fallback
    try:
        texts, labels = _load_labeled_text_from_db()
    except Exception as e:  # noqa: BLE001
        logger.warning("[LLM-NARR] DB load failed: %s", e)
        texts, labels = _synthetic_training_data()

    vectorizer, clf = _train_narrative_model(texts, labels)

    artifact = {
        "vectorizer": vectorizer,
        "classifier": clf,
        "label_mapping": {
            -1: "bearish",
             0: "neutral",
             1: "bullish",
        },
    }

    joblib.dump(artifact, ARTIFACT_PATH)
    logger.info("[LLM-NARR] Saved narrative model artifact to %s", ARTIFACT_PATH)
    logger.info("[LLM-NARR] Training complete.")


if __name__ == "__main__":
    main()
