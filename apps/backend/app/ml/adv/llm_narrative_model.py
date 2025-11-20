# app/ml/llm_narrative_model.py

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import joblib

logger = logging.getLogger(__name__)

# Try importing Gemini (live LLM)
try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
except Exception:  # optional dependency
    genai = None
    GenerativeModel = None  # type: ignore[assignment]

# ----------------------------------------------------------------------
# CONFIG & VERSIONING
# ----------------------------------------------------------------------

GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# Offline model artifact (trained by app/ml/adv/train_llm_narrative_model.py)
OFFLINE_MODEL_PATH = Path("model_artifacts/llm_narrative_model.pkl")

# Prompt + model versioning
PROMPT_VERSION = "v1.0"
LIVE_MODEL_NAME = "gemini-1.5-flash"

# Template used for live Gemini calls
PROMPT_TEMPLATE = (
    "You are a crypto narrative analyst for digital asset markets.\n"
    "Asset: {asset}\n"
    "Recent news / headlines / context (may be empty):\n"
    "{news_block}\n\n"
    "Task:\n"
    "  • Infer a single sentiment score in the range [-1, 1]\n"
    "    (-1 = strongly bearish, 0 = neutral, 1 = strongly bullish).\n"
    "  • Provide ONE most important headline or summary sentence.\n\n"
    "Respond STRICTLY in JSON with keys:\n"
    "{{ 'sentiment_score': float, 'key_headline': 'text' }}\n"
)

# Where to log narrative requests & responses (structured logs)
NARRATIVE_LOG_PREFIX = "[LLM-NARR-EVENT]"


# ----------------------------------------------------------------------
# LLM Narrative Engine
# ----------------------------------------------------------------------


class LLMNarrativeEngine:
    """
    Priority:
       1) Gemini live model (if API key available)
       2) Offline classifier (if llm_narrative_model.pkl exists)
       3) Neutral narrative (fallback)

    Additional responsibilities:
       - Version-aware: exposes prompt + model metadata.
       - Structured logging of incoming context & narrative outputs.
    """

    def __init__(self) -> None:
        self.use_live: bool = False
        self.use_offline: bool = False

        # Live LLM
        self.model: Optional[GenerativeModel] = None  # type: ignore[assignment]

        # Offline artifacts
        self.offline_vectorizer = None
        self.offline_classifier = None
        self.offline_label_mapping: Dict[int, str] = {}
        self.offline_metadata: Dict[str, Any] = {}

        # -----------------------------
        # Initialize LIVE Gemini
        # -----------------------------
        if GEMINI_KEY and genai is not None and GenerativeModel is not None:
            try:
                genai.configure(api_key=GEMINI_KEY)
                self.model = GenerativeModel(LIVE_MODEL_NAME)
                self.use_live = True
                logger.info(
                    "[LLM] Using LIVE Gemini model '%s' for narrative signal "
                    "(prompt_version=%s).",
                    LIVE_MODEL_NAME,
                    PROMPT_VERSION,
                )
            except Exception as e:  # noqa: BLE001
                logger.error("[LLM] Failed to initialize Gemini API: %s", e, exc_info=True)
                self.use_live = False

        # -----------------------------
        # Initialize OFFLINE model
        # -----------------------------
        if not self.use_live:
            self._load_offline_artifact()

    # ----------------------------------------------------------
    # Artifact loading
    # ----------------------------------------------------------

    def _load_offline_artifact(self) -> None:
        """
        Load offline narrative artifact produced by train_llm_narrative_model.py.

        Expected structure (recommended):

            artifact = {
                "model_name": "llm_narrative_offline",
                "version": "v1.0",
                "trained_at_utc": "...",
                "vectorizer": TfidfVectorizer,
                "classifier": LogisticRegression,
                "label_mapping": {...},
                "metrics": {...},
                "prompt_config": {...},   # optional
            }

        Older artifacts that are plain sklearn models are still supported but will
        not have metadata.
        """
        if not OFFLINE_MODEL_PATH.exists():
            logger.warning(
                "[LLM] No offline narrative model found at %s; narrative will be neutral.",
                OFFLINE_MODEL_PATH,
            )
            return

        try:
            artifact = joblib.load(OFFLINE_MODEL_PATH)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "[LLM] Could not load offline narrative model artifact: %s",
                e,
                exc_info=True,
            )
            return

        # New-style artifact: dict with vectorizer + classifier
        if isinstance(artifact, dict):
            self.offline_vectorizer = artifact.get("vectorizer")
            self.offline_classifier = artifact.get("classifier")
            self.offline_label_mapping = artifact.get("label_mapping", {}) or {}
            self.offline_metadata = {
                k: v
                for k, v in artifact.items()
                if k not in {"vectorizer", "classifier", "label_mapping"}
            }
        else:
            # Legacy: artifact is a classifier only
            self.offline_classifier = artifact
            self.offline_vectorizer = None
            self.offline_label_mapping = {}
            self.offline_metadata = {}

        if self.offline_classifier is not None:
            self.use_offline = True
            logger.info(
                "[LLM] Using OFFLINE narrative model from %s (version=%s).",
                OFFLINE_MODEL_PATH,
                self.offline_metadata.get("version", "unknown"),
            )
        else:
            logger.warning(
                "[LLM] Offline artifact at %s did not contain a classifier; "
                "narrative will be neutral.",
                OFFLINE_MODEL_PATH,
            )

    # ----------------------------------------------------------
    # STATUS FLAGS
    # ----------------------------------------------------------

    def is_model_loaded(self) -> bool:
        return self.use_live or self.use_offline

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return current narrative engine metadata (for debugging / monitoring).
        """
        return {
            "use_live": self.use_live,
            "use_offline": self.use_offline,
            "live_model_name": LIVE_MODEL_NAME if self.use_live else None,
            "prompt_version": PROMPT_VERSION,
            "offline_metadata": self.offline_metadata,
        }

    # ----------------------------------------------------------
    # MAIN ENTRYPOINT
    # ----------------------------------------------------------

    async def get_narrative_signal(
        self,
        asset: str,
        news_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entrypoint called by the orchestrator.

        Args:
            asset:     Symbol or asset name (e.g. "BTCUSDT").
            news_text: Optional aggregated news / headlines / tweets about the asset.
                       If omitted, the engine will rely on generic prompts.

        Returns:
            dict with at least:
                {
                  "sentiment_score": float in [-1, 1],
                  "key_headline": str,
                  "source": "live_gemini" | "offline_model" | "neutral",
                  "model_version": str,
                  "prompt_version": str,
                }
        """
        # 1) Live Gemini
        if self.use_live:
            result = await self._get_live_gemini_signal(asset, news_text)
        # 2) Offline classifier
        elif self.use_offline:
            result = self._get_offline_signal(asset, news_text)
        # 3) Neutral fallback
        else:
            result = self._neutral_narrative(asset)

        # Structured log of input + output
        self._log_narrative_event(asset, news_text, result)

        return result

    # ----------------------------------------------------------
    # LIVE LLM
    # ----------------------------------------------------------

    async def _get_live_gemini_signal(
        self,
        asset: str,
        news_text: Optional[str],
    ) -> Dict[str, Any]:
        """
        Use Gemini 1.5 Flash to generate a sentiment score and headline.
        """
        news_block = news_text or "(no explicit news provided)"
        prompt = PROMPT_TEMPLATE.format(asset=asset, news_block=news_block)

        try:
            if self.model is None:
                raise RuntimeError("Gemini model is not initialized")

            response = self.model.generate_content(prompt)
            raw_text = getattr(response, "text", "") or ""

            # Lenient JSON parsing; the model is instructed to return JSON
            data: Dict[str, Any] = {}
            try:
                data = json.loads(raw_text.replace("'", '"'))
            except Exception:
                logger.warning(
                    "[LLM] Gemini JSON parse failed; raw response: %s", raw_text
                )

            score = float(data.get("sentiment_score", 0.0))
            headline = str(data.get("key_headline", "")).strip()

            score = max(-1.0, min(1.0, score))

            return {
                "sentiment_score": score,
                "key_headline": headline or f"Narrative for {asset}",
                "source": "live_gemini",
                "model_version": LIVE_MODEL_NAME,
                "prompt_version": PROMPT_VERSION,
            }

        except Exception as e:  # noqa: BLE001
            logger.error("[LLM] Gemini call failed: %s", e, exc_info=True)
            return self._neutral_narrative(asset, source="live_error")

    # ----------------------------------------------------------
    # OFFLINE LLM (.pkl)
    # ----------------------------------------------------------

    def _get_offline_signal(
        self,
        asset: str,
        news_text: Optional[str],
    ) -> Dict[str, Any]:
        """
        Runs offline TF-IDF + LogisticRegression classifier trained earlier.

        The offline artifact must expose:
            - vectorizer.transform([text])
            - classifier.predict_proba or classifier.predict
        """
        if self.offline_classifier is None:
            return self._neutral_narrative(asset, source="offline_missing")

        try:
            # Build input text
            if news_text:
                sample_input = news_text
            else:
                # Generic template if no explicit news is provided
                sample_input = f"sentiment analysis for {asset}"

            if self.offline_vectorizer is not None:
                X = self.offline_vectorizer.transform([sample_input])
            else:
                # In rare legacy cases, classifier might accept raw text directly
                X = [sample_input]

            clf = self.offline_classifier

            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X)[0]
                # Expected 3 classes [-1, 0, 1]; if label_mapping exists,
                # we treat higher class index as more bullish.
                if len(proba) == 3:
                    # proba order assumed [-1, 0, 1]
                    score = float(proba[2] - proba[0])
                elif len(proba) == 2:
                    # binary: class 1 bullish vs 0 bearish/neutral
                    score = float(proba[1] - proba[0])
                else:
                    # fallback: mean-centered index
                    indices = list(range(len(proba)))
                    mean_idx = sum(i * p for i, p in zip(indices, proba))
                    score = (mean_idx - (len(proba) - 1) / 2.0) / (
                        (len(proba) - 1) / 2.0
                    )
            else:
                # Direct regression or signed output
                pred = clf.predict(X)[0]
                score = float(pred)

            score = max(-1.0, min(1.0, score))

            headline = (
                f"Offline narrative model opinion for {asset}"
                if not news_text
                else f"Offline narrative summary for {asset}"
            )

            return {
                "sentiment_score": score,
                "key_headline": headline,
                "source": "offline_model",
                "model_version": self.offline_metadata.get("version", "offline_unknown"),
                "prompt_version": PROMPT_VERSION,
            }

        except Exception as e:  # noqa: BLE001
            logger.error("[LLM] Offline narrative computation failed: %s", e, exc_info=True)
            return self._neutral_narrative(asset, source="offline_error")

    # ----------------------------------------------------------
    # FALLBACK + LOGGING
    # ----------------------------------------------------------

    def _neutral_narrative(
        self,
        asset: str,
        source: str = "neutral_fallback",
    ) -> Dict[str, Any]:
        return {
            "sentiment_score": 0.0,
            "key_headline": f"No narrative available for {asset}",
            "source": source,
            "model_version": self.offline_metadata.get("version", "none"),
            "prompt_version": PROMPT_VERSION,
        }

    def _log_narrative_event(
        self,
        asset: str,
        news_text: Optional[str],
        result: Dict[str, Any],
    ) -> None:
        """
        Emit a single structured log line with input + output context.

        This is designed to be ingested by log aggregators (ELK, Loki, etc.).
        """
        try:
            event = {
                "ts_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "asset": asset,
                "source": result.get("source"),
                "model_version": result.get("model_version"),
                "prompt_version": result.get("prompt_version"),
                "sentiment_score": result.get("sentiment_score"),
                "headline": result.get("key_headline"),
                # Truncate news text to avoid giant logs
                "news_excerpt": (
                    news_text[:500] + "…"
                    if news_text and len(news_text) > 500
                    else news_text
                ),
            }
            logger.info("%s %s", NARRATIVE_LOG_PREFIX, json.dumps(event))
        except Exception as e:  # noqa: BLE001
            logger.warning("[LLM] Failed to log narrative event: %s", e, exc_info=True)


# Singleton instance
llm_engine = LLMNarrativeEngine()
