# app/ml/llm_narrative_model.py

import os
import pickle
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Try importing Gemini (live LLM)
try:
    from google.generativeai import GenerativeModel
except Exception:
    GenerativeModel = None


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OFFLINE_MODEL_PATH = "model_artifacts/llm_narrative_model.pkl"


class LLMNarrativeEngine:
    """
    Priority:
       1) Gemini live model (if API key available)
       2) Offline classifier (if pkl exists)
       3) Neutral narrative (fallback)
    """

    def __init__(self):
        self.use_live = False
        self.use_offline = False

        # -----------------------------
        # Check Gemini (LIVE)
        # -----------------------------
        if GEMINI_KEY and GenerativeModel is not None:
            try:
                self.model = GenerativeModel("gemini-1.5-flash")
                self.use_live = True
                logger.info("[LLM] Using LIVE Gemini for narrative signal.")
            except Exception as e:
                logger.error(f"[LLM] Failed to initialize Gemini API: {e}")
                self.use_live = False

        # -----------------------------
        # Check OFFLINE .pkl model
        # -----------------------------
        if not self.use_live:
            if os.path.exists(OFFLINE_MODEL_PATH):
                try:
                    with open(OFFLINE_MODEL_PATH, "rb") as f:
                        self.offline_model = pickle.load(f)
                    self.use_offline = True
                    logger.info("[LLM] Using OFFLINE llm_narrative_model.pkl")
                except Exception as e:
                    logger.error(
                        f"[LLM] Could not load offline LLM model: {e}"
                    )
                    self.use_offline = False
            else:
                logger.warning(
                    "[LLM] No offline model found; narrative will be neutral."
                )

    # ----------------------------------------------------------
    # STATUS FLAGS
    # ----------------------------------------------------------

    def is_model_loaded(self) -> bool:
        return self.use_live or self.use_offline

    # ----------------------------------------------------------
    # MAIN ENTRYPOINT
    # ----------------------------------------------------------

    async def get_narrative_signal(self, asset: str) -> Dict[str, Any]:
        """
        Returns a dict:
            {
              "sentiment_score": float (-1 to 1),
              "key_headline": str
            }
        """

        # 1) Live Gemini
        if self.use_live:
            return await self._get_live_gemini_signal(asset)

        # 2) Offline classifier
        if self.use_offline:
            return self._get_offline_signal(asset)

        # 3) Neutral default
        return self._neutral_narrative(asset)

    # ----------------------------------------------------------
    # LIVE LLM
    # ----------------------------------------------------------

    async def _get_live_gemini_signal(self, asset: str) -> Dict[str, Any]:
        """
        Use Gemini 1.5 Flash to generate a sentiment score.
        """
        try:
            prompt = (
                f"You are a crypto analyst. Give a sentiment score (-1 to 1) "
                f"for {asset} right now, based ONLY on market mood, "
                "and include one most important headline.\n\n"
                "Respond strictly in JSON format:\n"
                "{ 'sentiment_score': float, 'key_headline': 'text' }"
            )

            response = self.model.generate_content(prompt)
            text = response.text

            # Very lenient parsing — your code may already have JSON parsing
            import json
            data = json.loads(text.replace("'", '"'))

            score = float(data.get("sentiment_score", 0.0))
            headline = str(data.get("key_headline", ""))

            return {
                "sentiment_score": max(-1.0, min(1.0, score)),
                "key_headline": headline,
            }

        except Exception as e:
            logger.error("[LLM] Gemini call failed: %s", e, exc_info=True)
            return self._neutral_narrative(asset)

    # ----------------------------------------------------------
    # OFFLINE LLM (.pkl)
    # ----------------------------------------------------------

    def _get_offline_signal(self, asset: str) -> Dict[str, Any]:
        """
        Runs offline logistic/ML classifier trained earlier.
        The offline model must expose:
            model.predict([text]) -> numeric score
        """
        try:
            # Offline model is usually a classifier with .predict_proba or .predict
            # Here we simulate with predict() for features based on asset name.
            sample_input = f"sentiment analysis for {asset}"

            if hasattr(self.offline_model, "predict_proba"):
                prob = self.offline_model.predict_proba([sample_input])[0]
                score = float(prob[1] - prob[0])  # convert to -1..1
            else:
                score = float(self.offline_model.predict([sample_input])[0])

            score = max(-1.0, min(1.0, score))

            return {
                "sentiment_score": score,
                "key_headline": f"Offline model narrative for {asset}",
            }

        except Exception as e:
            logger.error("[LLM] Offline model computation failed: %s", e)
            return self._neutral_narrative(asset)

    # ----------------------------------------------------------
    # FALLBACK
    # ----------------------------------------------------------

    def _neutral_narrative(self, asset: str) -> Dict[str, Any]:
        return {
            "sentiment_score": 0.0,
            "key_headline": f"No narrative available for {asset}",
        }


# Singleton instance
llm_engine = LLMNarrativeEngine()
