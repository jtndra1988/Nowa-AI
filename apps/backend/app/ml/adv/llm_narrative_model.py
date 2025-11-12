import os
import json
import logging
from typing import Optional, Dict, Any

import httpx
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class NarrativeOutput(BaseModel):
    """
    Strict schema for LLM narrative output.
    """
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    key_headline: str = Field(..., min_length=3, max_length=260)


class LLMNarrativeModel:
    """
    Layer 1: LIVE LLM-based Narrative Specialist.

    - Uses Gemini 1.5 (or compatible) over HTTP.
    - No mock / heuristic fallback.
    - If not correctly configured, `is_model_loaded()` is False and
      `get_narrative_signal()` will raise RuntimeError.

    Expected env:
      GEMINI_API_KEY: required
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash-latest",
        timeout: float = 8.0,
    ) -> None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.api_key: Optional[str] = api_key or None
        self.model = model
        self.api_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        )
        self.timeout = timeout

        if not self.api_key:
            self._loaded = False
            logger.error(
                "[LLMNarrativeModel] GEMINI_API_KEY not set. "
                "LLM narrative engine is DISABLED (no mock fallback)."
            )
        else:
            self._loaded = True
            logger.info(
                f"[LLMNarrativeModel] Initialized in LIVE mode with model '{model}'."
            )

    # -------- Public API used by InferenceService --------

    def is_model_loaded(self) -> bool:
        """
        Used by InferenceService for readiness.
        True ONLY if we have a valid API key and config.
        """
        return self._loaded

    def get_mode(self) -> str:
        """
        For debug / observability.
        """
        return "live" if self._loaded else "disabled"

    async def get_narrative_signal(self, asset: str) -> Dict[str, Any]:
        """
        Main entry called by InferenceService.

        Returns:
          {
            "sentiment_score": float (-1.0..1.0),
            "key_headline": str
          }

        Behavior:
          - If engine is not loaded → raises RuntimeError.
          - If API fails or response invalid → raises RuntimeError.
        """
        if not self.is_model_loaded():
            raise RuntimeError(
                "[LLMNarrativeModel] Called get_narrative_signal() but engine is not loaded."
            )

        asset = (asset or "BTC").upper()
        prompt = self._build_prompt(asset)
        api_result = await self._run_api_call(prompt)

        if not api_result:
            raise RuntimeError(
                "[LLMNarrativeModel] Empty/invalid response from LLM API."
            )

        try:
            parsed = NarrativeOutput(**api_result)
        except ValidationError as e:
            logger.error(
                f"[LLMNarrativeModel] Response validation failed: {e} | data={api_result}"
            )
            raise RuntimeError(
                "[LLMNarrativeModel] LLM response failed validation."
            ) from e

        return parsed.dict()

    # -------- Internal helpers --------

    def _build_prompt(self, asset: str) -> str:
        # Single, strict JSON-format instruction
        return (
            "You are an expert crypto/macro analyst for a quant fund.\n"
            f"Analyze the current market narrative specifically for {asset}.\n"
            "Consider: macro context, regulatory tone, derivatives positioning, "
            "on-chain flows, developer / ecosystem traction, social & news sentiment.\n"
            "Respond ONLY as strict JSON (no prose, no markdown):\n"
            "{\n"
            '  "sentiment_score": <float between -1.0 and 1.0>,\n'
            '  "key_headline": "<single concise narrative sentence>\n'
            "}\n"
        )

    async def _run_api_call(self, prompt: str) -> Optional[Dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.api_url}?key={self.api_key}",
                    headers=headers,
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.error(f"[LLMNarrativeModel] HTTP/API error: {e}", exc_info=True)
            return None

        return self._parse_gemini_response(data)

    def _parse_gemini_response(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Robustly extract the JSON object we requested.
        Supports:
        - Direct JSON (if model respects responseMimeType)
        - JSON string in candidates[0].content.parts[0].text
        """
        # Case 1: already what we need
        if isinstance(data, dict) and "sentiment_score" in data:
            return data

        try:
            candidates = data.get("candidates") or []
            if not candidates:
                return None
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return None
            raw = (parts[0].get("text") or "").strip()
            if not raw:
                return None
            # Some models wrap JSON in code fences; strip if present
            if raw.startswith("```"):
                raw = raw.strip("`")
                if "\n" in raw:
                    raw = raw.split("\n", 1)[1]
            return json.loads(raw)
        except Exception as e:
            logger.error(f"[LLMNarrativeModel] Failed to parse LLM JSON: {e}", exc_info=True)
            return None


# -------- Singleton for InferenceService --------

try:
    llm_engine = LLMNarrativeModel()
except Exception as e:
    # Hard fail: no mock. If this explodes, you WANT to see it.
    logger.critical(
        f"[LLMNarrativeModel] Failed to initialize llm_engine: {e}",
        exc_info=True,
    )
    # keep llm_engine undefined to surface issues early
    raise
