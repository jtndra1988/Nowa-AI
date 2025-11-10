import os
import json
import logging
from typing import Optional, Dict, Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(_name_)


class NarrativeOutput(BaseModel):
    """
    Output of the LLM Narrative Specialist (Layer 1).
    - sentiment_score: -1 (very negative) to +1 (very positive)
    - key_headline: short human-readable summary for logs / UI
    """
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    key_headline: str = Field(..., max_length=260)


class LLMNarrativeModel:
    """
    Layer 1: LLM-based Narrative Specialist

    Role:
      - Read external narrative (news, social, macro context).
      - Return a compact numeric sentiment + one synthesized headline.
      - Feeds into DecisionNet (Layer 2) & RL Head Trader (Layer 3).

    Notes:
      - If GEMINI_API_KEY is missing or any call fails → safe mock output.
      - You can later replace _mock_output and/or add real news fetching.
    """

    def _init_(
        self,
        model: str = "gemini-1.5-flash-latest",
        timeout: float = 8.0,
    ) -> None:
        key = os.getenv("GEMINI_API_KEY", "").strip()
        self.use_mock = not bool(key)
        self.model = model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

        if not self.use_mock:
            self.api_url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.model}:generateContent?key={key}"
            )
            logger.info("[LLM-Narrative] Using live Gemini model=%s", self.model)
        else:
            self.api_url = None
            logger.warning(
                "[LLM-Narrative] GEMINI_API_KEY missing or empty. "
                "Falling back to deterministic mock outputs."
            )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        """Call once on shutdown if you want to cleanly close the client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_narrative_model_output(self, asset: str) -> NarrativeOutput:
        """
        Main entrypoint.

        Returns a NarrativeOutput. Always succeeds:
        - Uses real Gemini call when configured.
        - Falls back to mock, deterministic sentiment when anything fails.
        """
        asset = (asset or "BTC").upper()

        if self.use_mock:
            return self._mock_output(asset)

        system_instruction = (
            "You are Nowa's narrative intelligence module for crypto markets.\n"
            "You read recent, reputable crypto news, macro headlines, and social context.\n"
            "You MUST respond with a single JSON object only, no extra text, in this schema:\n"
            "{\n"
            '  \"sentiment_score\": float   // between -1 and 1\n'
            '  \"key_headline\": string    // short concise summary headline\n'
            "}\n"
            "Rules:\n"
            "- sentiment_score < -0.3 = clearly negative narrative.\n"
            "- sentiment_score > 0.3 = clearly positive narrative.\n"
            "- Keep key_headline under 200 characters.\n"
            "- Do not include any other fields.\n"
        )

        body: Dict[str, Any] = {
            "systemInstruction": {
                "parts": [{"text": system_instruction}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                f"Asset: {asset}\n"
                                "Based only on trusted, non-spam sources, how positive or negative "
                                "is the current narrative around this asset right now?"
                            )
                        }
                    ],
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "object",
                    "properties": {
                        "sentiment_score": {"type": "number"},
                        "key_headline": {"type": "string"},
                    },
                    "required": ["sentiment_score", "key_headline"],
                },
                "temperature": 0.4,
                "topP": 0.9,
                "topK": 40,
            },
        }

        try:
            client = await self._get_client()
            resp = await client.post(self.api_url, json=body)

            if resp.status_code != 200:
                logger.warning(
                    "[LLM-Narrative] HTTP %s from Gemini: %s",
                    resp.status_code,
                    resp.text[:300],
                )
                return self._mock_output(asset)

            data = resp.json()
            payload = self._extract_payload(data)
            if not payload:
                logger.warning(
                    "[LLM-Narrative] Could not parse Gemini payload, using mock."
                )
                return self._mock_output(asset)

            out = NarrativeOutput(
                sentiment_score=float(max(-1.0, min(1.0, payload["sentiment_score"]))),
                key_headline=str(payload["key_headline"])[:260],
            )
            return out

        except Exception as e:  # noqa: BLE001
            logger.warning("[LLM-Narrative] Exception, using mock: %s", e)
            return self._mock_output(asset)

    @staticmethod
    def _extract_payload(data: Any) -> Optional[Dict[str, Any]]:
        """
        Gemini may either:
        - Return JSON directly, or
        - Wrap JSON as text in candidates[0].content.parts[0].text
        """
        if isinstance(data, dict) and "sentiment_score" in data:
            return data

        try:
            candidates = data.get("candidates") or []
            if not candidates:
                return None
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return None
            raw = parts[0].get("text", "").strip()
            if not raw:
                return None
            # If it's already JSON, parse it.
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            return None

    def _mock_output(self, asset: str) -> NarrativeOutput:
        """
        Safe deterministic-ish fallback for demos / missing API key.
        No external calls, no side effects.
        """
        # Cheap deterministic pseudo-random based on asset
        seed = sum(ord(c) for c in asset) % 200
        score = (seed - 100) / 150.0  # approx [-0.67, 0.67]
        score = max(-0.8, min(0.8, score))

        if score > 0.25:
            headline = f"{asset}: Constructive market narrative with supportive flows (demo)."
        elif score < -0.25:
            headline = f"{asset}: Cautious narrative with elevated risk signals (demo)."
        else:
            headline = f"{asset}: Mixed but balanced narrative, no extreme stress (demo)."

        return NarrativeOutput(
            sentiment_score=float(score),
            key_headline=headline,
        )