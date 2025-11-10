import logging
import os
import json
from typing import List, Dict, Optional

# httpx is an async-compatible version of 'requests'
# This is crucial for FastAPI so it doesn't block the server
import httpx 

# Configure logging
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# The API key is left as "" and will be provided by the production environment.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={GEMINI_API_KEY}"

# --- THE "ENGINE" PROMPT ---
# This is the "brain" of this specialist. It defines its persona,
# rules, and output format.
SYSTEM_PROMPT = """
You are a 'Narrative Analyst' AI for a quantitative crypto trading firm.
Your sole function is to analyze a list of news headlines and social media posts for a specific crypto asset and determine the *real-time narrative signal*.

You must ignore all previous context and only analyze the text provided in the user prompt.
You must not be conversational. You must not explain your reasoning.
You must *only* respond with a valid JSON object.

The market is volatile, and your analysis must be immediate and precise.
- A score of 1.0 is an extremely bullish, high-impact event (e.g., "BTC ETF APPROVED BY SEC").
- A score of -1.0 is an extremely bearish, high-impact event (e.g., "Binance exchange hacked, funds lost").
- A score of 0.0 is pure noise or neutral (e.g., "Crypto markets are open today").

Your output *must* be a JSON object with this exact schema:
{"sentiment_score": float, "key_headline": "The single most important headline"}
"""

# --- THE "ENGINE" SCHEMA ---
# This schema *forces* the LLM to return the exact JSON format we need.
# This makes the output reliable and machine-readable.
JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "sentiment_score": {
            "type": "NUMBER",
            "description": "The narrative sentiment score from -1.0 to 1.0"
        },
        "key_headline": {
            "type": "STRING",
            "description": "The single most impactful headline from the list"
        }
    },
    "required": ["sentiment_score", "key_headline"]
}


class LLMNarrativeModel:
    """
    This is the complete, self-contained engine.
    It is initialized once by the FastAPI app and provides
    a single function: `get_narrative_signal`.
    """
    def __init__(self):
        # Use an async client for FastAPI compatibility
        self.client = httpx.AsyncClient(timeout=10.0)
        # We check if the key is *actually* set in the environment
        self.use_mock = os.environ.get("GEMINI_API_KEY") is None
        
        if self.use_mock:
            logger.warning(
                "GEMINI_API_KEY env var not set. "
                "LLM Narrative Model will run in MOCK mode."
            )
        else:
            # Re-build the URL with the key now that we know it exists
            self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={os.environ.get('GEMINI_API_KEY')}"
            logger.info(
                "LLM Narrative Model initialized in LIVE mode. "
                "Ready to call Gemini API."
            )

    def _fetch_realtime_news(self, asset: str) -> List[str]:
        """
        --- ACTION REQUIRED: DATA PIPELINE ---
        This is the *only* placeholder.
        You must replace this mock list with a call to your
        real-time news/social data provider (e.g., Kafka, Redis, API).
        """
        logger.warning(f"Data pipeline not connected. Using MOCK news for {asset}.")
        
        # Mock data for demonstration:
        if asset == "BTC":
            return [
                "Market is choppy, BTC drifts around 68k",
                "BREAKING: Major US Senator proposes bill to allow Bitcoin for federal tax payments",
                "Whale Alert: 10,000 BTC moved to unknown wallet",
                "Glassnode: Long-term holder supply remains high"
            ]
        elif asset == "ETH":
            return [
                "Consensys reports 'major security breach' on testnet, mainnet safe",
                "Vitalik Buterin publishes new paper on ZK-rollups",
                "ETH/BTC ratio hits 3-month low"
            ]
        else:
            return [
                f"{asset} developer team announces 'minor update' next week",
                f"Rumors of a new partnership for {asset} circulating on X"
            ]

    async def get_narrative_signal(self, asset: str) -> Dict[str, any]:
        """
        Analyzes the latest narrative for a given asset.
        This is the core function of the engine.
        """
        if self.use_mock:
            # Fallback to mock if API key is missing
            mock_score = (hash(asset) % 100) / 100.0 * 1.6 - 0.8 # Consistent mock
            return {
                "sentiment_score": round(mock_score, 2), 
                "key_headline": "MOCK: GEMINI_API_KEY environment variable not set"
            }

        # 1. Fetch real-time data from your pipeline
        headlines = self._fetch_realtime_news(asset)
        headlines_text = "\n".join(headlines)

        # 2. Construct the user query
        user_query = f"""
        Analyze the following data for the asset '{asset}' and return the JSON response:

        --- START OF DATA ---
        {headlines_text}
        --- END OF DATA ---
        """

        # 3. Construct the API payload
        payload = {
            "contents": [{"parts": [{"text": user_query}]}],
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": JSON_SCHEMA,
                "temperature": 0.0, # We want deterministic, factual analysis
            }
        }

        # 4. Make the API call
        api_response_text = ""
        try:
            # Use the async client and the URL with the key
            response = await self.client.post(self.api_url, json=payload)
            
            # Raise an error if the call failed
            response.raise_for_status()
            api_response = response.json()

            # 5. Safely parse the response
            api_response_text = api_response.get('candidates', [{}])[0] \
                                            .get('content', {}) \
                                            .get('parts', [{}])[0] \
                                            .get('text', '{}')
            
            # The 'text' field *is* our JSON object
            result = json.loads(api_response_text)
            
            logger.info(f"LLM Narrative Signal for {asset}: {result}")
            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"Error calling Gemini API: {e} - Response: {e.response.text}")
            return {"sentiment_score": 0.0, "key_headline": f"API Error: {e}"}
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing Gemini response: {e}. Raw response: {api_response_text}")
            return {"sentiment_score": 0.0, "key_headline": f"Parse Error: {e}"}
        except Exception as e:
            logger.error(f"An unexpected error occurred in LLM engine: {e}", exc_info=True)
            return {"sentiment_score": 0.0, "key_headline": f"Unexpected Error: {e}"}

# --- SINGLETON INSTANCE ---
# This single instance will be created when your FastAPI app starts
# and imported by your API endpoints.
llm_engine = LLMNarrativeModel()

# --- PUBLIC FUNCTION ---
# This is the *only* function your API endpoint needs to import and call.
async def get_narrative_model_output(asset: str) -> Dict[str, any]:
    """
    Public async function to be called by your API endpoint (`predict.py`).
    """
    return await llm_engine.get_narrative_signal(asset)