# app/services/inference_service.py
import logging
from typing import Any, Dict
import random # For mock specialists
from datetime import datetime

# --- Imports for your NEW 3-Layer Architecture ---
from app.hybrid.schemas import MarketContext, HybridDecision, Layer2Prediction, RLAction
from app.ml.adv.llm_narrative_model import llm_engine # Import the REAL LLM engine
from app.ml.adv.rl_execution_agent import rl_agent     # Import the REAL RL agent

# (Mock) Imports for your *EXISTING* models
# In a real app, these would be your actual model inference functions
# from app.ml.adv.models_tft import get_tft_signal
# from app.ml.adv.models_tcn import get_tcn_signal
# ...etc
def get_tft_signal(asset: str) -> float: return random.uniform(-1, 1)
def get_tcn_signal(asset: str) -> float: return random.uniform(-1, 1)
def get_xgb_signal(asset: str) -> float: return random.uniform(-1, 1)
def get_options_signal(asset: str) -> float: return random.uniform(-1, 1)
def get_macro_signal(asset: str) -> float: return random.uniform(-1, 1)

# (Mock) Import for DecisionNet (Layer 2)
# from app.ml.adv.decision_net import run_decision_net
def run_decision_net(votes: Dict[str, float]) -> Dict:
    """
    MOCK of your DecisionNet ("The Judge").
    It takes all Layer 1 votes and fuses them into a single prediction.
    """
    if not votes:
        return {"direction": "flat", "price_confidence": 0.0}
        
    final_score = sum(votes.values()) / len(votes)
    direction = "flat"
    if final_score > 0.3: direction = "up"
    elif final_score < -0.3: direction = "down"
    
    confidence = min(1.0, abs(final_score) * 1.5) # Mock confidence
    return {
        "direction": direction,
        "price_confidence": round(confidence, 2)
    }

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Public-facing inference layer for Nowa.
    This IS the core HybridInferenceService (Layer 1 + Layer 2 + Layer 3).
    It exposes a clean API used by FastAPI routes.
    """

    def __init__(self) -> None:
        try:
            logger.info("[InferenceService] Initializing...")
            # Load the RL agent (which loads its model)
            self.rl_agent = rl_agent
            # Get the LLM engine
            self.llm_engine = llm_engine
            
            # Simple readiness check; extend with real model-loading checks
            self.is_ready: bool = self.rl_agent.is_model_loaded()
            if not self.is_ready:
                logger.warning("[InferenceService] RL model not found. Agent is in MOCK mode.")
            else:
                logger.info("[InferenceService] Initialization complete. RL model loaded.")
        except Exception as e:
            logger.error(
                f"[InferenceService] Failed to initialize core engine: {e}",
                exc_info=True,
            )
            self.is_ready: bool = False

    async def build_decision(self, ctx: MarketContext) -> HybridDecision:
        """
        Runs the full 3-Layer Hybrid AI logic to make a trading decision.
        """
        # Get "BTC" from "BTC-PERP" or "BTC/USDT"
        asset = ctx.symbol.split('/')[0].split('-')[0].upper()
        
        # --- LAYER 1: GATHER VOTES FROM ALL SPECIALISTS ---
        logger.debug(f"Layer 1: Gathering specialist votes for {asset}")
        
        # Run async LLM task
        llm_output_task = self.llm_engine.get_narrative_signal(asset)
        
        # Run sync model tasks (replace these with your real model calls)
        tft_signal = get_tft_signal(asset)
        tcn_signal = get_tcn_signal(asset)
        xgb_signal = get_xgb_signal(asset)
        options_signal = get_options_signal(asset)
        macro_signal = get_macro_signal(asset)
        
        # Wait for LLM task to complete
        llm_output = await llm_output_task
        
        model_votes = {
            "tft_visionary": tft_signal,
            "tcn_reflex": tcn_signal,
            "xgb_analyst": xgb_signal,
            "options_psychologist": options_signal,
            "macro_economist": macro_signal,
            "llm_narrative": llm_output["sentiment_score"],
        }
        
        # --- LAYER 2: FUSE PREDICTIONS WITH DECISIONNET ---
        logger.debug("Layer 2: Fusing votes with DecisionNet...")
        # NOTE: This uses the MOCK `run_decision_net` function above.
        # You must replace this with a call to your REAL DecisionNet model.
        layer_2_output = run_decision_net(model_votes)
        
        prediction = Layer2Prediction(
            asset=asset,
            direction=layer_2_output["direction"],
            price_confidence=layer_2_output["price_confidence"]
        )

        # --- LAYER 3: GET OPTIMAL ACTION FROM RL AGENT ---
        logger.debug("Layer 3: Getting optimal action from RL Agent...")
        # We need to update the context with the latest volatility/regime if it's missing
        # In a real system, this would come from your rt_adapt service
        if ctx.current_regime is None:
            ctx.current_regime = "mock_regime_neutral"
        if ctx.current_volatility is None:
            ctx.current_volatility = 0.5 # mock vol

        rl_action: RLAction = self.rl_agent.get_optimal_action(
            prediction=prediction, 
            context=ctx, # Pass the real context from the API
            model_votes=model_votes
        )

        # --- Build and return the final HybridDecision ---
        # This maps the 3-layer output to your existing HybridDecision schema
        
        confidence = prediction.price_confidence
        
        # Use RL agent's sizing if available, otherwise fallback
        if rl_action.optimal_size_pct > 0:
            size_factor = rl_action.optimal_size_pct
        else:
            # Fallback for "HOLD" or if RL agent fails
            size_factor = 0.0

        # Map RL action to a direction
        direction = "flat"
        if rl_action.optimal_action == "LONG":
            direction = "up"
        elif rl_action.optimal_action == "SHORT":
            direction = "down"

        # The new debug payload is much richer
        debug_payload = {
            "l1_votes": model_votes,
            "l2_prediction": prediction.dict(),
            "l3_action": rl_action.dict(),
            "context_in": ctx.dict()
        }

        return HybridDecision(
            symbol=ctx.symbol,
            instrument_type=ctx.instrument_type or "futures",
            timestamp=datetime.utcnow(),
            
            # --- From Layer 2 & 3 ---
            direction=direction,
            p_edge=confidence, # Using confidence as p_edge
            confidence=confidence,
            size_factor=size_factor,
            strategy_tag="nowa_hybrid_rl_v1",
            meta_execute=True, # RL agent's decision is always executable
            
            # --- From Layer 1 (Votes) ---
            model_votes={k: round(v, 2) for k, v in model_votes.items()},
            llm_headline=llm_output["key_headline"],
            
            # --- From Layer 3 (RL Action) ---
            rl_action=rl_action.optimal_action,
            rl_mode=self.rl_agent.get_mode(),
            rl_target_position=rl_action.optimal_size_pct if direction == "up" else -rl_action.optimal_size_pct,
            rl_execution_style=rl_action.execution_style,
            
            # Full debug info
            debug=debug_payload
        )

# --- SINGLETON INSTANCE ---
# Create one instance to be imported by the FastAPI app
try:
    inference_service = InferenceService()
except Exception as e:
    logger.critical(f"Failed to create InferenceService singleton: {e}", exc_info=True)
    inference_service = None # type: ignore