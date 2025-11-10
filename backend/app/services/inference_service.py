# app/services/inference_service.py
import logging
from typing import Any, Dict, List, Tuple
from datetime import datetime
from pathlib import Path
import random # We still use it for the *new* mock data function

# --- Model Loading Imports ---
import torch
import joblib
import numpy as np
import pandas as pd

# --- Imports for your REAL Models ---
from app.hybrid.schemas import MarketContext, HybridDecision, Layer2Prediction, RLAction
from app.ml.adv.llm_narrative_model import llm_engine # Import the REAL LLM engine
from app.ml.adv.rl_execution_agent import rl_agent     # Import the REAL RL agent

# --- Import Model Class Definitions ---
# We must import the model *classes* to load their .pth weights
from app.ml.adv.models_tft import TemporalFusionTransformer
from app.ml.adv.models_tcn import TemporalConvNet
# from app.ml.adv.decision_net import DecisionNet # Assuming you have a class
# from app.ml.adv.macro_onchain_model import MacroOnchainModel # Assuming you have a class
# from app.ml.adv.options_vol_model import OptionsVolatilityModel # Assuming you have a class


logger = logging.getLogger(__name__)

# --- 1. Define Model Artifact Paths ---
ARTIFACT_DIR = Path("./model_artifacts")
TFT_WEIGHTS_PATH = ARTIFACT_DIR / "tft_best_model.pth"
TCN_WEIGHTS_PATH = ARTIFACT_DIR / "tcn_best_model.pth"
XGB_PRICE_PATH = ARTIFACT_DIR / "xgb_price_model.joblib"
XGB_VOL_PATH = ARTIFACT_DIR / "xgb_vol_model.joblib"
DECISION_NET_PATH = ARTIFACT_DIR / "decision_net_model.joblib"
MACRO_MODEL_PATH = ARTIFACT_DIR / "macro_model.joblib"
OPTIONS_MODEL_PATH = ARTIFACT_DIR / "options_model.joblib"
SCALER_PATH = ARTIFACT_DIR / "tft_scaler.npz" # CRITICAL: The scaler used at training

# --- 2. Define Model Configurations (MUST match training) ---
# These are needed to create the model "skeletons" before loading weights
SEQ_LEN = 60 # Example: Must match your training sequence length

# Feature blocks (must match train_ensemble.py)
FEATURE_BLOCKS = {
    "price": ['open', 'high', 'low', 'close', 'volume'],
    "ob": ['ob_imbalance_1s', 'ob_spread', 'ob_depth_ask_1', 'ob_depth_bid_1'],
    "sent": ['sent_score_1m', 'sent_score_15m'],
}
# All sequential features
SEQ_FEATURE_COLS = [col for cols in FEATURE_BLOCKS.values() for col in cols]

# Tabular features (must match train_ensemble.py)
TABULAR_FEATURE_COLS = ['close', 'volume', 'open', 'high', 'low'] # Example
ROLL_WINDOWS = [5, 10, 20] # Example

# TFT Config (must match train_adv.py or train_ensemble.py)
TFT_CONFIG = {
    "feature_dims": { # Example dims, you must get this from your dataset
        'price': 5,
        'ob': 4,
        'sent': 2
    },
    "seq_len": SEQ_LEN,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 3,
    "dropout": 0.1
}

# TCN Config (must match train_tcn.py)
TCN_CONFIG = {
    "in_feat": len(SEQ_FEATURE_COLS), # Total num of sequential features
    "channels": (64, 128, 128),
    "kernel": 3,
    "dropout": 0.1
}

# --- 3. Internal Engine: Loads and runs all models ---

class _ModelInferenceEngine:
    """
    This private class is instantiated ONCE by InferenceService.
    It loads all models and scalers into memory and runs inference.
    """
    def __init__(self):
        logger.info("Initializing _ModelInferenceEngine...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_ready = True

        try:
            # --- Load Scaler ---
            logger.info(f"Loading scaler from {SCALER_PATH}")
            scaler = np.load(SCALER_PATH)
            self.scaler_mean = scaler['mean']
            self.scaler_std = scaler['std']
            
            # --- Load PyTorch Models ---
            logger.info("Loading PyTorch models...")
            self.tft_model = self._load_tft()
            self.tcn_model = self._load_tcn()
            
            # --- Load Joblib Models ---
            logger.info("Loading joblib models...")
            self.xgb_price_model = self._load_joblib("XGB Price", XGB_PRICE_PATH)
            self.xgb_vol_model = self._load_joblib("XGB Vol", XGB_VOL_PATH)
            self.decision_net = self._load_joblib("DecisionNet", DECISION_NET_PATH)
            self.macro_model = self._load_joblib("Macro", MACRO_MODEL_PATH)
            self.options_model = self._load_joblib("Options", OPTIONS_MODEL_PATH)
            
        except Exception as e:
            logger.error(f"Failed to load one or more models: {e}", exc_info=True)
            self.is_ready = False
        
        if self.is_ready:
            logger.info("_ModelInferenceEngine initialized successfully.")
        else:
            logger.warning("_ModelInferenceEngine initialized with MISSING models.")

    def _load_joblib(self, name: str, path: Path):
        if not path.exists():
            logger.warning(f"{name} model not found at {path}. Voting will be disabled.")
            return None
        logger.info(f"Loading {name} from {path}...")
        return joblib.load(path)

    def _load_tft(self):
        if not TFT_WEIGHTS_PATH.exists():
            logger.warning(f"TFT model not found at {TFT_WEIGHTS_PATH}.")
            return None
        model = TemporalFusionTransformer(**TFT_CONFIG).to(self.device)
        model.load_state_dict(torch.load(TFT_WEIGHTS_PATH, map_location=self.device))
        model.eval()
        return model

    def _load_tcn(self):
        if not TCN_WEIGHTS_PATH.exists():
            logger.warning(f"TCN model not found at {TCN_WEIGHTS_PATH}.")
            return None
        model = TemporalConvNet(**TCN_CONFIG).to(self.device)
        model.load_state_dict(torch.load(TCN_WEIGHTS_PATH, map_location=self.device))
        model.eval()
        return model

    def _preprocess_seq(self, features_df: pd.DataFrame) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepares data for TCN and TFT models."""
        # 1. Get last SEQ_LEN rows
        df_seq = features_df[SEQ_FEATURE_COLS].iloc[-SEQ_LEN:]
        
        # 2. Apply scaler
        df_scaled = (df_seq - self.scaler_mean) / (self.scaler_std + 1e-8)
        
        # 3. Create TCN input (B, C, L) -> (1, num_features, seq_len)
        tcn_tensor = torch.from_numpy(df_scaled.values.T).float().unsqueeze(0).to(self.device)
        
        # 4. Create TFT input (dict of tensors)
        tft_blocks = {}
        idx = 0
        for block_name, block_cols in FEATURE_BLOCKS.items():
            num_cols = len(block_cols)
            block_data = df_scaled.iloc[:, idx : idx + num_cols]
            tft_blocks[block_name] = torch.from_numpy(block_data.values).float().unsqueeze(0).to(self.device)
            idx += num_cols
            
        return tft_blocks, tcn_tensor

    def _preprocess_tab(self, features_df: pd.DataFrame) -> np.ndarray:
        """Prepares data for XGBoost models."""
        # TODO: This MUST match the feature engineering used for XGB training
        # This is a simplified example. You must add your rolling windows, etc.
        logger.warning("XGBoost preprocessing is simplified. Ensure it matches training!")
        
        # Simple example: just get the last row of tabular features
        # In reality, you need to compute rolling features here.
        last_row = features_df[TABULAR_FEATURE_COLS].iloc[-1].values
        
        # ... add your rolling features from ROLL_WINDOWS ...
        
        return last_row.reshape(1, -1) # (1, num_features)

    @torch.no_grad()
    def get_deep_signals(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Runs TFT and TCN models."""
        if not self.tft_model or not self.tcn_model:
            return {"tft_price": 0.0, "tcn_price": 0.0}
            
        tft_blocks, tcn_tensor = self._preprocess_seq(features_df)
        
        # Run models
        preds_tft = self.tft_model(tft_blocks)
        preds_tcn = self.tcn_model(tcn_tensor)
        
        # Return the 'price' prediction (assuming this is the output structure)
        return {
            "tft_price": preds_tft.get('price', torch.tensor(0.0)).item(),
            "tcn_price": preds_tcn.get('price', torch.tensor(0.0)).item()
        }

    def get_xgb_signals(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Runs XGBoost models."""
        if not self.xgb_price_model or not self.xgb_vol_model:
            return {"xgb_price": 0.0, "xgb_vol": 0.0}
            
        x_tabular = self._preprocess_tab(features_df)
        
        price_pred = self.xgb_price_model.predict(x_tabular)[0]
        vol_pred = self.xgb_vol_model.predict(x_tabular)[0]
        
        return {"xgb_price": float(price_pred), "xgb_vol": float(vol_pred)}

    def get_specialist_signals(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Runs other specialist models (Macro, Options)."""
        signals = {"options_price": 0.0, "macro_price": 0.0}
        
        # Prepare features (this is just an example)
        x_data = features_df.iloc[-1].values.reshape(1, -1) # adjust as needed
        
        if self.options_model:
            # You need to define what features this model expects
            # signals["options_price"] = self.options_model.predict(x_data)[0]
            pass # Placeholder
            
        if self.macro_model:
            # You need to define what features this model expects
            # signals["macro_price"] = self.macro_model.predict(x_data)[0]
            pass # Placeholder
            
        return signals

    def run_decision_net(self, votes: Dict[str, float]) -> Dict[str, Any]:
        """
        REAL replacement for the mock DecisionNet.
        Fuses Layer 1 votes into a single prediction.
        """
        if self.decision_net is None:
            logger.warning("DecisionNet not loaded. Using heuristic fallback.")
            # Fallback logic from your mock
            final_score = sum(votes.values()) / (len(votes) or 1)
            direction = "flat"
            if final_score > 0.3: direction = "up"
            elif final_score < -0.3: direction = "down"
            confidence = min(1.0, abs(final_score) * 1.5)
            return {"direction": direction, "price_confidence": round(confidence, 2)}
            
        # --- REAL INFERENCE ---
        # 1. Convert votes dict to DataFrame row (must match training)
        # Ensure the column order is IDENTICAL to the one used for training
        try:
            votes_df = pd.DataFrame([votes])
            
            # Example:
            # 'decision_net.predict_proba' returns [[P(class_0), P(class_1), P(class_2)]]
            # where class 0=down, 1=flat, 2=up
            # probas = self.decision_net.predict_proba(votes_df)[0]
            
            # This is a GUESS. You must adapt this to your DecisionNet's output.
            # Let's assume it's a simple regression model for this example:
            final_score = self.decision_net.predict(votes_df)[0]
            
            # Map score to direction/confidence
            direction = "flat"
            if final_score > 0.3: direction = "up"  # Adjust thresholds
            elif final_score < -0.3: direction = "down"
            confidence = min(1.0, abs(final_score) * 1.5)
            
            return {
                "direction": direction,
                "price_confidence": round(confidence, 2)
            }

        except Exception as e:
            logger.error(f"Error running DecisionNet: {e}", exc_info=True)
            return {"direction": "flat", "price_confidence": 0.0}


# --- 4. Main Public-Facing Service ---

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
            
            # --- THIS IS THE KEY ---
            # Instantiate the engine to load all models
            self.model_engine = _ModelInferenceEngine()
            
            self.is_ready: bool = (
                self.rl_agent.is_model_loaded() and self.model_engine.is_ready
            )
            
            if not self.is_ready:
                logger.warning("[InferenceService] Initialization incomplete. Check warnings.")
            else:
                logger.info("[InferenceService] Initialization complete. All models loaded.")
                
        except Exception as e:
            logger.error(
                f"[InferenceService] Failed to initialize core engine: {e}",
                exc_info=True,
            )
            self.is_ready: bool = False

    def _fetch_and_prepare_features(self, asset: str, seq_len: int) -> pd.DataFrame:
        """
        --- THIS IS THE NEW MOCK ---
        This function is responsible for getting the latest feature data
        for a given asset. You MUST replace this with a real implementation
        that queries your database or market data service.
        
        It must return a DataFrame with AT LEAST `seq_len` rows
        and all columns needed for:
        1. TCN/TFT (all columns in SEQ_FEATURE_COLS)
        2. XGB (all columns in TABULAR_FEATURE_COLS + data for rolling features)
        """
        logger.warning(f"--- USING MOCK FEATURE DATA for {asset} ---")
        # TODO: REPLACE THIS
        
        # 1. Get all feature columns
        all_cols = list(set(SEQ_FEATURE_COLS + TABULAR_FEATURE_COLS))
        
        # 2. Generate mock data
        # We need more than seq_len rows to calculate rolling features for XGB
        num_rows = seq_len + max(ROLL_WINDOWS) + 5 
        data = {col: np.random.randn(num_rows) for col in all_cols}
        
        # Add some realistic price data
        data['close'] = np.cumsum(np.random.randn(num_rows) * 0.1) + 100
        data['open'] = data['close'] - np.random.rand(num_rows) * 0.1
        data['high'] = data['close'] + np.random.rand(num_rows) * 0.1
        data['low'] = data['close'] - np.random.rand(num_rows) * 0.1
        data['volume'] = np.random.rand(num_rows) * 100
        
        return pd.DataFrame(data)

    async def build_decision(self, ctx: MarketContext) -> HybridDecision:
        """
        Runs the full 3-Layer Hybrid AI logic to make a trading decision.
        """
        # Get "BTC" from "BTC-PERP" or "BTC/USDT"
        asset = ctx.symbol.split('/')[0].split('-')[0].upper()
        
        # --- 1. Get Features ---
        # This is the ONE function you must replace with real data
        features_df = self._fetch_and_prepare_features(asset, SEQ_LEN)
        
        # --- LAYER 1: GATHER VOTES FROM ALL SPECIALISTS ---
        logger.debug(f"Layer 1: Gathering specialist votes for {asset}")
        
        # Run async LLM task
        llm_output_task = self.llm_engine.get_narrative_signal(asset)
        
        # --- Run REAL model tasks ---
        deep_signals = self.model_engine.get_deep_signals(features_df)
        xgb_signals = self.model_engine.get_xgb_signals(features_df)
        specialist_signals = self.model_engine.get_specialist_signals(features_df)
        
        # Wait for LLM task to complete
        llm_output = await llm_output_task
        
        model_votes = {
            "tft_visionary": deep_signals.get("tft_price", 0.0),
            "tcn_reflex": deep_signals.get("tcn_price", 0.0),
            "xgb_analyst": xgb_signals.get("xgb_price", 0.0),
            "xgb_vol_analyst": xgb_signals.get("xgb_vol", 0.0), # Added vol
            "options_psychologist": specialist_signals.get("options_price", 0.0),
            "macro_economist": specialist_signals.get("macro_price", 0.0),
            "llm_narrative": llm_output["sentiment_score"],
        }
        
        # --- LAYER 2: FUSE PREDICTIONS WITH DECISIONNET ---
        logger.debug("Layer 2: Fusing votes with DecisionNet...")
        # --- THIS IS THE REAL CALL ---
        layer_2_output = self.model_engine.run_decision_net(model_votes)
        
        prediction = Layer2Prediction(
            asset=asset,
            direction=layer_2_output["direction"],
            price_confidence=layer_2_output["price_confidence"]
        )

        # --- LAYER 3: GET OPTIMAL ACTION FROM RL AGENT ---
        logger.debug("Layer 3: Getting optimal action from RL Agent...")
        if ctx.current_regime is None:
            ctx.current_regime = "mock_regime_neutral"
        if ctx.current_volatility is None:
            ctx.current_volatility = xgb_signals.get("xgb_vol", 0.5) # Use predicted vol

        rl_action: RLAction = self.rl_agent.get_optimal_action(
            prediction=prediction, 
            context=ctx, # Pass the real context from the API
            model_votes=model_votes
        )

        # --- Build and return the final HybridDecision ---
        confidence = prediction.price_confidence
        
        if rl_action.optimal_size_pct > 0:
            size_factor = rl_action.optimal_size_pct
        else:
            size_factor = 0.0

        direction = "flat"
        if rl_action.optimal_action == "LONG":
            direction = "up"
        elif rl_action.optimal_action == "SHORT":
            direction = "down"

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
            direction=direction,
            p_edge=confidence, 
            confidence=confidence,
            size_factor=size_factor,
            strategy_tag="nowa_hybrid_rl_v1",
            meta_execute=True,
            model_votes={k: round(v, 2) for k, v in model_votes.items()},
            llm_headline=llm_output["key_headline"],
            rl_action=rl_action.optimal_action,
            rl_mode=self.rl_agent.get_mode(),
            rl_target_position=rl_action.optimal_size_pct if direction == "up" else -rl_action.optimal_size_pct,
            rl_execution_style=rl_action.execution_style,
            debug=debug_payload
        )

# --- SINGLETON INSTANCE ---
# Create one instance to be imported by the FastAPI app
try:
    inference_service = InferenceService()
except Exception as e:
    logger.critical(f"Failed to create InferenceService singleton: {e}", exc_info=True)
    inference_service = None # type: ignore