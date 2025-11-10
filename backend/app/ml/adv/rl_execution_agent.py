import logging
import gymnasium as gym
import numpy as np
import pandas as pd
import os
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

# --- NOTE ---
# This file contains the complete, automated RL system.
# 1. `TradingEnv`: The "flight simulator" for your agent.
# 2. `RLExecutionAgent`: The "pilot" (inference agent) your API uses.
# 3. `AutomatedTrainingPipeline`: The "flight school" (automated data scientist).
# 
# A Celery task will call `AutomatedTrainingPipeline.run_training()`
# weekly to automatically re-train the bot.
# -----------------------------------------------------------------

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# The final, trained "Head Trader" model will be saved here
MODEL_SAVE_PATH = "backend/models/rl_agent_ppo.zip"
# The master dataset (auto-generated) will be saved here
TRAINING_DATA_CSV = "backend/data/master_training_data.csv"
# The raw price data to use for training
HISTORICAL_PRICE_DATA = "backend/data/BTC_SPOT_1h.csv" # Example

# --- Data Structures (matching schemas.py) ---
class Layer2Prediction(BaseModel):
    """ The consensus prediction from DecisionNet (Layer 2) """
    asset: str
    direction: str
    price_confidence: float

class MarketContext(BaseModel):
    """ Real-time market state """
    current_regime: str
    current_volatility: float

class RLAction(BaseModel):
    """ The final, executable action from the RL Agent (Layer 3) """
    optimal_action: str
    optimal_size_pct: float
    execution_style: str


# --- 1. The Training Environment (The "Flight Simulator") ---

class TradingEnv(gym.Env):
    """
    A custom Gymnasium environment for training the RL "Head Trader".
    It takes a DataFrame of historical prices AND signals.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, feature_columns: List[str], price_column: str = 'price'):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.feature_columns = feature_columns
        self.price_column = price_column
        self.current_step = 0
        self.max_steps = len(self.df) - 2 # (n - 1 for index, -1 for next_price)

        # Observation space: All feature columns + 1 (for current_position)
        self.num_features = len(self.feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_features + 1,), dtype=np.float32
        )

        # Action space: 0 = HOLD, 1 = LONG, 2 = SHORT
        self.action_space = spaces.Discrete(3)
        self.action_map = {0: 0, 1: 1, 2: -1} # Map code to position

        # Portfolio state
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.current_position = 0 # -1 (short), 0 (flat), 1 (long)

    def _get_observation(self) -> np.ndarray:
        """Build the state vector for the current time step."""
        try:
            # Get all feature values for the current step
            obs = self.df.iloc[self.current_step][self.feature_columns].values
            # Add the agent's current position to the state
            obs_with_position = np.append(obs, [self.current_position])
            return obs_with_position.astype(np.float32)
        except Exception as e:
            logger.error(f"Error getting observation at step {self.current_step}: {e}")
            return np.zeros(self.num_features + 1).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.current_position = 0
        return self._get_observation(), {}

    def step(self, action):
        if self.current_step >= self.max_steps:
            # We are at the end of the data
            return self._get_observation(), 0, True, False, {}

        # 1. Get current state data
        current_price = self.df.iloc[self.current_step][self.price_column]
        target_position = self.action_map[action] # What the agent *wants* to do
        
        # 2. Advance time to the next step
        self.current_step += 1
        next_price = self.df.iloc[self.current_step][self.price_column]
        
        # 3. Calculate Reward (PnL)
        # We calculate the reward based on the position we *were* in
        reward = 0.0
        price_change_pct = (next_price - current_price) / current_price
        
        if self.current_position == 1: # We were LONG
            reward = price_change_pct
        elif self.current_position == -1: # We were SHORT
            reward = -price_change_pct
        
        # Simple transaction cost (0.1% per trade)
        if target_position != self.current_position:
             reward -= 0.001 
            
        # 4. Update portfolio
        self.balance *= (1 + reward) # Apply PnL
        self.current_position = target_position # Update to new position

        # 5. Check for termination
        done = self.current_step >= self.max_steps
        if self.balance < self.initial_balance * 0.5: # Stop if we lose 50%
            done = True
            logger.warning("RL Agent was 'liquidated' during training.")

        return self._get_observation(), reward, done, False, {}


# --- 2. The Automated "Data Scientist" (The "Flight School") ---

class AutomatedTrainingPipeline:
    """
    This is the "bot data scientist".
    It's responsible for fetching data, generating historical signals
    from all Layer 1 models, and training the Layer 3 RL agent.
    """
    def __init__(self):
        # This list defines the "state" for the RL agent.
        # It MUST match the features in `_generate_historical_signals`
        self.feature_columns = [
            'decisionnet_confidence', 'decisionnet_direction',
            'tft_signal', 'tcn_signal', 'xgb_signal',
            'options_signal', 'macro_signal', 'llm_signal'
        ]
        self.price_column = 'price'
        self.num_features = len(self.feature_columns)
        logger.info("AutomatedTrainingPipeline (Bot Data Scientist) initialized.")

    def _generate_historical_signals(self) -> pd.DataFrame:
        """
        --- ACTION REQUIRED: REAL DATA PIPELINE ---
        This is the most critical function you must build.
        You must replace this mock with your *actual* historical data pipeline.
        
        This function needs to:
        1. Load historical price data (e.g., from HISTORICAL_PRICE_DATA)
        2. Load all your trained Layer 1 models (TFT, TCN, XGB, etc.)
        3. Iterate through the *entire* price history (e.g., 5 years) and run 
           `model.predict()` for *every single timestep* to get the historical signals.
        4. (Mock) Run the DecisionNet (Layer 2) over those signals.
        5. (Mock) Run the LLM over historical news (this is complex/expensive) 
           or use a proxy (e.g., a simple sentiment score on headlines).
        6. Return a single, massive DataFrame with all signals aligned to the price.
        """
        logger.warning("--- MOCK DATA --- Using mock data for RL training.")
        logger.warning("--- ACTION REQUIRED --- Replace `_generate_historical_signals` with your real backtesting pipeline.")
        
        try:
            # 1. Load historical price data
            price_df = pd.read_csv(HISTORICAL_PRICE_DATA)
            price_df = price_df.rename(columns={'close': self.price_column}) # Adjust as needed
            size = len(price_df)
            
            # 2. Create a new DataFrame for all features
            df = pd.DataFrame(index=price_df.index)
            df[self.price_column] = price_df[self.price_column]
            
            # 3. (MOCK) Generate signals for all models
            #    Replace these random numbers with your *actual* model.predict() calls
            df['tft_signal'] = np.random.uniform(-1, 1, size)
            df['tcn_signal'] = np.random.uniform(-1, 1, size)
            df['xgb_signal'] = np.random.uniform(-1, 1, size)
            df['options_signal'] = np.random.uniform(-1, 1, size)
            df['macro_signal'] = np.random.uniform(-1, 1, size)
            df['llm_signal'] = np.random.uniform(-1, 1, size)
            
            # 4. (MOCK) Generate signals for DecisionNet (Layer 2)
            #    This would be your `decision_net.predict()`
            avg_signal = df[self.feature_columns[2:]].mean(axis=1) # Mock fusion
            df['decisionnet_confidence'] = np.abs(avg_signal)
            df['decisionnet_direction'] = np.sign(avg_signal)
            
            # 5. Save and return
            df.dropna(inplace=True) # Ensure no NaNs
            df.to_csv(TRAINING_DATA_CSV, index=False)
            logger.info(f"Mock training data generated and saved to {TRAINING_DATA_CSV}")
            return df
        
        except FileNotFoundError:
            logger.error(f"CRITICAL: Historical price data not found at {HISTORICAL_PRICE_DATA}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"CRITICAL: Failed to generate historical signals: {e}", exc_info=True)
            return pd.DataFrame()


    def run_training(self):
        """
        The main function called by the Celery task.
        This function *is* the automated data scientist.
        """
        logger.info("--- STARTING AUTOMATED RL AGENT TRAINING ---")
        
        # 1. Generate the master training dataset
        logger.info("Step 1/4: Generating historical signals...")
        df = self._generate_historical_signals()
        if df.empty:
            logger.error("Training halted: Historical signal generation failed.")
            return

        # 2. Create the environment
        logger.info("Step 2/4: Initializing training environment...")
        env = DummyVecEnv([lambda: TradingEnv(df, self.feature_columns, self.price_column)])

        # 3. Create or load the PPO model
        logger.info("Step 3/4: Initializing PPO model...")
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        
        if os.path.exists(MODEL_SAVE_PATH):
            logger.info(f"Loading existing model from {MODEL_SAVE_PATH} to continue training.")
            model = PPO.load(MODEL_SAVE_PATH, env=env)
        else:
            logger.info("No existing model found. Creating new PPO model.")
            # We use "MlpPolicy" because our state vector is a simple 1D array of numbers
            model = PPO("MlpPolicy", env, verbose=0)
        
        # 4. Train the model
        # 100k timesteps is a small-to-medium run, good for weekly updates.
        # A full, initial training might be 1M or 10M timesteps.
        training_timesteps = 100_000
        logger.info(f"Step 4/4: Starting training for {training_timesteps} timesteps...")
        model.learn(total_timesteps=training_timesteps, tb_log_name="rl_agent_run")
        logger.info("...Training complete.")

        # 5. Save the newly trained model
        model.save(MODEL_SAVE_PATH)
        logger.info(f"Trained RL model saved to {MODEL_SAVE_PATH}")
        logger.info("--- AUTOMATED RL AGENT TRAINING COMPLETE ---")


# --- 3. The Live Inference Engine (The "Pilot") ---

class RLExecutionAgent:
    """
    The "Head Trader" (Layer 3) Inference Engine.
    Loads the pre-trained PPO model and provides optimal actions.
    This class is instantiated once when the FastAPI app starts.
    """
    def __init__(self, model_path: str = MODEL_SAVE_PATH):
        self.model_path = model_path
        self.model: Optional[PPO] = None
        
        # This MUST match the feature_columns in the TrainingPipeline
        self.feature_columns = [
            'decisionnet_confidence', 'decisionnet_direction',
            'tft_signal', 'tcn_signal', 'xgb_signal',
            'options_signal', 'macro_signal', 'llm_signal'
        ]
        self.num_features = len(self.feature_columns)
        self.load_model()

    def load_model(self):
        """ Loads the trained model from disk. """
        try:
            if os.path.exists(self.model_path):
                self.model = PPO.load(self.model_path)
                logger.info(f"Successfully loaded trained RL model from {self.model_path}")
            else:
                logger.error(f"RL model file not found at {self.model_path}. Agent will run in MOCK mode.")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading RL model: {e}. Agent will run in MOCK mode.", exc_info=True)
            self.model = None

    def _format_state(
        self,
        prediction: Layer2Prediction,
        model_votes: Dict[str, float],
        current_position: int
    ) -> np.ndarray:
        """
        Formats the live API data into the 1D state vector
        that the trained RL model expects.
        
        The order MUST be identical to `self.feature_columns` + `current_position`.
        """
        direction_numeric = 1.0 if prediction.direction == 'up' else -1.0 if prediction.direction == 'down' else 0.0

        obs_values = [
            prediction.price_confidence,
            direction_numeric,
            model_votes.get('tft_visionary', 0.0),
            model_votes.get('tcn_reflex', 0.0),
            model_votes.get('xgb_analyst', 0.0),
            model_votes.get('options_psychologist', 0.0),
            model_votes.get('macro_economist', 0.0),
            model_votes.get('llm_narrative', 0.0),
        ]
        
        # Append the final piece of state: our current position
        obs_values.append(current_position)
        
        if len(obs_values) != self.num_features + 1:
            logger.error(f"State vector length mismatch! Expected {self.num_features + 1}, got {len(obs_values)}")
            # Return a "neutral" state
            return np.zeros(self.num_features + 1).astype(np.float32)

        return np.array(obs_values).astype(np.float32)

    def get_optimal_action(
        self,
        prediction: Layer2Prediction,
        context: MarketContext,
        model_votes: Dict[str, float]
    ) -> RLAction:
        """
        Takes the Layer 2 prediction and market state, and returns
        the optimal Layer 3 action.
        """
        if not self.model:
            # --- MOCK LOGIC (if model fails to load) ---
            logger.warning("RL Agent running in MOCK mode (model not loaded).")
            action, size = "HOLD", 0.0
            if prediction.direction == "up" and prediction.price_confidence > 0.7:
                action, size = "LONG", 0.5
            elif prediction.direction == "down" and prediction.price_confidence > 0.7:
                action, size = "SHORT", 0.5
            return RLAction(optimal_action=action, optimal_size_pct=size, execution_style="Market")

        # --- REAL INFERENCE ---
        # 1. Get current portfolio state (from your OrderManager service)
        #    !!! ACTION REQUIRED: Replace this with your *real* portfolio query !!!
        current_position = 0 # 0=flat, 1=long, -1=short
        
        # 2. Format the observation
        obs = self._format_state(prediction, model_votes, current_position)
        
        # 3. Predict the action from the state
        #    `deterministic=True` means we take the "best" action, not a random one
        action_code, _states = self.model.predict(obs, deterministic=True)
        
        # 4. Decode the action and return
        if action_code == 1:
            return RLAction(optimal_action="LONG", optimal_size_pct=1.0, execution_style="Market")
        elif action_code == 2:
            return RLAction(optimal_action="SHORT", optimal_size_pct=1.0, execution_style="Market")
        else: # action_code == 0
            return RLAction(optimal_action="HOLD", optimal_size_pct=0.0, execution_style="Market")

# --- 4. Singleton Instances ---
# These are loaded once by FastAPI on startup
rl_agent_engine = RLExecutionAgent()
automated_trainer_instance = AutomatedTrainingPipeline()

# --- 5. Public Functions for API and Celery ---

def get_optimal_action(
    prediction: Layer2Prediction,
    context: MarketContext,
    model_votes: Dict[str, float]
) -> RLAction:
    """Public function called by the API endpoint (`predict.py`)."""
    return rl_agent_engine.get_optimal_action(prediction, context, model_votes)

def run_automated_training():
    """Public function called by the Celery task (`training_tasks.py`)."""
    automated_trainer_instance.run_training()
    # After training, tell the live agent to reload the new model
    logger.info("Reloading live RL Agent with newly trained model...")
    rl_agent_engine.load_model()