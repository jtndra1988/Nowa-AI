import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

logger = logging.getLogger(_name_)

# Canonical feature layout shared by:
# - training data
# - TradingEnv
# - RLExecutionAgent live inference
FEATURE_COLUMNS: List[str] = [
    "decisionnet_confidence",
    "decisionnet_direction",  # -1, 0, 1
    "tft_visionary",
    "tcn_reflex",
    "xgb_analyst",
    "options_psychologist",
    "macro_economist",
    "llm_narrative",
]

# Where we store the trained policy
DEFAULT_POLICY_PATH = os.getenv(
    "NOWA_RL_POLICY_PATH", "backend/app/ml/adv/rl_head_trader_ppo.zip"
)


@dataclass
class Layer2Decision:
    """
    Minimal view of Layer 2 (DecisionNet) output consumed by RL Head Trader.
    """

    direction: str  # "up", "down", or "flat"
    confidence: float  # 0-1
    model_votes: Dict[str, float]  # keys MUST match FEATURE_COLUMNS where applicable


class TradingEnv(gym.Env):
    """
    Simple portfolio environment for PPO training.

    State = [FEATURE_COLUMNS..., current_position]
      - features: normalized signals from DecisionNet + all Layer 1 specialists.
      - current_position: -1.0 short, 0 flat, +1 long.

    Action space:
      0 -> flat (close / stay flat)
      1 -> long (e.g. +1x)
      2 -> short (e.g. -1x)

    Reward:
      - PnL based on "true" move (simulated) vs position
      - Penalty for churn & leverage
      - Heavy penalty for "liquidation" events
    """

    metadata = {"render_modes": []}

    def _init_(
        self,
        episode_len: int = 256,
        transaction_cost: float = 0.0005,
        liq_penalty: float = -5.0,
        seed: int = 42,
    ):
        super()._init_()
        self.episode_len = episode_len
        self.transaction_cost = transaction_cost
        self.liq_penalty = liq_penalty

        self.rng = np.random.default_rng(seed)

        # Observation = len(FEATURE_COLUMNS) + current_position
        obs_dim = len(FEATURE_COLUMNS) + 1
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32
        )
        # 0 flat, 1 long, 2 short
        self.action_space = spaces.Discrete(3)

        self._t = 0
        self._position = 0.0
        self._last_price = 1.0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._t = 0
        self._position = 0.0
        self._last_price = 1.0
        return self._sample_state(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action)

        prev_pos = self._position
        # Map discrete action to target position
        if action == 0:
            self._position = 0.0
        elif action == 1:
            self._position = 1.0
        else:
            self._position = -1.0

        # Simulate price move: small random plus weak drift from "true" signal
        true_signal = self._latent_trend()
        price_move = true_signal * 0.003 + self.rng.normal(0, 0.003)
        new_price = self._last_price * (1.0 + price_move)

        # PnL approx: position * return
        pnl = self._position * ((new_price / self._last_price) - 1.0)

        # Transaction cost on change in position
        cost = abs(self._position - prev_pos) * self.transaction_cost
        reward = pnl - cost

        # Rare "liquidation" event when highly misaligned
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        if abs(price_move) > 0.05 and np.sign(price_move) != np.sign(self._position):
            reward += self.liq_penalty
            terminated = True
            info["liquidation"] = True

        self._last_price = new_price
        self._t += 1

        if self._t >= self.episode_len:
            truncated = True

        obs = self._sample_state()
        return obs, float(reward), terminated, truncated, info

    # ----- helpers -----

    def _latent_trend(self) -> float:
        # Slow mean-reverting latent drift in [-1,1]
        phase = (self._t % 200) / 200.0
        return float(np.sin(2 * np.pi * phase))

    def _sample_state(self) -> np.ndarray:
        # For training scaffold: random-ish but bounded feature vector.
        # In real training, you will feed recorded features instead.
        core = self.rng.normal(0.0, 0.6, size=(len(FEATURE_COLUMNS),))
        state = np.concatenate([core, [self._position]]).astype(np.float32)
        return state


class AutomatedTrainingPipeline:
    """
    Automated RL training pipeline (offline).

    For now this uses a synthetic TradingEnv. In production:
    - Replace env with one that replays real historical hybrid decisions.
    - Use real Layer 1 + DecisionNet features as observations.
    """

    def _init_(
        self,
        policy_path: str = DEFAULT_POLICY_PATH,
        total_timesteps: int = 300_000,
    ) -> None:
        self.policy_path = policy_path
        self.total_timesteps = total_timesteps

    def train(self) -> None:
        logger.info("[RL-HeadTrader] Starting PPO training (synthetic env)...")
        env = TradingEnv()
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=None,
        )
        model.learn(total_timesteps=self.total_timesteps)
        os.makedirs(os.path.dirname(self.policy_path), exist_ok=True)
        model.save(self.policy_path)
        logger.info("[RL-HeadTrader] Saved PPO policy to %s", self.policy_path)


class RLExecutionAgent:
    """
    Layer 3: RL Head Trader

    Inputs:
      - Layer2Decision (DecisionNet output):
          direction: "up" / "down" / "flat"
          confidence: 0..1
          model_votes: {
            "tft_visionary": float,
            "tcn_reflex": float,
            "xgb_analyst": float,
            "options_psychologist": float,
            "macro_economist": float,
            "llm_narrative": float,
          }

    Behavior:
      - If trained PPO policy is available:
          uses it to map features -> optimal discrete action.
      - Else:
          falls back to a transparent rule-based policy.

    Output:
      dict with:
        - action: "FLAT" | "LONG_50" | "LONG_100" | "SHORT_50" | "SHORT_100"
        - target_position: float in [-1,1]
        - mode: "rl" or "rule_fallback"
    """

    def _init_(
        self,
        policy_path: str = DEFAULT_POLICY_PATH,
    ) -> None:
        self.policy_path = policy_path
        self.feature_columns = FEATURE_COLUMNS
        self.num_features = len(self.feature_columns)
        self._model: Optional[PPO] = None

        if os.path.exists(self.policy_path):
            try:
                self._model = PPO.load(self.policy_path)
                logger.info(
                    "[RL-HeadTrader] Loaded PPO policy from %s",
                    self.policy_path,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[RL-HeadTrader] Failed to load PPO policy (%s). "
                    "Falling back to rules.",
                    e,
                )
                self._model = None
        else:
            logger.warning(
                "[RL-HeadTrader] PPO policy not found at %s. Using rule-based fallback.",
                self.policy_path,
            )

    # ---------- PUBLIC API ----------

    def choose_action(
        self,
        decision: Layer2Decision,
    ) -> Dict[str, Any]:
        """
        Main entry for Layer 3.

        decision is the fused signal from DecisionNet (Layer 2).
        Returns a dict describing the chosen position.
        """
        obs = self._format_state(decision)

        if obs is None:
            # Bad input; safest is always FLAT.
            return {
                "action": "FLAT",
                "target_position": 0.0,
                "mode": "invalid_input",
            }

        if self._model is not None:
            return self._rl_action(obs, decision)
        else:
            return self._rule_based_action(decision)

    # ---------- INTERNALS ----------

    def _format_state(self, decision: Layer2Decision) -> Optional[np.ndarray]:
        try:
            dir_map = {"up": 1.0, "down": -1.0, "flat": 0.0}
            direction_num = dir_map.get(decision.direction.lower(), 0.0)
            conf = float(max(0.0, min(1.0, decision.confidence)))

            # model_votes keys must match these:
            mv = decision.model_votes or {}
            obs_values: List[float] = [
                conf,  # decisionnet_confidence
                direction_num,  # decisionnet_direction
                float(mv.get("tft_visionary", 0.0)),
                float(mv.get("tcn_reflex", 0.0)),
                float(mv.get("xgb_analyst", 0.0)),
                float(mv.get("options_psychologist", 0.0)),
                float(mv.get("macro_economist", 0.0)),
                float(mv.get("llm_narrative", 0.0)),
            ]

            if len(obs_values) != self.num_features:
                logger.warning(
                    "[RL-HeadTrader] Feature length mismatch. "
                    "Expected %d, got %d.",
                    self.num_features,
                    len(obs_values),
                )
                return None

            # append current_position for policy input (0 here; real impl can track)
            current_position = 0.0
            obs = np.array(
                obs_values + [current_position],
                dtype=np.float32,
            )
            return obs
        except Exception as e:  # noqa: BLE001
            logger.warning("[RL-HeadTrader] Failed to format state: %s", e)
            return None

    def _rl_action(
        self,
        obs: np.ndarray,
        decision: Layer2Decision,
    ) -> Dict[str, Any]:
        try:
            action_id, _ = self._model.predict(obs, deterministic=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[RL-HeadTrader] PPO predict failed (%s). "
                "Falling back to rule-based.",
                e,
            )
            return self._rule_based_action(decision)

        # Map PPO discrete action to a target position / label.
        # You can refine this mapping if you extend the action space.
        mapping = {
            0: ("FLAT", 0.0),
            1: ("LONG_50", 0.5),
            2: ("SHORT_50", -0.5),
        }
        label, pos = mapping.get(int(action_id), ("FLAT", 0.0))

        return {
            "action": label,
            "target_position": float(pos),
            "mode": "rl",
        }

    def _rule_based_action(self, decision: Layer2Decision) -> Dict[str, Any]:
        """
        Transparent fallback: behaves like a risk-aware head trader.

        - High confidence & strong alignment -> larger positions.
        - Medium -> smaller.
        - Low or conflicting -> flat.
        """
        d = (decision.direction or "flat").lower()
        c = max(0.0, min(1.0, float(decision.confidence)))

        if d == "up":
            if c >= 0.8:
                return {
                    "action": "LONG_100",
                    "target_position": 1.0,
                    "mode": "rule_fallback",
                }
            if c >= 0.6:
                return {
                    "action": "LONG_50",
                    "target_position": 0.5,
                    "mode": "rule_fallback",
                }
        elif d == "down":
            if c >= 0.8:
                return {
                    "action": "SHORT_100",
                    "target_position": -1.0,
                    "mode": "rule_fallback",
                }
            if c >= 0.6:
                return {
                    "action": "SHORT_50",
                    "target_position": -0.5,
                    "mode": "rule_fallback",
                }

        # Anything else → stand aside
        return {
            "action": "FLAT",
            "target_position": 0.0,
            "mode": "rule_fallback",
        }