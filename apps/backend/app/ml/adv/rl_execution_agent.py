# app/ml/adv/rl_execution_agent.py

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO

from app.hybrid.schemas import Layer2Prediction, MarketContext, RLAction

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: List[str] = [
    "decisionnet_confidence",
    "decisionnet_direction",  # -1, 0, 1
    "tft_visionary",
    "tcn_reflex",
    "xgb_analyst",
    "xgb_vol_analyst",
    "options_psychologist",
    "macro_economist",
    "llm_narrative",
    "current_position_size",
]

DEFAULT_POLICY_PATH = os.getenv(
    "NOWA_RL_POLICY_PATH",
    "model_artifacts/rl_agent_ppo.zip",
)


class RLAgent:
    """
    Layer 3: LIVE RL Execution Agent ("Head Trader").

    - Loads a trained PPO policy.
    - NO mock / rule fallback here.
    - If policy not loaded: is_model_loaded() = False and get_optimal_action() raises.
    """

    def __init__(self, policy_path: str = DEFAULT_POLICY_PATH) -> None:
        self.policy_path = Path(policy_path)
        self.feature_columns = FEATURE_COLUMNS
        self.policy: Optional[PPO] = self._load_policy()

        if self.policy is None:
            logger.error(
                f"[RLAgent] Failed to load RL policy from '{self.policy_path}'. "
                "RL agent is DISABLED."
            )
        else:
            logger.info(
                f"[RLAgent] RL policy loaded successfully from '{self.policy_path}'."
            )

    # -------- Internal: load policy --------

    def _load_policy(self) -> Optional[PPO]:
        if not self.policy_path.exists():
            logger.error(
                f"[RLAgent] Policy file not found at '{self.policy_path}'."
            )
            return None
        try:
            policy = PPO.load(self.policy_path, device="cpu")  # type: ignore[arg-type]
            return policy
        except Exception as e:
            logger.error(f"[RLAgent] Error loading PPO policy: {e}", exc_info=True)
            return None

    # -------- Public API used by InferenceService --------

    def is_model_loaded(self) -> bool:
        """True ONLY when PPO policy is successfully loaded."""
        return self.policy is not None

    def get_mode(self) -> str:
        """For observability."""
        return "rl_live" if self.policy is not None else "disabled"

    def get_optimal_action(
        self,
        prediction: Layer2Prediction,
        context: MarketContext,
        model_votes: Dict[str, float],
    ) -> RLAction:
        """
        Main entry called by InferenceService.
        """
        if not self.is_model_loaded():
            raise RuntimeError(
                "[RLAgent] get_optimal_action() called but RL policy is not loaded."
            )

        state = self._build_state(prediction, context, model_votes)
        action_idx, _ = self.policy.predict(state, deterministic=True)  # type: ignore[union-attr]
        rl_action = self._map_action(int(action_idx))
        # Ensure mode is set for downstream observability
        rl_action.mode = self.get_mode()
        return rl_action

    # -------- Internal helpers --------

    def _direction_to_int(self, direction: str) -> int:
        d = (direction or "flat").lower()
        if d == "up":
            return 1
        if d == "down":
            return -1
        return 0

    def _build_state(
        self,
        prediction: Layer2Prediction,
        context: MarketContext,
        model_votes: Dict[str, float],
    ) -> np.ndarray:
        obs: Dict[str, Any] = {
            "decisionnet_confidence": float(
                getattr(prediction, "price_confidence", 0.0) or 0.0
            ),
            "decisionnet_direction": self._direction_to_int(
                getattr(prediction, "direction", "flat")
            ),
            "tft_visionary": float(model_votes.get("tft_visionary", 0.0)),
            "tcn_reflex": float(model_votes.get("tcn_reflex", 0.0)),
            "xgb_analyst": float(model_votes.get("xgb_analyst", 0.0)),
            "xgb_vol_analyst": float(model_votes.get("xgb_vol_analyst", 0.0)),
            "options_psychologist": float(
                model_votes.get("options_psychologist", 0.0)
            ),
            "macro_economist": float(model_votes.get("macro_economist", 0.0)),
            "llm_narrative": float(model_votes.get("llm_narrative", 0.0)),
            "current_position_size": float(
                getattr(context, "current_position_size", 0.0) or 0.0
            ),
        }

        vec = np.array(
            [float(obs[c]) for c in self.feature_columns], dtype=np.float32
        )
        return vec

    def _map_action(self, action_idx: int) -> RLAction:
        """
        Map discrete PPO action index -> RLAction.

        This MUST match your training setup.
        """
        if action_idx == 0:
            return RLAction(
                optimal_action="FLAT",
                optimal_size_pct=0.0,
                execution_style="NONE",
            )
        elif action_idx == 1:
            return RLAction(
                optimal_action="LONG",
                optimal_size_pct=0.25,
                execution_style="TWAP_5M",
            )
        elif action_idx == 2:
            return RLAction(
                optimal_action="LONG",
                optimal_size_pct=1.0,
                execution_style="TWAP_15M",
            )
        elif action_idx == 3:
            return RLAction(
                optimal_action="SHORT",
                optimal_size_pct=0.25,
                execution_style="TWAP_5M",
            )
        elif action_idx == 4:
            return RLAction(
                optimal_action="SHORT",
                optimal_size_pct=1.0,
                execution_style="TWAP_15M",
            )

        logger.error(f"[RLAgent] Unknown action_idx={action_idx}, forcing FLAT.")
        return RLAction(
            optimal_action="FLAT",
            optimal_size_pct=0.0,
            execution_style="NONE",
        )


# -------- Singleton for InferenceService --------

try:
    rl_agent = RLAgent(policy_path=DEFAULT_POLICY_PATH)
except Exception as e:
    logger.critical(
        f"[RLAgent] Failed to initialize rl_agent: {e}", exc_info=True
    )
    raise
