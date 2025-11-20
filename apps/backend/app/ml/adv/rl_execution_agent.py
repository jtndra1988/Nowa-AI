# app/ml/adv/rl_execution_agent.py

from datetime import datetime
import json
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

    - Loads a trained PPO policy (if available).
    - If RL policy is missing or inference fails, gracefully falls back
      to a conservative, rules-based strategy.
    - Logs every action (RL or fallback) with model version & key inputs.
    """

    def __init__(self, policy_path: str = DEFAULT_POLICY_PATH) -> None:
        self.policy_path = Path(policy_path)
        self.feature_columns = FEATURE_COLUMNS
        self.policy: Optional[PPO] = None

        # Policy metadata (loaded from JSON written by train_rl_agent.py)
        self.metadata_path: Path = self.policy_path.with_name(
            self.policy_path.stem + "_metadata.json"
        )
        self.policy_metadata: Dict[str, Any] = {}
        self.policy_version: str = "unknown"

        # Load policy + metadata
        self.policy = self._load_policy()
        self._load_metadata()

        if self.policy is None:
            logger.error(
                "[RLAgent] Failed to load RL policy from '%s'. RL agent will use fallback strategy only.",
                self.policy_path,
            )
        else:
            logger.info(
                "[RLAgent] RL policy loaded successfully from '%s' (version=%s).",
                self.policy_path,
                self.policy_version,
            )

    # -------- Internal: load policy & metadata --------

    def _load_policy(self) -> Optional[PPO]:
        if not self.policy_path.exists():
            logger.error(
                "[RLAgent] Policy file not found at '%s'.", self.policy_path
            )
            return None
        try:
            policy = PPO.load(self.policy_path, device="cpu")  # type: ignore[arg-type]
            return policy
        except Exception as e:
            logger.error("[RLAgent] Error loading PPO policy: %s", e, exc_info=True)
            return None

    def _load_metadata(self) -> None:
        """
        Load RL policy metadata (version, hyperparams, etc.).
        """
        if not self.metadata_path.exists():
            logger.warning(
                "[RLAgent] Metadata file not found at '%s'; version will be 'unknown'.",
                self.metadata_path,
            )
            return

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.policy_metadata = meta or {}
            self.policy_version = str(meta.get("version", "unknown"))
        except Exception as e:
            logger.warning(
                "[RLAgent] Failed to parse metadata from '%s': %s",
                self.metadata_path,
                e,
                exc_info=True,
            )

    # -------- Public API used by InferenceService --------

    def is_model_loaded(self) -> bool:
        """True ONLY when PPO policy is successfully loaded."""
        return self.policy is not None

    def get_mode(self) -> str:
        """For observability."""
        return "rl_live" if self.policy is not None else "fallback_only"

    def get_policy_version(self) -> str:
        return self.policy_version

    def get_optimal_action(
        self,
        prediction: Layer2Prediction,
        context: MarketContext,
        model_votes: Dict[str, float],
    ) -> RLAction:
        """
        Main entry called by InferenceService.

        Behavior:
          - If PPO policy is loaded and inference succeeds, use RL decision.
          - Otherwise, fall back to a conservative heuristic strategy based on
            DecisionNet direction/confidence and current position.
        """
        # Build state vector (used for RL and logging)
        state = self._build_state(prediction, context, model_votes)

        source = "fallback"
        if self.is_model_loaded():
            try:
                action_idx, _ = self.policy.predict(state, deterministic=True)  # type: ignore[union-attr]
                rl_action = self._map_action(int(action_idx))
                rl_action.mode = self.get_mode()
                source = "rl_policy"
            except Exception as e:
                logger.error(
                    "[RLAgent] PPO inference failed, switching to fallback: %s",
                    e,
                    exc_info=True,
                )
                rl_action = self._fallback_action(
                    prediction, context, model_votes, reason="policy_error"
                )
        else:
            rl_action = self._fallback_action(
                prediction, context, model_votes, reason="policy_not_loaded"
            )

        # Log decision for audit
        self._log_action(prediction, context, model_votes, state, rl_action, source)

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

        logger.error("[RLAgent] Unknown action_idx=%s, forcing FLAT.", action_idx)
        return RLAction(
            optimal_action="FLAT",
            optimal_size_pct=0.0,
            execution_style="NONE",
        )

    def _fallback_action(
        self,
        prediction: Layer2Prediction,
        context: MarketContext,
        model_votes: Dict[str, float],
        reason: str = "fallback",
    ) -> RLAction:
        """
        Conservative default if RL policy is missing or fails.

        Heuristic:
          - Use DecisionNet direction and confidence.
          - Only open SMALL positions when confidence is high.
          - Otherwise stay FLAT.
        """
        direction = (getattr(prediction, "direction", "flat") or "flat").lower()
        confidence = float(getattr(prediction, "price_confidence", 0.0) or 0.0)
        current_pos = float(
            getattr(context, "current_position_size", 0.0) or 0.0
        )

        # Thresholds can be tuned
        CONF_WEAK = 0.5
        CONF_STRONG = 0.75

        # Default: stay flat
        optimal_action = "FLAT"
        optimal_size_pct = 0.0
        execution_style = "NONE"

        if confidence >= CONF_WEAK:
            if direction == "up":
                # Small long, scaled by confidence
                optimal_action = "LONG"
                optimal_size_pct = 0.25 if confidence < CONF_STRONG else 0.5
                execution_style = "TWAP_15M"
            elif direction == "down":
                # Small short, scaled by confidence
                optimal_action = "SHORT"
                optimal_size_pct = 0.25 if confidence < CONF_STRONG else 0.5
                execution_style = "TWAP_15M"

        action = RLAction(
            optimal_action=optimal_action,
            optimal_size_pct=optimal_size_pct,
            execution_style=execution_style,
        )

        # Mark that this came from fallback logic
        mode_suffix = reason.replace(" ", "_")
        action.mode = f"fallback_{mode_suffix}"

        logger.info(
            "[RLAgent] Using fallback action=%s size=%.3f reason=%s "
            "(direction=%s, confidence=%.3f, current_pos=%.3f)",
            optimal_action,
            optimal_size_pct,
            reason,
            direction,
            confidence,
            current_pos,
        )
        return action

    def _log_action(
        self,
        prediction: Layer2Prediction,
        context: MarketContext,
        model_votes: Dict[str, float],
        state: np.ndarray,
        action: RLAction,
        source: str,
    ) -> None:
        """
        Structured audit log for each decision.
        """
        try:
            event = {
                "ts_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "event": "rl_execution_decision",
                "source": source,
                "mode": getattr(action, "mode", None),
                "policy_version": self.policy_version,
                "policy_path": str(self.policy_path),
                "symbol": getattr(context, "symbol", None),
                "timeframe": getattr(context, "timeframe", None),
                "inputs": {
                    "decisionnet_confidence": float(
                        getattr(prediction, "price_confidence", 0.0) or 0.0
                    ),
                    "decisionnet_direction": getattr(prediction, "direction", "flat"),
                    "model_votes": model_votes,
                    "current_position_size": float(
                        getattr(context, "current_position_size", 0.0) or 0.0
                    ),
                    "state_vector": state.tolist(),
                },
                "action": {
                    "optimal_action": action.optimal_action,
                    "optimal_size_pct": action.optimal_size_pct,
                    "execution_style": action.execution_style,
                },
            }
            logger.info("[RLAgent-DECISION] %s", json.dumps(event))
        except Exception as e:
            logger.warning("[RLAgent] Failed to log RL action event: %s", e, exc_info=True)


# -------- Singleton for InferenceService --------

try:
    rl_agent = RLAgent(policy_path=DEFAULT_POLICY_PATH)
except Exception as e:
    logger.critical(
        "[RLAgent] Failed to initialize rl_agent: %s", e, exc_info=True
    )
    raise
