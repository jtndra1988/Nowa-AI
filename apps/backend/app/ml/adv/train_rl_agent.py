# app/ml/adv/train_rl_agent.py

import logging
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import func

from app.db.database import SessionLocal
from app.db import models
from app.core.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ARTIFACTS_DIR = Path(getattr(settings, "MODEL_ARTIFACTS_DIR", "model_artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_PATH = ARTIFACTS_DIR / "rl_agent_ppo.zip"


# Try to import stable-baselines3 + gym
try:
    import gym
    from gym import spaces
    from stable_baselines3 import PPO
except Exception as e:  # noqa: BLE001
    gym = None
    PPO = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_market_data(lookback_days: int = 60) -> pd.DataFrame:
    """
    Load recent hourly OHLCV from MarketData for a few symbols.
    """
    session = SessionLocal()
    try:
        if not hasattr(models, "MarketData"):
            raise RuntimeError("models.MarketData not found; adjust this script.")

        MD = getattr(models, "MarketData")

        last_ts = session.query(func.max(MD.timestamp)).scalar()
        if not last_ts:
            raise RuntimeError("[RL] No MarketData available in DB")

        start_ts = last_ts - timedelta(days=lookback_days)

        rows = (
            session.query(MD)
            .filter(MD.timestamp >= start_ts)
            .order_by(MD.symbol, MD.timestamp)
            .all()
        )
        if not rows:
            raise RuntimeError("[RL] No rows found in MarketData for given window")

        data = [
            {
                "symbol": r.symbol,
                "timestamp": r.timestamp,
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            }
            for r in rows
        ]

        df = pd.DataFrame(data).sort_values(["symbol", "timestamp"]).reset_index(
            drop=True
        )
        logger.info(
            "[RL] Loaded %d OHLCV rows for %d symbols",
            len(df),
            df["symbol"].nunique(),
        )
        return df
    finally:
        session.close()


def _build_price_series(df: pd.DataFrame, min_len: int = 200) -> List[pd.DataFrame]:
    """
    Split MarketData into per-symbol time series, keeping only reasonably long ones.
    """
    series: List[pd.DataFrame] = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) >= min_len:
            series.append(g)
            logger.info("[RL] Using symbol %s with %d rows", sym, len(g))
    if not series:
        raise RuntimeError(
            "[RL] No symbol has enough data for RL. "
            f"Need at least {min_len} rows per series."
        )
    return series


# ---------------------------------------------------------------------------
# Trading environment
# ---------------------------------------------------------------------------

if gym is not None:

    class SimpleTradingEnv(gym.Env):
        """
        Very simple 1D trading env:
          - Observes basic features of price/volume
          - Action: -1 (short), 0 (flat), 1 (long)
          - Reward: position * next_return - small transaction cost when changing pos
        """

        metadata = {"render.modes": ["human"]}

        def __init__(
            self,
            series_list: List[pd.DataFrame],
            transaction_cost: float = 0.0002,
        ):
            super().__init__()
            self.series_list = series_list
            self.transaction_cost = transaction_cost

            # Observation: [normalized_close, normalized_vol, last_ret]
            self.observation_space = spaces.Box(
                low=-5.0,
                high=5.0,
                shape=(3,),
                dtype=np.float32,
            )
            # Action: -1, 0, 1
            self.action_space = spaces.Discrete(3)

            self.curr_series: pd.DataFrame | None = None
            self.idx: int = 0
            self.position: int = 0  # -1 short, 0 flat, 1 long

        def reset(self, *, seed=None, options=None):  # type: ignore[override]
            super().reset(seed=seed)

            # Randomly pick a symbol series each episode
            self.curr_series = np.random.choice(self.series_list)
            self.idx = 1  # start from second bar so we can compute return
            self.position = 0

            obs = self._get_obs()
            return obs, {}

        def _get_obs(self) -> np.ndarray:
            assert self.curr_series is not None
            row = self.curr_series.iloc[self.idx]

            # Normalize by last 100 bars
            window = self.curr_series.iloc[max(0, self.idx - 100): self.idx + 1]
            close = row["close"]
            vol = row["volume"]
            close_norm = (
                (close - window["close"].mean()) / (window["close"].std() + 1e-6)
            )
            vol_norm = (vol - window["volume"].mean()) / (window["volume"].std() + 1e-6)
            last_ret = window["close"].pct_change().iloc[-1]

            return np.array(
                [close_norm, vol_norm, last_ret],
                dtype=np.float32,
            )

        def step(self, action):  # type: ignore[override]
            assert self.curr_series is not None

            # Map discrete action to position
            new_position = action - 1  # 0->-1, 1->0, 2->1

            # Compute reward based on next bar return
            done = False
            if self.idx >= len(self.curr_series) - 2:
                done = True

            row_now = self.curr_series.iloc[self.idx]
            row_next = self.curr_series.iloc[self.idx + 1]
            ret = row_next["close"] / row_now["close"] - 1.0

            reward = new_position * ret

            # Transaction cost if position changed
            if new_position != self.position:
                reward -= self.transaction_cost

            self.position = new_position
            self.idx += 1

            obs = self._get_obs() if not done else np.zeros(3, dtype=np.float32)
            info = {"ret": float(ret), "position": int(self.position)}
            return obs, float(reward), done, False, info

        def render(self, mode="human"):
            return


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def _train_ppo_agent(df: pd.DataFrame):
    if PPO is None or gym is None:
        raise RuntimeError(
            f"[RL] stable-baselines3/gym import failed: {_IMPORT_ERROR}. "
            "Install them inside the docker image."
        )

    series_list = _build_price_series(df, min_len=200)
    env = SimpleTradingEnv(series_list)

    logger.info("[RL] Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        ent_coef=0.01,
        clip_range=0.2,
        tensorboard_log=None,
    )

    total_timesteps = 50_000  # adjust up when you have more data/compute
    logger.info("[RL] Starting PPO training for %d timesteps...", total_timesteps)
    model.learn(total_timesteps=total_timesteps)

    logger.info("[RL] Saving PPO policy to %s", ARTIFACT_PATH)
    model.save(str(ARTIFACT_PATH))


def main():
    logger.info("[RL] ==== Training RL execution agent (PPO) ====")
    try:
        df = _load_market_data(lookback_days=90)
        _train_ppo_agent(df)
        logger.info("[RL] RL agent training complete.")
    except Exception as e:  # noqa: BLE001
        logger.error("[RL] Training failed: %s", e)
        # Re-raise to make Celery task fail visibly
        raise


if __name__ == "__main__":
    main()
