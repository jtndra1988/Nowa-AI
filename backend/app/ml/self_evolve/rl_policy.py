from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class Step:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class SimpleTradingEnv:
    """Tiny offline env using historical candles and (optionally) features.
    Actions: 0=flat, 1=long, 2=short; reward = position * return - penalty.
    """
    def __init__(self, prices: np.ndarray, features: np.ndarray | None = None, fee_bps: float = 2.0):
        self.p = prices.astype(np.float32)
        self.f = features.astype(np.float32) if features is not None else None
        self.t = 0
        self.pos = 0  # -1,0,1
        self.fee = fee_bps / 1e4

    def reset(self) -> np.ndarray:
        self.t = 0; self.pos = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        obs = [self.p[self.t]]
        if self.f is not None:
            obs.extend(self.f[self.t].tolist())
        obs.append(self.pos)
        return np.asarray(obs, dtype=np.float32)

    def step(self, action: int) -> Step:
        # fee when changing position
        fee = 0.0
        if action != self.pos:
            fee = self.fee
        ret = (self.p[min(self.t+1, len(self.p)-1)] - self.p[self.t]) / (self.p[self.t] + 1e-9)
        reward = (1 if action==1 else (-1 if action==2 else 0)) * ret - fee
        self.pos = {0:0, 1:1, 2:-1}[action]
        self.t += 1
        done = self.t >= len(self.p)-1
        return Step(self._obs(), action, float(reward), self._obs(), done)

# Lightweight policy gradient (REINFORCE)
import torch, torch.nn as nn

class PGPolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64, n_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, x):
        return self.net(x)


def train_pg(prices: np.ndarray, features: np.ndarray | None = None, epochs: int = 10, lr: float = 1e-3) -> PGPolicy:
    env = SimpleTradingEnv(prices, features)
    obs0 = env.reset()
    obs_dim = obs0.shape[0]
    policy = PGPolicy(obs_dim)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(epochs):
        logps = []; rewards = []
        obs = env.reset()
        done = False
        while not done:
            ob = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = policy(ob)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action = int(dist.sample().item())
            step = env.step(action)
            logps.append(dist.log_prob(torch.tensor(action)))
            rewards.append(step.reward)
            obs = step.next_state
            done = step.done
        # REINFORCE: baseline = mean reward
        R = torch.tensor(rewards, dtype=torch.float32)
        advantage = R - R.mean()
        loss = -(torch.stack(logps) * advantage).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return policy

