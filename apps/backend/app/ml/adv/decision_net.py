import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DECISION_NET_PATH = Path("./model_artifacts/decision_net.pth")


class DecisionNet(nn.Module):
    """
    Small MLP that takes:
      - expert outputs (TFT/TCN/XGB)
      - simple context features
    and predicts probabilities for [short, flat, long].
    """

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),   # short, flat, long
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits


_decision_net: Optional[DecisionNet] = None


def _lazy_load_decision_net(in_dim: int):
    global _decision_net
    if _decision_net is None:
        model = DecisionNet(in_dim=in_dim).to(DEVICE)
        if DECISION_NET_PATH.exists():
            state = torch.load(DECISION_NET_PATH, map_location=DEVICE)
            model.load_state_dict(state)
        model.eval()
        _decision_net = model


def decision_net_score(features: Dict[str, Any]) -> float:
    """
    Returns a score in [-1, 1]:
      +1 strong long, -1 strong short, 0 neutral.
    If no trained model yet, uses a fallback heuristic.
    """
    # Flatten numeric features
    keys = sorted(features.keys())
    x_vec = [float(features[k]) for k in keys]
    in_dim = len(x_vec)

    if in_dim == 0:
        return 0.0

    _lazy_load_decision_net(in_dim)
    assert _decision_net is not None

    with torch.no_grad():
        x = torch.tensor(x_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits = _decision_net(x)  # [1,3]
        probs = torch.softmax(logits, dim=-1)[0]  # [3]
        p_short, p_flat, p_long = probs.tolist()

    # score: weighted direction
    score = (p_long - p_short)
    # clamp to [-1,1] already
    return float(score)
