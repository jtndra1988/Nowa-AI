import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List

# Device + default runtime artifact
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DECISION_NET_PATH = Path("./model_artifacts/decision_net.pth")

# Canonical input routing from Layer 1 / experts:
# These keys define BOTH training and inference ordering.
DECISION_NET_INPUT_KEYS: List[str] = [
    "tft_vote",         # TFT (Visionary) price signal
    "tcn_vote",         # TCN (Reflex) price signal
    "tst_vote",         # TST (Transformer) price signal
    "xgb_price_vote",   # XGB tabular analyst (return)
    "xgb_vol_vote",     # XGB (volatility proxy or 0.0)
    "options_score",    # Options expert (Gamma/Vanna/Charm edge)
    "macro_score",      # Macro + on-chain expert
    "llm_narrative",    # LLM narrative sentiment score
]


class DecisionNet(nn.Module):
    """
    Small MLP that takes expert-level signals from Layer 1 and
    predicts logits for [short, flat, long].

    Input vector order is defined by DECISION_NET_INPUT_KEYS.
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
        """
        Args:
            x: Tensor of shape [batch_size, in_dim]
        Returns:
            logits: Tensor of shape [batch_size, 3]
        """
        return self.net(x)


_decision_net: Optional[DecisionNet] = None


def _lazy_load_decision_net():
    """
    Lazy-load DecisionNet using the fixed input dimension derived from
    DECISION_NET_INPUT_KEYS, and load state_dict from DECISION_NET_PATH
    if it exists.
    """
    global _decision_net
    if _decision_net is None:
        in_dim = len(DECISION_NET_INPUT_KEYS)
        model = DecisionNet(in_dim=in_dim).to(DEVICE)
        if DECISION_NET_PATH.exists():
            state = torch.load(DECISION_NET_PATH, map_location=DEVICE)
            model.load_state_dict(state)
        model.eval()
        _decision_net = model


def build_decision_feature_vector(expert_inputs: Dict[str, Any]) -> torch.Tensor:
    """
    Build a 1D feature vector (in the canonical order) from Layer-1 / expert outputs.

    expert_inputs is expected to contain (when available):
        {
            "tft_vote": float,
            "tcn_vote": float,
            "tst_vote": float,
            "xgb_price_vote": float,
            "xgb_vol_vote": float,
            "options_score": float,
            "macro_score": float,
            "llm_narrative": float,
        }

    Missing keys are filled with 0.0, and non-numeric values are safely coerced.

    Returns:
        torch.Tensor of shape [in_dim] on CPU (caller can move to device).
    """
    vec = []
    for key in DECISION_NET_INPUT_KEYS:
        v = expert_inputs.get(key, 0.0)
        try:
            vec.append(float(v))
        except Exception:
            vec.append(0.0)
    return torch.tensor(vec, dtype=torch.float32)


def decision_net_score(expert_inputs: Dict[str, Any]) -> float:
    """
    High-level scoring API used by ModelEngine / HybridInferenceService.

    Input:
        expert_inputs: dict of signals from TFT/TCN/TST/XGB/etc. keyed by
                       DECISION_NET_INPUT_KEYS (missing keys are 0.0).

    Output:
        score in [-1, 1]:
            +1  = strong long bias
            -1  = strong short bias
             0  = neutral / flat

    If no trained model is available yet, we fall back to a neutral score (0.0).
    """
    # Build canonical feature vector
    x_vec = build_decision_feature_vector(expert_inputs)
    in_dim = x_vec.shape[0]

    if in_dim == 0:
        return 0.0

    _lazy_load_decision_net()
    assert _decision_net is not None

    with torch.no_grad():
        x = x_vec.to(DEVICE).unsqueeze(0)  # [1, in_dim]
        logits = _decision_net(x)          # [1, 3]
        probs = torch.softmax(logits, dim=-1)[0]  # [3]
        p_short, p_flat, p_long = probs.tolist()

    # Directional score: long - short (clamped in [-1, 1] by construction)
    score = p_long - p_short
    return float(score)
