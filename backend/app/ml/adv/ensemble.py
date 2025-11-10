import torch
import torch.nn as nn


class StackingEnsemble(nn.Module):
    """
    Simple MLP ensemble:
      input: [B, N_models]
      output: [B, 1]
    """
    def __init__(self, n_models: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_models, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N_models]
        return self.net(x).squeeze(-1)
