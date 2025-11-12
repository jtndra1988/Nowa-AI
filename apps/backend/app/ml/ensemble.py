import torch
import torch.nn as nn

class StackingEnsemble(torch.nn.Module):
    """
    This is our simple, trainable blender.
    It learns a weighted average of pre-computed predictions.
    
    We will create two of these: one for price, one for volatility.
    'n_models' will be 3 (for TFT, TCN, and XGBoost).
    """
    def __init__(self, n_models: int):
        super().__init__()
        # Initialize with equal weights
        self.w = torch.nn.Parameter(torch.ones(n_models) / n_models)

    def forward(self, preds: torch.Tensor):  # preds [B, n_models]
        # Apply softmax to the weights to ensure they are
        # positive and sum to 1 (a convex blend)
        w = torch.softmax(self.w, dim=0)
        
        # Return the weighted sum
        return (preds * w).sum(dim=1)