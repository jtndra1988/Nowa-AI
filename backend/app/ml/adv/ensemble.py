import torch, numpy as np

class StackingEnsemble(torch.nn.Module):
    """
    Takes two (or more) model outputs and learns a convex blend on validation.
    You can load your existing LSTM head and blend with TCN/TST.
    """
    def __init__(self, n_models:int=2):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(n_models)/n_models)

    def forward(self, preds: torch.Tensor):  # preds [B, n_models]
        w = torch.softmax(self.w, dim=0)
        return (preds * w).sum(dim=1)
