import torch
import torch.nn.functional as F

def mse_direction_sharpe(pred, target, alpha=0.5, beta=0.2, lam=0.1):
    """
    Loss = alpha*MSE + beta*directional + lam*(1 - SharpeProxy)
    - directional: penalize wrong sign
    - SharpeProxy: mean(pred*target)/std(pred*target)
    """
    mse = F.mse_loss(pred, target)
    dir_term = (0.5*(1 - torch.sign(pred*target))).mean()  # 0 if same sign, 1 if opposite
    pnl = pred * target
    sharpe = pnl.mean() / (pnl.std(unbiased=False) + 1e-6)
    return alpha*mse + beta*dir_term + lam*(1 - sharpe)
