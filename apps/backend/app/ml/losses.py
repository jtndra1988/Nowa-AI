import torch
import torch.nn.functional as F


def _calculate_price_loss(pred, target, alpha=0.5, beta=0.2, lam=0.1):
    """Price loss:
    Loss = alpha*MSE + beta*directional + lam*(1 - SharpeProxy)
    - directional: penalize wrong sign
    - SharpeProxy: mean(pred*target)/std(pred*target)
    """
    mse = F.mse_loss(pred, target)
    # 0 if same sign, 1 if opposite
    dir_term = (0.5 * (1 - torch.sign(pred * target))).mean()

    pnl = pred * target

    # Add epsilon to std to prevent division by zero
    pnl_std = pnl.std(unbiased=False) + 1e-6
    sharpe = pnl.mean() / pnl_std

    # Handle potential NaNs early in training if std is zero
    if torch.isnan(sharpe):
        sharpe = torch.tensor(0.0, device=pred.device)

    return alpha * mse + beta * dir_term + lam * (1 - sharpe)


def _calculate_vol_loss(pred_vol, target_vol):
    """Simple MSE loss for the volatility prediction."""
    return F.mse_loss(pred_vol, target_vol)


def multitask_transformer_loss(pred_dict, target_dict, price_loss_params=None, vol_weight=0.2):
    """Combined loss function for the multi-task Transformer.

    Expects:
        pred_dict   = {"price": Tensor, "vol": Tensor}
        target_dict = {"price": Tensor, "vol": Tensor}

    price_loss_params can override alpha/beta/lam if provided.
    """
    if price_loss_params is None:
        price_loss_params = {}

    # 1. Price loss
    price_pred = pred_dict["price"]
    price_target = target_dict["price"]

    price_loss = _calculate_price_loss(
        price_pred,
        price_target,
        alpha=price_loss_params.get("alpha", 0.5),
        beta=price_loss_params.get("beta", 0.2),
        lam=price_loss_params.get("lam", 0.1),
    )

    # 2. Volatility loss
    vol_pred = pred_dict["vol"]
    vol_target = target_dict["vol"]

    vol_loss = _calculate_vol_loss(vol_pred, vol_target)

    # 3. Combine losses with a tunable weight for the volatility task
    total_loss = price_loss + (vol_weight * vol_loss)

    # Return all components for easier logging during training
    return {
        "total_loss": total_loss,
        "price_loss": price_loss,
        "vol_loss": vol_loss,
    }
