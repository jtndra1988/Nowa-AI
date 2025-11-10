import torch
import torch.nn as nn
from typing import Dict


class Chomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        if self.chomp == 0:
            return x
        return x[:, :, :-self.chomp].contiguous()


def tcn_block(in_ch: int, out_ch: int, k: int = 3, d: int = 1, p: float = 0.0) -> nn.Sequential:
    pad = (k - 1) * d
    return nn.Sequential(
        nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d)
        ),
        Chomp1d(pad),
        nn.ReLU(),
        nn.Dropout(p),
        nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d)
        ),
        Chomp1d(pad),
        nn.ReLU(),
        nn.Dropout(p),
    )


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN)

    Updated to be compatible with our multi-task, multi-input pipeline:
      1. Accepts a dictionary of feature blocks (x_blocks).
      2. Outputs {'price': ..., 'vol': ...}.
    """

    def __init__(
        self,
        in_feat: int,
        channels=(64, 128, 128),
        kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev = in_feat
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            block = tcn_block(prev, ch, k=kernel, d=dilation, p=dropout)
            down = nn.Conv1d(prev, ch, 1) if prev != ch else nn.Identity()
            layers.append(nn.Sequential(block, down))
            prev = ch

        # TCN body
        self.net = nn.ModuleList(layers)

        # Heads
        self.head_pool = nn.AdaptiveAvgPool1d(1)
        self.head_flatten = nn.Flatten()
        self.price_head = nn.Linear(prev, 1)
        self.vol_head = nn.Linear(prev, 1)

    def forward(self, x_blocks: Dict[str, torch.Tensor]):
        """
        x_blocks:
          {
            "price": [B, L, F_p],
            "ob":    [B, L, F_o],
            ...
          }
        """
        # concat blocks -> [B, L, F_total]
        x = torch.cat(list(x_blocks.values()), dim=-1)
        x = x.transpose(1, 2)  # [B, F_total, L]

        # residual TCN stack
        for blk in self.net:
            res = x
            h = blk[0](x)
            x = h + blk[1](res)

        x = self.head_pool(x)          # [B, C, 1]
        x = self.head_flatten(x)       # [B, C]

        price_pred = self.price_head(x).squeeze(-1)
        vol_pred = torch.nn.functional.softplus(self.vol_head(x).squeeze(-1))

        return {
            "price": price_pred,
            "vol": vol_pred,
        }
