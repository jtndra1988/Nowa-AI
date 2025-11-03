import torch
import torch.nn as nn
from typing import Dict

class Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp
    def forward(self, x):  # x: [B,C,L]
        return x[:, :, :-self.chomp].contiguous()

def tcn_block(in_ch, out_ch, k=3, d=1, p=0.0):
    pad = (k-1)*d
    return nn.Sequential(
        nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d)),
        Chomp1d(pad),
        nn.ReLU(),
        nn.Dropout(p),
        nn.utils.weight_norm(nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d)),
        Chomp1d(pad),
        nn.ReLU(),
        nn.Dropout(p),
    )

class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN)
    
    Updated to be compatible with our new multi-task, multi-input pipeline:
    1.  Accepts a dictionary of feature blocks (x_blocks) [cite: 31-33].
    2.  Outputs a dictionary of multi-task predictions (price and vol) .
    """
    def __init__(self, in_feat: int, channels=(64,128,128), kernel=3, dropout=0.1):
        super().__init__()
        layers = []
        prev = in_feat
        for i, ch in enumerate(channels):
            dilation = 2**i
            block = tcn_block(prev, ch, k=kernel, d=dilation, p=dropout)
            down = nn.Conv1d(prev, ch, 1) if prev != ch else nn.Identity()
            layers.append(nn.Sequential(block, down))
            prev = ch
        
        # --- 1. TCN Body ---
        self.net = nn.ModuleList(layers)
        
        # --- 2. Multi-Task Heads ---
        self.head_pool = nn.AdaptiveAvgPool1d(1)
        self.head_flatten = nn.Flatten()
        
        # Head 1: Price Prediction
        self.price_head = nn.Linear(prev, 1)
        
        # Head 2: Volatility Prediction
        self.vol_head = nn.Linear(prev, 1)
        # ------------------------

    def forward(self, x_blocks: Dict[str, torch.Tensor]):
        """
        Input 'x_blocks' is a dictionary of tensors, e.g.:
        {
            "price": tensor[B, L, 5],
            "ob": tensor[B, L, 4],
            ...
        }
        """
        
        # --- 1. Combine feature blocks ---
        # Concatenate all blocks to create the [B, L, F] tensor
        x = torch.cat(list(x_blocks.values()), dim=-1)

        # --- 2. TCN Body ---
        x = x.transpose(1,2)  # -> [B, F, L]
        for blk in self.net:
            res = x
            h = blk[0](x)
            x = h + blk[1](res)
        
        # --- 3. Pooling & Flatten ---
        x = self.head_pool(x) # [B, C, 1]
        x = self.head_flatten(x) # [B, C]
        
        # --- 4. Generate multi-task predictions ---
        price_pred = self.price_head(x).squeeze(-1) # [B]
        
        # Use softplus for non-negative volatility
        vol_pred = torch.nn.functional.softplus(self.vol_head(x).squeeze(-1)) # [B]
        
        # Return as a dictionary to match our loss function
        return {
            'price': price_pred,
            'vol': vol_pred
        }