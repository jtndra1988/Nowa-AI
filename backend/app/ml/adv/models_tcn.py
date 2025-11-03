import torch
import torch.nn as nn

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
        self.net = nn.ModuleList(layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(prev, 1))

    def forward(self, x):  # x: [B,L,F]
        x = x.transpose(1,2)  # -> [B,F,L]
        for blk in self.net:
            res = x
            h = blk[0](x)
            x = h + blk[1](res)
        return self.head(x).squeeze(-1)
