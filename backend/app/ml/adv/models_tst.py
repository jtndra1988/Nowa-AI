import torch
import torch.nn as nn

class TSTLite(nn.Module):
    """
    Simple Transformer encoder for time series regression.
    """
    def __init__(self, in_feat:int, d_model:int=128, nhead:int=4, num_layers:int=3, dropout:float=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_feat, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):  # [B,L,F]
        x = self.in_proj(x)
        x = self.encoder(x)
        x = x[:, -1]              # last token
        return self.head(x).squeeze(-1)
