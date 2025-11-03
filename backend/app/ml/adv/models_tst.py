import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding, as recommended in the audit[cite: 203, 235].
    This injects information about the relative or absolute position of the tokens.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1) # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe) # Register as buffer so it's not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)] # Add positional encoding
        return self.dropout(x)


class TSTLite(nn.Module):
    """
    Transformer encoder for time series regression, upgraded with:
    1. Positional Encoding (as recommended in the audit [cite: 201, 203, 235]).
    2. Multi-task heads for price and volatility (as discussed in the plan [cite: 286-287]).
    """
    def __init__(self, in_feat:int, seq_len:int, d_model:int=128, nhead:int=4, num_layers:int=3, dropout:float=0.1):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(in_feat, d_model)
        
        # 1. Add Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len + 1)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        # --- Multi-Task Heads ---
        # As per the plan to forecast both price and volatility [cite: 286]
        self.head_norm = nn.LayerNorm(d_model)
        
        # Head 1: Price Prediction
        self.price_head = nn.Linear(d_model, 1)
        
        # Head 2: Volatility Prediction
        self.vol_head = nn.Linear(d_model, 1)
        # ------------------------

    def forward(self, x):  # [B, L, F]
        # 1. Project input features to d_model and scale
        x = self.in_proj(x) * math.sqrt(self.d_model) # [B, L, d_model]
        
        # 2. Add Positional Encoding (The critical fix [cite: 201, 235])
        x = self.pos_encoder(x)
        
        # 3. Pass through Transformer Encoder
        x = self.encoder(x) # [B, L, d_model]
        
        # 4. Get output from the last time step
        x = x[:, -1] # [B, d_model]
        x = self.head_norm(x)
        
        # 5. Generate multi-task predictions
        price_pred = self.price_head(x).squeeze(-1)  # [B]
        
        # Use a softplus activation for volatility to ensure it's non-negative
        vol_pred = torch.nn.functional.softplus(self.vol_head(x).squeeze(-1)) # [B]
        
        # Return as a dictionary to be explicit
        return {
            'price': price_pred,
            'vol': vol_pred
        }