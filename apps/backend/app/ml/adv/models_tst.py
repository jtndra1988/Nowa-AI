from pathlib import Path
from typing import Dict
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

# Add this to the bottom of apps/backend/app/ml/adv/models_tst.py

from typing import Dict, Any

class TSTPredictor:
    def __init__(self, model_path="model_artifacts/tst_model.pth"):
        self.model = None
        # In real usage: 
        if Path(model_path).exists():
            self.model = torch.load(model_path)
            self.model.eval()
        pass

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def predict(self, features: Dict[str, Any]) -> float:
        """
        Runs inference for TST.
        CRITICAL: TSTLite.forward() expects a single tensor [B, L, F],
        so we must concatenate the feature blocks from the dictionary.
        """
        if not self.model:
            return 0.0
            
        self.model.eval()
        
        # Detect device
        device = next(self.model.parameters()).device
        
        tensors = []

        try:
            # 1. Process and collect all feature blocks
            for name, data in features.items():
                # Skip metadata like strings or single numbers
                if isinstance(data, (str, int, float, bool)) or data is None:
                    continue

                # Convert to Tensor
                if isinstance(data, torch.Tensor):
                    t = data.clone().detach()
                else:
                    t = torch.tensor(data, dtype=torch.float32)
                
                # Handle Dimensions: [L, F] -> [1, L, F]
                if t.ndim == 2:
                    t = t.unsqueeze(0)
                
                tensors.append(t)

            if not tensors:
                return 0.0

            # 2. Concatenate all blocks along the feature dimension (dim=2)
            # TST expects [Batch, Seq_Len, Total_Features]
            x_input = torch.cat(tensors, dim=-1).to(device)

            # 3. Run inference
            with torch.no_grad():
                prediction = self.model(x_input)
            
            # 4. Extract price prediction
            price_val = prediction.get("price")
            
            if price_val is not None:
                return float(price_val.item())
            
            return 0.0

        except Exception as e:
            print(f"[TSTPredictor] Prediction error: {e}")
            return 0.0


import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    """
    Time Series Transformer (TST) for sequence-to-sequence forecasting.
    Input: tensor of shape (batch_size, context_length, num_features)
    Output: tensor of shape (batch_size, prediction_length, target_size)
    Versioning and artifact loading supported via `load_from_artifact`.
    """

    def __init__(
        self,
        context_length: int,
        num_features: int,
        prediction_length: int,
        target_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super(TimeSeriesTransformer, self).__init__()
        # --- architecture definitions ---
        self.context_length = context_length
        self.num_features = num_features
        self.prediction_length = prediction_length
        self.target_size = target_size
        self.d_model = d_model

        # Input projection
        self.in_proj = nn.Linear(num_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=context_length + prediction_length)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection for sequence output
        self.out_proj = nn.Linear(d_model, target_size * prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, context_length, num_features)
        Returns:
            Tensor of shape (batch_size, prediction_length, target_size)
        """
        # 1. Project input features
        x = self.in_proj(x)  # [B, L, d_model]

        # 2. Add positional encoding
        x = self.pos_encoder(x)

        # 3. Transformer encoding
        x = self.transformer_encoder(x)  # [B, L, d_model]

        # 4. Use the last time step's representation
        #    (Alternatively, you could aggregate all time steps)
        last_hidden = x[:, -1, :]  # [B, d_model]

        # 5. Output projection
        out = self.out_proj(last_hidden)  # [B, target_size * prediction_length]

        # 6. Reshape to sequence form
        out = out.view(-1, self.prediction_length, self.target_size)  # [B, pred_len, target_size]

        return out

    @staticmethod
    def load_from_artifact(artifact_dir: str, device: torch.device = None):
        """
        Loads a TST model from the given artifact directory.
        Expects:
        - metadata file: metadata.json
        - weight file: tst_<version>_<timestamp>.pt
        """
        import os, json, glob, torch

        if device is None:
            device = torch.device("cpu")

        metadata_path = os.path.join(artifact_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        version = metadata.get("version")
        if version is None:
            raise KeyError("metadata.json must contain a 'version' field")

        pattern = os.path.join(artifact_dir, f"tst_{version}_*.pt")
        weight_files = glob.glob(pattern)
        if not weight_files:
            raise FileNotFoundError(f"No TST weight file found for version {version} in {artifact_dir}")

        weight_path = weight_files[0]

        hp = metadata.get("hyperparameters", {})
        context_length = hp["context_length"]
        num_features = hp["num_features"]
        prediction_length = metadata["prediction_length"]
        target_size = metadata["target_size"]

        model = TimeSeriesTransformer(
            context_length=context_length,
            num_features=num_features,
            prediction_length=prediction_length,
            target_size=target_size,
            d_model=hp.get("d_model", 128),
            nhead=hp.get("nhead", 4),
            num_layers=hp.get("num_layers", 3),
            dim_feedforward=hp.get("dim_feedforward", 512),
            dropout=hp.get("dropout", 0.1),
        )

        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        print(f"[INFO] Loaded TST model version {version} from {artifact_dir}. Hyperparams: {hp}")
        return model, metadata
