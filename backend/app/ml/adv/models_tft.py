import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict

class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU)
    This is a key component of the TFT's gating mechanism.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.linear_in = nn.Linear(in_features, out_features)
        self.linear_gate = nn.Linear(in_features, out_features)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [B, L, in_features]
        in_part = self.linear_in(x)      # [B, L, out_features]
        gate_part = self.linear_gate(x)  # [B, L, out_features]
        
        # Apply sigmoid to gate and multiply
        output = in_part * self.sigmoid(gate_part)
        output = self.dropout(output)
        return output

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN)
    The main processing block of the TFT, which applies a residual connection
    on top of a GatedLinearUnit.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_linear = nn.Linear(d_model, d_model)
        self.glu = GatedLinearUnit(d_model, d_model, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [B, L, d_model]
        
        # Residual connection
        residual = x 
        
        # Gating path
        hidden = self.hidden_linear(x)
        gated_output = self.glu(hidden)
        
        # Add residual and normalize
        output = self.layer_norm(residual + self.dropout(gated_output))
        
        return output

class FeatureEmbedding(nn.Module):
    """
    Handles embedding of all input features into the d_model space.
    This is a simplified version to start. A full TFT would differentiate
    between static, known, and observed features.
    """
    def __init__(self, feature_dims: Dict[str, int], d_model: int):
        super().__init__()
        # Create a linear projection layer for each feature block
        self.projections = nn.ModuleDict()
        self.feature_dims = feature_dims
        total_input_dim = 0
        
        for bname, dim in feature_dims.items():
            self.projections[bname] = nn.Linear(dim, d_model)
            total_input_dim += dim
            
        # We will also add a projection for the concatenated features
        self.concat_projection = nn.Linear(total_input_dim, d_model)

    def forward(self, x_blocks: Dict[str, torch.Tensor]):
        """
        Input is a dictionary of tensors, e.g.:
        {
            "price": tensor[B, L, 5],
            "ob": tensor[B, L, 4],
            "sent": tensor[B, L, 2]
        }
        """
        
        # Project each block and sum them up (a form of feature fusion)
        embeddings = []
        all_features_concat = []
        
        for bname in self.feature_dims.keys():
            x_b = x_blocks[bname]
            embeddings.append(self.projections[bname](x_b))
            all_features_concat.append(x_b)

        # Sum component-wise embeddings
        summed_embeddings = torch.stack(embeddings, dim=-1).sum(dim=-1)
        
        # Also pass all concatenated features through a single layer
        concat_features = torch.cat(all_features_concat, dim=-1)
        concat_embedding = self.concat_projection(concat_features)

        # Combine the two types of embeddings
        # This is a simple fusion, a real TFT would be more complex
        return summed_embeddings + concat_embedding
    
class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN)
    This is a key part of the TFT's interpretability. It learns weights
    for each input feature, allowing us to see which features the model
    considers most important for a given input.
    """
    def __init__(self, d_model: int, num_features: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        
        # A GRN is used to process the concatenated feature embeddings
        self.grn = GatedResidualNetwork(d_model * num_features, dropout)
        
        # A final linear layer outputs one weight per feature
        self.softmax_grn = nn.Sequential(
            nn.Linear(d_model * num_features, num_features),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor):
        # x shape: [B, L, F, d_model] - where F is num_features
        
        batch_size, seq_len, _, d_model = x.shape
        
        # Flatten features: [B, L, F * d_model]
        flat_x = x.view(batch_size, seq_len, -1)
        
        # Pass through GRN
        grn_output = self.grn(flat_x) # [B, L, F * d_model]
        
        # Get feature weights
        feature_weights = self.softmax_grn(grn_output) # [B, L, F]
        
        # Reshape weights to [B, L, F, 1] for broadcasting
        feature_weights = feature_weights.unsqueeze(-1)
        
        # Apply weights: (original features) * (learned weights)
        # weighted_features shape: [B, L, F, d_model]
        weighted_features = x * feature_weights
        
        # Sum weighted features to get final output
        # output shape: [B, L, d_model]
        output = weighted_features.sum(dim=2)
        
        return output, feature_weights.squeeze(-1) # Return weights for interpretability

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    (Copied from our TSTLite implementation)
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
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT) Model
    Combines GRNs and Variable Selection with a standard Transformer Encoder
    to achieve high accuracy and interpretability.
    
    This is a simplified implementation based on the audit's recommendations.
    """
    def __init__(self,
                 feature_dims: Dict[str, int],
                 seq_len: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # 1. Feature Embedding Layer
        self.embedder = FeatureEmbedding(feature_dims, d_model)
        self.num_features = len(feature_dims)
        
        # 2. Variable Selection Network
        # We need a VSN for our input features
        self.vsn = VariableSelectionNetwork(d_model, self.num_features, dropout)
        
        # 3. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len + 1)
        
        # 4. Core Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 5. Gating and Residuals around the Encoder
        self.pre_encoder_grn = GatedResidualNetwork(d_model, dropout)
        self.post_encoder_gate = GatedLinearUnit(d_model, d_model, dropout)
        self.post_encoder_norm = nn.LayerNorm(d_model)

        # 6. Final Output Heads (Multi-Task)
        self.head_norm = nn.LayerNorm(d_model)
        self.price_head = nn.Linear(d_model, 1)
        self.vol_head = nn.Linear(d_model, 1)

    def forward(self, x_blocks: Dict[str, torch.Tensor]):
        """
        Input 'x_blocks' is a dictionary of tensors, one for each feature block:
        {
            "price": tensor[B, L, 5],
            "ob": tensor[B, L, 4],
            ...
        }
        """
        
        # --- 1. Feature Embedding ---
        # Project each feature block to d_model
        # embeddings_dict = {"price": [B,L,D], "ob": [B,L,D], ...}
        embeddings_dict = {}
        for bname, x_b in x_blocks.items():
            embeddings_dict[bname] = self.embedder.projections[bname](x_b)

        # Stack features: [B, L, F, D] (F = num_features)
        stacked_embeddings = torch.stack(list(embeddings_dict.values()), dim=2)

        # --- 2. Variable Selection ---
        # Pass through VSN to get weighted features and feature weights
        # selected_features: [B, L, D]
        # feature_weights: [B, L, F] (for interpretability)
        selected_features, feature_weights = self.vsn(stacked_embeddings)
        
        # --- 3. Pre-Transformer Gating ---
        processed_features = self.pre_encoder_grn(selected_features)
        
        # --- 4. Positional Encoding ---
        # (Using the same scaling as TSTLite)
        x_in = processed_features * math.sqrt(self.d_model)
        x_in = self.pos_encoder(x_in)
        
        # --- 5. Transformer Encoder ---
        encoder_output = self.encoder(x_in) # [B, L, D]
        
        # --- 6. Post-Transformer Gating & Residual ---
        gated_output = self.post_encoder_gate(encoder_output)
        
        # Add residual connection from before the encoder
        residual_output = gated_output + selected_features 
        residual_output = self.post_encoder_norm(residual_output)
        
        # --- 7. Final Output ---
        # Get output from the last time step
        final_output = residual_output[:, -1] # [B, D]
        final_output = self.head_norm(final_output)

        # Generate multi-task predictions
        price_pred = self.price_head(final_output).squeeze(-1) # [B]
        vol_pred = torch.nn.functional.softplus(self.vol_head(final_output).squeeze(-1)) # [B]

        # Return predictions and interpretability weights
        return {
            'price': price_pred,
            'vol': vol_pred,
            'feature_weights': feature_weights # [B, L, F]
        }    