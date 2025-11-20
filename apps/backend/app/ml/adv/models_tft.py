import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU)"""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.linear_in = nn.Linear(in_features, out_features)
        self.linear_gate = nn.Linear(in_features, out_features)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        in_part = self.linear_in(x)
        gate_part = self.linear_gate(x)
        output = in_part * self.sigmoid(gate_part)
        output = self.dropout(output)
        return output


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN)"""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_linear = nn.Linear(d_model, d_model)
        self.glu = GatedLinearUnit(d_model, d_model, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        hidden = self.hidden_linear(x)
        gated_output = self.glu(hidden)
        output = self.layer_norm(residual + self.dropout(gated_output))
        return output


class FeatureEmbedding(nn.Module):
    """
    Embeds each feature block into d_model, plus a concat projection.
    """

    def __init__(self, feature_dims: Dict[str, int], d_model: int):
        super().__init__()
        self.feature_dims = feature_dims
        self.projections = nn.ModuleDict()
        total_input_dim = 0

        for bname, dim in feature_dims.items():
            self.projections[bname] = nn.Linear(dim, d_model)
            total_input_dim += dim

        self.concat_projection = nn.Linear(total_input_dim, d_model)

    def forward(self, x_blocks: Dict[str, torch.Tensor]):
        embeddings = []
        all_concat = []

        for bname in self.feature_dims.keys():
            x_b = x_blocks[bname]
            embeddings.append(self.projections[bname](x_b))
            all_concat.append(x_b)

        summed_embeddings = torch.stack(embeddings, dim=-1).sum(dim=-1)
        concat_features = torch.cat(all_concat, dim=-1)
        concat_embedding = self.concat_projection(concat_features)

        return summed_embeddings + concat_embedding


class VariableSelectionNetwork(nn.Module):
    """VSN for interpretability."""

    def __init__(self, d_model: int, num_features: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.grn = GatedResidualNetwork(d_model * num_features, dropout)
        self.softmax_grn = nn.Sequential(
            nn.Linear(d_model * num_features, num_features),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, L, F, d_model]
        b, l, f, d = x.shape
        flat_x = x.view(b, l, f * d)
        grn_output = self.grn(flat_x)
        weights = self.softmax_grn(grn_output)  # [B, L, F]
        weights = weights.unsqueeze(-1)         # [B, L, F, 1]
        weighted = x * weights                  # [B, L, F, d_model]
        output = weighted.sum(dim=2)           # [B, L, d_model]
        return output, weights.squeeze(-1)     # [B, L, F]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalFusionTransformer(nn.Module):
    """
    Simplified Temporal Fusion Transformer for your hybrid stack.
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        seq_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_features = len(feature_dims)

        self.embedder = FeatureEmbedding(feature_dims, d_model)
        self.vsn = VariableSelectionNetwork(d_model, self.num_features, dropout)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len + 1)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.pre_encoder_grn = GatedResidualNetwork(d_model, dropout)
        self.post_encoder_gate = GatedLinearUnit(d_model, d_model, dropout)
        self.post_encoder_norm = nn.LayerNorm(d_model)

        self.head_norm = nn.LayerNorm(d_model)
        self.price_head = nn.Linear(d_model, 1)
        self.vol_head = nn.Linear(d_model, 1)

    def forward(self, x_blocks: Dict[str, torch.Tensor]):
        # embed each block
        embeddings_dict = {
            bname: self.embedder.projections[bname](x_b)
            for bname, x_b in x_blocks.items()
        }
        stacked = torch.stack(list(embeddings_dict.values()), dim=2)  # [B,L,F,D]

        selected_features, feature_weights = self.vsn(stacked)         # [B,L,D], [B,L,F]

        processed = self.pre_encoder_grn(selected_features)

        x_in = processed * math.sqrt(self.d_model)
        x_in = self.pos_encoder(x_in)

        enc_out = self.encoder(x_in)

        gated = self.post_encoder_gate(enc_out)
        residual = gated + selected_features
        residual = self.post_encoder_norm(residual)

        final = residual[:, -1]      # [B,D]
        final = self.head_norm(final)

        price = self.price_head(final).squeeze(-1)
        vol = F.softplus(self.vol_head(final).squeeze(-1))

        return {
            "price": price,
            "vol": vol,
            "feature_weights": feature_weights,
        }

class TFTPredictor:
    def __init__(self, model_path="model_artifacts/tft_model.pth"):
        self.model = None
        # Logic to load self.model using torch.load(model_path) goes here
        # self.model = TemporalFusionTransformer(...)
        # self.model.load_state_dict(...)
        pass

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def predict(self, features: Dict[str, Any]) -> float:
        # 1. Transform 'features' dict into tensors expected by TemporalFusionTransformer
        # 2. Run self.model(x)
        # 3. Return float(output['price'])
        return 0.0 # Placeholder