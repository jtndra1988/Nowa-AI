import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU)."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.linear_in = nn.Linear(in_features, out_features)
        self.linear_gate = nn.Linear(in_features, out_features)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_part = self.linear_in(x)
        gate_part = self.linear_gate(x)
        output = in_part * self.sigmoid(gate_part)
        output = self.dropout(output)
        return output


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN)."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_linear = nn.Linear(d_model, d_model)
        self.glu = GatedLinearUnit(d_model, d_model, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x_blocks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        x_blocks: dict[name] -> Tensor[batch, seq_len, feat_dim]
        """
        embeddings = []
        all_concat = []

        for bname in self.feature_dims.keys():
            x_b = x_blocks[bname]
            embeddings.append(self.projections[bname](x_b))  # [B, L, d_model]
            all_concat.append(x_b)                           # [B, L, dim]

        summed_embeddings = torch.stack(embeddings, dim=-1).sum(dim=-1)  # [B, L, d_model]
        concat_features = torch.cat(all_concat, dim=-1)                  # [B, L, total_input_dim]
        concat_embedding = self.concat_projection(concat_features)       # [B, L, d_model]

        return summed_embeddings + concat_embedding


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN) for interpretability."""

    def __init__(self, d_model: int, num_features: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.grn = GatedResidualNetwork(d_model * num_features, dropout)
        self.softmax_grn = nn.Sequential(
            nn.Linear(d_model * num_features, num_features),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, L, F, d_model]
        Returns:
            output:   [B, L, d_model]
            weights:  [B, L, F]
        """
        b, l, f, d = x.shape
        flat_x = x.view(b, l, f * d)
        grn_output = self.grn(flat_x)
        weights = self.softmax_grn(grn_output)  # [B, L, F]
        weights = weights.unsqueeze(-1)         # [B, L, F, 1]
        weighted = x * weights                  # [B, L, F, d_model]
        output = weighted.sum(dim=2)            # [B, L, d_model]
        return output, weights.squeeze(-1)      # [B, L, F]


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
        # x: [B, L, d_model]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalFusionTransformer(nn.Module):
    """
    Simplified Temporal Fusion Transformer for your hybrid stack.

    Input:
        x_blocks: dict[name] -> Tensor[batch, seq_len, feat_dim]

    Output:
        {
            "price": Tensor[batch],
            "vol":   Tensor[batch],
            "feature_weights": Tensor[batch, seq_len, num_features]
        }
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
        self.seq_len = seq_len
        self.feature_dims = feature_dims
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

    def forward(self, x_blocks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 1. Per-block embeddings
        embeddings_dict = {
            bname: self.embedder.projections[bname](x_b)
            for bname, x_b in x_blocks.items()
        }
        # Stack into [B, L, F, D]
        stacked = torch.stack(list(embeddings_dict.values()), dim=2)

        # 2. Variable selection
        selected_features, feature_weights = self.vsn(stacked)  # [B, L, D], [B, L, F]

        # 3. Pre-encoder GRN
        processed = self.pre_encoder_grn(selected_features)

        # 4. Positional encoding + Transformer encoder
        x_in = processed * math.sqrt(self.d_model)
        x_in = self.pos_encoder(x_in)
        enc_out = self.encoder(x_in)  # [B, L, D]

        # 5. Post-encoder gating + residual
        gated = self.post_encoder_gate(enc_out)
        residual = gated + selected_features
        residual = self.post_encoder_norm(residual)

        # 6. Final head on last time-step
        final = residual[:, -1]  # [B, D]
        final = self.head_norm(final)

        price = self.price_head(final).squeeze(-1)                # [B]
        vol = F.softplus(self.vol_head(final).squeeze(-1))        # [B]

        return {
            "price": price,
            "vol": vol,
            "feature_weights": feature_weights,  # [B, L, F]
        }

    # ---------- Artifact loading for versioned TFT ----------

    @staticmethod
    def load_from_artifact(artifact_dir: str, device: torch.device = None):
        """
        Loads a TFT model from the given artifact directory.

        Expects:
            - metadata.json with:
                {
                  "model_name": "tft",
                  "version": "v1.0",
                  "feature_dims": { "block": dim, ... },
                  "seq_len": 60,
                  "hyperparameters": {
                      "d_model": ...,
                      "nhead": ...,
                      "num_layers": ...,
                      "dropout": ...
                  },
                  ...
                }
            - weights file: tft_<version>_<timestamp>.pt
        """
        import glob

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

        pattern = os.path.join(artifact_dir, f"tft_{version}_*.pt")
        weight_files = glob.glob(pattern)
        if not weight_files:
            raise FileNotFoundError(
                f"No TFT weight file found for version {version} in {artifact_dir}"
            )
        weight_path = weight_files[0]

        feature_dims = metadata["feature_dims"]
        seq_len = metadata["seq_len"]
        hp = metadata.get("hyperparameters", {})

        model = TemporalFusionTransformer(
            feature_dims=feature_dims,
            seq_len=seq_len,
            d_model=hp.get("d_model", 128),
            nhead=hp.get("nhead", 4),
            num_layers=hp.get("num_layers", 3),
            dropout=hp.get("dropout", 0.1),
        )

        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        print(
            f"[INFO] Loaded TFT model {metadata.get('model_name', 'tft')} "
            f"version {version} from {artifact_dir}"
        )
        return model, metadata


class TFTPredictor:
    """
    Simple production-friendly wrapper that uses the legacy
    full-model artifact: model_artifacts/tft_model.pth
    """

    def __init__(self, model_path: str = "model_artifacts/tft_model.pth"):
        self.model = None
        path = Path(model_path)
        if path.exists():
            self.model = torch.load(path)
            self.model.eval()

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def predict(self, features: Dict[str, Any]) -> float:
        """
        Runs inference on the TFT model.

        `features` is expected to be:
            {
                "block_name": array_or_tensor_like [L, F_block],
                ...
            }
        """
        if self.model is None:
            return 0.0

        self.model.eval()
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        x_blocks: Dict[str, torch.Tensor] = {}

        try:
            # 1. Convert features to tensors
            for name, data in features.items():
                if isinstance(data, (str, int, float, bool)) or data is None:
                    continue

                if isinstance(data, torch.Tensor):
                    tensor = data.clone().detach()
                else:
                    tensor = torch.tensor(data, dtype=torch.float32)

                # TFT expects [Batch, Seq_Len, Features]
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(0)  # [1, L, F]
                x_blocks[name] = tensor.to(device)

            if not x_blocks:
                return 0.0

            # 2. Run inference
            with torch.no_grad():
                prediction = self.model(x_blocks)

            # 3. Extract price
            price_val = prediction.get("price")
            if price_val is not None:
                return float(price_val.squeeze().item())

            return 0.0

        except Exception as e:
            print(f"[TFTPredictor] Prediction error: {e}")
            return 0.0
