import torch
import torch.nn as nn
from typing import Any, Dict


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

class TCNPredictor:
    def __init__(self, model_path="model_artifacts/tcn_model.pth"):
        self.model = None
        self.model = torch.load(model_path)
        self.model.eval()
        pass

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def predict(self, features: Dict[str, Any]) -> float:
        """
        Runs inference on the loaded TCN model.
        Expects 'features' to be a dict of {block_name: array_like} corresponding
        to the feature blocks the model was trained on.
        """
        if not self.model:
            return 0.0
            
        # Ensure model is in eval mode
        self.model.eval()
        
        # Detect device (CPU vs CUDA) from the model parameters
        device = next(self.model.parameters()).device
        
        x_blocks = {}

        # 1. Prepare tensors from features
        try:
            for name, data in features.items():
                # Skip metadata or non-array features if any exist in the dict
                if isinstance(data, (str, int, float, bool)) or data is None:
                    continue

                # Convert to Tensor
                # We assume data is compatible with numpy/torch (List or np.ndarray)
                if isinstance(data, torch.Tensor):
                    tensor = data.clone().detach()
                else:
                    tensor = torch.tensor(data, dtype=torch.float32)
                
                # Handle Dimensions:
                # Model expects [Batch, Seq_Len, Features]
                # If input is just [Seq_Len, Features], we add a Batch dim at index 0
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(0)
                
                x_blocks[name] = tensor.to(device)

            if not x_blocks:
                # If no valid feature blocks were found, return neutral
                return 0.0

            # 2. Run inference
            with torch.no_grad():
                # forward() returns {'price': ..., 'vol': ...}
                prediction = self.model(x_blocks)
            
            # 3. Return float(pred['price'])
            price_val = prediction.get("price")
            
            if price_val is not None:
                return float(price_val.item())
            
            return 0.0

        except Exception as e:
            # Log the error safely without crashing the whole engine
            print(f"[TCNPredictor] Prediction error: {e}")
            return 0.0