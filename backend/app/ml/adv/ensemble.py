import torch
import torch.nn as nn
from typing import Dict, Any

# Import the models we are going to ensemble
from .models_tft import TemporalFusionTransformer
from .models_tcn import TemporalConvNet

class StackingEnsemble(torch.nn.Module):
    """
    This is your original, simple blending class.
    It learns a weighted average of pre-computed predictions.
    We will use this *inside* our new ensemble model.
    """
    def __init__(self, n_models: int = 2):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(n_models) / n_models)

    def forward(self, preds: torch.Tensor):  # preds [B, n_models]
        w = torch.softmax(self.w, dim=0)
        return (preds * w).sum(dim=1)

class MultiTaskEnsemble(nn.Module):
    """
    This is the main ensemble model for Path 2.
    It loads the pre-trained TFT and TCN, freezes them,
    and learns to blend their multi-task outputs (price and vol).
    
    """
    def __init__(self, tft_config: Dict[str, Any], tcn_config: Dict[str, Any], tft_weights_path: str, tcn_weights_path: str):
        super().__init__()
        
        # --- 1. Load TFT Model ---
        self.tft_model = TemporalFusionTransformer(**tft_config)
        self.tft_model.load_state_dict(torch.load(tft_weights_path))
        self.tft_model.eval() # Set to eval mode
        # Freeze all TFT parameters
        for param in self.tft_model.parameters():
            param.requires_grad = False
            
        # --- 2. Load TCN Model ---
        self.tcn_model = TemporalConvNet(**tcn_config)
        self.tcn_model.load_state_dict(torch.load(tcn_weights_path))
        self.tcn_model.eval() # Set to eval mode
        # Freeze all TCN parameters
        for param in self.tcn_model.parameters():
            param.requires_grad = False
            
        print("TFT and TCN models loaded and frozen.")

        # --- 3. Create Blenders ---
        # We need two separate blenders, one for each task
        # The parameters 'w' in these blenders are the *only*
        # parameters that will be trained.
        self.price_blender = StackingEnsemble(n_models=2)
        self.vol_blender = StackingEnsemble(n_models=2)

    def forward(self, x_blocks: Dict[str, torch.Tensor]):
        """
        This model takes the same x_blocks input as the individual models.
        """
        
        # --- 1. Get predictions from both models ---
        # We don't need gradients here, as models are frozen
        with torch.no_grad():
            preds_tft = self.tft_model(x_blocks)
            preds_tcn = self.tcn_model(x_blocks)

        # --- 2. Stack predictions for blending ---
        
        # Stack price predictions: [B, 2]
        stacked_price = torch.stack([
            preds_tft['price'],
            preds_tcn['price']
        ], dim=1)
        
        # Stack volatility predictions: [B, 2]
        stacked_vol = torch.stack([
            preds_tft['vol'],
            preds_tcn['vol']
        ], dim=1)

        # --- 3. Blend the predictions ---
        # Only this part is trainable
        blended_price = self.price_blender(stacked_price)
        blended_vol = self.vol_blender(stacked_vol)
        
        # Return the final blended predictions in the same
        # dictionary format as our other models.
        return {
            'price': blended_price,
            'vol': blended_vol
        }