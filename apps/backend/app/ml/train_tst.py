import logging
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from app.ml.adv.models_tst import TSTLite
from app.ml.train_tft import load_training_data
from app.ml.adv.feature_engineering import FEATURE_CONFIG, process_market_data
from app.ml.dataset import MultiModalTS
from app.ml.losses import multitask_transformer_loss

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
    df = load_training_data(days=90)
    df_processed = df.groupby("symbol", group_keys=False).apply(process_market_data)
    
    df_processed["target_price"] = df_processed.groupby("symbol")["close"].shift(-1) / df_processed["close"] - 1
    df_processed["target_vol"] = df_processed.groupby("symbol")["roll_vol_6h"].shift(-1)
    df_processed = df_processed.dropna()

    ds = MultiModalTS(
        df=df_processed,
        feature_blocks=FEATURE_CONFIG,
        label_col="target_price",
        vol_label_col="target_vol",
        tabular_feature_cols=[],
        roll_windows=[],
        seq_len=60
    )
    
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    in_feat = sum(ds.get_feature_dims().values())
    
    model = TSTLite(in_feat=in_feat, seq_len=60).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(10):
        total_loss = 0
        for x_blocks, _, y_dict in loader:
            # Manual Concatenation for TST (mimicking wrapper logic)
            tensors = [v.to(DEVICE) for k, v in x_blocks.items()]
            x_input = torch.cat(tensors, dim=-1)
            y_dict = {k: v.to(DEVICE) for k, v in y_dict.items()}
            
            optimizer.zero_grad()
            pred = model(x_input)
            
            loss = multitask_transformer_loss(pred, y_dict)["total_loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        logger.info(f"[TST] Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

    torch.save(model, "model_artifacts/tst_model.pth")
    logger.info("Saved TST model.")

if __name__ == "__main__":
    train()