import os, math, time, json, torch, numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Dict, List
from .dataset import MultiModalTS
from .models_tcn import TemporalConvNet
from .models_tst import TSTLite
from .losses import mse_direction_sharpe

def build_model(arch: str, in_feat: int, cfg: Dict):
    if arch == "tcn":
        return TemporalConvNet(in_feat, channels=tuple(cfg.get("tcn_channels",[64,128,128])),
                               kernel=cfg.get("tcn_kernel",3), dropout=cfg.get("dropout",0.1))
    elif arch == "tst":
        return TSTLite(in_feat,
                       d_model=cfg.get("d_model",128),
                       nhead=cfg.get("nhead",4),
                       num_layers=cfg.get("num_layers",3),
                       dropout=cfg.get("dropout",0.1))
    else:
        raise ValueError("arch must be 'tcn' or 'tst'")

def train_one(df: pd.DataFrame, feature_blocks: Dict[str,List[str]], label_col: str, out_dir: str,
              arch="tst", seq_len=60, batch_size=256, epochs=20, lr=1e-3, 
              val_split_pct: float = 0.1, device=None, cfg:Dict=None):
    
    os.makedirs(out_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cfg = cfg or {}

    # --- 1. SEQUENTIAL DATA SPLIT (FIXED) ---
    # We split the DataFrame itself, not the Dataset object, to respect time.
    n_total = len(df)
    n_val = int(n_total * val_split_pct)
    n_train = n_total - n_val

    if n_train <= 0 or n_val <= 0:
        raise ValueError(f"Not enough data for train/val split. Total rows: {n_total}, Val split: {val_split_pct}")

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:]
    print(f"Data split: Total={n_total}, Train={len(train_df)}, Val={len(val_df)}")

    # --- 2. CREATE DATASET OBJECTS (using new dataset.py) ---
    train_ds = MultiModalTS(train_df, feature_blocks, label_col, seq_len=seq_len)
    val_ds = MultiModalTS(val_df, feature_blocks, label_col, seq_len=seq_len)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"Created datasets have 0 length. Check seq_len ({seq_len}) and data.")

    # --- 3. NORMALIZATION (FIXED) ---
    # Fit scaler ONLY on training data
    print("Fitting scaler on training data...")
    mean, std = train_ds.fit_scaler()
    
    # Save the scaler for inference
    scaler_path = os.path.join(out_dir, "scaler.pt")
    torch.save({"mean": mean, "std": std}, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Apply the fitted scaler to both train and val datasets
    train_ds.apply_scaler(mean, std)
    val_ds.apply_scaler(mean, std)
    
    # Get input feature count *after* creating the dataset
    in_feat = train_ds.X.shape[1] 

    # --- 4. CREATE DATALOADERS ---
    # We can now safely shuffle the training loader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False, pin_memory=True)

    # --- 5. MODEL, OPTIMIZER, SCALER ---
    model = build_model(arch, in_feat, cfg).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=cfg.get("weight_decay",1e-4))
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    best_val = math.inf
    patience = cfg.get("patience", 4)
    no_improve = 0
    ckpt_path = os.path.join(out_dir, f"{arch}_best.pt")

    print(f"Starting training for {epochs} epochs...")
    # --- 6. TRAINING LOOP (Unchanged) ---
    for epoch in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        t_start = time.time()
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                pred = model(xb)
                loss = mse_direction_sharpe(pred, yb, alpha=cfg.get("alpha",0.5),
                                            beta=cfg.get("beta",0.2), lam=cfg.get("lam",0.1))
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            scaler.step(opt); scaler.update()
            tr_loss += loss.item() * xb.size(0)

        # Validate
        model.eval()
        vl_loss, pnl = 0.0, []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                    pred = model(xb)
                    loss = mse_direction_sharpe(pred, yb, alpha=cfg.get("alpha",0.5),
                                                beta=cfg.get("beta",0.2), lam=cfg.get("lam",0.1))
                vl_loss += loss.item() * xb.size(0)
                pnl.append((pred*yb).detach().cpu().numpy())
        
        tr_loss /= len(train_ds); vl_loss /= len(val_ds)
        pnl = np.concatenate(pnl) if len(pnl) > 0 else np.asarray([0.0])
        val_sharpe = float(pnl.mean() / (pnl.std()+1e-6))
        
        epoch_time = time.time() - t_start

        print(f"Epoch {epoch:02d}/{epochs} | Time: {epoch_time:.1f}s | TrainLoss: {tr_loss:.4f} | ValLoss: {vl_loss:.4f} | ValSharpe: {val_sharpe:.4f}")

        # checkpoint on improvement (by loss)
        if vl_loss + 1e-9 < best_val:
            best_val = vl_loss; no_improve = 0
            torch.save({"model": model.state_dict(),
                        "cfg": {"arch":arch, "in_feat":in_feat, **cfg}}, ckpt_path)
            print(f"  -> New best val_loss: {best_val:.4f}. Checkpoint saved.")
        else:
            no_improve += 1

        # simple early stopping
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} due to no improvement for {patience} epochs.")
            break

        # write epoch log
        with open(os.path.join(out_dir, "train_log.jsonl"), "a") as f:
            f.write(json.dumps({"epoch":epoch, "train_loss":tr_loss, "val_loss":vl_loss,
                                "val_sharpe":val_sharpe, "best_val":best_val}) + "\n")

    print(f"Training complete. Best model saved to {ckpt_path}")
    return ckpt_path