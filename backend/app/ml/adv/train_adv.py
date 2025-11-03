import os, math, time, json, torch, numpy as np
from torch.utils.data import DataLoader, random_split
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

def train_one(df, feature_blocks: Dict[str,List[str]], label_col: str, out_dir: str,
              arch="tst", seq_len=60, batch_size=256, epochs=20, lr=1e-3, device=None, cfg:Dict=None):
    os.makedirs(out_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = cfg or {}

    ds = MultiModalTS(df, feature_blocks, label_col, seq_len=seq_len)
    in_feat = sum(len(v) for v in feature_blocks.values())
    n_val = max(1024, int(0.1*len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)

    model = build_model(arch, in_feat, cfg).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=cfg.get("weight_decay",1e-4))
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    best_val = math.inf
    patience = cfg.get("patience", 4)
    no_improve = 0
    ckpt_path = os.path.join(out_dir, f"{arch}_best.pt")

    for epoch in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
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
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = mse_direction_sharpe(pred, yb, alpha=cfg.get("alpha",0.5),
                                            beta=cfg.get("beta",0.2), lam=cfg.get("lam",0.1))
                vl_loss += loss.item() * xb.size(0)
                pnl.append((pred*yb).detach().cpu().numpy())
        tr_loss /= len(train_ds); vl_loss /= len(val_ds)
        pnl = np.concatenate(pnl) if len(pnl) else np.asarray([0.0])
        val_sharpe = float(pnl.mean() / (pnl.std()+1e-6))

        # checkpoint on improvement (by loss; sharpe is logged for selection later)
        if vl_loss + 1e-9 < best_val:
            best_val = vl_loss; no_improve = 0
            torch.save({"model": model.state_dict(),
                        "cfg": {"arch":arch, "in_feat":in_feat, **cfg}}, ckpt_path)
        else:
            no_improve += 1

        # simple early stopping
        if no_improve >= patience: break

        # write epoch log
        with open(os.path.join(out_dir, "train_log.jsonl"), "a") as f:
            f.write(json.dumps({"epoch":epoch, "train_loss":tr_loss, "val_loss":vl_loss,
                                "val_sharpe":val_sharpe, "best_val":best_val}) + "\n")

    return ckpt_path
