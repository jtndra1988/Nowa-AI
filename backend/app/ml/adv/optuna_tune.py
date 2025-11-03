import optuna, pandas as pd
from .train_adv import train_one
from .dataset import MultiModalTS

def objective(trial: optuna.Trial, df: pd.DataFrame, features, label_col, out_dir):
    arch = trial.suggest_categorical("arch", ["tst","tcn"])
    cfg = {
        "dropout": trial.suggest_float("dropout", 0.05, 0.4),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "alpha": trial.suggest_float("alpha", 0.2, 0.9),
        "beta": trial.suggest_float("beta", 0.0, 0.5),
        "lam": trial.suggest_float("lam", 0.05, 0.4),
        "patience": 3,
        "grad_clip": 1.0
    }
    if arch=="tst":
        cfg.update({
            "d_model": trial.suggest_categorical("d_model",[64,128,256]),
            "nhead":   trial.suggest_categorical("nhead",[2,4,8]),
            "num_layers": trial.suggest_int("num_layers",2,4)
        })
    else:
        cfg.update({
            "tcn_channels": trial.suggest_categorical("tcn_channels", [(64,128,128),(128,128,256)]),
            "tcn_kernel": trial.suggest_categorical("tcn_kernel",[3,5])
        })
    lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    seq_len = trial.suggest_int("seq_len", 30, 120, step=10)
    batch   = trial.suggest_categorical("batch",[128,256,512])
    ckpt = train_one(
        df, features, label_col, out_dir=out_dir, arch=arch, seq_len=seq_len,
        batch_size=batch, epochs=trial.suggest_int("epochs", 8, 20),
        lr=lr, cfg=cfg
    )
    # Use the best epoch's validated Sharpe from log
    sharpe = 0.0
    try:
        import json, os
        with open(os.path.join(out_dir, "train_log.jsonl")) as f:
            vals = [json.loads(x)["val_sharpe"] for x in f]
            sharpe = float(max(vals))
    except Exception:
        pass
    # We *maximize* Sharpe: Optuna minimizes by default => return negative
    return -sharpe

def run_tuning(df, features, label_col, out_dir, n_trials=25):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, df, features, label_col, out_dir), n_trials=n_trials)
    return study.best_params
