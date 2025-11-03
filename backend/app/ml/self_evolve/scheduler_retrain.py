from __future__ import annotations
import json
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import torch

from .datasets import build_features
from .feature_bank import add_extra_features
from .online_updates import XGBIncremental, finetune_lstm
from .ensemble_manager import DynamicEnsemble
from .utils import ensure_dir, out_dir, promote_if_better, FILENAMES, _save_manifest

# Reuse your canonical LSTM
from app.ml.model import LSTMSignalModel

# ===== One-shot full retrain =====

def full_retrain(symbol: str, table: str = "futures_market_data", seq_len: int = 60) -> Dict[str, Any]:
    df, feat_cols = build_features(symbol, table)
    df, extra = add_extra_features(df)
    feat_all = sorted(list(set(feat_cols + extra)))

    # target = next close
    df["target"] = df["close"].shift(-1)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    # train/val split on time
    split_idx = int(len(df)*0.8)
    df_tr, df_va = df.iloc[:split_idx], df.iloc[split_idx:]

    # scaler on train
    X_tr = df_tr[feat_all].values.astype(np.float32)
    X_va = df_va[feat_all].values.astype(np.float32)
    y_tr = df_tr["target"].values.astype(np.float32)
    y_va = df_va["target"].values.astype(np.float32)

    scaler = StandardScaler().fit(X_tr)
    # LSTM sequences
    def mk_seq(X: np.ndarray, y: np.ndarray, L: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(X) <= L: return np.empty((0,L,X.shape[1])), np.empty((0,1))
        Xs = np.stack([X[i-L:i] for i in range(L, len(X))])
        ys = y[L:]
        return Xs.astype(np.float32), ys.astype(np.float32)

    Xs_tr, ys_tr = mk_seq(scaler.transform(X_tr), y_tr, seq_len)
    Xs_va, ys_va = mk_seq(scaler.transform(X_va), y_va, seq_len)

    # Train LSTM
    model = LSTMSignalModel(input_size=Xs_tr.shape[-1], hidden_layer_size=128, num_layers=2, output_size=1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(12):
        # train
        model.train();
        bs=256
        for i in range(0, len(Xs_tr), bs):
            xb = torch.tensor(Xs_tr[i:i+bs]).to(dev)
            yb = torch.tensor(ys_tr[i:i+bs]).unsqueeze(-1).to(dev)
            pred = model(xb); loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # tiny val pass (optional early stop)
        model.eval()
        with torch.no_grad():
            if len(Xs_va) > 0:
                pv = model(torch.tensor(Xs_va).to(dev)).squeeze(-1).cpu().numpy()
                rmse = float(np.sqrt(((pv - ys_va)**2).mean()))
            else:
                rmse = float("inf")
    # Save to staging
    from joblib import dump
    staging = out_dir(symbol, "staging")
    ensure_dir(staging)
    (staging / FILENAMES["lstm"]).write_bytes(torch.save(model.state_dict(), staging / FILENAMES["lstm"]) or b"")
    dump(scaler, staging / FILENAMES["lstm_scaler"])  # type: ignore
    # write features.json
    (staging / FILENAMES["features"]).write_text(json.dumps({"lstm_features": feat_all}, indent=2))

    # XGB
    xgb = XGBRegressor(n_estimators=800, max_depth=8, subsample=0.8, colsample_bytree=0.8, learning_rate=0.05, tree_method="hist")
    xgb.fit(df_tr[feat_all].values, y_tr)
    # stage save
    xgb.save_model(str(staging / FILENAMES["xgb"]))
    (staging / FILENAMES["xgb_features"]).write_text(json.dumps({"xgb_features": feat_all}, indent=2))

    # Simple validation metric: directional accuracy
    def dir_acc(y_true, y_pred, ref):
        up_t = (y_true - ref) > 0; up_p = (y_pred - ref) > 0
        return float((up_t == up_p).mean())

    # Evaluate LSTM
    model.eval()
    with torch.no_grad():
        pv = model(torch.tensor(Xs_va).to(dev)).squeeze(-1).cpu().numpy() if len(Xs_va) else np.array([])
    lstm_da = dir_acc(ys_va, pv, df_va["close"].values[-len(ys_va):] if len(ys_va) else np.array([])) if len(ys_va) else 0.0

    # Evaluate XGB
    yx = xgb.predict(df_va[feat_all].values)
    xgb_da = dir_acc(y_va, yx, df_va["close"].values)

    _save_manifest(out_dir(symbol, "staging"), decision_params={"base_confidence_cutoff": 0.55})

    return {"lstm_dir_acc": lstm_da, "xgb_dir_acc": xgb_da}


# ===== Online / incremental hook (to be called daily) =====

def incremental_update(symbol: str, recent_rows: int = 10_000) -> Dict[str, Any]:
    df, feat = build_features(symbol)
    df = df.tail(recent_rows).copy()
    df["target"] = df["close"].shift(-1)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    X = df[feat].values.astype(np.float32)
    y = df["target"].values.astype(np.float32)

    # Load current prod artifacts
    from joblib import load
    prod = out_dir(symbol, "prod")
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(); xgb.load_model(str(prod / FILENAMES["xgb"]))
    except Exception:
        xgb = XGBRegressor()
    try:
        scaler = load(prod / FILENAMES["lstm_scaler"])  # type: ignore
    except Exception:
        scaler = StandardScaler().fit(X)

    # XGB warm start
    xincr = XGBIncremental(base_model=xgb)
    xgb2 = xincr.fit_more(X, y, n_additional=200)

    # LSTM fine-tune
    # load LSTM
    import torch
    from pathlib import Path
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lstm_state = None
    if (prod / FILENAMES["lstm"]).exists():
        lstm_state = torch.load(str(prod / FILENAMES["lstm"]), map_location=dev)
    model = LSTMSignalModel(input_size=X.shape[1], hidden_layer_size=128, num_layers=2, output_size=1)
    if isinstance(lstm_state, dict):
        state = lstm_state.get("state_dict", lstm_state.get("model_state_dict", lstm_state))
        try: model.load_state_dict(state, strict=False)
        except Exception: pass
    model = finetune_lstm(model, scaler, X, y, seq_len=60, epochs=3, lr=5e-4, device=dev)

    # Save back to staging and promote if better based on a quick holdout
    m = {"result": "ok"}
    staging = out_dir(symbol, "staging")
    ensure_dir(staging)
    xgb2.save_model(str(staging / FILENAMES["xgb"]))
    from joblib import dump
    dump(scaler, staging / FILENAMES["lstm_scaler"])  # type: ignore
    torch.save(model.state_dict(), staging / FILENAMES["lstm"])  # type: ignore
    (staging / FILENAMES["features"]).write_text(json.dumps({"lstm_features": feat, "xgb_features": feat}, indent=2))
    # Evaluate quick DA on last 2k rows
    k = min(2000, len(df)-61)
    da_new = 0.5
    if k > 0:
        X_sc = scaler.transform(X[-(k+60):])
        X_seq = np.stack([X_sc[i-60:i] for i in range(60, len(X_sc))])
        with torch.no_grad():
            pv = model(torch.tensor(X_seq).to(dev)).squeeze(-1).cpu().numpy()
        yv = y[-k:]
        ref = df["close"].values[-k:]
        da_new = float(((yv - ref) > 0) == ((pv - ref) > 0)).mean()
    prod_manifest = (out_dir(symbol, "prod") / FILENAMES["manifest"]).read_text() if (out_dir(symbol, "prod") / FILENAMES["manifest"]).exists() else "{}"
    try:
        last_metric = json.loads(prod_manifest).get("last_metric")
    except Exception:
        last_metric = None
    promoted = promote_if_better(symbol, da_new, last_metric, higher_is_better=True,
                                 files={"lstm": FILENAMES["lstm"], "lstm_scaler": FILENAMES["lstm_scaler"], "features": FILENAMES["features"], "xgb": FILENAMES["xgb"], "xgb_features": FILENAMES["xgb_features"]})
    m["promoted"] = promoted; m["new_dir_acc"] = da_new
    return m

