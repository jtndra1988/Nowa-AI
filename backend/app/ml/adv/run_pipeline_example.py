import pandas as pd
from .dataset import MultiModalTS
from .train_adv import train_one
from .optuna_tune import run_tuning

# 1) Prepare features mapping
FEATURES = {
  "price":   ["open","high","low","close","volume","ret_1","ret_5","atr_14","vol_10","rsi_14"],
  "orderbk": ["ob_imb_1s","ob_spread","ob_depth_ask_1","ob_depth_bid_1"],
  "sent":    ["sent_score","sent_vol","news_flow_1h"],
  "onchain": ["onch_active_addrs","onch_txn","onch_hashrate_z"]
}
LABEL = "next_return_15m"  # e.g., your forward return label

def load_aligned_frame(engine) -> pd.DataFrame:
    # join price + OB + sentiment + on-chain by timestamp & symbol
    # (replace with your SQL joins / cached parquet)
    raise NotImplementedError

def train_for_symbol(df_sym: pd.DataFrame, out_dir:str):
    best = run_tuning(df_sym, FEATURES, LABEL, out_dir=out_dir, n_trials=20)
    _ = train_one(df_sym, FEATURES, LABEL, out_dir=out_dir, arch=best["arch"],
                  seq_len=best["seq_len"], batch_size=best["batch"], epochs=best["epochs"],
                  lr=best["lr"], cfg={k:v for k,v in best.items() if k not in ["arch","seq_len","batch","epochs","lr"]})
