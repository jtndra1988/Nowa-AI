from __future__ import annotations
import argparse, json
from .scheduler_retrain import full_retrain, incremental_update
from .feedback_loop import adaptive_threshold_update


def main():
    ap = argparse.ArgumentParser("Self-evolving retrain CLI")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--table", default="futures_market_data")
    ap.add_argument("--mode", choices=["full","incremental"], default="full")
    args = ap.parse_args()

    if args.mode == "full":
        res = full_retrain(args.symbol, args.table)
    else:
        res = incremental_update(args.symbol)
    new_cut = adaptive_threshold_update(args.symbol)
    print(json.dumps({"result": res, "base_conf_cutoff": new_cut}, indent=2))

if __name__ == "__main__":
    main()
