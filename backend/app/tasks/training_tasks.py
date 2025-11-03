# app/tasks/training_tasks.py
from __future__ import annotations

import json
import logging
import math
import os
import shutil # <-- ADDED for potential cleanup
import subprocess
import sys
from collections import defaultdict # <-- ADDED for metric aggregation
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from app.worker import singleton_lock
from celery import shared_task
from app.celery_app.app import BaseTaskWithRetry, celery_app

# Prometheus metrics (safe if missing)
try:
    from app.monitoring.metrics import METRICS
except Exception:
    METRICS = None

# === paths ===
MODELS_ROOT = Path("/app/models").resolve()
CURRENT_SYMLINK = MODELS_ROOT / "current"        # -> /app/models/<timestamp>/
# CANDIDATES_DIR = MODELS_ROOT / "candidates"      # No longer primary, timestamped dirs used directly
MODELS_ROOT.mkdir(parents=True, exist_ok=True)
# CANDIDATES_DIR.mkdir(parents=True, exist_ok=True) # Ensure base exists if needed elsewhere

# === helpers ===
def _timestamp_dir(prefix: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    # Store candidates directly under MODELS_ROOT with timestamp
    d = MODELS_ROOT / f"{prefix}-{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _load_metrics(p: Path) -> Dict[str, Any]:
    """Loads metrics.json, checking target dir then parent if symlink."""
    f = p / "metrics.json"
    # If p is a symlink (like 'current'), check its target dir first
    if p.is_symlink():
         target_dir = p.resolve()
         f_target = target_dir / "metrics.json"
         if f_target.exists():
              f = f_target
         # Fallback check: Sometimes metrics might be in the parent of the target (the timestamped dir)
         elif (target_dir.parent / "metrics.json").exists():
              f = target_dir.parent / "metrics.json"

    # If still not found, check original path (in case it's not a symlink)
    if not f.exists():
         f_orig = p / "metrics.json"
         if f_orig.exists():
              f = f_orig
         else:
              print(f"[*] Metrics file not found near {p}")
              return {}
    try:
        print(f"[*] Loading metrics from {f}")
        return json.load(f.open())
    except Exception as e:
        print(f"[!] Error loading metrics from {f}: {e}")
        return {}

def _save_metrics(p: Path, metrics: Dict[str, Any]) -> None:
    try:
        filepath = p / "metrics.json"
        print(f"[*] Saving metrics to {filepath}")
        filepath.write_text(json.dumps(metrics, indent=2))
    except Exception as e:
        print(f"[!] Error saving metrics to {p}: {e}")

def _promote(new_dir: Path) -> None:
    # (Implementation remains the same - handles symlink atomically)
    tmp = CURRENT_SYMLINK.with_suffix(".tmp")
    print(f"[*] Attempting to promote candidate directory: {new_dir}")
    if tmp.exists() or tmp.is_symlink():
        print(f"[*] Removing existing temporary symlink: {tmp}")
        tmp.unlink()
    print(f"[*] Creating temporary symlink {tmp} -> {new_dir.resolve()}") # Use resolve for absolute path
    if not new_dir.exists():
         raise FileNotFoundError(f"Target directory for promotion does not exist: {new_dir}")
    # Create symlink relative to its parent directory if possible, else absolute
    try:
        relative_target = os.path.relpath(new_dir.resolve(), start=CURRENT_SYMLINK.parent)
        tmp.symlink_to(relative_target, target_is_directory=True)
        print(f"[*] Created relative symlink")
    except ValueError: # Target is on a different drive (unlikely in Docker)
        tmp.symlink_to(new_dir.resolve(), target_is_directory=True)
        print(f"[*] Created absolute symlink")

    print(f"[*] Atomically replacing {CURRENT_SYMLINK} with {tmp}")
    tmp.replace(CURRENT_SYMLINK) # Atomic rename/replace
    print(f"[+] Promotion successful. '{CURRENT_SYMLINK.name}' now points to '{new_dir.name}'")


def _is_better(new: Dict[str, Any], cur: Dict[str, Any]) -> bool:
    """
    Compares metrics, prioritizing aggregated metrics if available.
    Focuses on futures accuracy/loss for now, adapt if volatility metrics are added.
    """
    def pick(m: Dict[str, Any]) -> tuple:
        # Try aggregated metrics first
        # Use primary symbol (e.g. BTC) futures metrics as main driver if agg missing
        primary_acc = m.get("agg_fusion_val_accuracy", m.get("primary_fusion_val_accuracy", 0.0))
        primary_loss = m.get("agg_fusion_val_logloss", m.get("primary_fusion_val_logloss", float('inf')))

        # Lower loss is better, higher accuracy is better
        # Combine: prioritize lower loss, then higher accuracy
        # Convert loss to negative for maximization comparison
        return (primary_acc, -primary_loss) # Tuple comparison works element-wise

    new_k = pick(new)
    cur_k = pick(cur)
    # Ensure metrics are valid (not inf/-inf)
    new_k_valid = tuple(v for v in new_k if math.isfinite(v))
    cur_k_valid = tuple(v for v in cur_k if math.isfinite(v))

    print(f"[*] Comparing metrics: New={new_k_valid} vs Current={cur_k_valid}")

    # Check if new metrics are valid and non-empty
    if not new_k_valid:
         print("[*] Candidate metrics are invalid or empty. Cannot compare.")
         return False

    # Promote if current metrics are invalid/empty or if new metrics are strictly better
    is_strictly_better = not cur_k_valid or (new_k_valid > cur_k_valid and any(abs(a - b) > 1e-6 for a, b in zip(new_k_valid, cur_k_valid)))

    print(f"[*] Is candidate strictly better? {is_strictly_better}")
    return is_strictly_better

def _export_prom_metrics(stage: str, ok: bool, new_m: Dict[str, Any], cur_m: Dict[str, Any]) -> None:
    # (Implementation remains the same - handles metrics export)
    if METRICS is None: return
    try:
        METRICS.train_runs_total.labels(stage=stage, status="ok" if ok else "fail").inc()
        for k, v in (new_m or {}).items():
            if isinstance(v, (int, float)) and math.isfinite(v):
                METRICS.train_metric_gauge.labels(metric=k, model="candidate").set(float(v))
        for k, v in (cur_m or {}).items():
            if isinstance(v, (int, float)) and math.isfinite(v):
                METRICS.train_metric_gauge.labels(metric=k, model="current").set(float(v))
    except Exception as e: print(f"[!] Error exporting Prometheus metrics: {e}")

# === Import evaluator ===
# Ensure evaluate.py is updated to handle the structure and find models
try:
    from app.ml.evaluate import evaluate_model_dir
except ImportError as e:
    print(f"[!!!] Could not import evaluate_model_dir: {e}")
    # Define a dummy function to prevent crashes, but evaluation won't work
    def evaluate_model_dir(model_root: str, *args, **kwargs) -> Dict[str, Any]:
        print(f"[!!!] Using dummy evaluate_model_dir for {model_root}. Evaluation skipped!")
        return {"error": 1.0, "reason": "Evaluator import failed"}


# --- Main Retraining Task (MODIFIED) ---
@shared_task(base=BaseTaskWithRetry, name="tasks.retrain_models")
@singleton_lock("retrain.models.weekly", ttl=6*60*60)
def retrain_models_task(symbols: List[str] = ["BTC/USDT", "ETH/USDT"]) -> Dict[str, Any]:
    """
    Full re-train using the consolidated script:
      - trains futures (price) AND spot (volatility) models into a candidate dir
      - evaluates the entire candidate dir
      - compares vs current
      - promotes if better
    """
    stage = "retrain"
    candidate_dir = _timestamp_dir("candidate") # Main timestamped dir for this run
    out = {"candidate_dir": str(candidate_dir), "promoted": False, "reason": "", "trained_symbols": {}}
    python_executable = sys.executable
    train_script_path = Path(__file__).parent.parent / "ml" / "train_model.py"

    print(f"[*] Starting retraining run. Candidate dir: {candidate_dir}")
    print(f"[*] Target symbols: {symbols}")
    print(f"[*] Using trainer script: {train_script_path}")

    if not train_script_path.exists():
         out["reason"] = f"Training script not found: {train_script_path}"
         print(f"[!!!] {out['reason']}"); _export_prom_metrics(stage, False, {}, {}); return out

    training_successful_overall = True
    # --- Loop through each symbol ---
    for symbol in symbols:
        print(f"\n--- Training {symbol} ---")
        symbol_filename_part = symbol.replace("/", "_")
        # Initialize status for this symbol
        out["trained_symbols"][symbol] = {"futures_price": "failed", "spot_volatility": "failed"}

        # Define output directories matching evaluate_model_dir expectation
        spot_dir = candidate_dir / "spot" / symbol_filename_part
        fut_dir = candidate_dir / "futures" / symbol_filename_part
        spot_dir.mkdir(parents=True, exist_ok=True)
        fut_dir.mkdir(parents=True, exist_ok=True)

        symbol_fut_failed = False
        symbol_vol_failed = False

        # --- 1. Train Futures Price Model ---
        try:
            print(f"[*] Training Futures PRICE model ({symbol}) -> {fut_dir}")
            cmd_futures = [
                python_executable, str(train_script_path),
                "--symbol", symbol,
                "--table", "futures_market_data",
                "--prediction-target", "next_close", # Explicitly set target
                "--output-dir", str(fut_dir),
                # Add common args if needed: e.g., "--epochs", "20"
            ]
            print(f"[*] Running: {' '.join(cmd_futures)}")
            result_futures = subprocess.run(cmd_futures, capture_output=True, text=True, check=True, timeout=1800)
            print(f"[+] Futures price training completed for {symbol}.")
            # print(f"Stdout (last 500 chars):\n...{result_futures.stdout[-500:]}") # Optional: reduce verbosity
            if result_futures.stderr: print(f"[!] Stderr:\n{result_futures.stderr}")
            # Verify artifact existence
            if not (fut_dir / "lstm_best.pt").exists() or not (fut_dir / "manifest.json").exists():
                 raise RuntimeError("Futures model/manifest artifact missing after successful run.")
            out["trained_symbols"][symbol]["futures_price"] = "success"
        except subprocess.TimeoutExpired as e:
            print(f"[!] Futures PRICE training timed out for {symbol}: {e.timeout}s"); symbol_fut_failed = True
            out["trained_symbols"][symbol]["futures_price"] = "failed: Timeout"
        except subprocess.CalledProcessError as e:
            print(f"[!] Futures PRICE training script failed for {symbol} (Code: {e.returncode}):"); symbol_fut_failed = True
            print(f"Stderr:\n{e.stderr[-1000:]}"); print(f"Stdout:\n{e.stdout[-1000:]}") # Print last part of logs
            out["trained_symbols"][symbol]["futures_price"] = f"failed: Exit Code {e.returncode}"
        except Exception as e:
             print(f"[!] Unexpected error during Futures PRICE training ({symbol}): {e}"); symbol_fut_failed = True
             out["trained_symbols"][symbol]["futures_price"] = f"error: {e}"

        # --- 2. Train Spot Volatility Model ---
        # Decide which base table is best for volatility calculation (spot or futures underlying)
        # Using futures_market_data might give more consistent data if available
        vol_base_table = "futures_market_data" # Or "market_data"
        try:
            print(f"[*] Training Spot VOLATILITY model ({symbol}) -> {spot_dir}")
            cmd_volatility = [
                python_executable, str(train_script_path),
                "--symbol", symbol,
                "--table", vol_base_table, # Use appropriate table for vol calc
                "--prediction-target", "realized_volatility", # Set target
                "--output-dir", str(spot_dir),
                # Add specific args for volatility if needed: e.g., "--rv-window", "30"
            ]
            print(f"[*] Running: {' '.join(cmd_volatility)}")
            result_volatility = subprocess.run(cmd_volatility, capture_output=True, text=True, check=True, timeout=1800)
            print(f"[+] Spot volatility training completed for {symbol}.")
            # print(f"Stdout (last 500 chars):\n...{result_volatility.stdout[-500:]}")
            if result_volatility.stderr: print(f"[!] Stderr:\n{result_volatility.stderr}")
             # Verify artifact existence
            if not (spot_dir / "lstm_best.pt").exists() or not (spot_dir / "manifest.json").exists():
                 raise RuntimeError("Volatility model/manifest artifact missing after successful run.")
            out["trained_symbols"][symbol]["spot_volatility"] = "success"
        except subprocess.TimeoutExpired as e:
            print(f"[!] Spot VOLATILITY training timed out for {symbol}: {e.timeout}s"); symbol_vol_failed = True
            out["trained_symbols"][symbol]["spot_volatility"] = "failed: Timeout"
        except subprocess.CalledProcessError as e:
            print(f"[!] Spot VOLATILITY training script failed for {symbol} (Code: {e.returncode}):"); symbol_vol_failed = True
            print(f"Stderr:\n{e.stderr[-1000:]}"); print(f"Stdout:\n{e.stdout[-1000:]}")
            out["trained_symbols"][symbol]["spot_volatility"] = f"failed: Exit Code {e.returncode}"
        except Exception as e:
             print(f"[!] Unexpected error during Spot VOLATILITY training ({symbol}): {e}"); symbol_vol_failed = True
             out["trained_symbols"][symbol]["spot_volatility"] = f"error: {e}"

        # --- Update overall success flag ---
        if symbol_fut_failed or symbol_vol_failed:
             training_successful_overall = False
             # Optional: Stop processing further symbols if one fails
             # print(f"[!!!] Stopping retraining run due to failure for {symbol}.")
             # break

    # --- End Loop for Symbols ---

    if not training_successful_overall:
         out["reason"] = "One or more training runs failed. Skipping evaluation and promotion."
         print(f"[!!!] {out['reason']}")
         _export_prom_metrics(stage, False, {}, {})
         # Optionally clean up partially populated candidate dir
         # try: shutil.rmtree(candidate_dir); print(f"[*] Cleaned up failed candidate dir: {candidate_dir}")
         # except Exception as clean_e: print(f"[!] Error cleaning up failed candidate dir: {clean_e}")
         return out

    # --- 2) Evaluate candidate directory (contains all symbols/types trained) ---
    print(f"\n[*] Evaluating candidate directory: {candidate_dir}")
    cand_metrics = {}
    try:
        # evaluate_model_dir should handle the spot/<SYM>/ and futures/<SYM>/ structure
        cand_metrics = evaluate_model_dir(str(candidate_dir)) # Pass eval start/end dates if needed
        if not cand_metrics or cand_metrics.get("error") == 1.0:
             # Handle case where evaluation itself fails or returns no metrics
             eval_reason = cand_metrics.get("reason", "Evaluation produced no valid metrics")
             print(f"[!] Evaluation failed or returned empty metrics. Reason: {eval_reason}")
             raise ValueError(eval_reason)
        _save_metrics(candidate_dir, cand_metrics) # Save aggregated metrics to the timestamped dir
        print(f"[+] Candidate metrics calculated: {json.dumps(cand_metrics, indent=2)}")
    except Exception as e:
        print(f"[!] Evaluation failed: {e}")
        out["reason"] = f"Evaluation error: {e}"
        _export_prom_metrics(stage, False, {}, {}) # Export failure metric
        # Clean up candidate dir as evaluation failed? Optional.
        # shutil.rmtree(candidate_dir)
        return out

    # --- 3) Load current metrics ---
    print(f"[*] Loading metrics from current model directory: {CURRENT_SYMLINK}")
    cur_metrics = {}
    if CURRENT_SYMLINK.is_symlink(): # Check if it's a symlink before resolving
         current_target_dir = CURRENT_SYMLINK.resolve()
         cur_metrics = _load_metrics(current_target_dir) # Load from target dir
         print(f"[+] Current metrics: {json.dumps(cur_metrics, indent=2)}")
    else:
        print("[*] No current model symlink found or it's not a symlink.")


    # --- 4) Compare & maybe promote ---
    print("[*] Comparing candidate metrics vs current metrics...")
    try:
        # Use the _is_better function (adapt its logic based on key metrics if needed)
        should_promote = _is_better(cand_metrics, cur_metrics)

        if not cur_metrics: # Always promote if no current model exists
            out["reason"] = "promoted: first model trained"
            should_promote = True
            print(f"[*] {out['reason']}")

        if should_promote:
            print("[*] Candidate is better or no current model exists. Promoting...")
            _promote(candidate_dir) # Promote the entire timestamped candidate directory
            out["promoted"] = True
            if cur_metrics: out["reason"] = "promoted: candidate metrics are better than current"
        else:
            print("[*] Candidate is not strictly better than current. Keeping current model.")
            out["promoted"] = False
            out["reason"] = "kept current: candidate metrics not strictly better"
            # Optional: Clean up non-promoted candidate directory
            # try: shutil.rmtree(candidate_dir); print(f"[*] Cleaned up non-promoted candidate dir: {candidate_dir}")
            # except Exception as clean_e: print(f"[!] Error cleaning up non-promoted candidate dir: {clean_e}")

    except Exception as e:
        print(f"[!] Promotion failed: {e}")
        out["reason"] = f"Promotion error: {e}"
        _export_prom_metrics(stage, False, cand_metrics, cur_metrics) # Export failure
        return out

    # Export final metrics to Prometheus
    _export_prom_metrics(stage, True, cand_metrics, cur_metrics)
    print(f"\n[*] Retraining task finished. Promoted: {out['promoted']}. Reason: {out['reason']}")
    return out


# --- Evaluate Current Models Task (Remains the same) ---
@shared_task(base=BaseTaskWithRetry, name="tasks.evaluate_current_models")
@singleton_lock("eval.models.6h", ttl=2*60*60)
def evaluate_current_models_task() -> Dict[str, Any]:
    # (Implementation remains the same)
    stage = "evaluate_current"
    print(f"[*] Evaluating current models linked at: {CURRENT_SYMLINK}")
    if not CURRENT_SYMLINK.exists() or not CURRENT_SYMLINK.is_symlink():
        print("[!] No current model symlink found."); _export_prom_metrics(stage, False, {}, {}); return {"ok": False, "reason": "no current model symlink found"}
    current_target_dir = CURRENT_SYMLINK.resolve()
    print(f"[*] Current model target directory: {current_target_dir}")
    try:
        current_metrics = evaluate_model_dir(str(current_target_dir))
        if not current_metrics or current_metrics.get("error") == 1.0: raise ValueError(current_metrics.get("reason","Evaluation produced no metrics."))
        _save_metrics(current_target_dir, current_metrics); print(f"[+] Current evaluation metrics: {json.dumps(current_metrics, indent=2)}")
        _export_prom_metrics(stage, True, current_metrics, current_metrics)
        return {"ok": True, "metrics": current_metrics, "model_dir": str(current_target_dir)}
    except Exception as e:
        print(f"[!] Error evaluating current models: {e}"); _export_prom_metrics(stage, False, {}, {})
        return {"ok": False, "reason": str(e), "model_dir": str(current_target_dir)}

# --- Sentiment Fusion Task Placeholder (Remains the same) ---
@shared_task(base=BaseTaskWithRetry, name="tasks.run_sentiment_fusion")
@singleton_lock("sentiment.fusion.weekly", ttl=4*60*60)
def task_sentiment_fusion(hours_window: int = 6):
    """
    Celery task wrapper that calls the actual sentiment fusion logic.
    Uses the name 'tasks.run_sentiment_fusion' expected by the schedule.
    """
    if not _HAVE_SENTIMENT_FUSION:
        logger.error("[!!!] Cannot run Sentiment Fusion: Logic import failed.")
        raise ImportError("Sentiment Fusion logic could not be imported during task execution.")
    else:
        logger.info(f"[*] Calling execute_sentiment_fusion_logic (hours_window={hours_window})...")
        try:
            execute_sentiment_fusion_logic(hours_window=hours_window) # Call the imported logic
            logger.info("[+] Sentiment Fusion logic executed successfully.")
        except Exception as e:
            logger.error(f"[!!!] Error executing sentiment fusion logic: {e}", exc_info=True)
            raise # Re-raise the exception to mark the Celery task as failed
# Import necessary functions from your ML scripts
try:
    # Use the logic from scheduler_retrain for incremental update
    from app.ml.self_evolve.scheduler_retrain import incremental_update
    _HAVE_INCREMENTAL = True
except ImportError as e:
    print(f"[!!!] Could not import incremental_update: {e}")
    _HAVE_INCREMENTAL = False

try:
    # Import the feedback loop function
    from app.ml.self_evolve.feedback_loop import adaptive_threshold_update
    _HAVE_FEEDBACK = True
except ImportError as e:
    print(f"[!!!] Could not import adaptive_threshold_update: {e}")
    _HAVE_FEEDBACK = False

# --- XGBoost Training Task ---
@shared_task(base=BaseTaskWithRetry, name="tasks.train_xgboost")
@singleton_lock("retrain.xgb.weekly", ttl=3*60*60)
def train_xgboost_task(symbols: List[str] = ["BTC/USDT", "ETH/USDT"], table: str = "futures_market_data"):
    """Runs the standalone xgboost_train.py script via subprocess for specified symbols."""
    logger.info(f"[*] Starting XGBoost training task for symbols: {symbols}")
    python_executable = sys.executable
    script_path = Path(__file__).parent.parent / "ml" / "xgboost_train.py"
    results = {}
    overall_success = True

    if not script_path.exists():
        logger.error(f"[!!!] XGBoost training script not found: {script_path}")
        return {"error": "script_not_found"}

    for symbol in symbols:
        symbol_filename_part = symbol.replace("/", "_")
        # Define output dir consistent with retrain_models_task structure if possible
        # Or let xgboost_train.py handle its default output based on artifacts.py
        # output_dir = Path("/app/models/staging") / "xgboost" / symbol_filename_part # Example staging path
        # output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"--- Training XGBoost for {symbol} ---")
        try:
            cmd = [
                python_executable, str(script_path),
                "--symbol", symbol,
                "--base-table", table,
                # Add other relevant args for xgboost_train.py if needed (e.g., --output-dir)
                # "--output-dir", str(output_dir),
            ]
            logger.info(f"[*] Running: {' '.join(cmd)}")
            # Use a reasonable timeout
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1800) # 30 min timeout
            logger.info(f"[+] XGBoost training completed for {symbol}.")
            if result.stderr: logger.warning(f"[!] Stderr for {symbol}:\n{result.stderr}")
            results[symbol] = "success"
        except subprocess.TimeoutExpired as e:
            logger.error(f"[!!!] XGBoost training timed out for {symbol}: {e.timeout}s")
            results[symbol] = "failed: Timeout"
            overall_success = False
        except subprocess.CalledProcessError as e:
            logger.error(f"[!!!] XGBoost training script failed for {symbol} (Code: {e.returncode}):")
            logger.error(f"Stderr:\n{e.stderr[-1000:]}")
            logger.error(f"Stdout:\n{e.stdout[-1000:]}")
            results[symbol] = f"failed: Exit Code {e.returncode}"
            overall_success = False
        except Exception as e:
             logger.error(f"[!!!] Unexpected error during XGBoost training ({symbol}): {e}", exc_info=True)
             results[symbol] = f"error: {e}"
             overall_success = False

    return {"overall_status": "success" if overall_success else "failed", "details": results}

# --- Fusion Head Training Task ---
@shared_task(base=BaseTaskWithRetry, name="tasks.train_fusion_head")
@singleton_lock("retrain.fusion.weekly", ttl=3*60*60)
def train_fusion_head_task(symbols: List[str] = ["BTC/USDT", "ETH/USDT"], table: str = "futures_market_data"):
    """Runs the fusion_model.py script via subprocess to train the fusion head."""
    logger.info(f"[*] Starting Fusion Head training task for symbols: {symbols}")
    python_executable = sys.executable
    script_path = Path(__file__).parent.parent / "ml" / "fusion_model.py"
    results = {}
    overall_success = True

    if not script_path.exists():
        logger.error(f"[!!!] Fusion Head training script not found: {script_path}")
        return {"error": "script_not_found"}

    for symbol in symbols:
        logger.info(f"--- Training Fusion Head for {symbol} ---")
        try:
            cmd = [
                python_executable, str(script_path),
                "--symbol", symbol,
                "--base-table", table,
                # Add other relevant args for fusion_model.py if needed
            ]
            logger.info(f"[*] Running: {' '.join(cmd)}")
            # Fusion training might be quicker
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1200) # 20 min timeout
            logger.info(f"[+] Fusion Head training completed for {symbol}.")
            if result.stderr: logger.warning(f"[!] Stderr for {symbol}:\n{result.stderr}")
            results[symbol] = "success"
        except subprocess.TimeoutExpired as e:
            logger.error(f"[!!!] Fusion Head training timed out for {symbol}: {e.timeout}s")
            results[symbol] = "failed: Timeout"
            overall_success = False
        except subprocess.CalledProcessError as e:
            logger.error(f"[!!!] Fusion Head training script failed for {symbol} (Code: {e.returncode}):")
            logger.error(f"Stderr:\n{e.stderr[-1000:]}")
            logger.error(f"Stdout:\n{e.stdout[-1000:]}")
            results[symbol] = f"failed: Exit Code {e.returncode}"
            overall_success = False
        except Exception as e:
             logger.error(f"[!!!] Unexpected error during Fusion Head training ({symbol}): {e}", exc_info=True)
             results[symbol] = f"error: {e}"
             overall_success = False

    return {"overall_status": "success" if overall_success else "failed", "details": results}

# --- Incremental Update Task ---
@shared_task(base=BaseTaskWithRetry, name="tasks.incremental_model_update")
@singleton_lock("incremental.update.4h", ttl=90*60)
def incremental_model_update_task(symbols: List[str] = ["BTC/USDT", "ETH/USDT"]):
    """Performs an incremental update/fine-tuning of existing models."""
    if not _HAVE_INCREMENTAL:
        logger.warning("[!] Incremental update function not available. Skipping task.")
        return {"status": "skipped", "reason": "Import failed"}

    logger.info(f"[*] Starting incremental model update task for symbols: {symbols}")
    results = {}
    for symbol in symbols:
        try:
            logger.info(f"--- Incremental update for {symbol} ---")
            # Call the imported function directly
            result_dict = incremental_update(symbol)
            logger.info(f"[+] Incremental update finished for {symbol}: {result_dict}")
            results[symbol] = result_dict
        except Exception as e:
            logger.error(f"[!!!] Error during incremental update for {symbol}: {e}", exc_info=True)
            results[symbol] = {"error": str(e)}

    return results

# --- Feedback Loop Task ---
@shared_task(base=BaseTaskWithRetry, name="tasks.feedback_adaptive_threshold")
def feedback_adaptive_threshold_task(symbols: List[str] = ["BTC/USDT", "ETH/USDT"]):
    """Runs the adaptive threshold update based on trade performance."""
    if not _HAVE_FEEDBACK:
        logger.warning("[!] Adaptive threshold function not available. Skipping task.")
        return {"status": "skipped", "reason": "Import failed"}

    logger.info(f"[*] Starting adaptive threshold update task for symbols: {symbols}")
    results = {}
    for symbol in symbols:
        try:
            logger.info(f"--- Updating threshold for {symbol} ---")
            # Call the imported function
            new_cutoff = adaptive_threshold_update(symbol)
            logger.info(f"[+] Adaptive threshold update finished for {symbol}. New cutoff: {new_cutoff:.4f}")
            results[symbol] = {"new_cutoff": new_cutoff}
        except Exception as e:
            logger.error(f"[!!!] Error during adaptive threshold update for {symbol}: {e}", exc_info=True)
            results[symbol] = {"error": str(e)}
    return results

# --- Logger Setup (ensure logger is available globally in this file) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)