# app/tasks/training_tasks.py

import logging
import subprocess
import sys

from celery import chain, group

from app.celery_app.app import celery_app, BaseTaskWithRetry

logger = logging.getLogger(__name__)


def _run_training_script(script_module: str):
    """
    Helper to run a training script as a module with the current interpreter.
    Exits non-zero -> task failure (handled by BaseTaskWithRetry).
    """
    logger.info(f"[Training] Starting training module: {script_module}")
    try:
        process = subprocess.run(
            [sys.executable, "-m", script_module],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"[Training] {script_module} completed successfully.")
        if process.stdout:
            logger.info(f"[Training:{script_module}] STDOUT:\n{process.stdout}")
        if process.stderr:
            logger.debug(f"[Training:{script_module}] STDERR:\n{process.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(
            f"[Training] {script_module} failed "
            f"(code={e.returncode}). STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        )
        raise
    except Exception as e:
        logger.error(f"[Training] Unexpected error in {script_module}: {e}")
        raise


# ============================================================================
# Layer 1: Core Time-Series Models (TFT, TCN, TST, XGB)
# ============================================================================

@celery_app.task(name="app.tasks.training_tasks.train_tft_model", base=BaseTaskWithRetry)
def train_tft_model_task():
    return _run_training_script("app.ml.train_tft")

@celery_app.task(name="app.tasks.training_tasks.train_tcn_model", base=BaseTaskWithRetry)
def train_tcn_model_task():
    return _run_training_script("app.ml.train_tcn")

@celery_app.task(name="app.tasks.training_tasks.train_tst_model", base=BaseTaskWithRetry)
def train_tst_model_task():
    return _run_training_script("app.ml.train_tst")

@celery_app.task(name="app.tasks.training_tasks.train_xgb_model", base=BaseTaskWithRetry)
def train_xgb_model_task():
    return _run_training_script("app.ml.train_xgb")

@celery_app.task(name="app.tasks.training_tasks.retrain_all_core_models", base=BaseTaskWithRetry)
def retrain_all_core_models():
    """
    Retrain all L1 core models. 
    Since they are independent, we can run them in parallel (group) or sequence.
    Sequence is safer for memory.
    """
    logger.info("[Training] Retraining L1 Core Models (TFT, TCN, TST, XGB)...")
    chain(
        train_tft_model_task.s(),
        train_tcn_model_task.s(),
        train_tst_model_task.s(),
        train_xgb_model_task.s(),
    ).apply_async()
    return "Scheduled L1 core model retraining."


# ============================================================================
# Layer 3: Domain Experts (Options, Macro)
# Experts are trained *before* Fusion so fusion sees fresh expert signals.
# ============================================================================

@celery_app.task(name="app.tasks.training_tasks.train_options_expert", base=BaseTaskWithRetry)
def train_options_expert_task():
    return _run_training_script("app.ml.train_options_vol_model")

@celery_app.task(name="app.tasks.training_tasks.train_macro_expert", base=BaseTaskWithRetry)
def train_macro_expert_task():
    return _run_training_script("app.ml.train_macro_onchain_model")

@celery_app.task(name="app.tasks.training_tasks.retrain_experts", base=BaseTaskWithRetry)
def retrain_experts():
    logger.info("[Training] Retraining L3 Domain Experts...")
    chain(
        train_options_expert_task.s(),
        train_macro_expert_task.s(),
    ).apply_async()
    return "Scheduled L3 expert retraining."


# ============================================================================
# Layer 2: Fusion Layer (Ensemble, DecisionNet)
# Depends on L1 and L3 being ready.
# ============================================================================

@celery_app.task(name="app.tasks.training_tasks.train_ensemble_model", base=BaseTaskWithRetry)
def train_ensemble_model_task():
    return _run_training_script("app.ml.train_ensemble")

@celery_app.task(name="app.tasks.training_tasks.train_decision_net", base=BaseTaskWithRetry)
def train_decision_net_task():
    return _run_training_script("app.ml.train_decision_net")

@celery_app.task(name="app.tasks.training_tasks.retrain_fusion", base=BaseTaskWithRetry)
def retrain_fusion():
    """Retrain the Judges: Ensemble Stacker and DecisionNet."""
    logger.info("[Training] Retraining L2 Fusion Models...")
    chain(
        train_ensemble_model_task.s(),
        train_decision_net_task.s(),
    ).apply_async()
    return "Scheduled fusion retraining."


# ============================================================================
# Layer 4: Meta & Execution (LLM, RL)
# ============================================================================

@celery_app.task(name="app.tasks.training_tasks.retrain_llm_narrative_model", base=BaseTaskWithRetry)
def retrain_llm_narrative_model():
    logger.info("[Training] Retraining LLM Narrative model...")
    return _run_training_script("app.ml.train_llm_narrative_model")

@celery_app.task(name="app.tasks.training_tasks.retrain_rl_agent", base=BaseTaskWithRetry)
def retrain_rl_agent():
    logger.info("[Training] Retraining RL execution agent...")
    return _run_training_script("app.ml.train_rl_agent")


# ============================================================================
# Full Pipeline Orchestrator
# ============================================================================

@celery_app.task(
    name="app.tasks.training_tasks.run_full_retraining_pipeline",
    base=BaseTaskWithRetry,
)
def run_full_retraining_pipeline():
    """
    Master sequence:
    1. L1 Core Models + L3 Experts (Base signals)
    2. L2 Fusion (Learns how to combine L1+L3)
    3. L4 RL Agent (Learns policy based on L2 output)
    """
    logger.info("[Training] Starting FULL retraining pipeline (L1/L3 -> L2 -> L4).")
    
    # We can run L1 and L3 in parallel (group), then L2, then L4
    # Using a simple chain for reliability:
    pipeline = chain(
        # 1. Train Predictors
        train_tft_model_task.s(),
        train_tcn_model_task.s(),
        train_tst_model_task.s(),
        train_xgb_model_task.s(),
        
        # 2. Train Experts
        train_options_expert_task.s(),
        train_macro_expert_task.s(),
        
        # 3. Train Fusion (Needs predictors + experts ready)
        train_ensemble_model_task.s(),
        train_decision_net_task.s(),
        
        # 4. Train Execution (Needs fusion ready)
        retrain_rl_agent.s()
        
        # Note: LLM is usually asynchronous/independent, but can be added here if desired
    )
    
    pipeline.apply_async()
    return "Scheduled full 10-model retraining pipeline."