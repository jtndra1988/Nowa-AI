# app/tasks/training_tasks.py

import logging
import subprocess
import sys

from celery import chain

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
# L1 Core Models (TFT / TCN / XGB)
# ============================================================================

@celery_app.task(
    name="app.tasks.training_tasks.train_tft_model",
    base=BaseTaskWithRetry,
)
def train_tft_model_task():
    # TODO: point to your real TFT trainer
    return _run_training_script("app.ml.train_tft")


@celery_app.task(
    name="app.tasks.training_tasks.train_tcn_model",
    base=BaseTaskWithRetry,
)
def train_tcn_model_task():
    # TODO: point to your real TCN trainer
    return _run_training_script("app.ml.train_tcn")


@celery_app.task(
    name="app.tasks.training_tasks.train_xgb_model",
    base=BaseTaskWithRetry,
)
def train_xgb_model_task():
    # TODO: point to your real XGB trainer
    return _run_training_script("app.ml.train_xgb")


@celery_app.task(
    name="app.tasks.training_tasks.retrain_all_core_models",
    base=BaseTaskWithRetry,
)
def retrain_all_core_models():
    """
    Retrain all L1 core models sequentially:
      TFT -> TCN -> XGB
    Scheduled every 4h by worker.py.
    """
    logger.info("[Training] Retraining all L1 core models (TFT, TCN, XGB)...")
    chain(
        train_tft_model_task.s(),
        train_tcn_model_task.s(),
        train_xgb_model_task.s(),
    ).apply_async()
    return "Scheduled L1 core model retraining chain."


# ============================================================================
# L2 Ensemble / DecisionNet
# ============================================================================

@celery_app.task(
    name="app.tasks.training_tasks.train_ensemble_model",
    base=BaseTaskWithRetry,
)
def train_ensemble_model_task():
    # TODO: point to your real ensemble trainer
    return _run_training_script("app.ml.train_ensemble")


@celery_app.task(
    name="app.tasks.training_tasks.retrain_ensemble",
    base=BaseTaskWithRetry,
)
def retrain_ensemble():
    """
    Retrain L2 ensemble / DecisionNet.
    Scheduled 15 minutes after L1 models in worker.py.
    """
    logger.info("[Training] Retraining L2 ensemble / DecisionNet...")
    train_ensemble_model_task.apply_async()
    return "Scheduled ensemble retraining."


# ============================================================================
# L1 LLM Narrative Specialist
# ============================================================================

@celery_app.task(
    name="app.tasks.training_tasks.retrain_llm_narrative_model",
    base=BaseTaskWithRetry,
)
def retrain_llm_narrative_model():
    """
    Retrain / refresh the LLM narrative specialist.

    IMPORTANT:
    - Replace the script path below with your actual fine-tuning / RAG update logic.
    - Keeping it as its own module keeps infra clean.
    """
    logger.info("[Training] Retraining LLM Narrative model...")
    # Example placeholder – create this module in your repo:
    #   app/ml/adv/train_llm_narrative_model.py
    return _run_training_script("app.ml.adv.train_llm_narrative_model")


# ============================================================================
# L3 RL Execution Agent
# ============================================================================

@celery_app.task(
    name="app.tasks.training_tasks.retrain_rl_agent",
    base=BaseTaskWithRetry,
)
def retrain_rl_agent():
    """
    Retrain the RL execution agent (PPO policy).

    IMPORTANT:
    - Replace the script path below with your actual RL training pipeline.
    - That script should save model_artifacts/rl_agent_ppo.zip
      so RLAgent picks it up automatically.
    """
    logger.info("[Training] Retraining RL execution agent...")
    # Example placeholder – create this module in your repo:
    #   app/ml/adv/train_rl_agent.py
    return _run_training_script("app.ml.adv.train_rl_agent")


# ============================================================================
# Optional: Full pipeline wrapper
# ============================================================================

@celery_app.task(
    name="app.tasks.training_tasks.run_full_retraining_pipeline",
    base=BaseTaskWithRetry,
)
def run_full_retraining_pipeline():
    """
    Optional one-shot:
      1) L1 core models
      2) L2 ensemble
      3) L3 RL agent
    """
    logger.info("[Training] Starting full retraining pipeline (L1 -> L2 -> L3).")
    pipeline = chain(
        retrain_all_core_models.s(),
        retrain_ensemble.s(),
        retrain_rl_agent.s(),
    )
    pipeline.apply_async()
    return "Scheduled full retraining pipeline."
