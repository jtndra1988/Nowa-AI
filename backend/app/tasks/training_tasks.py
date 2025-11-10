import logging
import subprocess
import sys
from celery import chain
from app.celery_app.app import celery_app
from app.ml.train_adv import run_advanced_training
logger = logging.getLogger(__name__)

def _run_training_script(script_module: str):
    """
    Helper function to run a training script as a module.
    This ensures it uses the correct Python environment and paths.
    """
    logger.info(f"Starting training for: {script_module}")
    try:
        # We run the script as a module to ensure all imports work
        process = subprocess.run(
            [sys.executable, "-m", script_module],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Successfully ran {script_module}.")
        logger.info(f"Script output:\n{process.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run {script_module}.")
        logger.error(f"Return Code: {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred running {script_module}: {e}")
        raise e

# --- Individual Model Training Tasks ---
@celery_app.task(name="app.tasks.training_tasks.retrain_llm_narrative_model", base=BaseTaskWithRetry)
def retrain_llm_narrative_model():
    """
    Celery task to retrain the L1 LLM Narrative model.
    """
    logger.info("Starting LLM Narrative Model retraining task...")
    # TODO: Add your LLM fine-tuning/retraining logic here
    # Example: from app.ml.adv.llm_narrative_model import run_llm_retraining
    # run_llm_retraining()
    logger.info("LLM Narrative Model retraining task complete (STUB).")

@celery_app.task(name="app.tasks.training_tasks.retrain_rl_agent", base=BaseTaskWithRetry)
def retrain_rl_agent():
    """
    Celery task to retrain the L3 RL Execution Agent.
    """
    logger.info("Starting RL Agent retraining task...")
    # TODO: Add your RL agent retraining logic here
    # Example: from app.ml.adv.rl_execution_agent import run_rl_retraining
    # run_rl_retraining()
    logger.info("RL Agent retraining task complete (STUB).")
    
@celery_app.task(name="tasks.train_tft")
def train_tft_model_task():
    """
    Celery task to train the core Temporal Fusion Transformer model.
    """
    # --- FIX: Changed path to app.ml.adv ---
    return _run_training_script("app.ml.train_adv")

@celery_app.task(name="tasks.train_tcn")
def train_tcn_model_task():
    """
    Celery task to train the Temporal Convolutional Network model.
    """
    # --- FIX: Changed path to app.ml.adv ---
    return _run_training_script("app.ml.train_tcn")

@celery_app.task(name="tasks.train_xgb")
def train_xgb_model_task():
    """
    Celery task to train the XGBoost tabular models.
    """
    # --- FIX: Changed path to app.ml.adv ---
    return _run_training_script("app.ml.train_xgb")

@celery_app.task(name="tasks.train_ensemble")
def train_ensemble_model_task():
    """
    Celery task to train the final hybrid ensemble blender.
    This task should run *after* the individual models are trained.
    """
    # --- FIX: Changed path to app.ml.adv ---
    return _run_training_script("app.ml.train_ensemble")


# --- Main Retraining Pipeline ---

@celery_app.task(name="tasks.run_full_retraining_pipeline")
def run_full_retraining_pipeline():
    """
    Runs the complete, end-to-end model retraining pipeline.
    
    1. Trains the 3 base models (TFT, TCN, XGB).
    2. Trains the final ensemble blender.
    """
    logger.info("Starting full retraining pipeline...")
    
    # Create a chain of tasks.
    # The base models can run in parallel, but the ensemble
    # must wait for them. For simplicity, we run them sequentially.
    # A more advanced setup could use a Celery 'group' for parallel
    # execution of base models.
    
    training_chain = chain(
        train_tft_model_task.s(),
        train_tcn_model_task.s(),
        train_xgb_model_task.s(),
        train_ensemble_model_task.s()
    )
    
    logger.info("Executing training chain: TFT -> TCN -> XGB -> Ensemble")
    training_chain.apply_async()