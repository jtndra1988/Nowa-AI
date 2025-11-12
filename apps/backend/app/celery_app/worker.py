# app/worker.py
"""
Production-ready Celery worker wrapper:
- No DB queries at import time
- Prometheus metrics server on :9100
- Watchdog: restarts container if idle > 15 minutes
- Periodic refresh of risk settings every 60s
- Signals update LAST_TASK_TS so you don't have to touch each task
"""

from datetime import datetime, timezone
import os
import time
import threading
from typing import List, Optional, Any, Dict
import numpy as np
import pandas as pd
from celery import signals
from celery.schedules import crontab
from celery.signals import worker_ready, worker_shutdown, task_prerun, task_postrun, task_failure
from app.core.logging_setup import init_logging
import logging
from app.infra.alerts import alert_task_failure, alert_worker_event
from app.infra.metrics import time_task, WORKER_HEARTBEAT, APP_RESTARTS

# ---- Celery app / task base (no side effects, no DB) ----
from app.celery_app.app import celery_app, BaseTaskWithRetry

# ---- Domain imports (safe at import time) ----
from app.core.config import settings
from app.exchange.adapters import BybitAdapter
from app.db import models
# --- Use SessionLocal for task-level sessions ---
from app.db.database import SessionLocal, get_db

# --- Import Tasks ---
from app.tasks.collectors import (
    run_all_collectors,
    run_funding_collector,
    run_orderbook_collector,
    run_f_g_collector,
    run_macro_collector,
    run_options_collector,
    run_onchain_collector,
    run_github_collector,
    run_cross_asset_corr_collector,
)
from app.tasks.sentiment_scorer import run_sentiment_scorer
from app.tasks.sentiment_collector import run_all_sentiment_collectors
from app.tasks.sentiment_fusion_collector import run_sentiment_fusion

# --- UPDATED: Import ALL training tasks, including LLM and RL ---
from app.tasks.training_tasks import (
    retrain_all_core_models,
    retrain_ensemble,
    retrain_llm_narrative_model,  # <-- ADDED
    retrain_rl_agent              # <-- ADDED
)
# --- END UPDATE ---

from app.services.settings_service import RiskSettingsService

# ---- Globals (populated by worker_ready) ----
# This is a cache of risk settings, refreshed every 60s
RISK_SETTINGS: Dict[str, models.RiskSettings] = {}
LAST_RISK_REFRESH = 0

logger = logging.getLogger(__name__)


# ---- 1. Task Scheduling (Celery Beat) ----

@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """
    This is the Celery Beat scheduler.
    It runs in a separate process from the worker.
    """
    global RISK_SETTINGS
    logger.info("Configuring periodic tasks (Celery Beat)...")

    # --- Data Collection (Your existing tasks) ---
    sender.add_periodic_task(
        crontab(minute='*/15'),  # Every 15 minutes
        run_all_collectors.s(),
        name='[Data] Run All Collectors'
    )
    sender.add_periodic_task(
        crontab(minute='*/5'),  # Every 5 minutes
        run_sentiment_scorer.s(),
        name='[Data] Run Sentiment Scorer'
    )
    sender.add_periodic_task(
        crontab(minute='*/5'),
        run_all_sentiment_collectors.s(),
        name='[Data] Run Sentiment Collectors'
    )
    sender.add_periodic_task(
        crontab(minute='*/30'),
        run_sentiment_fusion.s(),
        name='[Data] Run Sentiment Fusion'
    )
    sender.add_periodic_task(
        crontab(minute='*/60'),
        run_github_collector.s(),
        name='[Data] Run GitHub Collector'
    )
    sender.add_periodic_task(
        crontab(hour='*/1'),
        run_funding_collector.s(),
        name='[Data] Run Funding Collector'
    )
    sender.add_periodic_task(
        crontab(minute='*/15'),
        run_orderbook_collector.s(),
        name='[Data] Run Orderbook Collector'
    )
    sender.add_periodic_task(
        crontab(hour='*/4'),
        run_f_g_collector.s(),
        name='[Data] Run F&G Collector'
    )
    sender.add_periodic_task(
        crontab(hour='*/4'),
        run_macro_collector.s(),
        name='[Data] Run Macro Collector'
    )
    sender.add_periodic_task(
        crontab(hour='*/1'),
        run_options_collector.s(),
        name='[Data] Run Options Collector'
    )
    sender.add_periodic_task(
        crontab(hour='*/4'),
        run_onchain_collector.s(),
        name='[Data] Run On-chain Collector'
    )
    sender.add_periodic_task(
        crontab(hour='*/1'),
        run_cross_asset_corr_collector.s(),
        name='[Data] Run Cross-Asset Corr Collector'
    )

    # --- Core Model Training Pipeline (Every 4 Hours) ---
    # This section is UPDATED to create a 3-stage pipeline.
    
    # 1. (L1) Train core specialists (TFT, TCN, XGB)
    # (Replaces your original 'retrain_all_core_models' task)
    sender.add_periodic_task(
        crontab(minute=0, hour='*/4'),  # Every 4 hours, on the hour
        retrain_all_core_models.s(),
        name='[ML] Retrain All Core L1 Models'
    )
    
    # 2. (L1) Train LLM specialist (runs in parallel with other L1)
    sender.add_periodic_task(
        crontab(minute=0, hour='*/4'),  # Every 4 hours, on the hour
        retrain_llm_narrative_model.s(), # <-- NEW TASK ADDED
        name='[ML] Retrain LLM Narrative L1 Model'
    )

    # 3. (L2) Train the "Judge" (DecisionNet/Ensemble)
    #    (Runs 15 mins later, needs L1 models to be done)
    # (Replaces your original 'retrain_ensemble' task)
    sender.add_periodic_task(
        crontab(minute=15, hour='*/4'),  # Every 4 hours, at 15 past
        retrain_ensemble.s(),
        name='[ML] Retrain L2 Ensemble/DecisionNet'
    )
    
    # 4. (L3) Train the "Actor" (RL Agent)
    #    (Runs 30 mins later, needs L2 model to be done)
    sender.add_periodic_task(
        crontab(minute=30, hour='*/4'),  # Every 4 hours, at 30 past
        retrain_rl_agent.s(), # <-- NEW TASK ADDED
        name='[ML] Retrain L3 RL Execution Agent'
    )
    
    logger.info("Periodic tasks configured.")


# ---- 2. Worker Lifecycle & Monitoring (All your original code) ----

LAST_TASK_TS = time.time()  # Last time a task finished
IS_READY = False  # Is this worker ready to accept tasks?

@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    """
    Runs ONCE when the worker process starts.
    - Start Prometheus
    - Start Watchdog
    - Load initial risk settings
    """
    global IS_READY
    logger.info(f"Worker ready (PID: {os.getpid()}). Initializing...")
    alert_worker_event("Worker ready", "INFO")
    
    # Start Prometheus server in a background thread
    prom_port = settings.PROMETHEUS_PORT
    prom_thread = threading.Thread(
        target=_start_prometheus, args=(prom_port,), daemon=True
    )
    prom_thread.start()

    # Start Watchdog in a background thread
    watchdog_thread = Watchdog(interval_min=15)  # Restarts if idle > 15 min
    watchdog_thread.start()

    # Load initial risk settings
    _refresh_risk_settings()
    
    IS_READY = True
    logger.info("Worker initialization complete.")


@worker_shutdown.connect
def on_worker_shutdown(sender, **kwargs):
    logger.warning("Worker shutting down...")
    alert_worker_event("Worker shutdown", "WARNING")


@task_prerun.connect
def on_task_prerun(task_id, task, args, kwargs, **z):
    """
    Before any task runs, refresh risk settings if cache is stale.
    """
    global LAST_RISK_REFRESH, RISK_SETTINGS
    if (time.time() - LAST_RISK_REFRESH) > 60:  # Cache for 60s
        _refresh_risk_settings()
        
    # --- Your existing pre-run logic ---
    task_name = task.name
    
    # Skip for settings refresh to avoid recursion
    if task_name == 'app.services.settings_service.refresh_settings':
        return
        
    # Check if task requires settings and if they are loaded
    if 'symbol' in kwargs:
        symbol = kwargs['symbol']
        if symbol not in RISK_SETTINGS:
            logger.warning(f"No risk settings found for {symbol}. Attempting refresh.")
            _refresh_risk_settings()
            if symbol not in RISK_SETTINGS:
                logger.error(f"FATAL: No risk settings for {symbol} after refresh. Task may fail.")
                # alert_task_failure(task_name, f"MissingRiskSettings: {symbol}", "") # Optional: alert
                
    elif task_name not in [
        'app.tasks.collectors.run_all_collectors',
        'app.tasks.sentiment_scorer.run_sentiment_scorer',
        'app.tasks.sentiment_collector.run_all_sentiment_collectors',
        'app.tasks.sentiment_fusion_collector.run_sentiment_fusion',
        'app.tasks.collectors.run_github_collector',
        'app.tasks.collectors.run_funding_collector',
        'app.tasks.collectors.run_orderbook_collector',
        'app.tasks.collectors.run_f_g_collector',
        'app.tasks.collectors.run_macro_collector',
        'app.tasks.collectors.run_options_collector',
        'app.tasks.collectors.run_onchain_collector',
        'app.tasks.collectors.run_cross_asset_corr_collector',
        'app.tasks.training_tasks.retrain_all_core_models',
        'app.tasks.training_tasks.retrain_ensemble',
        'app.tasks.training_tasks.retrain_llm_narrative_model', # <-- ADDED
        'app.tasks.training_tasks.retrain_rl_agent' # <-- ADDED
    ]:
        logger.debug(f"Task {task_name} does not seem to require risk settings.")


@task_postrun.connect
def on_task_postrun(task_id, task, args, kwargs, retval, state, **z):
    """
    After any task runs, update the LAST_TASK_TS for the watchdog.
    """
    global LAST_TASK_TS
    LAST_TASK_TS = time.time()
    WORKER_HEARTBEAT.set(int(LAST_TASK_TS))


@task_failure.connect
def on_task_failure(task_id, exception, args, kwargs, traceback, einfo, **z):
    """
    Global failure handler.
    """
    global LAST_TASK_TS
    LAST_TASK_TS = time.time()  # Update heartbeart even on failure
    WORKER_HEARTBEAT.set(int(LAST_TASK_TS))
    
    task_name = "unknown_task"
    try:
        # Try to get name from task object first
        if hasattr(args[0], 'name'):
            task_name = args[0].name
        # Fallback to string name if passed directly (e.g., from beat)
        elif isinstance(args[0], str):
            task_name = args[0]
        elif 'task_name' in kwargs:
            task_name = kwargs['task_name']
    except Exception:
        pass
        
    logger.error(f"Task {task_name} (ID: {task_id}) failed: {exception}")
    alert_task_failure(task_name, exception, traceback)


# ---- 3. Helper Functions (All your original code) ----

def _get_bybit_adapter():
    """
    Utility to get an initialized Bybit adapter.
    TODO: This is still here, but consider moving logic to a shared service
          to avoid direct instantiation in the worker.
    """
    try:
        return BybitAdapter(
            api_key=settings.BYBIT_API_KEY,
            api_secret=settings.BYBIT_API_SECRET,
            base_url=settings.BYBIT_BASE_URL,
        )
    except Exception as e:
        logger.error(f"Failed to initialize BybitAdapter: {e}")
        return None

def _refresh_risk_settings():
    """
    Internal: Refresh the global RISK_SETTINGS cache.
    """
    global RISK_SETTINGS, LAST_RISK_REFRESH
    logger.info("Refreshing risk settings cache...")
    try:
        db = SessionLocal()
        settings_svc = RiskSettingsService(db)
        all_settings = settings_svc.get_all_settings()
        RISK_SETTINGS = {s.symbol: s for s in all_settings}
        LAST_RISK_REFRESH = time.time()
        logger.info(f"Loaded {len(RISK_SETTINGS)} risk setting profiles.")
    except Exception as e:
        logger.error(f"Failed to refresh risk settings: {e}", exc_info=True)
        # Don't overwrite cache on failure, just log
    finally:
        if 'db' in locals() and db:
            db.close()

def _get_active_symbols_from_settings() -> List[str]:
    """
    Gets a list of symbols that are marked as active in the settings.
    """
    global RISK_SETTINGS
    if not RISK_SETTINGS:
        _refresh_risk_settings()
        
    active_symbols = [
        symbol for symbol, settings in RISK_SETTINGS.items() if settings.is_active
    ]
    logger.info(f"Found {len(active_symbols)} active symbols.")
    return active_symbols

def _get_risk_settings(symbol: str) -> Optional[models.RiskSettings]:
    """
    Safely get risk settings for a symbol from the cache.
    """
    return RISK_SETTINGS.get(symbol)


class Watchdog(threading.Thread):
    """
    Restarts the worker if it's been idle for too long.
    This protects against silent freezes (e.g., deadlocks, lost DB connection).
    """
    def __init__(self, interval_min=15):
        super().__init__()
        self.interval_sec = interval_min * 60
        self.daemon = True
        self.name = "WatchdogThread"
        logger.info(
            f"Watchdog initialized: will restart container if idle > {interval_min} min."
        )

    def run(self):
        global LAST_TASK_TS
        while True:
            time.sleep(self.interval_sec)
            idle_time = time.time() - LAST_TASK_TS
            
            if idle_time > self.interval_sec:
                APP_RESTARTS.inc()
                logger.critical(
                    f"[WATCHDOG] No task completed in {idle_time:.0f}s. "
                    f"Restarting container NOW."
                )
                alert_worker_event(
                    f"Watchdog restart: idle for {idle_time:.0f}s", "CRITICAL"
                )
                
                # Give logs a moment to flush
                time.sleep(5)
                
                # Force-quit the container (requires PID 1 / Tini)
                os._exit(1)


def _start_prometheus(port: int):
    """Start a Prometheus metrics server in a background thread."""
    logger.info(f"Attempting to start Prometheus server on port {port}...")
    try:
        from prometheus_client import start_http_server
        start_http_server(port)
        logger.info(f"[Prometheus] Metrics server started on port {port}")
    except ImportError:
        logger.warning(
            "[Prometheus] prometheus_client not found. Skipping metrics server."
        )
    except OSError as e:
        logger.error(
            f"[Prometheus] Failed to start metrics server on port {port}: {e}"
        )
    except Exception as e:
        logger.error(f"[Prometheus] Unknown error starting metrics server: {e}")

# --- Init logging ---
# init_logging() # Ensure this is called if running as a script, but usually handled by Celery boot

# --- Main execution (if needed, though typically run via `celery worker`) ---
if __name__ == "__main__":
    logger.info("Starting Celery worker from __main__...")
    # This is for debugging; in production, you'd run `celery -A app.celery_app.worker worker ...`
    
    # Start Prometheus in a background thread
    prom_thread = threading.Thread(target=_start_prometheus, args=(9100,), daemon=True)
    prom_thread.start()
    
    # Start Watchdog in a background thread
    watchdog_thread = Watchdog(interval_min=15)
    watchdog_thread.start()
    
    # Start the worker
    celery_app.worker_main(argv=['worker', '--loglevel=info'])