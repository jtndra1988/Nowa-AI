# app/celery_app/worker.py
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

from celery.schedules import crontab
from celery.signals import (
    worker_ready,
    worker_shutdown,
    task_prerun,
    task_postrun,
    task_failure,
)

from app.celery_app.app import celery_app
from app.core.logging_setup import init_logging
from app.core.config import settings
from app.infra.alerts import alert_task_failure, alert_worker_event
from app.infra.metrics import time_task, WORKER_HEARTBEAT, APP_RESTARTS
from app.exchange.adapters import BybitAdapter
from app.db import models
from app.db.database import SessionLocal

# --- Import Tasks that are known to exist ---
from app.tasks.collectors import (
    collect_all_assets_task,
    collect_fear_and_greed_task,
    collect_macro_data_task,
    collect_binance_historical_data,
)

# We keep only the fusion function import (this exists in your repo)
from app.tasks.sentiment_fusion_collector import run_sentiment_fusion

# ML training tasks (TFT/TCN/XGB/LLM/RL)
from app.tasks.training_tasks import (
    retrain_all_core_models,
    retrain_ensemble,
    retrain_llm_narrative_model,
    retrain_rl_agent,
)

# ---- Risk Settings service (safe import) ----
try:
    from app.services.settings_service import SettingsService as RiskSettingsService
except ImportError:
    # Fallback stub so worker import never explodes
    class RiskSettingsService:  # type: ignore[no-redef]
        def __init__(self, db):
            self.db = db

        def get_all_settings(self):
            return []


# Ensure we autodiscover only our app tasks
celery_app.autodiscover_tasks(["app.tasks"])

# ---- Globals (populated by worker_ready) ----
RISK_SETTINGS: Dict[str, models.RiskSettingsSymbol] = {}
LAST_RISK_REFRESH = 0
LAST_TASK_TS = time.time()
IS_READY = False

import logging

logger = logging.getLogger(__name__)


# ======================================================================
# 1. Task Scheduling (Celery Beat)
# ======================================================================

@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """
    This is the Celery Beat scheduler.
    It runs in a separate process from the worker.
    """
    global RISK_SETTINGS
    logger.info("Configuring periodic tasks (Celery Beat)...")

    # --- Data Collection ---

    # Master task: kicks off options, futures, sentiment, orderbook, etc
    sender.add_periodic_task(
        crontab(minute="*/15"),  # Every 15 minutes
        collect_all_assets_task.s(),
        name="[Data] Collect All Assets",
    )

    # Fear & Greed index (FNG API)
    sender.add_periodic_task(
        crontab(hour="*/4"),  # Every 4 hours
        collect_fear_and_greed_task.s(),
        name="[Data] Collect Fear & Greed Index",
    )

    # Macro data (DXY via yfinance)
    sender.add_periodic_task(
        crontab(hour="*/4"),  # Every 4 hours
        collect_macro_data_task.s(),
        name="[Data] Collect Macro Data (DXY)",
    )

    # Daily Binance historical backfill (spot & futures OHLCV)
    sender.add_periodic_task(
        crontab(minute=0, hour=0),  # Midnight UTC
        collect_binance_historical_data.s(),
        name="[Data] Collect Binance Historical Data",
    )

    # --- Sentiment stack (use string task names to avoid import crashes) ---

    # Sentiment scorer
    sender.add_periodic_task(
        300.0,  # every 5 minutes
        "app.tasks.sentiment_scorer.run_sentiment_scorer",
        name="sentiment_scorer",
    )

    # Sentiment collector (whatever you named the main collector)
    sender.add_periodic_task(
        60.0,  # every 60 seconds
        "app.tasks.sentiment_collector.run_sentiment_collector",
        name="sentiment_collector",
    )

    # Sentiment fusion (we know run_sentiment_fusion exists)
    sender.add_periodic_task(
        crontab(minute="*/30"),
        run_sentiment_fusion.s(),
        name="[Data] Run Sentiment Fusion",
    )

    # --- Other data collectors via string names (safe) ---

    sender.add_periodic_task(
        300.0,
        "app.tasks.cross_asset_collector.run_cross_asset_collector",
        name="cross_asset_correlation",
    )

    sender.add_periodic_task(
        300.0,
        "app.tasks.funding_collector.run_funding_collector",
        name="funding_collector",
    )

    sender.add_periodic_task(
        900.0,
        "app.tasks.github_collector.run_github_collector",
        name="github_collector",
    )

    sender.add_periodic_task(
        300.0,
        "app.tasks.options_metric_collector.run_options_metric_collector",
        name="options_metric_collector",
    )

    sender.add_periodic_task(
        10.0,
        "app.tasks.orderbook_collector.run_orderbook_collector",
        name="orderbook_collector",
    )

    sender.add_periodic_task(
        300.0,
        "app.tasks.onchain_collector.run_onchain_collector",
        name="onchain_collector",
    )

    # --- ML training pipeline ---

    sender.add_periodic_task(
        crontab(minute=0, hour="*/4"),
        retrain_all_core_models.s(),
        name="[ML] Retrain All Core L1 Models",
    )
    sender.add_periodic_task(
        crontab(minute=0, hour="*/4"),
        retrain_llm_narrative_model.s(),
        name="[ML] Retrain LLM Narrative L1 Model",
    )
    sender.add_periodic_task(
        crontab(minute=15, hour="*/4"),
        retrain_ensemble.s(),
        name="[ML] Retrain L2 Ensemble/DecisionNet",
    )
    sender.add_periodic_task(
        crontab(minute=30, hour="*/4"),
        retrain_rl_agent.s(),
        name="[ML] Retrain L3 RL Execution Agent",
    )

    logger.info("Periodic tasks configured.")


# ======================================================================
# 2. Worker Lifecycle & Monitoring
# ======================================================================

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

    task_name = task.name

    # Skip for settings refresh to avoid recursion
    if task_name == "app.services.settings_service.refresh_settings":
        return

    if "symbol" in kwargs:
        symbol = kwargs["symbol"]
        if symbol not in RISK_SETTINGS:
            logger.warning(f"No risk settings found for {symbol}. Attempting refresh.")
            _refresh_risk_settings()
            if symbol not in RISK_SETTINGS:
                logger.error(
                    f"FATAL: No risk settings for {symbol} after refresh. Task may fail."
                )
    elif task_name not in [
        "app.tasks.collectors.run_all_collectors",
        "app.tasks.sentiment_scorer.run_sentiment_scorer",
        "app.tasks.sentiment_collector.run_all_sentiment_collectors",
        "app.tasks.sentiment_fusion_collector.run_sentiment_fusion",
        "app.tasks.collectors.run_github_collector",
        "app.tasks.collectors.run_funding_collector",
        "app.tasks.collectors.run_orderbook_collector",
        "app.tasks.collectors.run_f_g_collector",
        "app.tasks.collectors.run_macro_collector",
        "app.tasks.collectors.run_options_collector",
        "app.tasks.collectors.run_onchain_collector",
        "app.tasks.collectors.run_cross_asset_corr_collector",
        "app.tasks.training_tasks.retrain_all_core_models",
        "app.tasks.training_tasks.retrain_ensemble",
        "app.tasks.training_tasks.retrain_llm_narrative_model",
        "app.tasks.training_tasks.retrain_rl_agent",
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
    LAST_TASK_TS = time.time()
    WORKER_HEARTBEAT.set(int(LAST_TASK_TS))

    task_name = getattr(args[0], "name", "unknown_task")
    logger.error(f"Task {task_name} (ID: {task_id}) failed: {exception}")
    alert_task_failure(task_name, exception, traceback)


# ======================================================================
# 3. Helper Functions
# ======================================================================

def _get_bybit_adapter():
    """
    Utility to get an initialized Bybit adapter.
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
    finally:
        if "db" in locals() and db:
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


def _get_risk_settings(symbol: str) -> Optional[models.RiskSettingsSymbol]:
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

                time.sleep(5)
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


if __name__ == "__main__":
    logger.info("Starting Celery worker from __main__...")
    prom_thread = threading.Thread(
        target=_start_prometheus, args=(9100,), daemon=True
    )
    prom_thread.start()

    watchdog_thread = Watchdog(interval_min=15)
    watchdog_thread.start()

    celery_app.worker_main(argv=["worker", "--loglevel=info"])
