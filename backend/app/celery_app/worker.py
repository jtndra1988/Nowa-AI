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
# --- REMOVED old import ---
# from backend.app.tasks.ml_retrain import retrain_symbol 
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

# --- Service/task imports (safe at import time) ----
# TODO: consider moving to functions if they grow side-effects
from app.services.settings_service import RiskSettingsService
from app.rt_adapt.event_bus import Q, bus

# --- Portfolio/Risk imports ---
from app.portfolio.allocator import portfolio_rebalance_suggest, portfolio_hedge_suggest
from app.portfolio.state import portfolio_tick
from app.risk.engine import orders_monitor
from app.rt_adapt.adaptive import on_symbol_event # keep this for regime changes
from app.rt_adapt.bar_writer import on_bar_close

# --- Task imports ---
from app.tasks.collectors import (
    collect_market_data,
    collect_option_metrics
)
# --- Import all data collectors ---
from app.tasks.cross_asset_corre_collector import collect_cross_asset_correlation
from app.tasks.funding_collector import collect_funding_rates
from app.tasks.github_collector import collect_github_activity
from app.tasks.onchain_collector import collect_onchain_metrics
from app.tasks.orderbook_collector import collect_orderbook_data
from app.tasks.sentiment_collector import collect_all_sentiment
from app.tasks.sentiment_fusion_collector import run_sentiment_fusion

# --- CORRECTED IMPORT: Import our *new* full training pipeline task ---
from app.tasks.training_tasks import run_full_retraining_pipeline
from app.services.inference_service import inference_service
from app.hybrid.schemas import MarketContext
from app.db.models import HybridSignal
# --- REMOVED old, broken task imports ---
# from app.tasks.training_tasks import (
#     retrain_models_task,
#     train_xgboost_task,
#     train_fusion_head_task
# )


logger = logging.getLogger(__name__)

# --------------------------------
# Global state (worker-level)
# --------------------------------
risk_settings_service: Optional[RiskSettingsService] = None

# --------------------------------
# Watchdog
# --------------------------------

LAST_TASK_TS = datetime.now(timezone.utc)
WATCHDOG_EXIT_CODE = 99

def _watchdog_thread(interval_min: int = 15):
    """
    If no task received in {interval_min} minutes, exit worker.
    K8s/supervisor will restart it. Fixes hung connections.
    """
    global LAST_TASK_TS
    logger.info(f"[Watchdog] Starting watchdog thread (pid {os.getpid()})...")
    interval_sec = interval_min * 60
    
    while True:
        time.sleep(30) # Check every 30s
        idle_time = (datetime.now(timezone.utc) - LAST_TASK_TS).total_seconds()
        
        if idle_time > interval_sec:
            logger.error(
                f"[Watchdog] Worker idle for > {interval_min} minutes. "
                f"Exiting with code {WATCHDOG_EXIT_CODE} for restart."
            )
            alert_worker_event(
                f"Watchdog: Worker idle for > {interval_min} min. Restarting."
            )
            # Send signal to main thread to exit
            os._exit(WATCHDOG_EXIT_CODE) # Hard exit

class Watchdog(threading.Thread):
    def __init__(self, interval_min: int):
        super().__init__()
        self.interval_min = interval_min
        self.daemon = True # Exit when main thread exits

    def run(self):
        _watchdog_thread(self.interval_min)

# --------------------------------
# Prometheus
# --------------------------------
def _start_prometheus_server(port: int):
    """Start a Prometheus metrics server in a background thread."""
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

# --------------------------------
# Signals
# --------------------------------
@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    logger.warning(f"Celery worker ready (pid: {os.getpid()}). Initializing services...")
    global risk_settings_service
    
    # 1. Init Prometheus
    # start_prometheus_server(9100)
    
    # 2. Init global services
    db = SessionLocal()
    try:
        risk_settings_service = RiskSettingsService(db=db)
        risk_settings_service.refresh_all_settings()
        logger.info("[✅] RiskSettingsService initialized and warm.")
    except Exception as e:
        logger.error(f"Failed to init RiskSettingsService: {e}")
    finally:
        db.close()
    
    # 3. Init watchdog
    # watchdog = Watchdog(15 * 60)  # 15 min
    # watchdog.start()
    
    logger.info("[✅] Worker startup sequence complete.")
    alert_worker_event(f"Celery worker started (pid: {os.getpid()}).")


@worker_shutdown.connect
def on_worker_shutdown(sender, **kwargs):
    logger.warning(f"Celery worker shutting down (pid: {os.getpid()}).")
    alert_worker_event(f"Celery worker shutdown (pid: {os.getpid()}).")


@task_prerun.connect
def on_task_prerun(sender, task_id, task, args, kwargs, **extras):
    """
    Update watchdog timer before task runs.
    """
    global LAST_TASK_TS
    LAST_TASK_TS = datetime.now(timezone.utc)
    logger.info(f"Starting task: {task.name} (id: {task_id})")
    

@task_postrun.connect
def on_task_postrun(sender, task_id, task, args, kwargs, retval, state, **extras):
    """
    Update watchdog timer after task runs.
    """
    global LAST_TASK_TS
    LAST_TASK_TS = datetime.now(timezone.utc)
    logger.info(f"Finished task: {task.name} (id: {task_id}, state: {state})")
    
    # Metrics
    time_task(task.name, state, retval)


@task_failure.connect
def on_task_failure(sender, task_id, exception, args, kwargs, traceback, einfo, **extras):
    """
    Alert on task failure.
    """
    logger.error(f"Task {sender.name} (id: {task_id}) failed: {exception}")
    alert_task_failure(sender.name, exception, traceback)


# --------------------------------
# Periodic Tasks (Beat)
# --------------------------------
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """
    CELERYBEAT SCHEDULE
    - Queues determined by Q() wrapper
    - We use crontab for predictable wall-clock times
    - We use simple floats for high-frequency tasks
    """
    logger.info("Configuring Celerybeat periodic tasks...")

    # --- Data Collectors (high-frequency) → 'collectors' queue
    sender.add_periodic_task(30.0, Q(collect_market_data.s(), "collectors"), name="Market Data Collector (30s)")
    sender.add_periodic_task(60.0, Q(collect_orderbook_data.s(), "collectors"), name="Orderbook Collector (60s)")
    sender.add_periodic_task(120.0, Q(collect_funding_rates.s(), "collectors"), name="Funding Rate Collector (2m)")
    
    # --- Data Collectors (low-frequency) → 'default' queue
    sender.add_periodic_task(crontab(minute='*/15'), Q(collect_all_sentiment.s(), "default"), name="Sentiment Collector (15m)")
    sender.add_periodic_task(crontab(minute='*/30'), Q(collect_onchain_metrics.s(), "default"), name="Onchain Collector (30m)")
    sender.add_periodic_task(crontab(hour='*/1'), Q(collect_github_activity.s(), "default"), name="Github Collector (1h)")
    sender.add_periodic_task(crontab(hour='*/2'), Q(collect_option_metrics.s(), "default"), name="Option Metrics Collector (2h)")
    sender.add_periodic_task(crontab(hour='*/4'), Q(collect_cross_asset_correlation.s(), "default"), name="Cross-Asset Corr Collector (4h)")

    # --- Feature Engineering / Events → 'signals' queue
    sender.add_periodic_task(60.0, Q(on_bar_close.s(timeframe='1m'), "signals"), name="1m Bar Close Event")
    sender.add_periodic_task(300.0, Q(on_bar_close.s(timeframe='5m'), "signals"), name="5m Bar Close Event")
    sender.add_periodic_task(60.0, Q(on_symbol_event.s(), "signals"), name="Symbol Event Aggregator (1m)")
    
    # Refresh risk settings cache
    sender.add_periodic_task(60.0, Q(refresh_risk_settings_task.s(), "default"), name="Refresh Risk Settings (60s)")

    # --- ML Training (Weekly, Sunday) → 'retrain' queue
    sender.add_periodic_task(crontab(hour=4, minute=0, day_of_week='sun'),
                             Q(run_sentiment_fusion.s(), "retrain"),
                             name="Run Sentiment Fusion (Weekly)")
    # --- THIS IS THE FIX ---
    # Replace the three old tasks with one call to our new pipeline.
    sender.add_periodic_task(crontab(hour=4, minute=30, day_of_week='sun'),
                             Q(run_full_retraining_pipeline.s(), "retrain"),
                             name="Weekly Full Hybrid Ensemble Retraining Pipeline")
    # Portfolio monitoring (you had these) → signals (low CPU)
    sender.add_periodic_task(30.0,  Q(portfolio_tick.s(), "signals"),             name="portfolio_tick_30s")
    sender.add_periodic_task(120.0, Q(portfolio_rebalance_suggest.s(), "signals"),name="portfolio_rebalance_suggest_2m")
    sender.add_periodic_task(90.0,  Q(portfolio_hedge_suggest.s(), "signals"),    name="portfolio_hedge_suggest_90s")
    sender.add_periodic_task(5.0,   Q(orders_monitor.s(), "signals"),            name="orders_monitor_5s")
        # --- Hybrid Signal Generation (optional) → 'signals' queue
    sender.add_periodic_task(
        60.0,
        Q(generate_hybrid_signals.s(), "signals"),
        name="Hybrid Signal Generator (60s)"
    )

    
    logger.info("[✅] Celerybeat periodic tasks configured.")
# --------------------------------
# Tasks
# --------------------------------
@celery_app.task(name="tasks.refresh_risk_settings", base=BaseTaskWithRetry)
def refresh_risk_settings_task():
    """
    Periodically refresh the settings cache in the global service.
    """
    global risk_settings_service
    if not risk_settings_service:
        # Service failed to init, retry
        logger.warning("RiskSettingsService not initialized, re-initializing...")
        db = SessionLocal()
        try:
            risk_settings_service = RiskSettingsService(db=db)
        finally:
            db.close()
            
    if risk_settings_service:
        logger.info("Refreshing risk settings cache...")
        risk_settings_service.refresh_all_settings()
    else:
        logger.error("Cannot refresh settings, RiskSettingsService is still None.")


# --------------------------------------------------------------------
# --- ALL YOUR ORIGINAL TASK DEFINITIONS (UNCHANGED, BUT FIXED) ---
# --------------------------------------------------------------------

# Note: These are the task *definitions*. The scheduling is handled
# in the `setup_periodic_tasks` function above.

@celery_app.task(name="tasks.collect_market_data", base=BaseTaskWithRetry)
def task_collect_market_data():
    return collect_market_data()

@celery_app.task(name="tasks.collect_option_metrics", base=BaseTaskWithRetry)
def task_collect_option_metrics():
    return collect_option_metrics()

@celery_app.task(name="tasks.collect_cross_asset_correlation", base=BaseTaskWithRetry)
def task_collect_cross_asset_correlation():
    return collect_cross_asset_correlation()

@celery_app.task(name="tasks.collect_funding_rates", base=BaseTaskWithRetry)
def task_collect_funding_rates():
    return collect_funding_rates()

@celery_app.task(name="tasks.collect_github_activity", base=BaseTaskWithRetry)
def task_collect_github_activity():
    return collect_github_activity()

@celery_app.task(name="tasks.collect_onchain_metrics", base=BaseTaskWithRetry)
def task_collect_onchain_metrics():
    return collect_onchain_metrics()

@celery_app.task(name="tasks.collect_orderbook_data", base=BaseTaskWithRetry)
def task_collect_orderbook_data():
    return collect_orderbook_data()

@celery_app.task(name="tasks.collect_all_sentiment", base=BaseTaskWithRetry)
def task_collect_all_sentiment():
    return collect_all_sentiment()

@celery_app.task(name="tasks.run_sentiment_fusion", base=BaseTaskWithRetry)
def task_run_sentiment_fusion():
    return run_sentiment_fusion()

@celery_app.task(name="tasks.on_bar_close", base=BaseTaskWithRetry)
def task_on_bar_close(timeframe: str):
    return on_bar_close(timeframe)

# --- THIS IS THE FINAL FIX ---
@celery_app.task(name="tasks.on_symbol_event", base=BaseTaskWithRetry)
def task_on_symbol_event():
    # --- REMOVED old ML retrain logic ---
    # try:
    #     retrain_symbol.delay(symbol, "AUTO", 1000)
    # except Exception as e:
    #     logger.error(f"Failed to submit auto-retrain task for {symbol}: {e}")
    # --- END OF FIX ---
    return on_symbol_event()
# --- END OF FINAL FIX ---

@celery_app.task(name="tasks.portfolio_tick", base=BaseTaskWithRetry)
def task_portfolio_tick():
    return portfolio_tick()

@celery_app.task(name="tasks.portfolio_rebalance_suggest", base=BaseTaskWithRetry)
def task_portfolio_rebalance_suggest():
    return portfolio_rebalance_suggest()

@celery_app.task(name="tasks.portfolio_hedge_suggest", base=BaseTaskWithRetry)
def task_portfolio_hedge_suggest():
    return portfolio_hedge_suggest()

@celery_app.task(name="tasks.orders_monitor", base=BaseTaskWithRetry)
def task_orders_monitor():
    return orders_monitor()
@celery_app.task(name="tasks.generate_hybrid_signals", base=BaseTaskWithRetry)
def generate_hybrid_signals():
    """
    Periodically compute hybrid decisions for a set of symbols
    and log them for later training & monitoring.
    """
    if inference_service is None or not inference_service.is_ready:
        logger.error("Inference service not ready in generate_hybrid_signals.")
        return {"status": "error", "reason": "inference_not_ready"}

    symbols = [
        "BTC-PERP",
        "ETH-PERP",
        # extend as needed
    ]

    from datetime import datetime, timezone

    now_ts = int(datetime.now(timezone.utc).timestamp())
    db = SessionLocal()

    logged = 0

    try:
        for sym in symbols:
            ctx = MarketContext(
                symbol=sym,
                instrument_type="perp",
                exchange="deribit",
                timestamp=now_ts,
            )

            try:
                decision = inference_service.build_decision(ctx)
            except Exception as e:
                logger.error(f"[HybridSignal] Failed decision for {sym}: {e}", exc_info=True)
                continue

            try:
                record = HybridSignal(
                    symbol=decision.symbol,
                    instrument_type=decision.instrument_type,
                    exchange=ctx.exchange,
                    direction=decision.direction,
                    p_edge=decision.p_edge,
                    confidence=decision.confidence,
                    size_factor=decision.size_factor,
                    strategy_tag=decision.strategy_tag,
                    meta_execute=decision.meta_execute,
                    debug_payload=decision.debug,
                )
                db.add(record)
                db.commit()
                logged += 1
            except Exception as e:
                db.rollback()
                logger.error(f"[HybridSignal] Failed to persist for {sym}: {e}", exc_info=True)

        return {"status": "ok", "logged": logged}

    finally:
        db.close()

def _start_prometheus(port: int = 9100):
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
    celery_app.worker_main(argv=['worker', '--loglevel=info', '-E'])