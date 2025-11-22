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
import logging
from app.tasks.rt_checks import run_all_regime_checks
from celery.schedules import crontab
from celery.signals import (
    worker_ready,
    worker_shutdown,
    task_prerun,
    task_postrun,
    task_failure,
    after_setup_logger,
)
try:
    import redis  # type: ignore
except Exception:  # noqa: BLE001
    redis = None

from app.celery_app.app import celery_app
from app.core.logging_setup import init_logging  # noqa: F401 (side effect)
from app.core.config import settings
from app.infra.alerts import (
    alert_task_failure,
    alert_worker_event,
    setup_telegram_logging,
    alert_info,
)
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

# Sentiment fusion (L2 sentiment)
from app.tasks.sentiment_fusion_collector import run_sentiment_fusion

# ML training tasks (The 10-Model Brain Architecture)
from app.tasks.training_tasks import (
    retrain_all_core_models,      # Layer 1: TFT, TCN, TST, XGB
    retrain_experts,              # Layer 3: Options, Macro/On-chain
    retrain_fusion,               # Layer 2: Ensemble, DecisionNet
    retrain_llm_narrative_model,  # Layer 4: Meta
    retrain_rl_agent,             # Layer 4: Execution
)

# Options derived metrics (per symbol)
from app.tasks.options_metrics_collector import (
    calculate_options_derived_metrics_task,
)

# --- Side-effect imports to ensure Celery registers ALL task names ---
# (These modules define @celery_app.task(...) but are not otherwise imported.)
import app.tasks.sentiment_collector          # defines tasks.collect_sentiment_all_sources & tasks.run_all_sentiment_collectors
import app.tasks.sentiment_scorer            # defines tasks.score_headlines_task & app.tasks.sentiment_scorer.run_sentiment_scorer
import app.tasks.orderbook_collector         # defines tasks.collect_orderbook_snapshot
import app.tasks.onchain_collector           # defines tasks.collect_onchain_data
import app.tasks.cross_asset_corre_collector # defines tasks.run_cross_asset_corre
import app.tasks.github_collector            # defines tasks.collect_github_activity
import app.tasks.funding_collector           # defines tasks.collect_funding_rates
import app.tasks.options_metrics_collector   # defines tasks.calculate_options_derived_metrics

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


# Ensure we autodiscover only our app tasks (kept for completeness)
celery_app.autodiscover_tasks(["app.tasks"])

# ---- Globals (populated by worker_ready) ----
RISK_SETTINGS: Dict[str, models.RiskSettingsSymbol] = {}
LAST_RISK_REFRESH = 0
LAST_TASK_TS = time.time()
IS_READY = False

# Cross-process watchdog state in Redis
REDIS_WATCHDOG_KEY = "mars:last_task_ts"
REDIS_CLIENT = None  # type: ignore[var-annotated]

logger = logging.getLogger(__name__)
# Enable Telegram logging for all logs >= INFO
setup_telegram_logging(level=logging.INFO)
# Optional: one-time bootstrap marker
alert_info(
    "Worker module imported",
    pid=os.getpid(),
    file=__file__,
)


def _get_redis_watchdog_client():
    """
    Lazily initialize a Redis client for the watchdog.

    Uses the same URL as the Celery broker (Redis) so we don't need extra config.
    If redis-py is not installed or connection fails, returns None and the
    watchdog falls back to process-local LAST_TASK_TS.
    """
    global REDIS_CLIENT

    if redis is None:
        # redis-py not installed; nothing to do
        return None

    if REDIS_CLIENT is not None:
        return REDIS_CLIENT

    try:
        client = redis.from_url(settings.CELERY_BROKER_URL)  # type: ignore[arg-type]
        REDIS_CLIENT = client
        logger.info(
            "[Watchdog] Redis client initialized for key %s",
            REDIS_WATCHDOG_KEY,
        )
        return client
    except Exception as e:  # noqa: BLE001
        logger.error("[Watchdog] Failed to init Redis client: %s", e)
        return None

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

    # --- Core market data collection ---

    # Master task: kicks off spot/futures/options collections etc.
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

    # --- Sentiment stack ---

    # Main multi-source sentiment collector (NewsAPI, CryptoPanic, Santiment, etc.)
    sender.add_periodic_task(
        60.0,  # every 60 seconds
        "tasks.run_all_sentiment_collectors",
        name="[Sentiment] Collect from all sources",
    )

    # Headline scorer (LLM/ML sentiment for raw headlines)
    sender.add_periodic_task(
        300.0,  # every 5 minutes
        "app.tasks.sentiment_scorer.run_sentiment_scorer",
        name="[Sentiment] Score headlines",
    )

    # Sentiment fusion → SentimentFusion table
    sender.add_periodic_task(
        crontab(minute="*/30"),
        run_sentiment_fusion.s(),
        name="[Sentiment] Run Sentiment Fusion",
    )

    # --- Other data collectors via registered task names ---

    # Cross-asset correlations vs ETH/DXY/NDX/GOLD
    sender.add_periodic_task(
        300.0,
        "tasks.run_cross_asset_corre",
        name="[Data] Cross-asset correlation",
    )

    # Orderbook snapshots + CDV/imbalance for top symbols
    sender.add_periodic_task(
        10.0,
        "tasks.collect_orderbook_snapshot",
        name="[Data] Orderbook snapshots",
    )

    # On-chain metrics (CoinMetrics + WhaleAlert)
    sender.add_periodic_task(
        300.0,
        "tasks.collect_onchain_data",
        name="[Data] On-chain metrics",
    )

    # Funding rates on perpetuals
    sender.add_periodic_task(
        300.0,
        "tasks.collect_funding_rates",
        name="[Data] Funding rates",
    )

    # GitHub / dev-activity metrics
    sender.add_periodic_task(
        900.0,
        "tasks.collect_github_activity",
        name="[Data] GitHub activity",
    )

    # --- Options-derived metrics per active symbol ---

    # We drive this off RiskSettings so that when you mark the
    # "top 100" symbols as active, they all get options metrics.
    try:
        active_symbols = _get_active_symbols_from_settings()
    except Exception as e:
        logger.error(f"Failed to load active symbols for options metrics: {e}")
        # Safe fallback to at least BTC/ETH
        active_symbols = ["BTC", "ETH"]

    if not active_symbols:
        active_symbols = ["BTC", "ETH"]
    
    # Schedule a derived-metrics ETL per symbol, once per hour.
    for sym in active_symbols:
        sender.add_periodic_task(
            crontab(minute=5, hour="*"),  # hh:05 every hour
            calculate_options_derived_metrics_task.s(symbol=sym),
            name=f"[Options] Derived metrics for {sym}",
        )
        
    sender.add_periodic_task(
            60.0, 
            run_all_regime_checks.s(),
            name=f"[RT] Adapt Regime & Params: {sym}"
        )    
    # --- ML training pipeline (The 10-Model Brain) ---
    # Scheduled to respect the data dependency chain: L1/L3 -> L2 -> L4

    # 1. Layer 1: Core Predictors (TFT, TCN, TST, XGB)
    # Starts at the top of the hour.
    sender.add_periodic_task(
        crontab(minute=0, hour="*/4"),
        retrain_all_core_models.s(),
        name="[ML] L1: Retrain Core Predictors",
    )

    # 2. Layer 3: Domain Experts (Options, Macro)
    # Starts 5 mins after core to distribute load (usually fast).
    # These must complete before Fusion starts.
    sender.add_periodic_task(
        crontab(minute=5, hour="*/4"),
        retrain_experts.s(),
        name="[ML] L3: Retrain Domain Experts",
    )

    # 3. Layer 2: Fusion (Ensemble, DecisionNet)
    # Starts 20 mins in. Expects L1 and L3 training to be done.
    sender.add_periodic_task(
        crontab(minute=20, hour="*/4"),
        retrain_fusion.s(),
        name="[ML] L2: Retrain Fusion Layer",
    )

    # 4. Layer 4: Execution (RL Agent)
    # Starts 40 mins in. Expects L2 fusion to be ready to provide environment state.
    sender.add_periodic_task(
        crontab(minute=40, hour="*/4"),
        retrain_rl_agent.s(),
        name="[ML] L4: Retrain RL Execution Agent",
    )

    # 5. Layer 4: Meta (LLM)
    # Independent. Runs every 12 hours as it's computationally expensive/slow.
    sender.add_periodic_task(
        crontab(minute=10, hour="*/12"),
        retrain_llm_narrative_model.s(),
        name="[ML] L4: Retrain LLM Narrative Specialist",
    )

    logger.info("Periodic tasks configured.")

@after_setup_logger.connect
def configure_celery_logger(logger, *args, **kwargs):
    """
    Called by Celery after it configures its own logging.
    We attach our TelegramLogHandler here so it doesn't get wiped.
    """
    setup_telegram_logging(level=logging.INFO)
    try:
        alert_info(
            "Telegram logging attached",
            celery_logger=logger.name,
            pid=os.getpid(),
        )
    except Exception:
        # Never break worker startup if Telegram misbehaves
        logger.debug("Failed to send Telegram logging attach notice", exc_info=True)

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
    alert_worker_event("Worker ready")

    # ---- Prometheus startup (safe / optional) ----
    prom_enabled = getattr(settings, "PROMETHEUS_ENABLED", False)
    prom_port = int(getattr(settings, "PROMETHEUS_PORT", 9100))

    if prom_enabled:
        prom_thread = threading.Thread(
            target=_start_prometheus, args=(prom_port,), daemon=True
        )
        prom_thread.start()
        logger.info(f"Prometheus metrics server requested on port {prom_port}")
    else:
        logger.info("Prometheus metrics disabled via settings.")

    # Start Watchdog in a background thread
    watchdog_thread = Watchdog(interval_min=15)
    watchdog_thread.start()

    # Load initial risk settings
    _refresh_risk_settings()

    IS_READY = True
    logger.info("Worker initialization complete.")

@worker_shutdown.connect
def on_worker_shutdown(sender, **kwargs):
    logger.warning("Worker shutting down...")
    alert_worker_event("Worker shutdown")


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
        "app.tasks.training_tasks.retrain_experts",  # Updated allowlist
        "app.tasks.training_tasks.retrain_fusion",   # Updated allowlist
        "app.tasks.training_tasks.retrain_llm_narrative_model",
        "app.tasks.training_tasks.retrain_rl_agent",
    ]:
        logger.debug(f"Task {task_name} does not seem to require risk settings.")


@task_postrun.connect
def on_task_postrun(task_id, task, args, kwargs, retval, state, **z):
    """
    After any task runs, update the LAST_TASK_TS for the watchdog.

    IMPORTANT:
    - We update the process-local LAST_TASK_TS (for metrics, logging).
    - We ALSO write a timestamp into Redis so the Watchdog thread
      in the main process can see activity from all forked workers.
    """
    global LAST_TASK_TS

    ts = time.time()
    LAST_TASK_TS = ts
    ts_int = int(ts)

    # existing Prometheus heartbeat
    WORKER_HEARTBEAT.set(ts_int)

    # cross-process heartbeat via Redis
    client = _get_redis_watchdog_client()
    if client is not None:
        try:
            # TTL is just a safety net; key will be constantly refreshed anyway.
            client.set(REDIS_WATCHDOG_KEY, ts_int, ex=24 * 60 * 60)
        except Exception as e:  # noqa: BLE001
            logger.error("[Watchdog] Failed to update Redis heartbeat: %s", e)

@task_failure.connect
def on_task_failure(task_id, exception, args, kwargs, traceback, einfo, **z):
    """
    Global failure handler.
    """
    global LAST_TASK_TS
    LAST_TASK_TS = time.time()
    WORKER_HEARTBEAT.set(int(LAST_TASK_TS))

    task_obj = args[0] if args else None
    task_name = getattr(task_obj, "name", "unknown_task")
    
    # Convert exception to string for checking
    exc_str = str(exception).lower()

    # --- CRITICAL FIX: FILTER NOISY ALERTS ---
    # If we are banned (418) or rate limited (429) or network down,
    # DO NOT send a Telegram alert. Just log locally.
    if "418" in exc_str or "429" in exc_str or "network is unreachable" in exc_str or "timed out" in exc_str:
        logger.warning(f"Task {task_name} failed due to Network/API Limit. Alert suppressed. Error: {exc_str}")
        return
    # -----------------------------------------

    logger.error(
        "Task %s (ID: %s) failed: %s",
        task_name,
        task_id,
        exception,
        exc_info=einfo,
    )

    err_text = f"{exception}\n{traceback}"
    alert_task_failure(str(task_name), str(task_id), err_text)

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
    Gets a list of symbols that are marked as active in risk_settings_symbol.
    This is what drives the “top-100 hourly prediction universe”.
    """
    db = SessionLocal()
    try:
        rows = (
            db.query(models.RiskSettingsSymbol)
              .filter(models.RiskSettingsSymbol.is_active == True)  # noqa: E712
              .order_by(models.RiskSettingsSymbol.symbol)
              .all()
        )
        symbols = [r.symbol for r in rows]
        logger.info(
            "Found %d active symbols for periodic tasks. Sample: %s",
            len(symbols),
            symbols[:10],
        )
        return symbols
    except Exception as e:
        logger.error(f"Failed to load active symbols from risk_settings_symbol: {e}", exc_info=True)
        return []
    finally:
        db.close()
def _get_risk_settings(symbol: str) -> Optional[models.RiskSettingsSymbol]:
    """
    Safely get risk settings for a symbol from the cache.
    """
    return RISK_SETTINGS.get(symbol)


class Watchdog(threading.Thread):
    """
    Restarts the worker if it's been idle for too long.

    Uses a Redis-backed heartbeat so that:
    - Any forked worker process that finishes a task updates the timestamp.
    - The main process Watchdog thread reads the same timestamp.
    """

    def __init__(self, interval_min: int = 15):
        super().__init__()
        self.interval_sec = interval_min * 60
        self.daemon = True
        self.name = "WatchdogThread"
        logger.info(
            "Watchdog initialized: will restart container if idle > %s min.",
            interval_min,
        )

    def _get_last_task_ts(self) -> float:
        """
        Read the last-task timestamp, preferring Redis (cross-process)
        but falling back to the local LAST_TASK_TS if Redis is not available.
        """
        # Start with local fallback
        last_ts = LAST_TASK_TS

        client = _get_redis_watchdog_client()
        if client is not None:
            try:
                raw = client.get(REDIS_WATCHDOG_KEY)
                if raw is not None:
                    try:
                        last_ts = float(raw)
                    except (TypeError, ValueError):
                        logger.warning(
                            "[Watchdog] Invalid last_task_ts in Redis: %r", raw
                        )
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "[Watchdog] Failed to read Redis key %s: %s",
                    REDIS_WATCHDOG_KEY,
                    e,
                )

        return last_ts

    def run(self) -> None:
        logger.info(
            "[Watchdog] Thread started; checking idle time every %ss.",
            self.interval_sec,
        )
        while True:
            time.sleep(self.interval_sec)

            last_ts = self._get_last_task_ts()
            idle_time = time.time() - last_ts

            if idle_time > self.interval_sec:
                APP_RESTARTS.inc()
                logger.critical(
                    "[WATCHDOG] No task completed in %.0fs "
                    "(threshold=%ss). Restarting container NOW.",
                    idle_time,
                    self.interval_sec,
                )
                alert_worker_event(
                    f"Watchdog restart: idle for {idle_time:.0f}s"
                )

                # give logs/alerts a moment to flush
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