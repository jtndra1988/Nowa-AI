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

# If you import typed ORM tables directly elsewhere, keep them here
from app.db.models import MarketData, FuturesMarketData, SentimentData  # noqa

# Inference (ensemble runner)
from app.ml.inference_ensemble import run as run_ensemble

# Risk-aware executor (singleton) – should not hit DB at import time
from app.services.execution import ExecutionService
from app.services.order_manager import OrderMonitor, OrderPolicy
# --- Import ALL tasks that will be scheduled ---
# Collectors
from app.tasks.sentiment_collector import collect_sentiment_all_sources
from app.tasks.sentiment_scorer import score_headlines_task
from app.tasks.orderbook_collector import collect_orderbook_snapshot_task
from app.tasks.onchain_collector import collect_onchain_data_task
from app.tasks.github_collector import collect_github_activity_task
from app.tasks.options_metrics_collector import calculate_options_derived_metrics_task
from app.tasks.cross_asset_corre_collector import run_cross_asset_corre_task
import app.tasks.collectors as collectors

# Utils (make sure _utcnow is here if used by tasks)
from app.utils import _norm_futures_symbol, _base_from_pair, _safe_float, _utcnow

# Training & Feedback Tasks
# (Import all tasks defined in training_tasks.py)
from app.tasks.training_tasks import (
    CURRENT_SYMLINK,
    retrain_models_task,
    evaluate_current_models_task,
    run_sentiment_fusion_task_wrapper as task_sentiment_fusion, # Use the wrapper
    train_xgboost_task,
    train_fusion_head_task,
    incremental_model_update_task,
    feedback_adaptive_threshold_task
)
from app.rt_adapt.regime_detector import detect_and_store
from app.rt_adapt.orderflow_monitor import scan_orderflow
from app.rt_adapt.dynamic_params import tune
from app.rt_adapt.cooldown_scheduler import update_cooldowns
from app.rt_adapt.event_bus import set_kv
from app.rt_adapt.adaptive import adaptive_task

from app.portfolio.state import PortfolioState
from app.portfolio.rebalancer import suggest_rebalance
from app.portfolio.hedger import compute_hedge
# === Logger Setup ===
init_logging()  # JSON structured logs from the first log line onward
logger = logging.getLogger(__name__)
logger.info("worker_boot", extra={"event": "boot"})
from app.infra.metrics import APP_RESTARTS
APP_RESTARTS.inc()
# =============================================================================
# Prometheus exporter (starts HTTP server on :9100).
# =============================================================================
try:
    from prometheus_client import start_http_server
    from app.monitoring.metrics import METRICS
    from prometheus_client import Counter, Gauge  # type: ignore

    start_http_server(9100)
    logger.info("[Monitoring] Worker Prometheus exporter on :9100") # Use logger
    _METRICS_OK = True

    # --- Training/promotion specific metrics ---
    TRAIN_MODEL_PROMOTED = Counter(
        "model_promoted_total", "Count of successful model promotions", ["task", "symbol"],
    )
    TRAIN_MODEL_DIRACC = Gauge(
        "model_diracc_latest", "Latest validation metric from training tasks", ["task", "symbol", "metric"],
    )
    TRAIN_TASK_RUNS = Counter(
        "train_task_runs_total", "Total training/evaluation task runs by status", ["task", "status"],
    )
    REGIME_LABELS = Gauge("rt_regime_state", "Current regime vol and slope attr", ["symbol", "key"])
    ORDERFLOW = Gauge("rt_orderflow", "Orderflow metrics", ["symbol", "metric"])
    DYNPARAM = Gauge("rt_dyn_param", "Dynamic param scalars", ["symbol","param"])
    PF_GAUGE = Gauge("portfolio_stats", "Unified portfolio stats", ["key"])
    ORDER_MONITOR_SECONDS = Gauge("order_monitor_seconds", "Duration of order monitor cycle (s)")
    ADAPT_TICKS = Counter("rt_adapt_ticks_total", "How many adaptation cycles", ["symbol"])

    SYMBOLS = os.getenv("SYMBOLS", "BTC,ETH").split(",")
except Exception as _e:
    _METRICS_OK = False
    class _Dummy:
        def __getattr__(self, _): return self
        def labels(self, *_, **__): return self
        def inc(self, *_, **__): pass
        def observe(self, *_, **__): pass
        def set(self, *_, **__): pass
    METRICS = _Dummy()
    logger.warning(f"[Monitoring] Prometheus exporter disabled: {_e}") # Use logger
    TRAIN_MODEL_PROMOTED = _Dummy(); TRAIN_MODEL_DIRACC = _Dummy(); TRAIN_TASK_RUNS = _Dummy()
# === Queueing, reliability & autoscale-friendly settings ===
try:
    from kombu import Queue, Exchange
except Exception:
    Queue = Exchange = None  # keeps import-safe in minimal envs

# Queues (declare even if you don't route yet; safe & idempotent)
if Queue and Exchange:
    celery_app.conf.task_queues = (
        Queue("default",   Exchange("default"),   routing_key="default"),
        Queue("signals",   Exchange("signals"),   routing_key="signals"),
        Queue("retrain",   Exchange("retrain"),   routing_key="retrain"),
        Queue("portfolio", Exchange("portfolio"), routing_key="portfolio"),
        Queue("io",        Exchange("io"),        routing_key="io"),
        Queue("highprio",  Exchange("highprio"),  routing_key="highprio"),
    )

celery_app.conf.update(
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",

    # Reliability + worker lifecycle
    task_acks_late=True,                   # ack AFTER task runs (so task is redelivered on crash)
    task_reject_on_worker_lost=True,       # avoid duplicates on worker death
    task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "900")),      # hard cap (s)
    task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "840")),  # graceful warn
    broker_transport_options={
        "visibility_timeout": int(os.getenv("CELERY_VISIBILITY_TIMEOUT", "3600"))
    },

    # Throughput / autoscale friendliness
    worker_prefetch_multiplier=int(os.getenv("CELERY_PREFETCH_MULTIPLIER", "1")),  # prefer fairness
    task_ignore_result=True,               # cut Redis/DB writes for most tasks
    result_expires=int(os.getenv("CELERY_RESULT_EXPIRES", "3600")),
)

# Smart routing WITHOUT renaming your tasks:
# We route by substring so your current task names keep working as-is.
def _dynamic_router(name, args, kwargs, options, task=None, **kw):
    n = (name or "").lower()
    if "retrain"   in n or "train" in n:     return {"queue": "retrain",   "routing_key": "retrain"}
    if "portfolio" in n:                     return {"queue": "portfolio", "routing_key": "portfolio"}
    if "order"     in n and "monitor" in n:  return {"queue": "highprio",  "routing_key": "highprio"}
    if "collect"   in n or "ingest" in n or "writer" in n:  # I/O-heavy
                                             return {"queue": "io",        "routing_key": "io"}
    if "signal"    in n or "analyz" in n:    return {"queue": "signals",   "routing_key": "signals"}
    return {"queue": "default", "routing_key": "default"}

celery_app.conf.task_routes = (_dynamic_router,)
# ---- Celery signals: worker lifecycle ----
@worker_ready.connect
def _on_ready(sender=None, **kw):
    logging.getLogger("worker").info("worker_ready")
    alert_worker_event("ready")

@worker_shutdown.connect
def _on_shutdown(sender=None, **kw):
    logging.getLogger("worker").info("worker_shutdown")
    alert_worker_event("shutdown")

# ---- Celery signals: task timing & failure alerts ----
_current_task_name = {}

@task_prerun.connect
def _on_task_start(task_id=None, task=None, **kw):
    if task:
        _current_task_name[task_id] = task.name
    WORKER_HEARTBEAT.set_to_current_time()
    logging.getLogger("task").info("task_start", extra={"task_name": getattr(task, "name", ""), "task_id": task_id})

@task_postrun.connect
def _on_task_end(task_id=None, task=None, retval=None, **kw):
    WORKER_HEARTBEAT.set_to_current_time()
    logging.getLogger("task").info("task_end", extra={"task_name": getattr(task, "name", ""), "task_id": task_id})
    _current_task_name.pop(task_id, None)

@task_failure.connect
def _on_task_failure(task_id=None, exception=None, traceback=None, einfo=None, sender=None, **kw):
    name = getattr(sender, "name", _current_task_name.get(task_id, "unknown"))
    logging.getLogger("task").error("task_failed",
        extra={"task_name": name, "task_id": task_id, "error": str(exception)})
    alert_task_failure(name, task_id, str(exception))

# ---------- Optional: anti-overlap lock for periodic jobs (safe no-op if Redis missing) ----------
import functools, uuid
try:
    import redis as _redis_lib
    _LOCK_REDIS = _redis_lib.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))
except Exception:
    _LOCK_REDIS = None

def singleton_lock(key: str, ttl: int = 900):
    """
    Decorate periodic tasks you don't want to overlap.
    Example:
        @celery_app.task(name="retrain.daily", queue="retrain")
        @singleton_lock("retrain.daily", ttl=3600)
        def retrain_daily(): ...
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            if _LOCK_REDIS is None:
                # No Redis? Just run; keeps environments without Redis working.
                return fn(*a, **kw)
            token = str(uuid.uuid4())
            if not _LOCK_REDIS.set(f"lock:{key}", token, nx=True, ex=ttl):
                # Already running; skip quietly
                return {"skipped": True, "reason": "locked"}
            try:
                return fn(*a, **kw)
            finally:
                try:
                    cur = _LOCK_REDIS.get(f"lock:{key}")
                    if cur and cur.decode() == token:
                        _LOCK_REDIS.delete(f"lock:{key}")
                except Exception:
                    pass
        return wrapper
    return deco
# === END ADD BLOCK ===

# =============================================================================
# Watchdog (idle restart) + Celery signals
# =============================================================================
LAST_TASK_TS = time.time()
IDLE_RESTART_SECONDS = int(os.getenv("WORKER_IDLE_RESTART_SECONDS", str(15 * 60)))

def _watchdog_loop():
    while True:
        time.sleep(30)
        idle = time.time() - LAST_TASK_TS
        if idle > IDLE_RESTART_SECONDS:
            logger.critical(f"[Watchdog] No tasks processed for {int(idle)}s (> {IDLE_RESTART_SECONDS}). Exiting for restart.")
            os._exit(1)

# Start watchdog in a daemon thread
threading.Thread(target=_watchdog_loop, daemon=True).start()

_task_start_times = {}

@signals.task_prerun.connect
def _on_task_prerun(task_id=None, task=None, **kwargs):
    _task_start_times[task_id] = time.time()
    logger.debug(f"[Task Prerun] Starting task: {task.name} ({task_id})") # Use logger

@signals.task_postrun.connect
def _on_task_postrun(task_id=None, task=None, state=None, **kwargs):
    global LAST_TASK_TS
    LAST_TASK_TS = time.time()
    logger.debug(f"[Task Postrun] Finished task: {task.name} ({task_id}), State: {state}") # Use logger
    try:
        started = _task_start_times.pop(task_id, None)
        if _METRICS_OK and started is not None:
            METRICS.task_runtime_seconds.labels(task=getattr(task, "name", "unknown")).observe(LAST_TASK_TS - started)
    except Exception:
        pass # Never let metrics break the worker
# =============================================================================
# Order Management
# =============================================================================
@celery_app.task(name="orders.monitor_tick")
def orders_monitor_tick():
    """
    Lightweight order monitor: breakeven, trailing, TIF.
    Uses the same adapter as execution; acts on ledger entries.
    """
    svc = ExecutionService(paper_mode=True)  # or False in live
    mon = OrderMonitor(adapter=svc.adapter, risk_engines=svc.risk_engines)

    policy = OrderPolicy(
        breakeven=OrderPolicy().breakeven,      # r_trigger=1.0 (defaults)
        trailing=OrderPolicy().trailing,        # ATR x2 or set mode="pct", pct=0.75
        tif=OrderPolicy().tif                   # 120s default
    )

    t0 = time.time()
    # The portfolio state isn’t strictly necessary here; you can pass None
    out = mon.tick(ledger=svc.ledger, portfolio_state=None, policy=policy)
    ORDER_MONITOR_SECONDS.set(time.time() - t0)
    return out

# --- Task Success/Failure Hooks for Metrics ---
@signals.task_success.connect
def _on_task_success(sender=None, result: Any = None, **kwargs):
    try:
        tname = getattr(sender, "name", "unknown")
        TRAIN_TASK_RUNS.labels(task=tname, status="success").inc()
        
        # List of all training/feedback task names
        TRAINING_TASKS = {
            "tasks.retrain_models",
            "tasks.train_xgboost",
            "tasks.train_fusion_head",
            "tasks.incremental_model_update",
            "tasks.feedback_adaptive_threshold",
            "ml.self_evolve.full_retrain", # Add names from self-evolve tasks if used
            "ml.self_evolve.incremental_update"
        }
        if tname not in TRAINING_TASKS:
            return

        payload = result or {}
        
        def _emit(sym: str, d: Dict[str, Any]):
            if "promoted" in d and d.get("promoted") is True:
                TRAIN_MODEL_PROMOTED.labels(task=tname, symbol=sym).inc()
            for k in ("new_dir_acc", "lstm_dir_acc", "xgb_dir_acc", "new_cutoff", "base_conf_cutoff"):
                if k in d and d[k] is not None:
                    try: TRAIN_MODEL_DIRACC.labels(task=tname, symbol=sym, metric=k).set(float(d[k]))
                    except Exception: pass

        # Handle different result structures
        if "details" in payload and isinstance(payload["details"], dict): # For train_xgboost/fusion_head
             for sym, val in payload["details"].items():
                 if isinstance(val, dict): _emit(sym, val)
        elif "result" in payload and isinstance(payload["result"], dict): # For self-evolve tasks
             if "promoted" in payload["result"]: # Incremental
                 _emit("*", payload["result"]) 
             else: # Full retrain (res)
                 _emit("*", payload["result"])
             if "base_conf_cutoff" in payload: _emit("*", {"new_cutoff": payload["base_conf_cutoff"]})
        elif "trained_symbols" in payload: # For retrain_models_task
             if payload.get("promoted") is True:
                 for sym in payload["trained_symbols"]: TRAIN_MODEL_PROMOTED.labels(task=tname, symbol=sym).inc()
        elif "new_cutoff" in payload: # For feedback_adaptive_threshold_task
            _emit("*", payload) # Handle symbol-keyed dict if present
            if isinstance(payload.get("details"), dict):
                for sym, val in payload["details"].items():
                     if isinstance(val, dict): _emit(sym, val)

    except Exception as e:
        logger.warning(f"[Metrics] Error processing task_success signal: {e}", exc_info=True)

@signals.task_failure.connect
def _on_task_failure(sender=None, **kwargs):
    try:
        tname = getattr(sender, "name", "unknown")
        TRAIN_TASK_RUNS.labels(task=tname, status="failure").inc()
    except Exception:
        pass

# =============================================================================
# Circuit breakers & analytics helpers
# =============================================================================
def _model_artifact_age_days() -> Optional[float]:
    # (Implementation remains the same as previous version)
    p = CURRENT_SYMLINK
    if not p.is_symlink():
         logger.warning(f"Cannot check model age: '{p}' is not a symlink.")
         return None
    try:
        target_dir = p.resolve()
        if not target_dir.exists():
             logger.warning(f"Cannot check model age: Symlink target '{target_dir}' does not exist.")
             return None
        manifest_path = target_dir / "manifest.json"
        if not manifest_path.exists():
             mtime = target_dir.stat().st_mtime
             logger.warning(f"Cannot find manifest in {target_dir}, using directory mtime.")
        else:
             mtime = manifest_path.stat().st_mtime
        age_days = (_utcnow() - datetime.fromtimestamp(mtime, tz=timezone.utc)).total_seconds() / 86400.0
        return age_days
    except Exception as e:
        logger.error(f"Error checking model artifact age for '{p}': {e}", exc_info=True)
        return None

def _calc_missing_bars(df: pd.DataFrame, timeframe: str) -> int:
    # (Implementation remains the same)
    if df.empty or "timestamp" not in df.columns: return 0
    df = df.sort_values("timestamp"); dt = df["timestamp"].diff().dropna().dt.total_seconds()
    expected = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}.get(timeframe, 60)
    gaps = (dt > expected * 1.5).sum(); return int(gaps)

def _vol_zscore_from_df(df: pd.DataFrame) -> Optional[float]:
    # (Implementation remains the same)
    if df.empty or "close" not in df.columns: return None
    df = df.sort_values("timestamp"); px = df["close"].astype(float); rets = px.pct_change().dropna()
    if len(rets) < 50: return None
    short = rets.tail(24); long = rets.tail(24 * 14) if len(rets) >= 24 * 14 else rets
    sv = short.std(ddof=0); lv = long.rolling(24).std(ddof=0).dropna()
    if len(lv) < 10: return None
    mean_lv, std_lv = lv.mean(), lv.std(ddof=0)
    if std_lv <= 1e-12: return None
    return float((sv - mean_lv) / std_lv)

def _load_recent_orderbook_spread_bps(db, symbol: str) -> Optional[float]:
    # (Implementation remains the same)
    try:
        q = (db.query(models.OrderbookSnapshot).filter(models.OrderbookSnapshot.symbol == symbol)
             .order_by(models.OrderbookSnapshot.timestamp.desc()).limit(1))
        ob = q.first()
        if ob and hasattr(ob, 'best_bid') and hasattr(ob, 'best_ask') and ob.best_bid and ob.best_ask and ob.best_ask > 0:
            return (float(ob.best_ask) - float(ob.best_bid)) / float(ob.best_ask) * 1e4
    except Exception as e: logger.warning(f"Failed to query OrderbookSnapshot for spread: {e}")
    return None

def _passes_circuit_breakers(adapter: BybitAdapter, norm_symbol: str, confidence: float) -> bool:
    """DB-driven circuit breakers. Returns True if OK to trade, False to SKIP."""
    from app.services.settings_service import load_risk_settings
    base = _base_from_pair(norm_symbol)
    db = SessionLocal() # Use SessionLocal for a fresh, isolated session
    try:
        eff = load_risk_settings(db, base)
        
        # 1) Confidence cutoff
        conf_cut = float(eff.get("base_confidence_cutoff", 0.55))
        if _METRICS_OK: METRICS.confidence_cutoff.labels(symbol=base).set(conf_cut)
        if confidence < conf_cut:
            logger.info(f"[Circuit] {base}: confidence {confidence:.3f} < cutoff {conf_cut:.3f} → SKIP")
            if _METRICS_OK: METRICS.signals_filtered.labels(symbol=base, reason="low_confidence").inc()
            return False

        # 2) Model staleness
        max_model_age_days = int(eff.get("model_max_age_days", 7))
        model_age = _model_artifact_age_days()
        if model_age is not None:
            if _METRICS_OK: METRICS.model_age_days.labels(symbol="global").set(model_age)
            if model_age > max_model_age_days:
                logger.warning(f"[Circuit] Model too old: {model_age:.1f}d > {max_model_age_days}d → SKIP")
                if _METRICS_OK: METRICS.signals_filtered.labels(symbol=base, reason="model_stale").inc()
                return False
        
        # 3) Ticker age
        max_ticker_age_sec = int(eff.get("max_ticker_age_sec", 60))
        try:
            t = adapter.get_ticker(norm_symbol, params={"category": "linear"})
            ts_ms = t.get("timestamp") or t.get("datetime")
            if ts_ms:
                t_ts = (datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc) if isinstance(ts_ms, (int, float)) else pd.to_datetime(ts_ms, utc=True).to_pydatetime())
                age = (_utcnow() - t_ts).total_seconds()
                if _METRICS_OK: METRICS.ticker_age_seconds.labels(symbol=base).set(age)
                if age > max_ticker_age_sec:
                    logger.info(f"[Circuit] {base} ticker age {int(age)}s > {max_ticker_age_sec}s → SKIP")
                    if _METRICS_OK: METRICS.signals_filtered.labels(symbol=base, reason="stale_ticker").inc()
                    return False
        except Exception as e_t: logger.warning(f"Circuit breaker ticker check failed: {e_t}")

        # 4) Missing bars
        timeframe = "1m"; max_missing_bars = int(eff.get("max_missing_bars", 2))
        try:
            df = adapter.fetch_ohlcv(norm_symbol, timeframe=timeframe, limit=300)
            if df is None or df.empty: timeframe = "5m"; df = adapter.fetch_ohlcv(norm_symbol, timeframe=timeframe, limit=300)
            if isinstance(df, pd.DataFrame) and not df.empty:
                if "timestamp" not in df.columns: df = df.rename(columns={"ts": "timestamp"})
                missing = _calc_missing_bars(df, timeframe)
                if _METRICS_OK: METRICS.missing_bars.labels(symbol=base, timeframe=timeframe).set(missing)
                if missing > max_missing_bars:
                    logger.info(f"[Circuit] {base} missing bars {missing} > {max_missing_bars} on {timeframe} → SKIP")
                    if _METRICS_OK: METRICS.signals_filtered.labels(symbol=base, reason="missing_bars").inc()
                    return False
        except Exception as e_b: logger.warning(f"Circuit breaker missing bars check failed: {e_b}")
        
        # 5) Volatility spike (Z)
        vol_z_max = float(eff.get("vol_z_max", 3.0))
        try:
            dfh = adapter.fetch_ohlcv(norm_symbol, timeframe="1h", limit=24 * 30)
            if isinstance(dfh, pd.DataFrame) and not dfh.empty:
                if "timestamp" not in dfh.columns: dfh = dfh.rename(columns={"ts": "timestamp"})
                z = _vol_zscore_from_df(dfh)
                if z is not None:
                    if _METRICS_OK: METRICS.volatility_zscore.labels(symbol=base).set(z)
                    if z > vol_z_max:
                        logger.info(f"[Circuit] {base} vol z-score {z:.2f} > {vol_z_max} → SKIP")
                        if _METRICS_OK: METRICS.signals_filtered.labels(symbol=base, reason="vol_spike").inc()
                        return False
        except Exception as e_v: logger.warning(f"Circuit breaker volatility check failed: {e_v}")

        # 6) Liquidity / Spread guard
        max_spread_bps = float(eff.get("max_spread_bps", 25.0))
        spread_bps: Optional[float] = _load_recent_orderbook_spread_bps(db, norm_symbol)
        if spread_bps is None: # Fallback to ticker
            try:
                t2 = adapter.get_ticker(norm_symbol, params={"category": "linear"})
                bid = _safe_float(t2.get("bid") or t2.get("bidPrice") or t2.get("bid1Price"))
                ask = _safe_float(t2.get("ask") or t2.get("askPrice") or t2.get("ask1Price"))
                if ask and bid and ask > 0 and bid > 0: spread_bps = (ask - bid) / ask * 1e4
            except Exception as e_s: logger.warning(f"Circuit breaker spread check (ticker) failed: {e_s}")
        if spread_bps is not None:
            if _METRICS_OK: METRICS.spread_bps.labels(symbol=base).set(spread_bps)
            if spread_bps > max_spread_bps:
                logger.info(f"[Circuit] {base} spread {spread_bps:.1f} bps > {max_spread_bps} bps → SKIP")
                if _METRICS_OK: METRICS.signals_filtered.labels(symbol=base, reason="wide_spread").inc()
                return False

        # 7) Funding shock
        funding_abs_bps_max = float(eff.get("funding_abs_bps_max", 20.0))
        try:
            # Assumes FundingRate model exists
            latest = (db.query(models.FundingRate).filter(models.FundingRate.symbol == norm_symbol)
                      .order_by(models.FundingRate.timestamp.desc()).limit(1).first())
            if latest and latest.funding_rate is not None:
                fr_bps = float(latest.funding_rate) * 1e4
                if _METRICS_OK: METRICS.funding_bps.labels(symbol=base).set(fr_bps)
                if abs(fr_bps) > funding_abs_bps_max:
                    logger.info(f"[Circuit] {base} |funding| {abs(fr_bps):.1f} bps > {funding_abs_bps_max} bps → SKIP")
                    if _METRICS_OK: METRICS.signals_filtered.labels(symbol=base, reason="funding_shock").inc()
                    return False
        except Exception as e_f: logger.warning(f"Circuit breaker funding check failed: {e_f}")

        # 8) Correlation stress
        corr_z_max = float(eff.get("corr_z_max", 3.0))
        try:
             # Assumes CrossAssetCorr model exists
            corr_row = (db.query(models.CrossAssetCorr).filter(models.CrossAssetCorr.base_symbol == base)
                        .order_by(models.CrossAssetCorr.timestamp.desc()).limit(1).first())
            if corr_row and getattr(corr_row, "zscore", None) is not None:
                zc = float(corr_row.zscore)
                if _METRICS_OK: METRICS.correlation_zscore.labels(symbol=base).set(zc)
                if abs(zc) > corr_z_max:
                    logger.info(f"[Circuit] {base} |corr_z| {abs(zc):.2f} > {corr_z_max} → SKIP")
                    if _METRICS_OK: METRICS.signals_filtered.labels(symbol=base, reason="correlation_stress").inc()
                    return False
        except Exception as e_c: logger.warning(f"Circuit breaker correlation check failed: {e_c}")

        # 9) Toggles
        if bool(eff.get("maintenance_mode", False)):
            logger.info(f"[Circuit] maintenance_mode = True → SKIP")
            if _METRICS_OK: METRICS.signals_filtered.labels(symbol=base, reason="maintenance").inc()
            return False
        if bool(eff.get("symbol_kill_switch", False)):
            logger.info(f"[Circuit] {base} kill_switch = True → SKIP")
            if _METRICS_OK: METRICS.signals_filtered.labels(symbol=base, reason="kill_switch").inc()
            return False

        return True # All checks passed
    
    except Exception as e_outer:
         logger.error(f"[Circuit] Outer circuit breaker check failed: {e_outer}", exc_info=True)
         return False # Default to safe (SKIP)
    finally:
         if db:
             try: db.close() # Always close the session
             except: pass


# =============================================================================
# Risk-aware execution service (singleton)
# =============================================================================
PAPER_MODE = bool(getattr(settings, "PAPER_TRADING", True))
EXEC = ExecutionService(adapter=BybitAdapter(paper_mode=PAPER_MODE), paper_mode=PAPER_MODE)

# =============================================================================
# Tasks (Core analysis/reconciliation loops)
# =============================================================================
@celery_app.task(base=BaseTaskWithRetry)
def scan_top_currencies_task():
    logger.info("[Scan] Top symbols by volume …") # Use logger
    try:
        adapter = BybitAdapter(paper_mode=PAPER_MODE)
        symbols = adapter.get_top_symbols_by_volume(limit=10) or []
        for coin in symbols:
            analyze_futures_market_task.delay(coin) # Pass base symbol (e.E., BTC)
    except Exception as e:
         logger.error(f"Error in scan_top_currencies_task: {e}", exc_info=True)


@celery_app.task(base=BaseTaskWithRetry)
def analyze_options_chain_task(underlying_symbol: str):
    logger.info(f"[Options] Analysis requested for {underlying_symbol} (WIP)") # Use logger


@celery_app.task(base=BaseTaskWithRetry)
@adaptive_task(settings.REDIS_URL, symbol_arg="underlying_symbol")
def analyze_futures_market_task(underlying_symbol: str, *_args, _dyn_params=None, **_kwargs):  # Expects base symbol (e.g., BTC)
    """
    Adaptive version:
    - Decorated by @adaptive_task to apply self-tuning cooldowns
    - Reads _dyn_params to gate by min_model_conf and pass sizing/SL/TP hints
    """
    params = _dyn_params or {}
    min_conf = float(params.get("min_model_conf", 0.55))
    size_factor = float(params.get("size_factor", 1.0))
    sl_mult = float(params.get("sl_atr_mult", 1.5))
    tp_mult = float(params.get("tp_atr_mult", 2.2))
    model_profile = params.get("model_profile", "default")

    adapter = BybitAdapter(paper_mode=PAPER_MODE)
    futures_symbol = f"{underlying_symbol.upper()}/USDT"  # Construct pair symbol

    ticker = {}
    try:
        ticker = adapter.get_ticker(futures_symbol, params={"category": "linear"})
        if not ticker or not ticker.get("last"):
            logger.warning(f"No valid ticker data for {futures_symbol}. Skipping analysis.")
            return {"skipped": True, "reason": "no_ticker"}
    except Exception as e_t:
        logger.error(f"Failed to fetch ticker for {futures_symbol}: {e_t}")
        return {"skipped": True, "reason": "ticker_error"}

    norm_symbol = _norm_futures_symbol(ticker.get("symbol") or futures_symbol)
    price = _safe_float(ticker.get("last") or ticker.get("lastPrice"))
    if not price:
        logger.warning(f"Ticker for {norm_symbol} missing 'last' price. Skipping.")
        return {"skipped": True, "reason": "no_price"}

    # --- Save Ticker Data (unchanged from your version) ---
    db = SessionLocal()
    try:
        ts_ms = ticker.get("timestamp")
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else _utcnow()

        exists = db.query(models.FuturesMarketData.id).filter(
            models.FuturesMarketData.symbol == norm_symbol,
            models.FuturesMarketData.timestamp == ts
        ).first()

        if not exists:
            rec = models.FuturesMarketData(
                symbol=norm_symbol, timestamp=ts,
                open=_safe_float(ticker.get("openPrice") or ticker.get("open")),
                high=_safe_float(ticker.get("highPrice") or ticker.get("high")),
                low=_safe_float(ticker.get("lowPrice") or ticker.get("low")),
                close=price,
                volume=_safe_float(ticker.get("volume24h") or ticker.get("baseVolume")),
            )
            db.add(rec)
            db.commit()
    except Exception as e_db:
        logger.error(f"[DB] Error saving futures ticker for {norm_symbol}: {e_db}")
        db.rollback()
    finally:
        db.close()

    # --- Run Inference (unchanged call; model_profile is made available via params) ---
    try:
        model_output = run_ensemble(symbol=norm_symbol, base_table="futures_market_data")
        final_signal = model_output.get("decision", "HOLD").upper()
        confidence = float(model_output.get("confidence", 0.0))
        regime = int(model_output.get("regime", 0))
    except Exception as e_ml:
        logger.error(f"[ML] Ensemble prediction error for {norm_symbol}: {e_ml}", exc_info=True)
        if _METRICS_OK: METRICS.trade_errors.labels(symbol=_base_from_pair(norm_symbol), stage="inference").inc()
        return {"skipped": True, "reason": "inference_error"}

    # --- Adaptive early gate by min_model_conf ---
    if final_signal != "HOLD" and confidence < min_conf:
        if _METRICS_OK:
            METRICS.signals_total.labels(symbol=_base_from_pair(norm_symbol), decision="FILTERED_LOWCONF").inc()
        logger.info(f"[AdaptiveGate] {norm_symbol}: conf {confidence:.3f} < min_conf {min_conf:.3f} → SKIP")
        return {"skipped": True, "reason": "low_conf", "conf": confidence, "min_conf": min_conf, "params": params}

    # --- Circuit Breakers (unchanged) ---
    if final_signal != "HOLD":
        if not _passes_circuit_breakers(adapter, norm_symbol, confidence):
            if _METRICS_OK: METRICS.signals_total.labels(symbol=_base_from_pair(norm_symbol), decision="FILTERED").inc()
            return {"skipped": True, "reason": "circuit_filtered"}
    else:
        if _METRICS_OK: METRICS.signals_total.labels(symbol=_base_from_pair(norm_symbol), decision="HOLD").inc()
        return {"skipped": True, "reason": "hold"}

    # --- Build adaptive order payload (adds hints; executor may use/ignore safely) ---
    order_payload = {
        "symbol": norm_symbol,
        "action": final_signal,
        "confidence": confidence,
        "regime": regime,
        "price": float(price),
        "type": "futures",
        # adaptive hints:
        "model_profile": model_profile,   # e.g., "trend" / "meanrev" / "default"
        "size_factor": size_factor,       # scale base position size
        "sl_atr_mult": sl_mult,           # suggested SL multiplier
        "tp_atr_mult": tp_mult,           # suggested TP multiplier
    }

    if _METRICS_OK:
        METRICS.signals_total.labels(symbol=_base_from_pair(norm_symbol), decision=final_signal).inc()
        METRICS.model_confidence.labels(symbol=_base_from_pair(norm_symbol)).observe(confidence)

    try:
        out = EXEC.execute_trade_signal_sync(order_payload)
        if out:
            logger.info(f"[TRADE] (Paper={PAPER_MODE}) {norm_symbol} {final_signal}: {out}")
            if _METRICS_OK: METRICS.trades_placed.labels(symbol=_base_from_pair(norm_symbol), side=final_signal).inc()
        return {"ok": True, "applied_params": params, "exec": out}
    except Exception as e_exec:
        logger.error(f"[Exec] Execution error (futures) for {norm_symbol}: {e_exec}", exc_info=True)
        if _METRICS_OK: METRICS.trade_errors.labels(symbol=_base_from_pair(norm_symbol), stage="execution").inc()
        return {"skipped": True, "reason": "exec_error"}

@celery_app.task
def reconcile_positions_task():
    try:
        out = EXEC.reconcile_and_learn()
        if out and out.get("closed"):
            logger.info(f"[Risk/Reconcile] Closed: {len(out['closed'])} trades") # Use logger
            if _METRICS_OK:
                for tr in out.get("closed", []):
                    METRICS.trades_closed.labels(
                        symbol=_base_from_pair(tr.get("symbol", "?")),
                        side=tr.get("side", "?"), reason=tr.get("reason", "?"),
                    ).inc()
                    METRICS.trade_pnl_usd.labels(
                        symbol=_base_from_pair(tr.get("symbol", "?")),
                        side=tr.get("side", "?"),
                    ).observe(float(tr.get("pnl_usd", 0.0)))
    except Exception as e:
        logger.error(f"[Risk/Reconcile] error: {e}", exc_info=True) # Use logger
@celery_app.task(name="rt_regime_scan")

def rt_regime_scan():
    for sym in SYMBOLS:
        snap = detect_and_store(settings.SQLALCHEMY_DATABASE_URI, settings.REDIS_URL, sym)
        REGIME_LABELS.labels(sym, "atr_pct").set(snap.atr_pct)
        REGIME_LABELS.labels(sym, "trend_slope").set(snap.trend_slope)
        # encode regime/vol as numbers for a gauge
        REGIME_LABELS.labels(sym, "regime_num").set({"bear":-1,"chop":0,"bull":1}[snap.regime])
        REGIME_LABELS.labels(sym, "vol_num").set({"low_vol":0,"mid_vol":1,"high_vol":2}[snap.vol])
        ADAPT_TICKS.labels(sym).inc()
    return {"ok": True}

@celery_app.task(name="rt_orderflow_scan")
def rt_orderflow_scan():
    for sym in SYMBOLS:
        of = scan_orderflow(settings.SQLALCHEMY_DATABASE_URI, settings.REDIS_URL, sym)
        ORDERFLOW.labels(sym, "obi").set(of["obi"])
        ORDERFLOW.labels(sym, "aggr").set(of["aggr"])
        ORDERFLOW.labels(sym, "vpin").set(of["vpin"])
        ADAPT_TICKS.labels(sym).inc()
    return {"ok": True}

@celery_app.task(name="rt_tune_params")
def rt_tune_params():
    for sym in SYMBOLS:
        p = tune(settings.SQLALCHEMY_DATABASE_URI, settings.REDIS_URL, sym)
        # expose a few safe scalars
        for k in ("min_model_conf","size_factor","sl_atr_mult","tp_atr_mult","cooldown_sec"):
            DYNPARAM.labels(sym, k).set(float(p.get(k,0)))
        ADAPT_TICKS.labels(sym).inc()
    return {"ok": True}

@celery_app.task(name="rt_update_cooldowns")
def rt_update_cooldowns():
    for sym in SYMBOLS:
        state = update_cooldowns(settings.REDIS_URL, sym)
        # also stash for quick reads
        set_kv(settings.REDIS_URL, f"cooldown:view:{sym}", state, ttl=180)
        ADAPT_TICKS.labels(sym).inc()
    return {"ok": True}

@celery_app.task(name="portfolio.tick")
def portfolio_tick():
    st = PortfolioState.load()
    PF_GAUGE.labels("equity_usd").set(st.equity_usd)
    PF_GAUGE.labels("cash_usd").set(st.cash_usd)
    PF_GAUGE.labels("gross_exposure").set(st.gross_exposure_usd)
    PF_GAUGE.labels("net_exposure").set(st.net_exposure_usd)
    return {"equity": st.equity_usd, "gross": st.gross_exposure_usd, "net": st.net_exposure_usd}

@celery_app.task(name="portfolio.rebalance_suggest")
def portfolio_rebalance_suggest():
    return suggest_rebalance()

@celery_app.task(name="portfolio.hedge_suggest")
def portfolio_hedge_suggest():
    return compute_hedge(net_cap_ratio=0.30)
@celery_app.task(name="internal.refresh_risk_settings")
def _refresh_risk_settings():
    db_s = None
    try:
        from app.db.database import SessionLocal
        from app.services.settings_service import refresh_cache
        db_s = SessionLocal()
        refresh_cache(db_s)
        logger.info("[RiskSettings] cache refreshed")
    except Exception as e:
        logger.error(f"[RiskSettings] refresh error: {e}", exc_info=True)
    finally:
        if db_s:
            try: db_s.close()
            except Exception: pass

# =============================================================================
# Periodic schedule (Beat) - CONSOLIDATED & DE-DUPLICATED
# =============================================================================
def Q(sig, queue: str):
    return sig.set(queue=queue, routing_key=queue)

@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    logger.info("--- Configuring Celery Beat Schedule (Hybrid Approach) ---")

    # Realtime light loops → signals
    sender.add_periodic_task(30.0, Q(rt_regime_scan.s(), "signals"),      name="rt_regime_scan_30s")
    sender.add_periodic_task(15.0, Q(rt_orderflow_scan.s(), "signals"),   name="rt_orderflow_scan_15s")
    sender.add_periodic_task(30.0, Q(rt_tune_params.s(), "signals"),      name="rt_tune_params_30s")
    sender.add_periodic_task(20.0, Q(rt_update_cooldowns.s(), "signals"), name="rt_update_cooldowns_20s")

    # Housekeeping → io
    sender.add_periodic_task(60.0, Q(_refresh_risk_settings.s(), "io"),
                             name="Refresh Risk Settings Cache (60s)")

    # Minute scanner (spawns per-symbol analyze tasks) → signals
    sender.add_periodic_task(60.0, Q(scan_top_currencies_task.s(), "signals"),
                             name="Scan Top Currencies Every Minute")

    # Optional: reconcile loop (you had this in legacy block) → signals (short & frequent)
    sender.add_periodic_task(30.0, Q(reconcile_positions_task.s(), "signals"),
                             name="Reconcile Positions & Learn (every 30s)")

    # Orderbook (mostly IO) → io
    sender.add_periodic_task(crontab(minute="*/10"),
                             Q(collect_orderbook_snapshot_task.s(), "io"),
                             name="Orderbook Collector (10min)")

    # Options metrics (medium) → signals
    sender.add_periodic_task(crontab(minute='*/15'),
                             Q(calculate_options_derived_metrics_task.s(symbol="BTC"), "signals"),
                             name="Calculate Options Derived Metrics (BTC 15min)")
    sender.add_periodic_task(crontab(minute='*/15'),
                             Q(calculate_options_derived_metrics_task.s(symbol="ETH"), "signals"),
                             name="Calculate Options Derived Metrics (ETH 15min)")

    # Sentiment collection (IO) + scoring (signals)
    sender.add_periodic_task(crontab(minute=0, hour="*"),
                             Q(collect_sentiment_all_sources.s(), "io"),
                             name="Collect All Sentiment Sources (Hourly)")
    sender.add_periodic_task(crontab(minute="*/30"),
                             Q(score_headlines_task.s(), "signals"),
                             name="Score Headlines Batch (30min)")

    # Feedback loop (medium) → signals
    sender.add_periodic_task(crontab(minute=45, hour='*'),
                             Q(feedback_adaptive_threshold_task.s(), "signals"),
                             name="Adaptive Threshold Update (Hourly)")

    # Other data sources (mostly IO) → io
    sender.add_periodic_task(crontab(hour='*/2', minute=5),
                             Q(collectors.collect_fear_and_greed_task.s(), "io"),
                             name="Collect Fear and Greed Index (2h)")
    sender.add_periodic_task(crontab(hour='*/4', minute=5),
                             Q(collectors.collect_macro_data_task.s(), "io"),
                             name="Collect Macro Data (DXY 4h)")
    sender.add_periodic_task(crontab(hour="*/6", minute=10),
                             Q(collect_onchain_data_task.s(), "io"),
                             name="Onchain Metrics Collector (6h)")
    sender.add_periodic_task(crontab(hour="*/3", minute=0),
                             Q(run_cross_asset_corre_task.s(), "signals"),
                             name="Run Cross Asset Correlation Task (3h)")
    sender.add_periodic_task(crontab(hour='*/4', minute=15),
                             Q(collectors.collect_binance_historical_data.s(), "io"),
                             name="Collect Binance Historical Data (4h)")

    # Incremental model updates / eval → retrain
    sender.add_periodic_task(crontab(minute=15, hour='*/4'),
                             Q(incremental_model_update_task.s(), "retrain"),
                             name="Incremental Model Update (4h)")
    sender.add_periodic_task(crontab(minute=0, hour="*/6"),
                             Q(evaluate_current_models_task.s(), "retrain"),
                             name="Evaluate Current Models (6h)")

    # Weekly infrequent jobs
    sender.add_periodic_task(crontab(hour=2, minute=10, day_of_week=0),
                             Q(collect_github_activity_task.s(), "io"),
                             name="Github Activity Collector (Weekly)")

    # Weekly heavy processing → retrain
    sender.add_periodic_task(crontab(hour=5, minute=0, day_of_week=0),
                             Q(task_sentiment_fusion.s(), "retrain"),
                             name="Run Sentiment Fusion (Weekly)")

    # Full training pipeline (sequenced on Sunday) → retrain
    sender.add_periodic_task(crontab(hour=4, minute=30, day_of_week='sun'),
                             Q(retrain_models_task.s(), "retrain"),
                             name="Weekly Full LSTM Retrain")
    sender.add_periodic_task(crontab(hour=5, minute=30, day_of_week='sun'),
                             Q(train_xgboost_task.s(), "retrain"),
                             name="Weekly XGBoost Training")
    sender.add_periodic_task(crontab(hour=6, minute=0, day_of_week='sun'),
                             Q(train_fusion_head_task.s(), "retrain"),
                             name="Weekly Fusion Head Training")

    # Portfolio monitoring (you had these) → signals (low CPU)
    sender.add_periodic_task(30.0,  Q(portfolio_tick.s(), "signals"),             name="portfolio_tick_30s")
    sender.add_periodic_task(120.0, Q(portfolio_rebalance_suggest.s(), "signals"),name="portfolio_rebalance_suggest_2m")
    sender.add_periodic_task(90.0,  Q(portfolio_hedge_suggest.s(), "signals"),    name="portfolio_hedge_suggest_90s")
    sender.add_periodic_task(5.0,   Q(orders_monitor_tick.s(), "signals"),        name="orders_monitor_tick_5s")

    logger.info("--- Celery Beat Schedule Configured ---")

    logger.info("--- Configuring Celery Beat Schedule (Hybrid Approach) ---")

    # Realtime light loops → signals
    sender.add_periodic_task(30.0, Q(rt_regime_scan.s(), "signals"),      name="rt_regime_scan_30s")
    sender.add_periodic_task(15.0, Q(rt_orderflow_scan.s(), "signals"),   name="rt_orderflow_scan_15s")
    sender.add_periodic_task(30.0, Q(rt_tune_params.s(), "signals"),      name="rt_tune_params_30s")
    sender.add_periodic_task(20.0, Q(rt_update_cooldowns.s(), "signals"), name="rt_update_cooldowns_20s")

    # Housekeeping → io
    sender.add_periodic_task(60.0, Q(_refresh_risk_settings.s(), "io"),
                             name="Refresh Risk Settings Cache (60s)")

    # Minute scanner (spawns per-symbol analyze tasks) → signals
    sender.add_periodic_task(60.0, Q(scan_top_currencies_task.s(), "signals"),
                             name="Scan Top Currencies Every Minute")

    # Orderbook (mostly IO) → io
    sender.add_periodic_task(crontab(minute="*/10"),
                             Q(collect_orderbook_snapshot_task.s(), "io"),
                             name="Orderbook Collector (10min)")

    # Options metrics (medium) → signals  (per symbol)
    sender.add_periodic_task(crontab(minute='*/15'),
                             Q(calculate_options_derived_metrics_task.s(symbol="BTC"), "signals"),
                             name="Calculate Options Derived Metrics (BTC 15min)")
    sender.add_periodic_task(crontab(minute='*/15'),
                             Q(calculate_options_derived_metrics_task.s(symbol="ETH"), "signals"),
                             name="Calculate Options Derived Metrics (ETH 15min)")

    # Sentiment collection/scoring (IO + some CPU) → io for collect, signals for score
    sender.add_periodic_task(crontab(minute=0, hour="*"),
                             Q(collect_sentiment_all_sources.s(), "io"),
                             name="Collect All Sentiment Sources (Hourly)")
    sender.add_periodic_task(crontab(minute="*/30"),
                             Q(score_headlines_task.s(), "signals"),
                             name="Score Headlines Batch (30min)")

    # Feedback loop (medium) → signals
    sender.add_periodic_task(crontab(minute=45, hour='*'),
                             Q(feedback_adaptive_threshold_task.s(), "signals"),
                             name="Adaptive Threshold Update (Hourly)")

    # Other data sources (mostly IO) → io
    sender.add_periodic_task(crontab(hour='*/2', minute=5),
                             Q(collectors.collect_fear_and_greed_task.s(), "io"),
                             name="Collect Fear and Greed Index (2h)")
    sender.add_periodic_task(crontab(hour='*/4', minute=5),
                             Q(collectors.collect_macro_data_task.s(), "io"),
                             name="Collect Macro Data (DXY 4h)")

    # Incremental model updates / eval (medium→heavy) → retrain
    sender.add_periodic_task(crontab(minute=15, hour='*/4'),
                             Q(incremental_model_update_task.s(), "retrain"),
                             name="Incremental Model Update (4h)")
    sender.add_periodic_task(crontab(minute=0, hour="*/6"),
                             Q(evaluate_current_models_task.s(), "retrain"),
                             name="Evaluate Current Models (6h)")

    # Weekly infrequent jobs
    sender.add_periodic_task(crontab(hour=2, minute=10, day_of_week=0),
                             Q(collect_github_activity_task.s(), "io"),
                             name="Github Activity Collector (Weekly)")

    # Weekly heavy processing → retrain
    sender.add_periodic_task(crontab(hour=5, minute=0, day_of_week=0),
                             Q(task_sentiment_fusion.s(), "retrain"),
                             name="Run Sentiment Fusion (Weekly)")

    # Full training pipeline (sequenced on Sunday) → retrain
    sender.add_periodic_task(crontab(hour=4, minute=30, day_of_week='sun'),
                             Q(retrain_models_task.s(), "retrain"),
                             name="Weekly Full LSTM Retrain")
    sender.add_periodic_task(crontab(hour=5, minute=30, day_of_week='sun'),
                             Q(train_xgboost_task.s(), "retrain"),
                             name="Weekly XGBoost Training")
    sender.add_periodic_task(crontab(hour=6, minute=0, day_of_week='sun'),
                             Q(train_fusion_head_task.s(), "retrain"),
                             name="Weekly Fusion Head Training")

    logger.info("--- Celery Beat Schedule Configured ---")

    
    logger.info("--- Configuring Celery Beat Schedule (Hybrid Approach) ---")
    # Every 30s: regime scan
    sender.add_periodic_task(30.0, rt_regime_scan.s(), name="rt_regime_scan_30s")
    # Every 15s: orderflow scan
    sender.add_periodic_task(15.0, rt_orderflow_scan.s(), name="rt_orderflow_scan_15s")
    # Every 30s: tune dynamic params
    sender.add_periodic_task(30.0, rt_tune_params.s(), name="rt_tune_params_30s")
    # Every 20s: update cooldowns
    sender.add_periodic_task(20.0, rt_update_cooldowns.s(), name="rt_update_cooldowns_20s")
    # --- 1. High Frequency (Real-time Loops) ---
    # Refresh risk settings from DB
    sender.add_periodic_task(60.0, _refresh_risk_settings.s(),
                             name="Refresh Risk Settings Cache (60s)")
    # Scan for new signals
    sender.add_periodic_task(60.0, scan_top_currencies_task.s(),
                             name="Scan Top Currencies Every Minute")
    # Check positions, PnL, and learn from trades
    sender.add_periodic_task(30.0, reconcile_positions_task.s(),
                             name="Reconcile Positions & Learn (every 30s)")

    # --- 2. Medium Frequency (Data Collection & Feedback) ---
    # Core data for analysis (triggers options/futures)
    sender.add_periodic_task(crontab(minute='*/15'), # Every 15 minutes
                             collectors.collect_all_assets_task.s(),
                             name="Collect All Main Assets (15min)")
    # Orderbook data
    sender.add_periodic_task(crontab(minute="*/10"), # Every 10 minutes
                             collect_orderbook_snapshot_task.s(),
                             name="Orderbook Collector (10min)")
    # Derived options metrics
    sender.add_periodic_task(crontab(minute='*/15'), # Every 15 minutes
                         calculate_options_derived_metrics_task.s(symbol="BTC"),
                         name="Calculate Options Derived Metrics (BTC 15min)")
    sender.add_periodic_task(crontab(minute='*/15'), # Every 15 minutes
                         calculate_options_derived_metrics_task.s(symbol="ETH"),
                         name="Calculate Options Derived Metrics (ETH 15min)")
    # Sentiment data collection
    sender.add_periodic_task(crontab(minute=0, hour="*"), # Hourly
                             collect_sentiment_all_sources.s(),
                             name="Collect All Sentiment Sources (Hourly)")
    # Sentiment scoring
    sender.add_periodic_task(crontab(minute="*/30"), # Every 30 mins
                             score_headlines_task.s(),
                             name="Score Headlines Batch (30min)")
    # **Feedback Loop**
    sender.add_periodic_task(crontab(minute=45, hour='*'), # Every hour at XX:45
                             feedback_adaptive_threshold_task.s(),
                             name="Adaptive Threshold Update (Hourly)")

    # --- 3. Low Frequency (Heavy Data & Incremental Training) ---
    # Other data sources
    sender.add_periodic_task(crontab(hour='*/2', minute=5), # Every 2 hours
                             collectors.collect_fear_and_greed_task.s(),
                             name="Collect Fear and Greed Index (2h)")
    sender.add_periodic_task(crontab(hour='*/4', minute=5), # Every 4 hours
                             collectors.collect_macro_data_task.s(),
                             name="Collect Macro Data (DXY 4h)")
    sender.add_periodic_task(crontab(hour="*/6", minute=10), # Every 6 hours
                             collect_onchain_data_task.s(),
                             name="Onchain Metrics Collector (6h)")
    sender.add_periodic_task(crontab(hour="*/3", minute=0), # Every 3 hours
                             run_cross_asset_corre_task.s(),
                             name="Run Cross Asset Correlation Task (3h)")
    # Historical data (for ML training)
    sender.add_periodic_task(crontab(hour='*/4', minute=15), # Every 4 hours
                             collectors.collect_binance_historical_data.s(),
                             name="Collect Binance Historical Data (4h)")
    
    # **Incremental Model Updates**
    sender.add_periodic_task(crontab(minute=15, hour='*/4'), # Every 4 hours at XX:15
                             incremental_model_update_task.s(),
                             name="Incremental Model Update (4h)")
    
    # Evaluate the current model's performance
    sender.add_periodic_task(crontab(minute=0, hour="*/6"), # Every 6 hours
                             evaluate_current_models_task.s(),
                             name="Evaluate Current Models (6h)")

    # --- 4. Very Low Frequency (Full Retraining - Weekly on Sunday) ---
    # Infrequent data
    sender.add_periodic_task(crontab(hour=2, minute=10, day_of_week=0), # Sunday @ 02:10 UTC
                             collect_github_activity_task.s(),
                             name="Github Activity Collector (Weekly)")
    # Heavy processing
    sender.add_periodic_task(crontab(hour=5, minute=0, day_of_week=0), # Sunday @ 05:00 UTC
                             task_sentiment_fusion.s(),
                             name="Run Sentiment Fusion (Weekly)")
    
    # **Full Training Pipeline (Sequenced on Sunday)**
    # 04:30 UTC Sunday: Full LSTM Retrain
    sender.add_periodic_task(crontab(hour=4, minute=30, day_of_week='sun'),
                             retrain_models_task.s(),
                             name="Weekly Full LSTM Retrain")
    # 05:30 UTC Sunday: XGBoost Training
    sender.add_periodic_task(crontab(hour=5, minute=30, day_of_week='sun'),
                             train_xgboost_task.s(),
                             name="Weekly XGBoost Training")
    # 06:00 UTC Sunday: Fusion Head Training
    sender.add_periodic_task(crontab(hour=6, minute=0, day_of_week='sun'),
                             train_fusion_head_task.s(),
                             name="Weekly Fusion Head Training")

    sender.add_periodic_task(30.0, portfolio_tick.s(), name="portfolio_tick_30s")
    sender.add_periodic_task(120.0, portfolio_rebalance_suggest.s(), name="portfolio_rebalance_suggest_2m")
    sender.add_periodic_task(90.0, portfolio_hedge_suggest.s(), name="portfolio_hedge_suggest_90s")
    sender.add_periodic_task(5.0, orders_monitor_tick.s(), name="orders_monitor_tick_5s")
    # ---- Periodic refresh of risk settings (every 60s) ----
    @celery_app.task(name="internal.refresh_risk_settings")
    def _refresh_risk_settings():
        db_s = None
        try:
            from app.db.database import SessionLocal # Use SessionLocal
            from app.services.settings_service import refresh_cache
            db_s = SessionLocal() # Create session
            refresh_cache(db_s) # Pass session
            logger.info("[RiskSettings] cache refreshed") # Use logger
        except Exception as e:
            logger.error(f"[RiskSettings] refresh error: {e}", exc_info=True) # Use logger
        finally:
            if db_s:
                try: db_s.close()
                except Exception: pass
    
    # This task is already defined and scheduled above (line 399)
    # sender.add_periodic_task(60.0, _refresh_risk_settings.s(),
    #                          name="Refresh Risk Settings Cache (60s)")
    
    logger.info("--- Celery Beat Schedule Configured ---")