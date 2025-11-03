# app/monitoring/metrics.py
from __future__ import annotations
from prometheus_client import Counter, Gauge

class _M:
    train_runs_total = Counter(
        "bot_train_runs_total",
        "Count of training runs",
        labelnames=("stage", "status"),
    )
    train_metric_gauge = Gauge(
        "bot_train_metric",
        "Training/Evaluation metrics gauge",
        labelnames=("metric", "model"),
    )
    tasks_total = Counter(
        "bot_celery_tasks_total",
        "Celery task runs",
        labelnames=("task", "status"),
    )
    signals_total = Counter(
        "bot_signals_total",
        "Signals produced for trading",
        labelnames=("kind", "decision"),
    )
    trade_errors = Counter(
        "bot_trade_errors_total",
        "Total count of trade-related errors",
        labelnames=("symbol", "stage"),
    )

METRICS = _M()
