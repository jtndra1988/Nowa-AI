# app/infra/metrics.py
import time
from contextlib import contextmanager
from prometheus_client import Counter, Histogram, Gauge

TASK_EXCEPTIONS = Counter("app_task_exceptions_total", "Task exceptions", ["task"])
TASK_DURATION   = Histogram("app_task_duration_seconds", "Task duration (s)", ["task"])
WORKER_HEARTBEAT = Gauge("app_worker_heartbeat_ts", "Unix ts of last worker heartbeat")
STATE_CHECKPOINTS = Counter("app_state_checkpoints_total", "State saves", ["key"])
STATE_RECOVERIES  = Counter("app_state_recoveries_total", "State loads", ["key"])
APP_RESTARTS      = Counter("app_process_restarts_total", "Process restarts")

@contextmanager
def time_task(name: str):
    t0 = time.perf_counter()
    try:
        yield
        TASK_DURATION.labels(name).observe(time.perf_counter() - t0)
    except Exception:
        TASK_EXCEPTIONS.labels(name).inc()
        raise
