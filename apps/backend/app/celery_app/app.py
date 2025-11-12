from celery import Celery
from app.core.config import settings
import os
celery_app = Celery(
    "worker",
    broker=os.getenv("CELERY_BROKER_URL", f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0"),
)

celery_app.conf.update(
    task_routes={"app.tasks.*": {"queue": "default"}},
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_always_eager=False,
    timezone="UTC",
)

# --------------------------------
# Autodiscover and log confirmation
# --------------------------------
celery_app.autodiscover_tasks(["app.tasks"])

print("[✅] Celery autodiscovery initialized for app.tasks")

# --------------------------------
# Retryable Base Task
# --------------------------------
class BaseTaskWithRetry(celery_app.Task):
    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3, "countdown": 5}
    retry_backoff = True
    retry_jitter = True

# --------------------------------
# Fallback explicit imports
# --------------------------------
try:
    import app.tasks.collectors
    import app.tasks.sentiment_collector
    import app.tasks.cross_asset_corre_collector
    import app.tasks.funding_collector
    import app.tasks.github_collector
    import app.tasks.onchain_collector
    import app.tasks.options_metrics_collector
    import app.tasks.sentiment_fusion_collector
    import app.tasks.training_tasks
    import app.tasks.orderbook_collector
    import app.tasks.sentiment_scorer
    print("[✅] All fallback task modules imported successfully.")
except Exception as e:
    print(f"[!] Task import failed during Celery init: {e}")