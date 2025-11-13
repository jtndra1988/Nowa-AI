# app/celery_app/app.py

from celery import Celery, Task # <--- Import Task
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Single global Celery app
celery_app = Celery("mars_worker")

# ✅ Use UPPERCASE attributes from settings.py
celery_app.conf.broker_url = settings.CELERY_BROKER_URL
celery_app.conf.result_backend = settings.CELERY_RESULT_BACKEND

# Basic config
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Only look in our app.tasks package
celery_app.autodiscover_tasks(["app.tasks"])

# ==========================================
# ✅ ADD THIS CLASS DEFINITION
# ==========================================
class BaseTaskWithRetry(Task):
    """
    Base task that retries automatically on failure.
    """
    autoretry_for = (Exception,)
    retry_backoff = True
    retry_backoff_max = 600  # 10 minutes
    retry_jitter = True
    max_retries = 3

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {self.name} (ID: {task_id}) failed: {exc}")
        super().on_failure(exc, task_id, args, kwargs, einfo)