# app/infra/alerts.py
import os
import json
import logging
import urllib.request
import urllib.parse
from typing import Any, Dict, Optional

log = logging.getLogger("alerts")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Environment labels used in pretty alerts
APP_ENV = os.getenv("APP_ENV", "DEV").upper()
SERVICE_NAME = os.getenv("SERVICE_NAME", "CELERY")


def send_telegram(text: str) -> bool:
    """
    Low-level Telegram sender. Uses HTML parse_mode.
    Returns True on success, False on any error.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        data = urllib.parse.urlencode(payload).encode()

        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            log.debug("Telegram response: %s", body)
            return True
    except Exception as e:
        # Avoid crashing worker on Telegram issue
        log.warning("telegram_failed: %s", e, extra={"event": "telegram_failed"})
        return False


def fmt_kv(**kw: Any) -> str:
    """
    Simple key=value formatter into HTML <b>k</b>: v lines.
    """
    return "\n".join([f"<b>{k}</b>: {v}" for k, v in kw.items()])


# ----------------------------------------------------------------------
# High-level alert helpers (backwards compatible + pretty)
# ----------------------------------------------------------------------

def alert_task_failure(task_name: str, task_id: str, err: str) -> None:
    """
    Backwards-compatible task failure alert with nicer formatting.
    """
    header = "⚠️ <b>Task failed</b>\n"
    meta = fmt_kv(task=task_name, id=task_id)
    # Truncate error to avoid hitting Telegram message limits
    err_snippet = (err or "").strip()
    if len(err_snippet) > 1500:
        err_snippet = err_snippet[:1500] + "\n[...]"

    body = f"{header}{meta}\n\n<pre>{err_snippet}</pre>"
    send_telegram(body)


def alert_worker_event(ev: str) -> None:
    """
    Simple worker lifecycle events.
    """
    send_telegram(f"👷 <b>Worker</b> {ev} on node.")


# ----------------------------------------------------------------------
# Generic pretty log alert API
# ----------------------------------------------------------------------

def alert_log(
    title: str,
    level: str = "INFO",
    meta: Optional[Dict[str, Any]] = None,
    details: Optional[str] = None,
) -> None:
    """
    Generic pretty Telegram log formatter.
    Used by alert_info/success/warning/error and by the logging handler.
    """
    meta = meta or {}
    level = level.upper()

    emoji = {
        "DEBUG": "🧪",
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "🛑",
        "CRITICAL": "🔥",
    }.get(level, "ℹ️")

    header = f"{emoji} <b>{title}</b>\n"
    env_line = f"<b>Env</b>: {APP_ENV} | <b>Service</b>: {SERVICE_NAME}\n"

    meta_lines = ""
    if meta:
        meta_lines = "\n".join([f"<b>{k}</b>: {v}" for k, v in meta.items()]) + "\n"

    details_block = ""
    if details:
        d = details.strip()
        if len(d) > 3500:
            d = d[:3500] + "\n[...]"
        details_block = f"\n<pre>{d}</pre>"

    text = header + env_line + (meta_lines or "") + details_block
    send_telegram(text)


def alert_info(title: str, **meta: Any) -> None:
    alert_log(title=title, level="INFO", meta=meta)


def alert_success(title: str, **meta: Any) -> None:
    alert_log(title=title, level="SUCCESS", meta=meta)


def alert_warning(title: str, **meta: Any) -> None:
    alert_log(title=title, level="WARNING", meta=meta)


def alert_error(title: str, error: str, **meta: Any) -> None:
    alert_log(title=title, level="ERROR", meta=meta, details=error)


# ----------------------------------------------------------------------
# Logging handler to mirror Python logs to Telegram
# ----------------------------------------------------------------------

class TelegramLogHandler(logging.Handler):
    """
    Logging handler that forwards log records to Telegram using send_telegram().
    Attach it to the root logger via setup_telegram_logging().
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Avoid infinite recursion from our own alerts logger
        if record.name.startswith("alerts"):
            return

        try:
            msg = self.format(record)
            level = record.levelname.upper()
            title = f"{record.levelname} | {record.name}"

            meta = {
                "logger": record.name,
                "module": record.module,
                "line": record.lineno,
            }

            alert_log(
                title=title,
                level=level,
                meta=meta,
                details=msg,
            )
        except Exception:
            self.handleError(record)


def setup_telegram_logging(level: int = logging.INFO) -> None:
    """
    Attach TelegramLogHandler to the root logger.
    Call this once during worker startup.
    """
    root = logging.getLogger()

    # Ensure root level is not higher than our handler level
    root.setLevel(level)

    # Don't attach twice
    for h in root.handlers:
        if isinstance(h, TelegramLogHandler):
            return

    handler = TelegramLogHandler()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(fmt)
    handler.setLevel(level)

    root.addHandler(handler)
