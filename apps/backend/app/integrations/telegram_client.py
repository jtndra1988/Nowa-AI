# app/integrations/telegram_client.py
import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_message(text: str) -> None:
    """Send a Telegram message, with strong safety + logging.

    This will NO-OP if env vars are missing, and will log
    full error bodies on HTTP 400 to help debug chat_id/token issues.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning(
            "[Telegram] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID; "
            "skipping notification."
        )
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            logger.info("[Telegram] Sent successfully. Response: %s", body)
    except urllib.error.HTTPError as e:
        # <-- This is where we see detailed reason for HTTP 400
        body = ""
        try:
            body = e.read().decode("utf-8") if e.fp else ""
        except Exception:
            pass

        logger.error(
            "[Telegram] HTTPError %s while sending message: %s. Body=%s",
            e.code,
            e.reason,
            body,
        )
    except Exception as e:
        logger.exception("[Telegram] Unexpected error while sending message: %s", e)
