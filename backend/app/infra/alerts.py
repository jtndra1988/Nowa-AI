# app/infra/alerts.py
import os, json, time, logging, urllib.request, urllib.parse
log = logging.getLogger("alerts")

TELEGRAM_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TG_CHAT_ID", "")

def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}).encode()
        with urllib.request.urlopen(urllib.request.Request(url, data=data)) as _:
            return True
    except Exception as e:
        log.warning("telegram_failed: %s", e, extra={"event": "telegram_failed"})
        return False

def fmt_kv(**kw) -> str:
    return "\n".join([f"<b>{k}</b>: {v}" for k, v in kw.items()])

def alert_task_failure(task_name: str, task_id: str, err: str):
    send_telegram(f"⚠️ <b>Task failed</b>\n{fmt_kv(task=task_name, id=task_id)}\n<pre>{err[:900]}</pre>")

def alert_worker_event(ev: str):
    send_telegram(f"👷 <b>Worker</b> {ev} on node.")
