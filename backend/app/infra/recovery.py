# app/infra/recovery.py
import json, os, tempfile, logging, time
from pathlib import Path
from typing import Any, Optional
from .metrics import STATE_CHECKPOINTS, STATE_RECOVERIES

log = logging.getLogger("recovery")

class StateStore:
    """
    Very light atomic JSON store for resume tokens/offsets.
    Use per-task keys, e.g., 'collector.news', 'ohlcv.BTCUSDT.1m'.
    """
    def __init__(self, root: str = "/app/state/checkpoints"):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_")
        return self.root / f"{safe}.json"

    def save(self, key: str, state: Any) -> None:
        path = self._path(key)
        tmp = Path(tempfile.mkstemp(dir=str(self.root), prefix=".tmp_")[1])
        json.dump({"ts": int(time.time()), "state": state}, open(tmp, "w"))
        tmp.replace(path)
        STATE_CHECKPOINTS.labels(key).inc()
        log.info("checkpoint_saved", extra={"job_key": key})

    def load(self, key: str) -> Optional[Any]:
        path = self._path(key)
        if not path.exists(): return None
        try:
            payload = json.load(open(path))
            STATE_RECOVERIES.labels(key).inc()
            log.info("checkpoint_loaded", extra={"job_key": key})
            return payload.get("state")
        except Exception as e:
            log.warning("checkpoint_load_failed", extra={"job_key": key, "err": str(e)})
            return None

STATE = StateStore()
