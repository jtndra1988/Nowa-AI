import json
import time
from typing import Any, Dict, Optional
import redis

_redis = None

def get_redis(url: str):
    global _redis
    if _redis is None:
        _redis = redis.from_url(url, decode_responses=True)
    return _redis

def publish(url: str, topic: str, payload: Dict[str, Any]):
    r = get_redis(url)
    msg = {"ts": time.time(), "topic": topic, "payload": payload}
    r.publish(topic, json.dumps(msg))

def set_kv(url: str, key: str, value: Any, ttl: Optional[int] = None):
    r = get_redis(url)
    r.set(key, json.dumps(value))
    if ttl:
        r.expire(key, ttl)

def get_kv(url: str, key: str, default=None):
    r = get_redis(url)
    raw = r.get(key)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default
