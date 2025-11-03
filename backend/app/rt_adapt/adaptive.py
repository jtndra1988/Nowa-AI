from functools import wraps
from typing import Callable
from .cooldown_scheduler import should_skip, mark_ran
from .event_bus import get_kv
from .dynamic_params import get_params

def adaptive_task(redis_url: str, symbol_arg: str = "symbol"):
    """
    Use like:
    @celery.task
    @adaptive_task(settings.REDIS_URL, symbol_arg="symbol")
    def my_signal_task(symbol: str):
        ...
    """
    def deco(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            symbol = kwargs.get(symbol_arg) or (args[0] if args else "BTC")
            task_name = fn.__name__
            if should_skip(redis_url, symbol, task_name):
                return {"skipped": True, "reason": "cooldown"}

            # you can pull params to gate logic inside your task
            params = get_params(redis_url, symbol) or {}
            out = fn(*args, **kwargs, _dyn_params=params)
            mark_ran(redis_url, symbol, task_name)
            return out
        return wrapper
    return deco
