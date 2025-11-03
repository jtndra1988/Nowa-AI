# app/rt/aggregator.py
import os, json, time, redis
from collections import defaultdict, deque

REDIS_URL = os.getenv("REDIS_URL","redis://redis:6379/0")
r = redis.Redis.from_url(REDIS_URL)

def run(symbol="BTCUSDT"):
    stream = f"ticks:{symbol}"
    out_stream = f"bars:{symbol}:1s"
    last_id = "0-0"
    bucket = None
    o=h=l=c=None
    while True:
        resp = r.xread({stream: last_id}, count=500, block=1000)
        if not resp:
            continue
        for _stream, entries in resp:
            for _id, fields in entries:
                last_id = _id.decode()
                data = json.loads(fields[b"data"].decode())
                ts = data["ts"]//1000
                p = float(data["last"] or data["bid"] or data["ask"] or 0)
                if p <= 0: continue
                if bucket is None:
                    bucket = ts; o=h=l=c = p,p,p,p
                if ts == bucket:
                    c = p
                    h = max(h,p); l = min(l,p)
                else:
                    # flush prior bar
                    bar = {"ts": bucket, "open": o, "high": h, "low": l, "close": c}
                    r.xadd(out_stream, {"data": json.dumps(bar)}, maxlen=200000, approximate=True)
                    # start new bucket
                    bucket = ts; o=h=l=c = p,p,p,p

if __name__ == "__main__":
    run()
