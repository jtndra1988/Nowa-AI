# app/rt/producer.py
import asyncio, os, json, time, redis
import ccxt.pro as ccxtpro

REDIS_URL = os.getenv("REDIS_URL","redis://redis:6379/0")
r = redis.Redis.from_url(REDIS_URL)

async def stream_ticker(symbol="BTC/USDT", exchange="bybit"):
    ex = getattr(ccxtpro, exchange)()
    try:
        while True:
            t = await ex.watch_ticker(symbol)
            payload = {
                "ts": int(time.time()*1000),
                "symbol": symbol,
                "last": t.get("last"),
                "bid": t.get("bid"),
                "ask": t.get("ask"),
            }
            r.xadd(f"ticks:{symbol.replace('/','')}", {"data": json.dumps(payload)}, maxlen=100000, approximate=True)
    finally:
        await ex.close()

if __name__ == "__main__":
    asyncio.run(stream_ticker())
