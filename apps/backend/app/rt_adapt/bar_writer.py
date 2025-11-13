# app/rt/bar_writer.py
import os, json, time, redis
import psycopg2, psycopg2.extras as extras

REDIS_URL = os.getenv("REDIS_URL","redis://redis:6379/0")
PG_URL = os.getenv("SQLALCHEMY_DATABASE_URI","postgresql://nowa:nowa@db:5432/nowa").replace("+psycopg2","")
r = redis.Redis.from_url(REDIS_URL)

def write_loop(symbol="BTCUSDT"):
    stream = f"bars:{symbol}:1s"
    last_id = "0-0"
    conn = psycopg2.connect(PG_URL); conn.autocommit=True
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_1s (
          symbol text, ts timestamptz, open double precision, high double precision,
          low double precision, close double precision,
          PRIMARY KEY(symbol, ts)
        );
    """)
    while True:
        resp = r.xread({stream: last_id}, count=1000, block=1500)
        if not resp:
            continue
        rows=[]
        for _stream, entries in resp:
            for _id, fields in entries:
                last_id = _id.decode()
                data = json.loads(fields[b"data"].decode())
                rows.append((
                    symbol, time.strftime("%Y-%m-%d %H:%M:%S+00", time.gmtime(data["ts"])),
                    data["open"], data["high"], data["low"], data["close"]
                ))
        if rows:
            extras.execute_batch(cur, """
              INSERT INTO ohlcv_1s(symbol, ts, open, high, low, close)
              VALUES (%s,%s,%s,%s,%s,%s)
              ON CONFLICT (symbol, ts) DO UPDATE
              SET open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close;
            """, rows, page_size=500)

if __name__ == "__main__":
    write_loop()
