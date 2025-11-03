import os, json, time
from celery import shared_task
from prometheus_client import Counter, Gauge
from sqlalchemy import create_engine, text
import pandas as pd

from app.core.config import settings
from app.ml.adv.run_pipeline_example import train_for_symbol

RETRAIN_RUNS = Counter("ml_retrain_runs_total", "Number of ML retrain runs", ["symbol"])
LAST_SHARPE   = Gauge("ml_last_val_sharpe", "Last validation Sharpe for a symbol", ["symbol"])

@shared_task(name="ml.retrain_symbol")
def retrain_symbol(symbol:str="BTC"):
    start = time.time()
    engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
    # TODO: load & align all features for the symbol here
    # df = load_aligned_frame(engine) ; df_sym = df[df["symbol"]==symbol].sort_values("timestamp")
    # Placeholder: query your materialized training view/table:
    df_sym = pd.read_sql(text("SELECT * FROM training_view WHERE symbol=:s ORDER BY ts"), engine, params={"s":symbol})

    out_dir = f"/app/models/{symbol}"
    os.makedirs(out_dir, exist_ok=True)
    train_for_symbol(df_sym, out_dir)
    # read best Sharpe from log
    sharpe = 0.0
    try:
        with open(os.path.join(out_dir, "train_log.jsonl")) as f:
            sharpe = max(float(json.loads(x)["val_sharpe"]) for x in f)
    except Exception:
        pass
    RETRAIN_RUNS.labels(symbol).inc()
    LAST_SHARPE.labels(symbol).set(sharpe)
    return {"symbol":symbol, "val_sharpe":sharpe, "took_s": round(time.time()-start,2)}
