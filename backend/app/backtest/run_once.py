from app.backtest.engine import BacktestEngine
from app.backtest.strategies_ensemble import EnsemblePaths, EnsembleModelAdapter, EnsembleBacktestStrategy, EnsembleSignalConfig

paths = EnsemblePaths(root="/app/models/BTCUSDT_1h")  # adjust to your saved model folder
adapter = EnsembleModelAdapter(paths).load()
cfg = EnsembleSignalConfig(buy_threshold=0.58, sell_threshold=0.58, sl_atr_mult=1.8, tp_atr_mult=3.0)

eng = BacktestEngine("BTC/USDT", fee_bps=6.0, start_equity=100000.0)
res = eng.run_once(df, EnsembleBacktestStrategy(adapter, "BTC/USDT", cfg))
print(res)
