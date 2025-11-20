import logging
from celery import group
from app.celery_app.app import celery_app
from app.core.config import settings
from app.rt_adapt.regime_detector import detect_and_store
from app.rt_adapt.dynamic_params import tune
# Adapter to fetch top symbols
from app.exchange.adapters import BybitAdapter 

logger = logging.getLogger(__name__)

@celery_app.task(name="app.tasks.rt_checks.update_market_regime")
def update_market_regime_task(symbol: str):
    """
    Worker Task: Runs for a SINGLE symbol.
    1. Detects Regime (Bull/Bear/Chop).
    2. Tunes Dynamic Parameters (SL/TP multipliers).
    """
    # Safety fallback if called without arg
    if not symbol:
        logger.warning("update_market_regime_task called without symbol")
        return None

    try:
        # 1. Detect Regime
        snap = detect_and_store(
            engine_url=str(settings.SQLALCHEMY_DATABASE_URI),
            redis_url=settings.CELERY_BROKER_URL, 
            symbol=symbol
        )
        
        # 2. Tune Params based on new regime
        tune(
            engine_url=str(settings.SQLALCHEMY_DATABASE_URI),
            redis_url=settings.CELERY_BROKER_URL,
            symbol=symbol
        )
        
        return f"[{symbol}] Regime: {snap.regime} ({snap.vol})"
    except Exception as e:
        logger.error(f"RT Check failed for {symbol}: {e}")
        return None


@celery_app.task(name="app.tasks.rt_checks.run_all_regime_checks")
def run_all_regime_checks():
    """
    Master Task: Runs every minute.
    1. Fetches Top 100 Symbols by Volume from Exchange.
    2. Fans out update_market_regime_task for each.
    """
    try:
        # 1. Dynamic Discovery
        # You can switch this to BinanceDataAdapter if you prefer
        adapter = BybitAdapter() 
        
        # Get top 100 volatile/liquid assets (bases like 'BTC', 'ETH')
        top_bases = adapter.get_top_symbols_by_volume(limit=100)
        
        if not top_bases:
            logger.warning("[RT] No symbols found from adapter.")
            return
            
        # 2. Format as Tradeable Symbols (e.g. BTC/USDT)
        # Ensure this matches what is stored in your DB ('market_data')
        symbols = [f"{base}/USDT" for base in top_bases]

        logger.info(f"[RT] Triggering regime checks for {len(symbols)} symbols.")

        # 3. Fan Out (Parallel Execution)
        # We use a Celery group to run these in parallel workers
        job = group(update_market_regime_task.s(sym) for sym in symbols)
        job.apply_async()
        
        return f"Scheduled {len(symbols)} regime checks."

    except Exception as e:
        logger.error(f"[RT] Master check failed: {e}", exc_info=True)