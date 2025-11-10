from sqlalchemy import JSON, Column, Index, Integer, String, Float, DateTime, BigInteger, Boolean, UniqueConstraint, func
from sqlalchemy.sql import func
from .base import Base
from datetime import datetime
class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), index=True, nullable=False)

    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    # Optional extra fields (keep them, they won’t interfere)
    last_price = Column(Float, nullable=True)
    bid_price = Column(Float, nullable=True)
    ask_price = Column(Float, nullable=True)
    iv = Column(Float, nullable=True)
    delta = Column(Float, nullable=True)
    gamma = Column(Float, nullable=True)
    vega = Column(Float, nullable=True)
    theta = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_market_symbol_timestamp", "symbol", "timestamp", unique=True),
    )

class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    trade_id = Column(String, unique=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String) # e.g., 'buy' or 'sell'
    price = Column(Float)
    quantity = Column(Float)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    pnl = Column(Float, nullable=True)


class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True)
    quantity = Column(Float)
    average_entry_price = Column(Float)
    side = Column(String) # 'long' or 'short'
    is_open = Column(Boolean, default=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
class FuturesMarketData(Base):
    __tablename__ = "futures_market_data"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), index=True, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    __table_args__ = (
        Index("ix_futures_symbol_timestamp", "symbol", "timestamp", unique=True),
    )
class SentimentData(Base):
    """Table to store news headlines and their sentiment score."""
    __tablename__ = 'sentiment_data'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String, index=True, nullable=False)
    source = Column(String, nullable=False)
    headline = Column(String, nullable=False)
    sentiment_score = Column(Float, nullable=True)

class FundingRate(Base):
    __tablename__ = 'funding_rates'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String, index=True, nullable=False)
    funding_rate = Column(Float, nullable=False)

class FearAndGreedIndex(Base):
    __tablename__ = 'fear_and_greed_index'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    value = Column(Integer, nullable=False)
    value_classification = Column(String, nullable=False)

class MacroData(Base):
    __tablename__ = 'macro_data'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    indicator = Column(String, index=True, nullable=False) # e.g., 'DXY'
    value = Column(Float, nullable=False)
class OnchainMetrics(Base):
    __tablename__ = 'onchain_metrics'

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    whale_tx_count = Column(Integer)
    whale_volume_usd = Column(Float)
    exchange_net_flow_usd = Column(Float)
    active_addresses = Column(Integer)
    total_tx = Column(Integer)
    source = Column(String(50))
class DeveloperActivity(Base):
    __tablename__ = 'developer_activity'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String, index=True, nullable=False)
    repo_url = Column(String, nullable=False)
    stars = Column(Integer)
    forks = Column(Integer)
    open_issues = Column(Integer)   

class OrderbookSnapshot(Base):
    __tablename__ = "orderbook_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), index=True, nullable=False)
    symbol = Column(String(32), index=True, nullable=False)  # e.g. "BTC/USDT"
    # aggregated over top N levels at snapshot time
    bid_volume = Column(Float, nullable=False)
    ask_volume = Column(Float, nullable=False)
    mid_price = Column(Float, nullable=False)

    # derived features
    bid_ask_imb = Column(Float, nullable=False)     # (bid - ask) / (bid + ask)
    vw_price_skew = Column(Float, nullable=False)   # (vwap_asks - vwap_bids) / mid
    cdv_1m = Column(Float, nullable=False, default=0.0)  # cumulative delta volume over last 1 min

    __table_args__ = (
        Index("ix_obs_ts_sym", "timestamp", "symbol"),
    )


class CrossAssetCorr(Base):
    __tablename__ = "cross_asset_corr"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), index=True, nullable=False)
    base_symbol = Column(String(16), index=True, nullable=False)  # e.g. "BTC"
    window_minutes = Column(Integer, nullable=False)              # e.g. 60, 240, 1440

    # rolling correlations on returns
    corr_btc_eth  = Column(Float, nullable=False)
    corr_btc_dxy  = Column(Float, nullable=False)
    corr_btc_ndx  = Column(Float, nullable=False)
    corr_btc_gold = Column(Float, nullable=False)

    __table_args__ = (
        Index("ix_cac_ts_sym_win", "timestamp", "base_symbol", "window_minutes"),
    )
class SentimentFusion(Base):
    """Aggregated sentiment score per symbol per time window."""
    __tablename__ = "sentiment_fusion"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), index=True, nullable=False)
    symbol = Column(String(32), index=True, nullable=False)

    avg_news_sentiment = Column(Float, default=0.0)
    whale_emotion = Column(Float, default=0.0)
    fear_greed_index = Column(Float, default=0.0)
    social_score = Column(Float, default=0.0)
    final_sentiment = Column(Float, default=0.0)

    __table_args__ = (
        Index("ix_sentiment_fusion_symbol_ts", "symbol", "timestamp"),
    )
class OptionsDerivedMetrics(Base):
    __tablename__ = 'options_derived_metrics'
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False) # Underlying symbol (e.g., BTC)
    timestamp = Column(DateTime(timezone=True), index=True, nullable=False, default=datetime.utcnow)
    total_put_volume = Column(Float, nullable=True)
    total_call_volume = Column(Float, nullable=True)
    put_call_volume_ratio = Column(Float, nullable=True)
    total_put_oi = Column(Float, nullable=True) # Open Interest
    total_call_oi = Column(Float, nullable=True)
    put_call_oi_ratio = Column(Float, nullable=True)
    avg_iv_near_term = Column(Float, nullable=True) # Average IV for options expiring soon
    avg_iv_mid_term = Column(Float, nullable=True)
    iv_skew_25d = Column(Float, nullable=True) # E.g., (25d Put IV - 25d Call IV) / ATM IV

    # --- New ML-friendly fields ---
    iv_atm_near = Column(Float, nullable=True)  # ATM IV for ~7D bucket
    iv_atm_mid  = Column(Float, nullable=True)  # ATM IV for ~30D bucket
    iv_atm_far  = Column(Float, nullable=True)  # ATM IV for ~90D bucket

    # Term structure slopes
    iv_term_slope_near_mid = Column(Float, nullable=True)  # iv_mid - iv_near
    iv_term_slope_near_far = Column(Float, nullable=True)  # iv_far - iv_near
    iv_term_slope_reg_logT = Column(Float, nullable=True)  # slope from OLS(IV ~ a + b*log(T_days))

    # Smile (same expiry) — 25-delta convexity / curvature
    smile_25d_put_iv  = Column(Float, nullable=True)
    smile_atm_iv      = Column(Float, nullable=True)  # duplicate of iv_atm_mid if you prefer mid bucket
    smile_25d_call_iv = Column(Float, nullable=True)
    smile_curvature   = Column(Float, nullable=True)  # 0.5*(IV_P25 + IV_C25) - IV_ATM

    # Aggregate Gamma Exposure
    gex_total = Column(Float, nullable=True)    # sum over all options
    gex_calls = Column(Float, nullable=True)
    gex_puts  = Column(Float, nullable=True)
    __table_args__ = (UniqueConstraint('symbol', 'timestamp', name='_symbol_timestamp_uc_options_derived'),)    

class OptionsChain(Base):
    __tablename__ = "options_chain"

    id = Column(Integer, primary_key=True, index=True)

    # Underlying symbol key you pass at training time (e.g., "BTC" or "BTC/USDT")
    symbol = Column(String, index=True, nullable=False)

    # Contract identifiers
    expiry = Column(DateTime(timezone=True), index=True, nullable=False)  # or use Date if you prefer
    strike = Column(Float, index=True, nullable=False)
    option_type = Column(String(4), index=True, nullable=False)  # 'C'/'P' or 'CALL'/'PUT'

    # Time series
    timestamp = Column(DateTime(timezone=True), index=True, nullable=False)

    # Prices & marketability
    bid = Column(Float, nullable=True)
    ask = Column(Float, nullable=True)
    last_price = Column(Float, nullable=True)
    mark_price = Column(Float, nullable=True)

    # Liquidity & OI
    volume = Column(Float, nullable=True)
    open_interest = Column(Float, nullable=True)

    # Implied vol & Greeks (optional but useful)
    iv = Column(Float, nullable=True)
    delta = Column(Float, nullable=True)
    gamma = Column(Float, nullable=True)
    vega  = Column(Float, nullable=True)
    theta = Column(Float, nullable=True)

    __table_args__ = (
        # fast lookups & de-duplication per contract per timestamp
        UniqueConstraint("symbol", "expiry", "strike", "option_type", "timestamp",
                         name="uq_optchain_sym_ex_str_ot_ts"),
        Index("ix_optchain_sym_ts", "symbol", "timestamp"),
        Index("ix_optchain_contract", "symbol", "expiry", "strike", "option_type"),
    )
class RiskSettingsGlobal(Base):
    __tablename__ = "risk_settings_global"
    id = Column(Integer, primary_key=True)
    # global knobs (defaults you want at system level)
    account_equity_usd = Column(Float, nullable=False, default=100000.0)
    max_portfolio_leverage = Column(Float, nullable=False, default=2.0)
    max_concurrent_positions = Column(Integer, nullable=False, default=10)

    target_daily_vol = Column(Float, nullable=False, default=0.01)
    min_position_usd = Column(Float, nullable=False, default=200.0)

    atr_length = Column(Integer, nullable=False, default=14)
    sl_atr_mult_init = Column(Float, nullable=False, default=1.8)
    tp_atr_mult_init = Column(Float, nullable=False, default=3.0)

    base_confidence_cutoff = Column(Float, nullable=False, default=0.55)
    max_confidence_boost = Column(Float, nullable=False, default=1.6)
    regime_risk_multipliers = Column(JSON, nullable=False, default={"0":0.8,"1":1.1,"2":1.0})

    daily_loss_limit_pct = Column(Float, nullable=False, default=0.03)
    rolling_max_dd_pct = Column(Float, nullable=False, default=0.15)
    per_trade_risk_pct_cap = Column(Float, nullable=False, default=0.01)

    ewma_alpha = Column(Float, nullable=False, default=0.2)
    min_trades_to_learn = Column(Integer, nullable=False, default=20)
    bandit_lr = Column(Float, nullable=False, default=0.05)

    min_tp_sl_ratio = Column(Float, nullable=False, default=1.3)
    min_sl_atr_mult = Column(Float, nullable=False, default=0.8)
    max_sl_atr_mult = Column(Float, nullable=False, default=4.0)
    min_tp_atr_mult = Column(Float, nullable=False, default=1.2)
    max_tp_atr_mult = Column(Float, nullable=False, default=6.0)

    taker_fee_bps = Column(Float, nullable=False, default=6.0)  # 6 bps = 0.06%
    default_adapter = Column(String, nullable=False, default="BYBIT")  # or "BINANCE"
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class RiskSettingsSymbol(Base):
    __tablename__ = "risk_settings_symbol"
    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True, unique=True, nullable=False)
    # any field here overrides global when non-null
    max_symbol_leverage = Column(Float, nullable=True)
    max_symbol_exposure_pct = Column(Float, nullable=True)
    atr_length = Column(Integer, nullable=True)
    sl_atr_mult_init = Column(Float, nullable=True)
    tp_atr_mult_init = Column(Float, nullable=True)
    base_confidence_cutoff = Column(Float, nullable=True)
    regime_risk_multipliers = Column(JSON, nullable=True)  # e.g. {"0":0.9,"1":1.2,"2":0.95}
    min_position_usd = Column(Float, nullable=True)
    per_trade_risk_pct_cap = Column(Float, nullable=True)
    taker_fee_bps = Column(Float, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class RiskLearnedState(Base):
    __tablename__ = "risk_learned_state"
    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True, unique=True, nullable=False)
    sl_atr_mult = Column(Float, nullable=False, default=1.8)
    tp_atr_mult = Column(Float, nullable=False, default=3.0)
    trades = Column(Integer, nullable=False, default=0)
    wins = Column(Integer, nullable=False, default=0)
    gross_pnl = Column(Float, nullable=False, default=0.0)
    gross_pnl_after_fees = Column(Float, nullable=False, default=0.0)
    avg_win = Column(Float, nullable=False, default=0.0)
    avg_loss = Column(Float, nullable=False, default=0.0)
    trading_day = Column(String, nullable=False, default="")
    day_pnl = Column(Float, nullable=False, default=0.0)
    rolling_equity_peak = Column(Float, nullable=False, default=100000.0)
    rolling_equity_min = Column(Float, nullable=False, default=100000.0)
    last_regime = Column(Integer, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())    
class HybridSignal(Base):
    __tablename__ = "hybrid_signals"

    id = Column(Integer, primary_key=True, index=True)

    # When we generated this decision
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
        nullable=False,
    )

    # What instrument we’re talking about
    symbol = Column(String, index=True, nullable=False)
    instrument_type = Column(String, index=True, nullable=False)  # spot/perp/future/option
    exchange = Column(String, index=True, nullable=True)

    # Core decision fields
    direction = Column(String, nullable=False)        # long/short/flat
    p_edge = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    size_factor = Column(Float, nullable=False)
    strategy_tag = Column(String, nullable=False)
    meta_execute = Column(Boolean, nullable=False)

    # Full debug payload (experts, specialists, meta outputs, weights)
    debug_payload = Column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_hybrid_signals_symbol_created_at", "symbol", "created_at"),
    )

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True, nullable=False)   # e.g. 'HybridEnsemble'
    version = Column(String, nullable=False)                  # e.g. '2025-11-10_001'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    metadata = Column(JSON, nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, index=True, nullable=True)
    symbol = Column(String, index=True, nullable=False)
    prediction_time = Column(DateTime(timezone=True), index=True, nullable=False)
    prediction = Column(Float, nullable=False)        # e.g. price or log-return forecast
    raw_score = Column(Float, nullable=True)          # e.g. vol forecast, or other
    model_inputs = Column(JSON, nullable=True)        # feature_importance / context

    __table_args__ = (
        Index("ix_predictions_sym_time", "symbol", "prediction_time"),
    )
class AIExecutionLog(Base):
    _tablename_ = "ai_execution_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # --- Linkage to Hybrid Signal / Prediction ---
    hybrid_signal_id = Column(Integer, ForeignKey("hybrid_signals.id"), nullable=False)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True)

    # --- Decision Metadata ---
    decided_action = Column(String(50), nullable=False)   # LONG / SHORT / HOLD / FLAT
    execution_style = Column(String(50), nullable=True)   # RL, conservative, aggressive, rule_fallback, etc.
    target_position = Column(Float, nullable=True)        # Suggested net exposure (-1 to 1)
    executed_position = Column(Float, nullable=True)      # What was actually taken
    executed_price = Column(Float, nullable=True)
    exchange = Column(String(50), nullable=True)
    symbol = Column(String(50), nullable=True)

    # --- Performance Metrics ---
    realized_pnl = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, nullable=True)
    slippage = Column(Float, nullable=True)
    latency_ms = Column(Integer, nullable=True)           # time from signal → execution
    confidence = Column(Float, nullable=True)
    p_edge = Column(Float, nullable=True)

    # --- Diagnostic Payloads ---
    rl_diagnostics = Column(JSON, nullable=True)          # RL debug (policy weights, reward components)
    exec_meta = Column(JSON, nullable=True)               # Exchange/trade details (order ids, etc.)

    # --- Relations ---
    hybrid_signal = relationship("HybridSignal", backref="execution_logs")
    model_version = relationship("ModelVersion")

    def _repr_(self):
        return f"<AIExecutionLog(id={self.id}, symbol={self.symbol}, action={self.decided_action})>"

