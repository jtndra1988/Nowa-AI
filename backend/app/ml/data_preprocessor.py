# unchanged (rich FE + joins); this feeds the calibrated fusion pipeline

import logging
from typing import List, Tuple
from venv import logger
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from app.core.config import settings
from app.db.models import OptionsDerivedMetrics
import sys
# -------------------------------------------------------------------------
# Database Engine
# -------------------------------------------------------------------------
engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)


# -------------------------------------------------------------------------
# Phase 1 — Technical Indicators
# -------------------------------------------------------------------------
def add_technical_indicators(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Adds indicator cluster with consistent names.
    Works with OHLCV; if some columns are missing, falls back safely.
    """
    import pandas_ta as ta

    out = df.copy()
    out["close"] = out[price_col]
    high = out["high"] if "high" in out.columns else out["close"]
    low  = out["low"]  if "low"  in out.columns else out["close"]
    vol  = (out["volume"] if "volume" in out.columns else pd.Series(0, index=out.index)).astype(float)

    # --- Trend indicators ---
    out["EMA_20"] = ta.ema(out["close"], length=20)
    out["EMA_50"] = ta.ema(out["close"], length=50)
    macd = ta.macd(out["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        out["MACD"]      = macd["MACD_12_26_9"]
        out["MACD_hist"] = macd["MACDh_12_26_9"]
        out["MACD_sig"]  = macd["MACDs_12_26_9"]

    # --- Momentum indicators ---
    out["RSI_14"] = ta.rsi(out["close"], length=14)
    stoch = ta.stoch(high, low, out["close"])
    if stoch is not None:
        out["STOCH_K"] = stoch["STOCHk_14_3_3"]
        out["STOCH_D"] = stoch["STOCHd_14_3_3"]
    try:
       # Initialize the column with NaN and specify float dtype FIRST
       out["MFI_14"] = pd.Series(np.nan, index=out.index, dtype=float)
       # Now assign the calculated MFI values
       mfi_result = ta.mfi(high, low, out["close"], vol, length=14)
       if mfi_result is not None: # Check if ta.mfi returned something
          out["MFI_14"] = mfi_result

    except Exception as e: # Catch the specific exception if possible
       def warn(msg: str):
         print(f"[preprocess] MFI calculation error: {msg}", file=sys.stderr) # Log the error
       # Ensure column exists even if calculation fails
       if "MFI_14" not in out.columns:
           out["MFI_14"] = pd.Series(np.nan, index=out.index, dtype=float)
    # --- Volatility cluster ---
    bb = ta.bbands(out["close"], length=20)
    if bb is not None:
        out["BB_w"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / (out["close"] + 1e-9)
    out["ATR_14"] = ta.atr(high, low, out["close"], length=14)

    # --- Volume cluster ---
    try:
        out["OBV"] = ta.obv(out["close"], vol)
    except Exception:
        def warn(msg: str):
          print(f"[preprocess] {msg}", file=sys.stderr)
        out["OBV"] = np.nan
    try:
       # Ensure inputs are float64 BEFORE calling ta.mfi
       high_f = high.astype(float)
       low_f = low.astype(float)
       close_f = out["close"].astype(float)
       vol_f = vol.astype(float) # vol should already be float from earlier fix

       # Initialize the column first (as before)
       out["MFI_14"] = pd.Series(np.nan, index=out.index, dtype=float)
       mfi_result = ta.mfi(high_f, low_f, close_f, vol_f, length=14)
       if mfi_result is not None:
         out["MFI_14"] = mfi_result

    except Exception as e:
       logger = logging.getLogger(__name__) # Use logger
       logger.warning(f"MFI calculation error: {e}") # Log the error
       if "MFI_14" not in out.columns:
         out["MFI_14"] = pd.Series(np.nan, index=out.index, dtype=float)    
    out["VOL_PCT_CHANGE"] = vol.pct_change()
    out = out.bfill().ffill()
    out = out.iloc[50:].reset_index(drop=True) # warm-up burn-in
    return out


# -------------------------------------------------------------------------
# Market Regime Detection
# -------------------------------------------------------------------------
def detect_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Detects regimes using ADX + trend slope."""
    import pandas_ta as ta
    df = df.copy()
    try:
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["ADX_14"] = adx["ADX_14"]
    except Exception:
        df["ADX_14"] = 0

    df["trend"] = df["close"].pct_change().rolling(5).mean()
    df["regime"] = np.select(
        [
            df["ADX_14"] < 20,
            (df["ADX_14"] >= 20) & (df["trend"] > 0),
            (df["ADX_14"] >= 20) & (df["trend"] <= 0),
        ],
        [0, 1, 2],  # 0=range, 1=uptrend, 2=downtrend
    )
    return df


# -------------------------------------------------------------------------
# Multi-Timeframe Resampling
# -------------------------------------------------------------------------
def resample_multi_timeframes(df: pd.DataFrame, price_col: str = "close") -> dict:
    """
    Resamples dataframe into multiple timeframes (MTF) and returns dict of indicators.
    Uses updated time codes and pct_change fill_method.
    """
    mtf_dict = {}
    # Use NEW time codes: 'min', 'h', 'D'
    for tf_in, tf_out in [("5T", "5min"), ("1H", "1h"), ("4H", "4h"), ("1D", "1D")]:
        # Use the output code (e.g., '5min') for resampling
        dfr = df[price_col].resample(tf_out).ohlc()
        # Add fill_method=None to pct_change
        dfr[f"ret_{tf_in}"] = dfr["close"].pct_change(fill_method=None)
        dfr[f"ema_{tf_in}"] = dfr["close"].ewm(span=10, min_periods=5).mean()
        mtf_dict[tf_in] = dfr # Keep original keys ('5T', '1H') if subsequent code expects them
        # Or use new keys: mtf_dict[tf_out] = dfr
    return mtf_dict
# -------------------------------------------------------------------------
# Main Preprocessor (Phase 1 → 3)
# -------------------------------------------------------------------------
def preprocess_raw_chunk(
    df_chunk: pd.DataFrame,
    symbol: str,
    engine=None,
    include_mtf: bool = True,
    mtf_list: List[str] = ("5min", "1h", "4h", "1D"),
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Full feature-engineering pipeline:
      - Technicals + regime
      - Multi-timeframe joins
      - Funding, Macro, On-chain, Dev, Orderbook
      - Cross-asset correlations (+ MA + vol)
      - Sentiment / Fear-Greed / Fusion
    """
    if engine is None:
        engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)

    df = df_chunk.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # ---- Phase 1: indicators + regime
    df = add_technical_indicators(df, price_col="close" if "close" in df.columns else "last_price")
    df = detect_regime(df)

    # ---- Multi-timeframe
    if include_mtf:
        base = df.set_index("timestamp")
        mtf = resample_multi_timeframes(base, price_col="close")
        for tf, dft in mtf.items():
            df = df.set_index("timestamp").join(dft, how="left", rsuffix="_resampled").reset_index()

    # ---- Funding rates
    try:
        fr = pd.read_sql("SELECT timestamp,symbol,funding_rate FROM funding_rates",
                         engine, parse_dates=["timestamp"])
        fr = fr[fr["symbol"].str.upper() == f"{symbol.upper()}/USDT"].drop(columns=["symbol"])
        df = pd.merge_asof(df.sort_values("timestamp"), fr.sort_values("timestamp"),
                           on="timestamp", direction="backward")
    except Exception:
        def warn(msg: str):
         print(f"[preprocess] {msg}", file=sys.stderr)
        df["funding_rate"] = 0.0

    # ---- Macro (DXY)
    try:
        macro = pd.read_sql("SELECT timestamp,indicator,value FROM macro_data",
                            engine, parse_dates=["timestamp"])
        dxy = macro[macro["indicator"] == "DXY"].drop(columns=["indicator"]).sort_values("timestamp")
        df = pd.merge_asof(df.sort_values("timestamp"), dxy, on="timestamp", direction="backward")
        df = df.rename(columns={"value": "DXY"})
    except Exception:
        def warn(msg: str):
         print(f"[preprocess] {msg}", file=sys.stderr)
        df["DXY"] = 0.0

    # ---- On-chain
    try:
        onc = pd.read_sql("SELECT * FROM onchain_metrics", engine, parse_dates=["timestamp"])
        onc = onc[onc["symbol"].str.lower() == symbol.lower()].drop(columns=["id", "symbol", "source"], errors="ignore")
        df = pd.merge_asof(df.sort_values("timestamp"), onc.sort_values("timestamp"),
                           on="timestamp", direction="backward")
    except Exception:
        for c in ["whale_tx_count","whale_volume_usd","exchange_net_flow_usd","active_addresses","total_tx"]:
            if c not in df.columns: df[c] = 0.0

    # ---- Developer Activity
    try:
        dev = pd.read_sql("SELECT timestamp,symbol,stars,forks,open_issues FROM developer_activity",
                          engine, parse_dates=["timestamp"])
    # ...

        dev = dev[dev["symbol"].str.upper() == symbol.upper()].drop(columns=["symbol"])
        df = pd.merge_asof(df.sort_values("timestamp"), dev.sort_values("timestamp"),
                           on="timestamp", direction="backward")
    except Exception:
        for c in ["stars","forks","open_issues"]:
            if c not in df.columns: df[c] = 0.0

    # ---- Orderbook
    try:
        obs = pd.read_sql("SELECT timestamp,symbol,bid_ask_imb,vw_price_skew,cdv_1m FROM orderbook_snapshots",
                          engine, parse_dates=["timestamp"])
        obs = obs[obs["symbol"].str.upper() == f"{symbol.upper()}/USDT"].drop(columns=["symbol"])
        df = pd.merge_asof(df.sort_values("timestamp"), obs.sort_values("timestamp"),
                           on="timestamp", direction="backward")
    except Exception:
        for c in ["bid_ask_imb","vw_price_skew","cdv_1m"]:
            if c not in df.columns: df[c] = 0.0

    # ---- Cross-asset correlations (with MA + vol)
    try:
        cac = pd.read_sql(text("""
        SELECT timestamp,base_symbol,window_minutes,
               corr_btc_eth,corr_btc_dxy,corr_btc_ndx,corr_btc_gold
        FROM cross_asset_corr WHERE base_symbol=:sym
        """), engine, params={"sym": symbol.split("/")[0].upper()}, parse_dates=["timestamp"])
        if not cac.empty:
            cac = cac.sort_values(["timestamp","window_minutes"])
            piv = cac.pivot_table(index="timestamp", columns="window_minutes",
                                  values=["corr_btc_eth","corr_btc_dxy","corr_btc_ndx","corr_btc_gold"])
            piv.columns = [f"{m}_{w}" for m,w in piv.columns.to_list()]
            piv = piv.sort_index().reset_index()
            for base in ["corr_btc_eth","corr_btc_dxy","corr_btc_ndx","corr_btc_gold"]:
                for w in [60,240,1440]:
                    col=f"{base}_{w}"
                    if col in piv.columns:
                        piv[f"{col}_ma3"]=piv[col].rolling(3,min_periods=1).mean()
                        piv[f"{col}_vol"]=piv[col].rolling(10,min_periods=3).std()
            df=pd.merge_asof(df.sort_values("timestamp"),piv.sort_values("timestamp"),
                             on="timestamp",direction="backward")
    except Exception as e:
        print(f"[!] Cross-asset corr merge failed for {symbol}: {e}")
        for base in ["corr_btc_eth","corr_btc_dxy","corr_btc_ndx","corr_btc_gold"]:
            for w in [60,240,1440]:
                for suffix in ["","_ma3","_vol"]:
                    c=f"{base}_{w}{suffix}"
                    if c not in df.columns: df[c]=0.0

    # ---- Sentiment
    try:
        news = pd.read_sql("SELECT timestamp,symbol,sentiment_score FROM sentiment_data",
                           engine, parse_dates=["timestamp"])
        news = news[news["symbol"].str.upper() == symbol.upper()].drop(columns=["symbol"])
        news = news.set_index("timestamp").sort_index()
        news["sentiment_score"]=news["sentiment_score"].fillna(0.0)
        news_1h=news.resample("1h").agg({"sentiment_score":["mean","count"]})
        news_1h.columns=["sentiment_mean_1h","news_count_1h"]
        news_24h=news.resample("1h").agg({"sentiment_score":"mean"}).rolling(24,min_periods=1).mean()
        news_24h.columns=["sentiment_mean_24h"]
        senti=news_1h.join(news_24h,how="outer").reset_index()
        df=pd.merge_asof(df.sort_values("timestamp"),senti.sort_values("timestamp"),
                         on="timestamp",direction="backward")
    except Exception:
        for c in ["sentiment_mean_1h","news_count_1h","sentiment_mean_24h"]:
            if c not in df.columns: df[c]=0.0

    # ---- Fear & Greed Index
    try:
        fng=pd.read_sql("SELECT timestamp,value FROM fear_and_greed_index",engine,parse_dates=["timestamp"])
        fng=fng.sort_values("timestamp")
        df=pd.merge_asof(df.sort_values("timestamp"),fng,on="timestamp",direction="backward")
        df=df.rename(columns={"value":"fear_greed"})
        df["fg_extreme_fear"] =(df["fear_greed"]<25).astype(int)
        df["fg_fear"]         =((df["fear_greed"]>=25)&(df["fear_greed"]<45)).astype(int)
        df["fg_neutral"]      =((df["fear_greed"]>=45)&(df["fear_greed"]<=55)).astype(int)
        df["fg_greed"]        =((df["fear_greed"]>55)&(df["fear_greed"]<=75)).astype(int)
        df["fg_extreme_greed"]=(df["fear_greed"]>75).astype(int)
    except Exception:
        df["fear_greed"]=50.0
        for c in ["fg_extreme_fear","fg_fear","fg_neutral","fg_greed","fg_extreme_greed"]:
            if c not in df.columns: df[c]=0

    # ---- Fused Sentiment
    try:
        sf=pd.read_sql("SELECT timestamp,symbol,final_sentiment FROM sentiment_fusion",
                       engine,parse_dates=["timestamp"])
        sf=sf[sf["symbol"].str.upper()==f"{symbol.upper()}/USDT"].drop(columns=["symbol"])
        df=pd.merge_asof(df.sort_values("timestamp"),sf.sort_values("timestamp"),
                         on="timestamp",direction="backward")
    except Exception:
        df["final_sentiment"]=0.0
    # ---- Options Derived Metrics ---
    try:
        # Fetch latest derived metrics for the underlying symbol (e.g., 'BTC' from 'BTC/USDT')
        underlying_symbol = symbol.split('/')[0].upper()
        odm_query = text("""
            SELECT timestamp, put_call_volume_ratio, put_call_oi_ratio,
                   avg_iv_near_term, avg_iv_mid_term, iv_skew_25d
            FROM options_derived_metrics
            WHERE symbol = :underlying
            ORDER BY timestamp DESC
            LIMIT 200 -- Fetch a bit of history for joining
        """)
        odm_df = pd.read_sql(odm_query, engine, params={"underlying": underlying_symbol}, parse_dates=["timestamp"])

        if not odm_df.empty:
            odm_df = odm_df.sort_values("timestamp")
            # Rename columns slightly to avoid potential conflicts
            odm_df = odm_df.rename(columns={
                "put_call_volume_ratio": "opt_pcr_vol",
                "put_call_oi_ratio": "opt_pcr_oi",
                "avg_iv_near_term": "opt_iv_near",
                "avg_iv_mid_term": "opt_iv_mid",
                "iv_skew_25d": "opt_skew"
            })
            # Merge as-of, similar to other features
            df = pd.merge_asof(df.sort_values("timestamp"),
                               odm_df.sort_values("timestamp"),
                               on="timestamp",
                               direction="backward",
                               tolerance=pd.Timedelta('1hour')) # Allow matching within an hour

            # Fill NaNs potentially introduced by merge_asof or if metrics are missing
            opt_cols = ["opt_pcr_vol", "opt_pcr_oi", "opt_iv_near", "opt_iv_mid", "opt_skew"]
            for c in opt_cols:
                if c not in df.columns:
                    df[c] = np.nan # Add column if merge failed
            # Forward fill first, then fill remaining NaNs (e.g., at the start) with 0 or median
            df[opt_cols] = df[opt_cols].ffill().fillna(0)

        else:
            # Add columns with default values if no metrics found
            opt_cols = ["opt_pcr_vol", "opt_pcr_oi", "opt_iv_near", "opt_iv_mid", "opt_skew"]
            for c in opt_cols:
                if c not in df.columns: df[c] = 0.0

    except Exception as e:
        print(f"[!] Error joining options derived metrics for {symbol}: {e}")
        # Ensure columns exist even if join fails
        opt_cols = ["opt_pcr_vol", "opt_pcr_oi", "opt_iv_near", "opt_iv_mid", "opt_skew"]
        for c in opt_cols:
            if c not in df.columns: df[c] = 0.0    

    # ---- Final cleanup
    df=df.sort_values("timestamp").bfill().ffill().fillna(0)
    # CORRECTED LINE - 'close' is NO LONGER dropped from the feature list
    drop_cols={"symbol","timestamp","open","high","low","last_price","ret"} # Removed "close"
    # Ensure 'close' is actually in df.columns before creating feature_columns
    if 'close' not in df.columns:
     # Handle case where 'close' might be missing earlier (e.g., options data)
     # Maybe rename 'last_price' if appropriate or raise error
     if 'last_price' in df.columns:
          df = df.rename(columns={'last_price': 'close'})
          logger.warning("Renamed 'last_price' to 'close' as 'close' was missing.")
     else:
          logger.error("'close' column not found in DataFrame before final feature selection.")
          # Return empty list or raise error depending on desired behavior
          return df, []
    feature_columns=[c for c in df.columns if c not in drop_cols]
    # Optionally explicitly add 'close' if it wasn't dropped but might be missing?
    # if 'close' not in feature_columns and 'close' in df.columns:
    #     feature_columns.append('close')
    return df,feature_columns