# app/exchange/adapters.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import re

import ccxt
import pandas as pd

from app.core.config import settings
from app.exchange.base import BaseExchangeAdapter


# ---------------------------
# Helpers
# ---------------------------
def _norm_usdt_symbol(sym: str) -> str:
    """Normalize to 'COIN/USDT'."""
    if not sym:
        return sym
    s = sym.upper().replace("USDT/", "").replace("/", "")
    if s.endswith("USDT"):
        s = s[:-4]
    return f"{s}/USDT"


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


# ============================================================
# Bybit (paper mode) — used read-only for data/learning
# ============================================================
class BybitAdapter(BaseExchangeAdapter):
    """
    Bybit adapter (sandbox mode). Provides:
      - get_top_symbols_by_volume()
      - get_options_chain()
      - get_ticker()
      - fetch_ohlcv() / get_futures_ohlcv()
    """

    def __init__(self, paper_mode: bool = True):
        api_key = getattr(settings, "BYBIT_TESTNET_API_KEY", None)
        api_secret = getattr(settings, "BYBIT_TESTNET_API_SECRET", None)

        self.exchange = ccxt.bybit({
            "apiKey": api_key,
            "secret": api_secret,
            "options": {
                "defaultType": "swap",
                "recvWindow": 20000,
            },
        })
        if paper_mode:
            try:
                self.exchange.set_sandbox_mode(True)
            except Exception:
                pass

        try:
            self.exchange.load_markets(True)
            print("[Bybit] markets loaded.")
        except Exception as e:
            print(f"[Bybit] load_markets failed: {e}")

    def get_top_symbols_by_volume(self, limit: int = 10) -> List[str]:
        try:
            tickers = self.exchange.fetch_tickers()
        except Exception as e:
            print(f"[Bybit] fetch_tickers failed: {e}")
            return ["BTC", "ETH"]

        perps = []
        for t in tickers.values():
            if t.get("type") == "swap" and "USDT" in t.get("symbol", "") and t.get("quoteVolume"):
                perps.append(t)

        if not perps:
            return ["BTC", "ETH"]

        perps = sorted(perps, key=lambda x: x["quoteVolume"], reverse=True)[: max(1, limit)]
        return [p["symbol"].split("/")[0] for p in perps]

    def get_options_chain(self, underlying_symbol: str) -> List[Dict[str, Any]]:
        """
        Returns list[dict] rows:
          expiry, strike, option_type ('CALL'/'PUT'), timestamp,
          bid, ask, last_price, mark_price, volume, open_interest,
          iv, delta, gamma, theta, vega
        """
        base = underlying_symbol.upper()
        try:
            res = self.exchange.public_get_v5_market_tickers({
                "category": "option",
                "baseCoin": base,  # e.g., 'BTC'
            })
        except Exception as e:
            print(f"[Bybit] options tickers failed: {e}")
            return []

        rows = (res or {}).get("result", {}).get("list", []) or []
        if not rows:
            return []

        df = pd.DataFrame(rows)
        out: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            try:
                expiry_ms = r.get("deliveryTime") or r.get("expiryDate")
                ts_ms = r.get("updatedTime") or r.get("timestamp")
                option_type = (r.get("optionType") or r.get("type") or "").upper()
                if option_type.startswith("CALL"):
                    option_type = "CALL"
                elif option_type.startswith("PUT"):
                    option_type = "PUT"

                out.append({
                    "expiry": pd.to_datetime(expiry_ms, unit="ms", utc=True).to_pydatetime() if expiry_ms else None,
                    "strike": _safe_float(r.get("strikePrice")),
                    "option_type": option_type,
                    "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True).to_pydatetime() if ts_ms else None,
                    "bid": _safe_float(r.get("bid1Price")),
                    "ask": _safe_float(r.get("ask1Price")),
                    "last_price": _safe_float(r.get("lastPrice")),
                    "mark_price": _safe_float(r.get("markPrice")),
                    "volume": _safe_float(r.get("volume24h")),
                    "open_interest": _safe_float(r.get("openInterest")),
                    "iv": _safe_float(r.get("impliedVolatility")),
                    "delta": _safe_float(r.get("delta")),
                    "gamma": _safe_float(r.get("gamma")),
                    "theta": _safe_float(r.get("theta")),
                    "vega": _safe_float(r.get("vega")),
                })
            except Exception:
                continue
        return out

    def get_ticker(self, symbol: str, params: Optional[dict] = None) -> Dict[str, Any]:
        sym = symbol if "/" in symbol else _norm_usdt_symbol(symbol)
        try:
            t = self.exchange.fetch_ticker(sym, params or {})
            return t or {}
        except Exception as e:
            print(f"[Bybit] fetch_ticker({sym}) failed: {e}")
            return {}

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
        sym = symbol if "/" in symbol else _norm_usdt_symbol(symbol)
        try:
            ohlcv = self.exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
            if not ohlcv:
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["symbol"] = sym
            return df
        except Exception as e:
            print(f"[Bybit] fetch_ohlcv({sym}) failed: {e}")
            return pd.DataFrame()

    def get_futures_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
        return self.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


# ============================================================
# Binance (read-only) — spot/futures + options (EAPI) for data
# ============================================================
class BinanceDataAdapter(BaseExchangeAdapter):
    """
    Binance data adapter (read-only) for training/ETL:
      - get_top_symbols_by_volume()
      - get_options_chain()   (via EAPI — returns normalized rows)
      - get_ticker()
      - fetch_ohlcv() / get_futures_ohlcv()
    """

    BIN_OPT_REGEX = re.compile(
        r"^(?P<base>[A-Z]+)-(?P<y>\d{2})(?P<m>\d{2})(?P<d>\d{2})-(?P<strike>\d+(?:\.\d+)?)-(?P<cp>[CP])$"
    )
    # Example: BTC-241227-50000-C

    def __init__(self):
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",   # USD-M futures/linear for OHLCV/tickers
            }
        })
        try:
            self.exchange.load_markets(True)
            print("[Binance] markets loaded.")
        except Exception as e:
            print(f"[Binance] load_markets failed: {e}")

    def get_top_symbols_by_volume(self, limit: int = 10) -> List[str]:
        try:
            tickers = self.exchange.fetch_tickers()
        except Exception as e:
            print(f"[Binance] fetch_tickers failed: {e}")
            return ["BTC", "ETH"]

        perps = []
        for t in tickers.values():
            typ = t.get("type")
            sym = t.get("symbol", "")
            qv = t.get("quoteVolume")
            if typ in ("future", "swap") and "USDT" in sym and qv:
                perps.append(t)

        if not perps:
            return ["BTC", "ETH"]

        perps = sorted(perps, key=lambda x: x["quoteVolume"], reverse=True)[: max(1, limit)]
        return [p["symbol"].split("/")[0] for p in perps]

    # ---------- Binance Options (EAPI) ----------
    def _parse_option_symbol(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Parse Binance option instrument names like 'BTC-241227-50000-C'
        """
        if not name:
            return None
        m = self.BIN_OPT_REGEX.match(name)
        if not m:
            return None
        gd = m.groupdict()
        base = gd["base"]
        y, mth, d = int(gd["y"]), int(gd["m"]), int(gd["d"])
        # 20xx assumption for yy
        year = 2000 + y
        expiry = pd.Timestamp(year=year, month=mth, day=d, tz="UTC")
        return {
            "base": base,
            "expiry": expiry.to_pydatetime(),
            "strike": _safe_float(gd["strike"]),
            "option_type": "CALL" if gd["cp"] == "C" else "PUT",
        }

    def get_options_chain(self, underlying_symbol: str) -> List[Dict[str, Any]]:
        """
        Returns normalized rows for Binance options via EAPI.
        We try multiple EAPI endpoints (ccxt maps) and fall back gracefully to [].
        """
        base = underlying_symbol.upper()

        # Attempt several public EAPI endpoints ccxt may expose.
        # We expect a list of tickers where each item includes 'symbol' (instrument name) and prices.
        candidates = [
            "eapiPublicGetTicker",             # common in ccxt for binance options
            "eapiPublicGetMark",               # alt
            "eapiPublicGetOpenInterest",       # alt (less pricing info)
        ]
        data: List[Dict[str, Any]] = []

        for meth in candidates:
            if hasattr(self.exchange, meth):
                try:
                    res = getattr(self.exchange, meth)()
                    # res could be dict or list; normalize to list
                    tickers = res if isinstance(res, list) else res.get("data") or res.get("result") or res.get("tickers") or []
                    if isinstance(tickers, dict):
                        tickers = tickers.get("list", []) or []
                    if tickers:
                        data = tickers
                        break
                except Exception as e:
                    print(f"[Binance] {meth} failed: {e}")

        if not data:
            # As a last resort, try ccxt's generic fetch method if exposed
            try:
                # Some ccxt versions expose: self.exchange.eapiPublicGetTickerBookTicker()
                if hasattr(self.exchange, "eapiPublicGetTickerBookTicker"):
                    res = self.exchange.eapiPublicGetTickerBookTicker()
                    data = res if isinstance(res, list) else res.get("data", [])  # best effort
            except Exception as e:
                print(f"[Binance] eapiPublicGetTickerBookTicker failed: {e}")

        if not data:
            return []

        out: List[Dict[str, Any]] = []
        for item in data:
            try:
                # instrument name can be under 'symbol' or 'instrumentId'
                name = item.get("symbol") or item.get("instrumentId") or item.get("symbolName")
                meta = self._parse_option_symbol(name)
                if not meta or meta["base"] != base:
                    continue

                # timestamps may be 'time' or 'updateTime' in ms
                ts_ms = item.get("time") or item.get("updateTime") or item.get("timestamp")
                # prices could be best bid/ask (bidPrice/askPrice) and last/mark
                out.append({
                    "expiry": meta["expiry"],
                    "strike": meta["strike"],
                    "option_type": meta["option_type"],
                    "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True).to_pydatetime() if ts_ms else None,
                    "bid": _safe_float(item.get("bidPrice") or item.get("bestBidPrice") or item.get("bid")),
                    "ask": _safe_float(item.get("askPrice") or item.get("bestAskPrice") or item.get("ask")),
                    "last_price": _safe_float(item.get("lastPrice") or item.get("last")),
                    "mark_price": _safe_float(item.get("markPrice")),
                    "volume": _safe_float(item.get("volume") or item.get("volume24h")),
                    "open_interest": _safe_float(item.get("openInterest")),
                    "iv": _safe_float(item.get("impliedVolatility") or item.get("iv")),
                    "delta": _safe_float(item.get("delta")),
                    "gamma": _safe_float(item.get("gamma")),
                    "theta": _safe_float(item.get("theta")),
                    "vega": _safe_float(item.get("vega")),
                })
            except Exception:
                continue
        return out

    def get_ticker(self, symbol: str, params: Optional[dict] = None) -> Dict[str, Any]:
        sym = symbol if "/" in symbol else _norm_usdt_symbol(symbol)
        try:
            t = self.exchange.fetch_ticker(sym, params or {})
            return t or {}
        except Exception as e:
            print(f"[Binance] fetch_ticker({sym}) failed: {e}")
            return {}

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
        sym = symbol if "/" in symbol else _norm_usdt_symbol(symbol)
        try:
            ohlcv = self.exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
            if not ohlcv:
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["symbol"] = sym
            return df
        except Exception as e:
            print(f"[Binance] fetch_ohlcv({sym}) failed: {e}")
            return pd.DataFrame()

    def get_futures_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
        return self.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
