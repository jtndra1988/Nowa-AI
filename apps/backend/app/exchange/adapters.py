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
# Bybit Adapter
# ============================================================
class BybitAdapter(BaseExchangeAdapter):
    def __init__(self, paper_mode: bool = True):
        self.is_paper = paper_mode
        
        if self.is_paper:
            api_key = getattr(settings, "BYBIT_TESTNET_API_KEY", None)
            api_secret = getattr(settings, "BYBIT_TESTNET_API_SECRET", None)
        else:
            api_key = getattr(settings, "BYBIT_API_KEY", None)
            api_secret = getattr(settings, "BYBIT_API_SECRET", None)

        self.exchange = ccxt.bybit({
            "apiKey": api_key,
            "secret": api_secret,
            "options": {
                "defaultType": "swap",
                "recvWindow": 20000,
            },
        })
        
        if self.is_paper:
            try:
                self.exchange.set_sandbox_mode(True)
            except Exception:
                pass

        try:
            self.exchange.load_markets(True)
            print(f"[Bybit] ({'Paper' if self.is_paper else 'Live'}) markets loaded.")
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
            # Relaxed Check: accept if type is missing OR matches swap
            typ = t.get("type")
            sym = t.get("symbol", "")
            qv = t.get("quoteVolume")
            
            is_perp = (typ is None or typ in ("swap", "future"))
            if is_perp and "USDT" in sym and qv:
                perps.append(t)

        if not perps:
            return ["BTC", "ETH"]

        perps = sorted(perps, key=lambda x: x["quoteVolume"], reverse=True)[: max(1, limit)]
        return [p["symbol"].split("/")[0] for p in perps]

    def get_options_chain(self, underlying_symbol: str) -> List[Dict[str, Any]]:
        base = underlying_symbol.upper()
        try:
            res = self.exchange.public_get_v5_market_tickers({
                "category": "option",
                "baseCoin": base,
            })
        except Exception as e:
            print(f"[Bybit] options tickers failed: {e}")
            return []

        rows = (res or {}).get("result", {}).get("list", []) or []
        if not rows:
            return []

        out: List[Dict[str, Any]] = []
        for r in pd.DataFrame(rows).to_dict("records"):
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
# Binance Adapter
# ============================================================
class BinanceDataAdapter(BaseExchangeAdapter):
    """
    Binance adapter using LIVE data.
    """
    BIN_OPT_REGEX = re.compile(
        r"^(?P<base>[A-Z]+)-(?P<y>\d{2})(?P<m>\d{2})(?P<d>\d{2})-(?P<strike>\d+(?:\.\d+)?)-(?P<cp>[CP])$"
    )

    def __init__(self):
        api_key = getattr(settings, "BINANCE_API_KEY", "")
        api_secret = getattr(settings, "BINANCE_API_SECRET", "")
        
        config = {
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",   # USD-M futures
            }
        }
        
        if api_key and api_secret:
            config["apiKey"] = api_key
            config["secret"] = api_secret
            print("[Binance] Initializing with API keys (Authenticated).")
        else:
            print("[Binance] Initializing in public mode (Unauthenticated).")

        self.exchange = ccxt.binance(config)
        try:
            self.exchange.load_markets(True)
            print("[Binance] Markets loaded.")
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
            
            # ✅ FIXED: Allow 'type' to be None (defaultType='future' handles context)
            is_perp = (typ is None or typ in ("future", "swap"))
            
            # Ensure we only pick valid USDT futures
            if is_perp and "USDT" in sym and qv is not None:
                perps.append(t)

        if not perps:
            print("[Binance] No symbols found after filtering! Falling back to BTC, ETH.")
            return ["BTC", "ETH"]

        # Sort descending by 24h Quote Volume
        perps = sorted(perps, key=lambda x: x["quoteVolume"], reverse=True)[: max(1, limit)]
        
        # Safe extraction of base symbol
        bases = []
        for p in perps:
            s = p["symbol"]
            # Handle 'BTC/USDT', 'BTC/USDT:USDT', 'BTCUSDT'
            if "/" in s:
                bases.append(s.split("/")[0])
            else:
                bases.append(s.replace("USDT", "").replace(":USDT", ""))
        
        # De-duplicate while preserving order
        return list(dict.fromkeys(bases))

    # ---------- Binance Options (EAPI) ----------
    def _parse_option_symbol(self, name: str) -> Optional[Dict[str, Any]]:
        if not name: return None
        m = self.BIN_OPT_REGEX.match(name)
        if not m: return None
        gd = m.groupdict()
        base = gd["base"]
        y, mth, d = int(gd["y"]), int(gd["m"]), int(gd["d"])
        year = 2000 + y
        expiry = pd.Timestamp(year=year, month=mth, day=d, tz="UTC")
        return {
            "base": base,
            "expiry": expiry.to_pydatetime(),
            "strike": _safe_float(gd["strike"]),
            "option_type": "CALL" if gd["cp"] == "C" else "PUT",
        }

    def get_options_chain(self, underlying_symbol: str) -> List[Dict[str, Any]]:
        base = underlying_symbol.upper()
        candidates = ["eapiPublicGetTicker", "eapiPublicGetMark"]
        data: List[Dict[str, Any]] = []

        for meth in candidates:
            if hasattr(self.exchange, meth):
                try:
                    res = getattr(self.exchange, meth)()
                    tickers = res if isinstance(res, list) else res.get("data") or []
                    if tickers:
                        data = tickers
                        break
                except Exception as e:
                    print(f"[Binance] {meth} failed: {e}")

        if not data:
            return []

        out: List[Dict[str, Any]] = []
        for item in data:
            try:
                name = item.get("symbol") or item.get("instrumentId")
                meta = self._parse_option_symbol(name)
                if not meta or meta["base"] != base:
                    continue

                ts_ms = item.get("time") or item.get("updateTime")
                out.append({
                    "expiry": meta["expiry"],
                    "strike": meta["strike"],
                    "option_type": meta["option_type"],
                    "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True).to_pydatetime() if ts_ms else None,
                    "bid": _safe_float(item.get("bidPrice")),
                    "ask": _safe_float(item.get("askPrice")),
                    "last_price": _safe_float(item.get("lastPrice")),
                    "mark_price": _safe_float(item.get("markPrice")),
                    "volume": _safe_float(item.get("volume")),
                    "open_interest": _safe_float(item.get("openInterest")),
                    "iv": _safe_float(item.get("impliedVolatility")),
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