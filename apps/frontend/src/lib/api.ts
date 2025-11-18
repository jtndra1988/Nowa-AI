// src/lib/api.ts
// API layer: Added ALL_ASSETS to enable full search

export type MarketMode = "futures" | "spot" | "options";
export type SymbolCode = string;

export const EXCHANGES = ["Binance", "OKX", "Bybit", "Deribit", "Kraken"] as const;
export type ExchangeCode = (typeof EXCHANGES)[number];

export type Asset =
  | "BTC" | "ETH" | "SOL" | "BNB" | "XRP" | "ADA" | "DOGE" | "TRX" | "TON"
  | "DOT" | "MATIC" | "LTC" | "SHIB" | "AVAX" | "LINK" | "UNI" | "ATOM"
  | "XLM" | "ETC" | "AAVE" | "NEAR" | "OP" | "ARB" | "ICP" | "HBAR" | "SUI"
  | "APT" | "FIL" | "RNDR" | "TIA" | "INJ" | "FTM" | "GRT" | "EGLD" | "KAS"
  | "SEI" | "ALGO" | "IMX" | "MKR" | "RUNE" | "QNT" | "FLOW" | "BCH"
  | "PEPE" | "WIF" | "JTO" | "PYTH" | "JUP" | "SAND" | "MANA" | "AXS" | "ENS"
  | "DYDX" | "SFP" | "XMR" | "ZEC" | "GMT" | "WLD" | "LDO" | "BLUR" | "ZRX"
  | "CELR" | "STX" | "BAT" | "1INCH" | "COMP" | "CRV" | "KAVA" | "KLAY"
  | "CELO" | "CHZ" | "SNX" | "SKL" | "AR" | "ROSE" | "ANKR" | "GMX" | "APE"
  | "OND" | "STRK" | "AEVO" | "ALT" | "TAO" | "ORDI" | "BONK" | "NEO"
  | "IOTA" | "XEC" | "TFUEL" | "THETA" | "PYR" | "GHST" | "OCEAN" | "GLM"
  | "CKB";

// NEW: Full list of all supported assets for search
export const ALL_ASSETS: Asset[] = [
  "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "TRX", "TON",
  "DOT", "MATIC", "LTC", "SHIB", "AVAX", "LINK", "UNI", "ATOM",
  "XLM", "ETC", "AAVE", "NEAR", "OP", "ARB", "ICP", "HBAR", "SUI",
  "APT", "FIL", "RNDR", "TIA", "INJ", "FTM", "GRT", "EGLD", "KAS",
  "SEI", "ALGO", "IMX", "MKR", "RUNE", "QNT", "FLOW", "BCH",
  "PEPE", "WIF", "JTO", "PYTH", "JUP", "SAND", "MANA", "AXS", "ENS",
  "DYDX", "SFP", "XMR", "ZEC", "GMT", "WLD", "LDO", "BLUR", "ZRX",
  "CELR", "STX", "BAT", "1INCH", "COMP", "CRV", "KAVA", "KLAY",
  "CELO", "CHZ", "SNX", "SKL", "AR", "ROSE", "ANKR", "GMX", "APE",
  "OND", "STRK", "AEVO", "ALT", "TAO", "ORDI", "BONK", "NEO",
  "IOTA", "XEC", "TFUEL", "THETA", "PYR", "GHST", "OCEAN", "GLM",
  "CKB"
];

export const TOP_ASSETS: Asset[] = [
  "BTC","ETH","SOL","BNB","XRP","ADA","DOGE","TRX","TON","DOT",
  "MATIC","LTC","SHIB","AVAX","LINK","UNI","ATOM","XLM","ETC","AAVE",
];

export const MOCK = true;

// --- Types ---
export type FeatureImportance = { name: string; weight: number };
export type PredictResponse = {
  symbol: SymbolCode;
  asset: string;
  timestamp: string;
  direction: "up" | "down" | "flat";
  price_confidence: number;
  volatility_pct: number;
  regime: string;
  risk: string;
  votes?: Record<string, number>;
  feature_importance: FeatureImportance[];
};
export type Ticker = {
  symbol: SymbolCode;
  price: number;
  change: number; // %
};
export type DashboardMetrics = {
  funding_rate: number;
  order_flow: number;
  put_call_ratio: number;
  pnl_today_percent: number;
  unrealized_pnl_percent: number;
};
export type MarketIntel = {
  sentimentHistory: { t: number | string; score: number }[];
  onChainHistory: { t: number | string; active: number }[];
  devActivity: { t: number | string; commits: number }[];
};
export type LiveTrade = {
  id: string;
  ts: string;
  symbol: SymbolCode;
  market: MarketMode;
  side: "BUY" | "SELL" | "LONG" | "SHORT";
  size: number;
  price: number;
  pnl?: number;
  exchange?: string;
};
export type OptionsChainRow = {
  strike: number;
  c_price: number; c_iv: number; c_oi: number; c_vol: number;
  c_delta: number; c_gamma: number; c_theta: number; c_vega: number;
  p_price: number; p_iv: number; p_oi: number; p_vol: number;
  p_delta: number; p_gamma: number; p_theta: number; p_vega: number;
};
export type OptionsChain = Record<string, OptionsChainRow>;
export type OrderSide = "BUY" | "SELL";
export type OrderType = "market" | "limit";
export type PlaceOrderParams = {
  symbol: SymbolCode;
  market: MarketMode;
  side: OrderSide;
  type: OrderType;
  size: number;
  price?: number;
  exchange: string;
};
export type Order = {
  id: string;
  symbol: SymbolCode;
  market: MarketMode;
  side: OrderSide;
  type: OrderType;
  size: number;
  price: number;
  exchange?: string;
  timestamp: string;
};

// --- Helpers ---
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const rand = () => Math.random();
const pick = <T,>(xs: readonly T[]): T => xs[Math.floor(rand() * xs.length)];
const randomId = () => Math.random().toString(36).slice(2);
const nowIso = () => new Date().toISOString();

// --- Caching & Mocks ---
const priceCache: Record<SymbolCode, Ticker> = {};
const basePrices: Record<string, number> = {};

const basePriceFor = (symbol: string): number => {
  const s = symbol.toUpperCase();
  if (s.startsWith("BTC")) return 68000;
  if (s.startsWith("ETH")) return 3200;
  if (s.startsWith("SOL")) return 180;
  if (s.startsWith("BNB")) return 600;
  if (s.startsWith("XRP")) return 0.6;
  return 100;
};

const getBasePrice = (symbol: string): number => {
  if (basePrices[symbol]) return basePrices[symbol];
  const base = basePriceFor(symbol);
  basePrices[symbol] = base;
  return base;
};

const getInitialTicker = (symbol: SymbolCode): Ticker => {
  if (priceCache[symbol]) return priceCache[symbol];
  const base = getBasePrice(symbol);
  const price = base * (1 + (rand() - 0.5) * 0.02); 
  const change = ((price - base) / base) * 100;
  const t = { symbol, price, change };
  priceCache[symbol] = t;
  return t;
};

const updateLiveTicker = (symbol: SymbolCode): Ticker => {
  const current = getInitialTicker(symbol);
  const base = getBasePrice(symbol);
  const move = (rand() - 0.5) * 0.005; 
  const newPrice = current.price * (1 + move);
  const newChange = ((newPrice - base) / base) * 100;
  const t = { symbol, price: newPrice, change: newChange };
  priceCache[symbol] = t;
  return t;
};

// --- Binance Utils ---
function toBinanceSymbol(s: string): string {
  const base = s.toUpperCase()
    .replace(/-PERP$/, "")
    .replace(/-OPT$/, "")
    .replace(/\/USDT$/, "")
    .replace(/-USD$/, "");
  return `${base.toLowerCase()}usdt`;
}

export function formatSymbol(asset: Asset | string, mode: MarketMode): SymbolCode {
  const base = String(asset).toUpperCase();
  if (base.includes("-PERP") || base.includes("-OPT") || base.includes("/")) return base as SymbolCode;
  
  if (mode === "futures") return (base + "-PERP") as SymbolCode;
  if (mode === "spot") return (base + "/USDT") as SymbolCode;
  if (mode === "options") return (base + "-OPT") as SymbolCode;
  return base as SymbolCode;
}

export function toAsset(symbol: SymbolCode): Asset {
  if (!symbol) return "" as Asset;
  return symbol.split(/[-/]/)[0].toUpperCase() as Asset;
}

// --- Public API Methods ---

export async function getTicker(symbol: SymbolCode): Promise<Ticker> {
  const bs = toBinanceSymbol(symbol);
  try {
    const res = await fetch(`https://api.binance.com/api/v3/ticker/24hr?symbol=${bs.toUpperCase()}`);
    if (!res.ok) throw new Error("Fetch failed");
    const d = await res.json();
    const t: Ticker = {
      symbol,
      price: parseFloat(d.lastPrice),
      change: parseFloat(d.priceChangePercent)
    };
    priceCache[symbol] = t;
    return t;
  } catch (e) {
    return getInitialTicker(symbol);
  }
}

export function streamTicker(
  symbol: SymbolCode,
  cb: (t: Ticker) => void
): () => void {
  cb(priceCache[symbol] || getInitialTicker(symbol));

  let ws: WebSocket | null = null;
  let alive = true;
  let fallbackTimer: any = null;

  const bs = toBinanceSymbol(symbol);
  const url = `wss://stream.binance.com:9443/ws/${bs}@ticker`;

  try {
    ws = new WebSocket(url);
    ws.onopen = () => { if (fallbackTimer) clearInterval(fallbackTimer); };
    ws.onmessage = (ev) => {
      if (!alive) return;
      try {
        const d = JSON.parse(ev.data);
        const price = parseFloat(d.c);
        const change = parseFloat(d.P);
        if (!isNaN(price)) {
          const t = { symbol, price, change };
          priceCache[symbol] = t;
          cb(t);
        }
      } catch { }
    };
    ws.onerror = () => { startFallback(); };
    ws.onclose = () => { if (alive) startFallback(); };
  } catch (e) {
    startFallback();
  }

  function startFallback() {
    if (fallbackTimer) return;
    fallbackTimer = setInterval(() => {
      if (!alive) return;
      cb(updateLiveTicker(symbol));
    }, 1500);
  }

  return () => {
    alive = false;
    if (ws) ws.close();
    if (fallbackTimer) clearInterval(fallbackTimer);
  };
}

export async function placeOrder(params: PlaceOrderParams): Promise<Order> {
  await new Promise((r) => setTimeout(r, 500));
  const t = await getTicker(params.symbol);
  return {
    id: `ord-${Date.now()}`,
    symbol: params.symbol,
    market: params.market,
    side: params.side,
    type: params.type,
    size: params.size,
    price: params.price || t.price,
    exchange: params.exchange,
    timestamp: nowIso(),
  };
}

export function streamPredict(asset: string, cb: (p: Partial<PredictResponse>) => void): () => void {
  const s = formatSymbol(asset, "futures");
  cb(mockPredict(asset, s));
  const id = setInterval(() => cb(mockPredict(asset, s)), 5000);
  return () => clearInterval(id);
}

const mockPredict = (asset: string, symbol: SymbolCode): PredictResponse => ({
  symbol, asset, timestamp: nowIso(),
  direction: pick(["up", "down", "flat"]),
  price_confidence: 0.55 + rand() * 0.35,
  volatility_pct: 20 + rand() * 40,
  regime: pick(["Trend", "Choppy", "Volatile"]),
  risk: pick(["Low", "Medium", "High"]),
  feature_importance: [{ name: "Momentum", weight: 0.5 }, { name: "Vol", weight: 0.3 }],
});

export async function getDashboardMetrics(symbol: SymbolCode): Promise<DashboardMetrics> {
  return {
    funding_rate: (rand() - 0.5) * 0.001,
    order_flow: (rand() - 0.5),
    put_call_ratio: 0.8 + rand() * 0.4,
    pnl_today_percent: (rand() - 0.5) * 5,
    unrealized_pnl_percent: (rand() - 0.2) * 10,
  };
}

export async function getMarketIntel(symbol: SymbolCode): Promise<MarketIntel> {
  return {
    sentimentHistory: Array.from({ length: 20 }, (_, i) => ({ t: i, score: (rand() - 0.5) })),
    onChainHistory: Array.from({ length: 20 }, (_, i) => ({ t: i, active: 100 + rand() * 50 })),
    devActivity: Array.from({ length: 20 }, (_, i) => ({ t: i, commits: Math.floor(rand() * 10) })),
  };
}

export async function getLiveTrades(asset: string, mode: MarketMode): Promise<LiveTrade[]> {
  return Array.from({ length: 15 }).map(() => ({
    id: randomId(),
    ts: new Date().toLocaleTimeString(),
    symbol: formatSymbol(asset, mode),
    market: mode,
    side: rand() > 0.5 ? "BUY" : "SELL",
    size: rand() * 2,
    price: getBasePrice(formatSymbol(asset, mode)) * (1 + (rand() - 0.5) * 0.01),
  }));
}

export async function getOptionsChain(asset: Asset): Promise<OptionsChain> {
  return {};
}

export function tickLiveTrades(asset: string, mode: MarketMode, apply: (fn: any) => void): () => void {
  const id = setInterval(() => { }, 3000);
  return () => clearInterval(id);
}

const API = {
  MOCK, TOP_ASSETS, ALL_ASSETS, EXCHANGES, formatSymbol, toAsset,
  getTicker, streamTicker, streamPredict, getDashboardMetrics,
  getMarketIntel, getLiveTrades, getOptionsChain, tickLiveTrades, placeOrder
};

export default API;