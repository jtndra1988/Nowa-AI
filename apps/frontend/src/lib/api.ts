// src/lib/api.ts
// API layer with Live Binance Tickers + Real System Health

export type MarketMode = "futures" | "spot" | "options";
export type SymbolCode = string;

export const EXCHANGES = ["Binance", "OKX", "Bybit", "Deribit", "Kraken"] as const;
export type ExchangeCode = (typeof EXCHANGES)[number];

// Relaxed Asset type to string to allow dynamic lists
export type Asset = string;

// Initial Fallback (Used before API loads)
export const ALL_ASSETS: Asset[] = [
  "BTC",
  "ETH",
  "SOL",
  "BNB",
  "XRP",
  "DOGE",
  "TON",
  "ADA",
  "AVAX",
  "LINK",
];
export const TOP_ASSETS: Asset[] = ALL_ASSETS;

export const MOCK = false;

// --- Dynamic Asset Fetcher (Top 10 from backend) ---
export async function fetchActiveAssets(): Promise<string[]> {
  const base = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
  try {
    const res = await fetch(`${base}/api/v1/assets?limit=10`, {
      method: "GET",
      cache: "no-store",
    });
    if (!res.ok) throw new Error("Failed to fetch assets");
    return await res.json();
  } catch (e) {
    console.warn("Asset fetch failed, using fallback:", e);
    // Fallback is also just 10 assets
    return ALL_ASSETS;
  }
}

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
export type SentimentPoint = { t: number | string; score: number };
export type OnChainPoint = { t: number | string; active: number };
export type SimpleSeriesPoint = { t: number | string; v: number };
export type CorrelationPoint = { name: string; val: number };
export type RadarIntel = {
  regime: "MOMENTUM" | "MEAN-REVERT" | "BALANCED";
  crowding: string;
  trend: "UP" | "DOWN" | "FLAT";
  liquidity: "LOW" | "NORMAL" | "HIGH";
  warning: string;
};
export type SentimentBreakdown = {
  composite: number;
  news: number;
  social: number;
  global_score: number;
};

export type MarketIntel = {
  symbol: string;
  mode: string;
  sentimentHistory: SentimentPoint[];
  onChainHistory: OnChainPoint[];
  ivHistory: SimpleSeriesPoint[];
  fundingHistory: SimpleSeriesPoint[];
  oiHistory: SimpleSeriesPoint[];
  cvdHistory: SimpleSeriesPoint[];
  correlations: CorrelationPoint[];
  radar: RadarIntel;
  sentimentBreakdown?: SentimentBreakdown;
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

// --- System Health Types ---
export type SystemStatus = {
  status: "OK" | "DEGRADED" | "DOWN";
  api_latency: number;
  brain: {
    ready: boolean;
    l2_model: boolean;
    llm_engine: boolean;
    rl_agent: boolean;
  };
  resources: {
    cpu_load: number;
    ram_usage: number;
    gpu_util: number;
  };
  services?: {
    db?: ServiceHealth;
    redis?: ServiceHealth;
    celery?: ServiceHealth;
    binance?: ServiceHealth;
    llm_provider?: ServiceHealth;
  };
  timestamp?: string;
};

export type ServiceStatus = "ok" | "degraded" | "down" | "unknown";
export type ServiceHealth = {
  status: ServiceStatus;
  detail?: string;
};

// --- Helpers ---
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

// --- System Health & Status (backend snapshot) ---
function normalizeService(svc: any | undefined): ServiceHealth | undefined {
  if (!svc) return undefined;
  const raw = (svc.status ?? "unknown").toString().toLowerCase();
  return {
    status: (raw === "ok" || raw === "degraded" || raw === "down") ? raw as ServiceStatus : "unknown",
    detail: svc.detail,
  };
}

function normalizeSystemSnapshot(raw: any, latency: number): SystemStatus {
  let status: SystemStatus["status"] = "DOWN";
  if (typeof raw.status === "string") {
    const up = raw.status.toUpperCase();
    if (up === "OK" || up === "DEGRADED" || up === "DOWN") status = up as SystemStatus["status"];
  }

  const brainSource = raw.brain || raw;
  const brain = {
    ready: Boolean(brainSource.ready || brainSource.ai_brain === "ready" || brainSource.is_ready),
    l2_model: Boolean(brainSource.l2_model || brainSource.l2_ensemble),
    llm_engine: Boolean(brainSource.llm_engine || brainSource.llm_ready),
    rl_agent: Boolean(brainSource.rl_agent || brainSource.rl_ready),
  };

  const resourcesRaw = raw.resources || {};
  const resources = {
    cpu_load: resourcesRaw.cpu_load || 0,
    ram_usage: resourcesRaw.ram_usage || 0,
    gpu_util: resourcesRaw.gpu_util || 0,
  };

  const servicesRaw = raw.services || {};
  const services: SystemStatus["services"] = {
    db: normalizeService(servicesRaw.db),
    redis: normalizeService(servicesRaw.redis),
    celery: normalizeService(servicesRaw.celery),
    binance: normalizeService(servicesRaw.binance),
    llm_provider: normalizeService(servicesRaw.llm_provider),
  };

  return {
    status,
    api_latency: latency,
    brain,
    resources,
    services,
    timestamp: raw.timestamp || new Date().toISOString(),
  };
}

export async function getSystemStatus(): Promise<SystemStatus> {
  const start = performance.now();
  const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
  const url = `${apiBase.replace(/\/$/, "")}/api/v1/system-health`;

  try {
    const res = await fetch(url, { method: "GET", cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const raw = await res.json();
    const latency = Math.round(performance.now() - start);
    return normalizeSystemSnapshot(raw, latency);
  } catch (e) {
    const latency = Math.round(performance.now() - start);
    return {
      status: "DOWN",
      api_latency: latency,
      brain: { ready: false, l2_model: false, llm_engine: false, rl_agent: false },
      resources: { cpu_load: 0, ram_usage: 0, gpu_util: 0 },
      services: {},
      timestamp: new Date().toISOString(),
    };
  }
}

// --- Real-time System Telemetry via SSE ---
function normalizeSystemSnapshotFromSSE(raw: any): SystemStatus {
  const brain = {
    ready: Boolean(raw.ai_ready ?? raw.is_ready),
    l2_model: Boolean(raw.l2 ?? raw.l2_model),
    llm_engine: Boolean(raw.llm ?? raw.llm_engine),
    rl_agent: Boolean(raw.rl ?? raw.rl_agent),
  };

  const resourcesRaw = raw.resources || {};
  const resources = {
    cpu_load: resourcesRaw.cpu_load || 0,
    ram_usage: resourcesRaw.ram_usage || 0,
    gpu_util: resourcesRaw.gpu_util || 0,
  };

  return {
    status: brain.ready ? "OK" : "DOWN",
    api_latency: 0,
    brain,
    resources,
    services: {},
    timestamp: raw.timestamp || new Date().toISOString(),
  };
}

export function subscribeSystemStream(onUpdate: (snapshot: SystemStatus) => void, onError?: (err: any) => void): () => void {
  const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
  const url = `${apiBase.replace(/\/$/, "")}/api/v1/system-stream`;
  let es: EventSource | null = null;

  try {
    es = new EventSource(url, { withCredentials: false });
    es.onmessage = (ev) => {
      try {
        onUpdate(normalizeSystemSnapshotFromSSE(JSON.parse(ev.data)));
      } catch (e) { console.error(e); }
    };
    es.onerror = (err) => { onError?.(err); es?.close(); };
  } catch (e) { onError?.(e); }

  return () => es?.close();
}

// --- Mocks & Wrappers ---
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
  const base = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
  const res = await fetch(`${base}/api/v1/market-intel/${encodeURIComponent(symbol)}`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Failed to load market intel (${res.status})`);
  return (await res.json()) as MarketIntel;
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

// Clean API Export
const API = {
  MOCK,
  TOP_ASSETS,
  ALL_ASSETS,
  EXCHANGES,
  fetchActiveAssets, // ✅ Exported for use in UI
  formatSymbol,
  toAsset,
  getTicker,
  streamTicker,
  streamPredict,
  getDashboardMetrics,
  getMarketIntel,
  getLiveTrades,
  getOptionsChain,
  tickLiveTrades,
  placeOrder,
  getSystemStatus,
  subscribeSystemStream,
};

export default API;