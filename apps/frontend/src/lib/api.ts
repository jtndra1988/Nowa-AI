// src/lib/api.ts
// Mocked API layer to power the Nowa frontend for demos.
// Matches existing components: NeonHeader, tabs, System, etc.

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

// Symbols shown in various parts of the UI
export const TOP_ASSETS: Asset[] = [
  "BTC","ETH","SOL","BNB","XRP","ADA","DOGE","TRX","TON","DOT",
  "MATIC","LTC","SHIB","AVAX","LINK","UNI","ATOM","XLM","ETC","AAVE",
];

// Demo-only flag (SystemTab reads this)
export const MOCK = true;

export type FeatureImportance = {
  name: string;
  weight: number; // 0–1
};

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
  order_flow: number;            // -1..1
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

// ---------- helpers ----------

const clamp = (v: number, lo: number, hi: number) =>
  Math.max(lo, Math.min(hi, v));

const rand = () => {
  if (typeof crypto !== "undefined" && "getRandomValues" in crypto) {
    const a = new Uint32Array(1);
    crypto.getRandomValues(a);
    return a[0] / 0xffffffff;
  }
  return Math.random();
};

const pick = <T,>(xs: readonly T[]): T =>
  xs[Math.floor(rand() * xs.length)];

const randomId = () =>
  (typeof crypto !== "undefined" && "randomUUID" in crypto)
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);

const nowIso = () => new Date().toISOString();

// --- Live Price Mocking ---
// In a real app, this would be replaced by a WebSocket connection
// that pushes live price updates into the cache.
const priceCache: Record<SymbolCode, Ticker> = {};
const basePrices: Record<string, number> = {};
// --- End Live Price Mocking ---

const basePriceFor = (symbol: string): number => {
  const s = symbol.toUpperCase();
  if (s.startsWith("BTC")) return 68000;
  if (s.startsWith("ETH")) return 3200;
  if (s.startsWith("SOL")) return 180;
  if (s.startsWith("BNB")) return 600;
  if (s.startsWith("XRP")) return 0.6;
  if (s.startsWith("ADA")) return 0.6;
  if (s.startsWith("DOGE")) return 0.2;
  return 100;
};

// Gets the base price and stores it for 24h change calculation
const getBasePrice = (symbol: string): number => {
  if (basePrices[symbol]) return basePrices[symbol];
  const base = basePriceFor(symbol);
  basePrices[symbol] = base;
  return base;
};

// Generates the *first* ticker for a symbol or retrieves the cached one
const getInitialTicker = (symbol: SymbolCode): Ticker => {
  if (priceCache[symbol]) {
    return priceCache[symbol];
  }

  const base = getBasePrice(symbol);
  const price = base * (0.99 + rand() * 0.02); // Start with a random price
  const change = ((price - base) / base) * 100;

  const ticker: Ticker = {
    symbol,
    price: +price.toFixed(4), // Use 4 decimals for price
    change: +change.toFixed(2),
  };

  priceCache[symbol] = ticker;
  return ticker;
};

// Simulates a live price "walk"
const updateLiveTicker = (symbol: SymbolCode): Ticker => {
  // Ensure we have a starting point
  const currentTicker = getInitialTicker(symbol);
  const base = getBasePrice(symbol);

  // Simulate a random walk (e.g., +/- 0.5% of the current price)
  const prevPrice = currentTicker.price;
  const movePercent = (rand() - 0.495) * 0.005; // Small random move
  let newPrice = prevPrice * (1 + movePercent);

  // Add a small chance of a larger "jump"
  if (rand() > 0.98) {
    newPrice = newPrice * (1 + (rand() - 0.5) * 0.02);
  }

  // Prevent price from going negative or too wild
  if (newPrice <= 0) {
    newPrice = base * 0.01;
  }

  // Recalculate 24h change based on the initial base price
  const newChange = ((newPrice - base) / base) * 100;

  const newTicker: Ticker = {
    symbol,
    price: +newPrice.toFixed(4), // Use 4 decimals for price
    change: +newChange.toFixed(2),
  };

  // Store the new price in the cache for the next update
  priceCache[symbol] = newTicker;
  return newTicker;
};


// Match NeonHeader expectations: format asset+mode → SymbolCode
export function formatSymbol(asset: Asset | string, mode: MarketMode): SymbolCode {
  const base = String(asset).toUpperCase();

  if (base.includes("-PERP") || base.includes("-OPT") || base.includes("/")) {
    if (mode === "futures") {
      if (base.includes("-PERP")) return base as SymbolCode;
      if (base.includes("/"))
        return (base.split("/")[0] + "-PERP") as SymbolCode;
    }
    if (mode === "spot") {
      if (base.includes("/")) return base as SymbolCode;
      if (base.includes("-PERP") || base.includes("-OPT"))
        return (base.split("-")[0] + "/USDT") as SymbolCode;
    }
    if (mode === "options") {
      if (base.includes("-OPT")) return base as SymbolCode;
      if (base.includes("/"))
        return (base.split("/")[0] + "-OPT") as SymbolCode;
    }
  }

  const sym = base.replace(/[-/].*$/, "");
  if (mode === "futures") return (sym + "-PERP") as SymbolCode;
  if (mode === "spot") return (sym + "/USDT") as SymbolCode;
  if (mode === "options") return (sym + "-OPT") as SymbolCode;
  return sym as SymbolCode;
}

// Helper to reverse formatSymbol
export function toAsset(symbol: SymbolCode): Asset {
  if (!symbol) return "" as Asset;
  return symbol.split(/[-/]/)[0].toUpperCase() as Asset;
}

// ---------- mock generators ----------

const mockTicker = (symbol: SymbolCode): Ticker => {
  // This function is now just a wrapper for the live simulation
  return updateLiveTicker(symbol);
};

const mockFeatureImportance = (): FeatureImportance[] => [
  { name: "OrderBookImbalance(1m)", weight: 0.22 },
  { name: "FundingRateDelta(8h)", weight: 0.18 },
  { name: "OnChainFlows(24h)", weight: 0.14 },
  { name: "RealizedVol(30m)", weight: 0.11 },
  { name: "BasisSpread(Perp-Spot)", weight: 0.10 },
  { name: "OIAcceleration", weight: 0.08 },
];

const mockPredict = (asset: string, symbol: SymbolCode): PredictResponse => {
  const dir = pick<"up" | "down" | "flat">(["up", "down", "flat"]);
  const conf = 0.55 + rand() * 0.35;

  return {
    symbol,
    asset,
    timestamp: nowIso(),
    direction: dir,
    price_confidence: +conf.toFixed(2),
    volatility_pct: +(20 + rand() * 40).toFixed(1),
    regime: pick([
      "Hybrid Ensemble",
      "Bull Trend",
      "Choppy",
      "High Vol",
    ]),
    risk: pick([
      "All Clear",
      "Hedge Recommended",
      "Reduce Size",
      "Event Risk Window",
    ]),
    votes: {
      TFT: +(0.5 + rand() * 0.5).toFixed(2),
      TCN: +(0.5 + rand() * 0.5).toFixed(2),
      XGB: +(0.5 + rand() * 0.5).toFixed(2),
    },
    feature_importance: mockFeatureImportance(),
  };
};

const mockDashboardMetrics = (symbol: SymbolCode): DashboardMetrics => ({
  funding_rate: +(0.0006 +
    symbol.length * 0.00001 +
    (rand() - 0.5) * 0.0004
  ).toFixed(5),
  order_flow: +clamp((rand() - 0.5) * 0.6, -1, 1).toFixed(2),
  put_call_ratio: +clamp(
    0.9 + (rand() - 0.5) * 0.3,
    0.5,
    2
  ).toFixed(2),
  pnl_today_percent: +((rand() - 0.4) * 5).toFixed(2),
  unrealized_pnl_percent: +((rand() - 0.3) * 15).toFixed(2),
});

const mockMarketIntel = (symbol: SymbolCode): MarketIntel => {
  const len = 60;
  return {
    sentimentHistory: Array.from({ length: len }, (_, i) => ({
      t: i,
      score: +(Math.sin(i / 9) * 0.4 + (rand() - 0.5) * 0.2).toFixed(2),
    })),
    onChainHistory: Array.from({ length: len }, (_, i) => ({
      t: i,
      active: Math.round(
        90 + Math.sin(i / 7) * 25 + rand() * 18
      ),
    })),
    devActivity: Array.from({ length: len }, (_, i) => ({
      t: i,
      commits: Math.round(
        2 + (Math.sin(i / 11) + 1) * 3 + rand() * 2
      ),
    })),
  };
};

const mockLiveTrades = (
  asset: string,
  mode: MarketMode,
  count = 30
): LiveTrade[] => {
  const sym =
    mode === "futures"
      ? `${asset}-PERP`
      : mode === "options"
      ? `${asset}-OPT`
      : `${asset}/USDT`;

  return Array.from({ length: count }).map((_, i) => {
    const side =
      mode === "spot"
        ? rand() > 0.5
          ? "BUY"
          : "SELL"
        : rand() > 0.5
        ? "LONG"
        : "SHORT";

    const base = basePriceFor(sym);
    const price = base * (0.99 + rand() * 0.02);

    return {
      id: `${sym}-${i}-${randomId()}`,
      ts: new Date(
        Date.now() - Math.floor(rand() * 120_000)
      ).toLocaleTimeString(),
      symbol: sym as SymbolCode,
      market: mode,
      side,
      size: +(0.1 + rand() * 2.5).toFixed(3),
      price: +price.toFixed(2),
      pnl:
        mode === "futures"
          ? +((rand() - 0.45) * 80).toFixed(2)
          : undefined,
      exchange: pick(EXCHANGES),
    };
  });
};

const mockOptionsChain = (asset: Asset): OptionsChain => {
  const base = basePriceFor(asset);
  const strikes = Array.from({ length: 10 }, (_, i) =>
    Math.round((base * (0.9 + i * 0.02)) / 5) * 5
  );
  const chain: OptionsChain = {};
  strikes.forEach((strike) => {
    chain[strike] = {
      strike,
      c_price: +(base * 0.05 * rand()).toFixed(2),
      c_iv: +(0.4 + rand() * 0.3).toFixed(2),
      c_oi: +(10 + rand() * 50).toFixed(0),
      c_vol: +(5 + rand() * 20).toFixed(0),
      c_delta: +(0.4 + rand() * 0.2).toFixed(2),
      c_gamma: +(0.01 + rand() * 0.02).toFixed(3),
      c_theta: +(-0.05 - rand() * 0.05).toFixed(3),
      c_vega: +(0.1 + rand() * 0.1).toFixed(3),
      p_price: +(base * 0.04 * rand()).toFixed(2),
      p_iv: +(0.35 + rand() * 0.3).toFixed(2),
      p_oi: +(8 + rand() * 40).toFixed(0),
      p_vol: +(4 + rand() * 15).toFixed(0),
      p_delta: +(-0.5 + rand() * 0.2).toFixed(2),
      p_gamma: +(0.01 + rand() * 0.02).toFixed(3),
      p_theta: +(-0.04 - rand() * 0.05).toFixed(3),
      p_vega: +(0.1 + rand() * 0.1).toFixed(3),
    };
  });
  return chain;
};


// ---------- public (used by components) ----------

export async function getTicker(symbol: SymbolCode): Promise<Ticker> {
  // Return the *current* cached price, or generate one if it doesn't exist
  return Promise.resolve(priceCache[symbol] || getInitialTicker(symbol));
}
export async function placeOrder(params: PlaceOrderParams): Promise<Order> {
  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 500));

  // Determine price (mock fetch if market order)
  let fillPrice = params.price;
  if (!fillPrice) {
     // In a real app, this would come from the matching engine
     const ticker = await getTicker(params.symbol);
     fillPrice = ticker.price;
  }

  return {
    id: `ord-${Date.now()}`, // Simple ID generation
    symbol: params.symbol,
    market: params.market,
    side: params.side,
    type: params.type,
    size: params.size,
    price: fillPrice,
    exchange: params.exchange,
    timestamp: new Date().toISOString(),
  };
}
export function streamTicker(
  symbol: SymbolCode,
  cb: (t: Ticker) => void
): () => void {
  // In a real app, you would subscribe to a WebSocket channel here, e.g.:
  // const ws = new WebSocket('wss://api.exchange.com/v1/stream');
  // ws.onopen = () => ws.send(JSON.stringify({ op: 'subscribe', args: [`ticker:${symbol}`] }));
  // ws.onmessage = (event) => {
  //   const data = JSON.parse(event.data);
  //   if (data.type === 'ticker' && data.symbol === symbol) {
  //     const newTicker = { ... }; // transform data to Ticker type
  //     priceCache[symbol] = newTicker; // Update cache
  //     cb(newTicker); // Push to consumer
  //   }
  // };
  //
  // const stop = () => ws.close();
  // return stop;

  // --- MOCK IMPLEMENTATION ---
  // Call once immediately with the current price
  cb(getInitialTicker(symbol));

  // Then, start "live" updates
  const id = setInterval(() => {
    // Each interval, we calculate a new "walked" price
    const newTicker = updateLiveTicker(symbol);
    cb(newTicker);
  }, 1500); // Update price more frequently (e.g., every 1.5s)

  return () => clearInterval(id);
}

export function streamPredict(
  asset: string,
  cb: (p: Partial<PredictResponse>) => void
): () => void {
  const symbol = formatSymbol(asset, "futures");
  cb(mockPredict(asset, symbol));
  const id = setInterval(
    () => cb(mockPredict(asset, symbol)),
    5_000
  );
  return () => clearInterval(id);
}

export async function getDashboardMetrics(
  symbol: SymbolCode
): Promise<DashboardMetrics> {
  return mockDashboardMetrics(symbol);
}

export async function getMarketIntel(
  symbol: SymbolCode
): Promise<MarketIntel> {
  return mockMarketIntel(symbol);
}

export async function getLiveTrades(
  asset: string,
  mode: MarketMode
): Promise<LiveTrade[]> {
  return mockLiveTrades(asset, mode, 30);
}

export async function getOptionsChain(asset: Asset): Promise<OptionsChain> {
  return mockOptionsChain(asset);
}

export function tickLiveTrades(
  asset: string,
  mode: MarketMode,
  apply: (fn: (prev: LiveTrade[]) => LiveTrade[]) => void
): () => void {
  let current = mockLiveTrades(asset, mode, 30);
  apply(() => current);
  const id = setInterval(() => {
    const [next] = mockLiveTrades(asset, mode, 1);
    current = [next, ...current].slice(0, 80);
    apply(() => current);
  }, 2_000);
  return () => clearInterval(id);
}

// Attach everything the components expect onto default export:
const API = {
  MOCK,
  TOP_ASSETS,
  EXCHANGES,
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
};

export default API;