"use client";

import dynamic from "next/dynamic";
import React, {
  useEffect,
  useMemo,
  useState,
} from "react";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { Toaster } from "sonner";
import NeonHeader from "@/components/NeonHeader";
import {
  Activity,
  Brain,
  History,
  Signal,
  Wallet,
  ChartArea,
  ShieldCheck,
  SlidersHorizontal,
  PanelLeftClose,
  Sun,
  Moon,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  Info,
  ChevronLeft,
  ChevronRight,
  BrainCircuit
} from "lucide-react";

import API, {
  MarketMode,
  PredictResponse,
  SymbolCode,
  type Asset,
  type LiveTrade,
} from "@/lib/api";
import { Switch as UiSwitch } from "@/components/ui/switch";

/* ================================
   Shared visual tokens
   ================================ */

export const glassPanel =
  "backdrop-blur-xl bg-white/8 dark:bg-white/5 border border-white/15 shadow-[0_8px_30px_rgb(0,0,0,0.25)]";
const glassCard = "rounded-2xl " + glassPanel;
export const glassHover =
  "transition-transform duration-300 hover:-translate-y-0.5 hover:shadow-[0_12px_40px_rgba(0,0,0,0.35)]";
export const softText = "text-slate-600 dark:text-slate-300";
export const faintText = "text-slate-500/80 dark:text-slate-400/80";

/* ================================
   Utilities
   ================================ */

export function useIsMobile() {
  const [isMobile, setIsMobile] = useState(false);
  useEffect(() => {
    const fn = () => setIsMobile(window.innerWidth < 1024);
    fn();
    window.addEventListener("resize", fn);
    return () => window.removeEventListener("resize", fn);
  }, []);
  return isMobile;
}

export function useLocalStorage<T>(key: string, initialValue: T): [T, (v: T) => void] {
  const [storedValue, setStoredValue] = useState<T>(initialValue);

  useEffect(() => {
    try {
      const item = window.localStorage.getItem(key);
      if (item != null) setStoredValue(JSON.parse(item) as T);
    } catch {
      // ignore
    }
  }, [key]);

  const setValue = (value: T) => {
    try {
      setStoredValue(value);
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // ignore
    }
  };

  return [storedValue, setValue];
}

/** Seeded PRNG (mulberry32) used for stable mock data */
export function makeSeeded(seedStr: string) {
  let h = 1779033703 ^ seedStr.length;
  for (let i = 0; i < seedStr.length; i++) {
    h = Math.imul(h ^ seedStr.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return () => {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    return ((h ^= h >>> 16) >>> 0) / 4294967296;
  };
}

/* ================================
   Primitives
   ================================ */
export const DirectionLabel: React.FC<{
  value: number;          // e.g. +3.5 or -1.2 (percent or bps)
  className?: string;
  ariaLabel?: string;
}> = ({ value, className = "", ariaLabel }) => {
  const dir = value > 0 ? "up" : value < 0 ? "down" : "flat";
  const abs = Math.abs(value).toFixed(2);

  const label =
    dir === "up"
      ? `Up ${abs}`
      : dir === "down"
        ? `Down ${abs}`
        : "No change";

  return (
    <span
      className={`inline-flex items-center gap-1 text-xs ${className}`}
      aria-label={ariaLabel || label}
    >
      {dir === "up" && (
        <ArrowUpRight
          className="w-3 h-3 text-emerald-400"
          aria-hidden="true"
        />
      )}
      {dir === "down" && (
        <ArrowDownRight
          className="w-3 h-3 text-rose-400"
          aria-hidden="true"
        />
      )}
      {dir === "flat" && (
        <Minus
          className="w-3 h-3 text-slate-400"
          aria-hidden="true"
        />
      )}
      <span>
        {dir === "flat"
          ? "0.00"
          : `${dir === "up" ? "+" : "-"}${abs}`}
      </span>
    </span>
  );
};

export const HelpTooltip: React.FC<{ label: string; className?: string }> = ({
  label,
  className = "",
}) => (
  <span
    className={`ml-1 inline-flex items-center justify-center w-3 h-3 rounded-full border border-white/25 text-[8px] text-slate-500 dark:text-slate-400 ${className}`}
    aria-label={label}
    title={label}
  >
    ?
  </span>
);


type ButtonVariant = "primary" | "secondary" | "ghost" | "outline";
type ButtonSize = "default" | "icon";

export const Button: React.FC<{
  variant?: ButtonVariant;
  size?: ButtonSize;
  onClick?: () => void | Promise<void>;
  children: React.ReactNode;
  className?: string;
  id?: string;
  title?: string;
  type?: "button" | "submit" | "reset";
  disabled?: boolean;
}> = ({
  variant = "primary",
  size = "default",
  onClick,
  children,
  className = "",
  id,
  title,
  type = "button",
  disabled = false,
}) => {
    const base =
      "inline-flex items-center justify-center rounded-xl text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/70 focus-visible:ring-offset-0 disabled:opacity-50 disabled:pointer-events-none transition-colors";
    const variants: Record<ButtonVariant, string> = {
      primary:
        "bg-indigo-600/90 hover:bg-indigo-500/90 text-white shadow-[0_6px_24px_rgba(99,102,241,0.35)]",
      secondary: glassPanel + " hover:bg-white/12 " + softText,
      ghost: "hover:bg-white/10 " + softText,
      outline: "border border-white/20 hover:bg-white/10 " + softText,
    };
    const sizes: Record<ButtonSize, string> = {
      default: "px-4 py-2",
      icon: "h-10 w-10",
    };
    return (
      <button
        type={type}
        id={id}
        title={title}
        className={`${base} ${variants[variant]} ${sizes[size]} ${className}`}
        onClick={onClick}
        disabled={disabled}
      >
        {children}
      </button>
    );
  };

export const Input: React.FC<React.InputHTMLAttributes<HTMLInputElement>> = (props) => (
  <input
    {...props}
    className={`h-11 w-full rounded-xl border border-white/20 bg-white/5 px-3 text-sm text-slate-900 dark:text-white placeholder:text-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/60 ${props.className || ""}`}
  />
);

export const Label: React.FC<React.LabelHTMLAttributes<HTMLLabelElement>> = (props) => (
  <label {...props} className={`text-sm ${softText} ${props.className || ""}`} />
);

export const Tag: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <span className="text-[11px] px-2 py-0.5 rounded-lg border border-white/15 bg-white/8 mr-1">
    {children}
  </span>
);

export const Badge: React.FC<{
  variant?: "primary" | "secondary" | "outline" | "positive" | "negative" | "muted";
  children: React.ReactNode;
  className?: string;
}> = ({ variant = "primary", children, className = "" }) => {
  const v = {
    primary:
      "bg-gradient-to-b from-white/60 to-white/20 dark:from-white/10 dark:to-white/5 text-slate-900 dark:text-white border border-white/30",
    secondary: glassPanel + " " + faintText,
    outline: "border border-white/30 text-slate-900 dark:text-white",
    positive: "bg-emerald-500/15 text-emerald-300 border border-emerald-500/25",
    negative: "bg-rose-500/15 text-rose-300 border border-rose-500/25",
    muted: "bg-white/8 text-slate-300 border border-white/15",
  } as const;
  return (
    <span
      className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold ${v[variant]} ${className}`}
    >
      {children}
    </span>
  );
};

export const Card: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = "",
}) => <div className={`${glassCard} ${glassHover} p-4 ${className}`}>{children}</div>;

export const CardHeader: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = "",
}) => (
  <div className={`pb-3 border-b border-white/10 ${className}`}>
    {children}
  </div>
);

export const CardTitle: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = "",
}) => (
  <h3 className={`text-lg font-semibold text-slate-900 dark:text-white ${className}`}>
    {children}
  </h3>
);

export const CardDescription: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = "",
}) => (
  <p className={`text-sm ${faintText} ${className}`}>
    {children}
  </p>
);

export const CardContent: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = "",
}) => <div className={`pt-4 ${className}`}>{children}</div>;

export const KpiTile: React.FC<{
  label: string;
  value: React.ReactNode;
  hint?: string;
  className?: string;
}> = ({ label, value, hint, className = "" }) => (
  <div className={`${glassPanel} rounded-xl p-3 ${className}`} title={hint}>
    <div className={`text-[11px] ${faintText}`}>{label}</div>
    <div className="text-xl font-semibold mt-0.5">{value}</div>
  </div>
);

/** Re-export Switch for convenience in tabs */
export const Switch = UiSwitch;

/* ================================
   Data helpers & hooks from old file
   ================================ */

export type Ticker = { symbol: SymbolCode; price: number; change: number };

export function toAsset(symbol: SymbolCode): Asset {
  if (symbol.includes("-PERP") || symbol.includes("-OPT")) return symbol.split("-")[0] as Asset;
  if (symbol.includes("/")) return symbol.split("/")[0] as Asset;
  return symbol as Asset;
}

export function useTicker(symbol: SymbolCode) {
  const [t, setT] = useState<Ticker>({ symbol, price: 0, change: 0 });

  useEffect(() => {
    let alive = true;
    let stop: (() => void) | undefined;

    // Initial fetch (safe)
    (async () => {
      try {
        const initial = await API.getTicker(symbol);
        if (alive && initial) setT(initial);
      } catch {
        // swallow: fallback UI is fine
      }
    })();

    // Streaming (only if available + only if it returns a function)
    if (API.streamTicker) {
      const result = API.streamTicker(symbol, (next: Ticker) => {
        if (alive && next) setT(next);
      });

      if (typeof result === "function") {
        stop = result;
      }
    }

    // Cleanup
    return () => {
      alive = false;
      if (stop) {
        stop();
      }
    };
  }, [symbol]);

  return t;
}

type DashboardMetricsLite = {
  funding_rate: number;
  order_flow: number;
  put_call_ratio: number;
};

export function usePredict(symbol: SymbolCode): PredictResponse {
  const [p, setP] = useState<PredictResponse>({
    symbol,
    direction: "flat",
    price_confidence: 0.5,
    volatility_pct: 35,
    regime: "NEUTRAL",
    risk: "OK",
    feature_importance: [
      { name: "OrderBookImbalance(1m)", weight: 0.22 },
      { name: "FundingRateDelta(8h)", weight: 0.18 },
      { name: "OnChainFlows(24h)", weight: 0.14 },
      { name: "RealizedVol(30m)", weight: 0.11 },
    ],
  } as unknown as PredictResponse);

  useEffect(() => {
    let stop: (() => void) | undefined;

    try {
      const maybeStop = API.streamPredict(
        symbol.includes("-")
          ? (symbol.split("-")[0] as Asset)
          : (symbol as Asset),
        (next) => setP((prev) => ({ ...prev, ...next, symbol })),
      );

      // Only keep it if backend actually returns an unsubscribe fn
      if (typeof maybeStop === "function") {
        stop = maybeStop;
      }
    } catch {
      // ignore streaming errors; keep last known prediction
    }

    return () => {
      if (stop) {
        stop();
      }
    };
  }, [symbol]);


  return p;
}

export function useDashboardMetrics(symbol: SymbolCode): DashboardMetricsLite {
  const [m, setM] = useState<DashboardMetricsLite>({
    funding_rate: 0.0008,
    order_flow: 0.1,
    put_call_ratio: 0.92,
  });

  useEffect(() => {
    let alive = true;
    (API as any)?.getDashboardMetrics?.(symbol).then((x: any) => {
      if (alive && x) setM(x);
    });
    const id = setInterval(() => {
      setM((prev) => ({
        funding_rate: prev.funding_rate * (0.98 + Math.random() * 0.04),
        order_flow: Math.max(-1, Math.min(1, prev.order_flow + (Math.random() - 0.5) * 0.1)),
        put_call_ratio: Math.max(
          0.5,
          Math.min(1.8, prev.put_call_ratio + (Math.random() - 0.5) * 0.04),
        ),
      }));
    }, 2500);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [symbol]);

  return m;
}

// keep other imports/utilities as they are above

export function useMarketIntel(symbol: SymbolCode) {
  type Intel = {
    sentimentHistory: { t: number; score: number }[];
    onChainHistory: { t: number; active: number }[];
    devActivity: { t: number; commits: number }[];
    radar?: {
      regime?: string;
      trend?: string;
      liquidity?: string;
      warning?: string;
      crowding?: string;
    };
    correlations?: { name: string; val: number }[];
  };

  const [intel, setIntel] = useState<Intel>(() => {
    // nice looking default so UI has content instantly
    const baseLen = 60;
    const sentimentHistory = Array.from({ length: baseLen }, (_, i) => ({
      t: i,
      score: Math.sin(i / 9) * 0.4,
    }));
    const onChainHistory = Array.from({ length: baseLen }, (_, i) => ({
      t: i,
      active: 100 + Math.sin(i / 7) * 20,
    }));
    const devActivity = Array.from({ length: baseLen }, (_, i) => ({
      t: i,
      commits: 3 + (i % 5),
    }));
    return {
      sentimentHistory,
      onChainHistory,
      devActivity,
      correlations: [
        { name: "BTC.D", val: 0.42 },
        { name: "ETH.D", val: 0.35 },
        { name: "SPX", val: 0.18 },
        { name: "DXY", val: -0.27 },
        { name: "VIX", val: -0.31 },
      ],
    };
  });

  // Seed from API once (mocked or real)
  useEffect(() => {
    let alive = true;
    (API as any)?.getMarketIntel?.(symbol)
      ?.then((x: any) => {
        if (!alive || !x) return;
        setIntel((prev) => ({
          ...prev,
          ...x,
          sentimentHistory: x.sentimentHistory ?? prev.sentimentHistory,
          onChainHistory: x.onChainHistory ?? x.onchainHistory ?? prev.onChainHistory,
          devActivity: x.devActivity ?? prev.devActivity,
          correlations: x.correlations ?? prev.correlations,
          radar: x.radar ?? prev.radar,
        }));
      })
      .catch(() => {
        /* demo-safe */
      });
    return () => {
      alive = false;
    };
  }, [symbol]);

  // Live drift: extend timeseries every 2.5s so charts move
  useEffect(() => {
    let alive = true;
    const id = setInterval(() => {
      if (!alive) return;
      setIntel((prev) => {
        const maxLen = 160;

        const lastSent = prev.sentimentHistory.at(-1) ?? { t: 0, score: 0 };
        const lastOn = prev.onChainHistory.at(-1) ?? { t: 0, active: 100 };
        const lastDev = prev.devActivity.at(-1) ?? { t: 0, commits: 3 };

        const tNext =
          (Number(lastSent.t ?? lastOn.t ?? lastDev.t) || 0) + 1;

        const nextSent = {
          t: tNext,
          score: Math.max(
            -1,
            Math.min(
              1,
              (Number(lastSent.score) || 0) + (Math.random() - 0.5) * 0.08
            )
          ),
        };

        const nextOn = {
          t: tNext,
          active: Math.max(
            40,
            (Number(lastOn.active) || 100) *
            (1 + (Math.random() - 0.5) * 0.06)
          ),
        };

        const nextDev = {
          t: tNext,
          commits: Math.max(
            0,
            (Number(lastDev.commits) || 3) +
            Math.round((Math.random() - 0.3) * 2)
          ),
        };

        return {
          ...prev,
          sentimentHistory: [
            ...prev.sentimentHistory.slice(-maxLen + 1),
            nextSent,
          ],
          onChainHistory: [
            ...prev.onChainHistory.slice(-maxLen + 1),
            nextOn,
          ],
          devActivity: [
            ...prev.devActivity.slice(-maxLen + 1),
            nextDev,
          ],
        };
      });
    }, 2500);

    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [symbol]);

  const latestSent =
    intel.sentimentHistory.at(-1)?.score ?? 0;
  const latestOn =
    intel.onChainHistory.at(-1)?.active ?? 0;

  const latest = useMemo(
    () => ({
      sentiment: latestSent,
      onchain: latestOn,
    }),
    [latestSent, latestOn]
  );

  const sentimentLabel =
    latest.sentiment > 0.25
      ? "Bullish"
      : latest.sentiment < -0.25
        ? "Bearish"
        : "Neutral";

  const chainLabel =
    latest.onchain > 120
      ? "Hot"
      : latest.onchain < 80
        ? "Cool"
        : "Normal";

  const devLabel = "Stable";

  return { intel, latest, sentimentLabel, chainLabel, devLabel };
}


export function useLiveTrades(asset: Asset, mode: MarketMode) {
  const [rows, setRows] = useState<LiveTrade[]>([]);

  useEffect(() => {
    let alive = true;
    let stop: (() => void) | undefined;

    try {
      const maybeStop = API.tickLiveTrades(
        asset,
        mode,
        (updater: (prev: LiveTrade[]) => LiveTrade[]) => {
          if (!alive) return;
          setRows(updater);
        },
      );

      // Only store if backend actually returns an unsubscribe fn
      if (typeof maybeStop === "function") {
        stop = maybeStop;
      }
    } catch {
      // ignore errors in mock/demo mode
    }

    return () => {
      alive = false;
      if (stop) {
        stop();
      }
    };
  }, [asset, mode]);

  return rows;
}


/* ================================
   Sidebar + Shell
   ================================ */

const NAV_ITEMS = [
  { key: "dashboard", icon: Activity, label: "Dashboard", hash: "#dashboard" },
  { key: "strategies", icon: Brain, label: "Strategies", hash: "#strategies" },
  { key: "backtests", icon: History, label: "Backtests", hash: "#backtests" },
  { key: "live", icon: Signal, label: "Live Trades", hash: "#live" },
  { key: "portfolio", icon: Wallet, label: "Portfolio", hash: "#portfolio" },
  { key: "intel", icon: ChartArea, label: "Market Intel", hash: "#intel" },
  { key: "system", icon: ShieldCheck, label: "System", hash: "#system" },
  { key: "settings", icon: SlidersHorizontal, label: "Settings", hash: "#settings" },
] as const;
// const [collapsed, setCollapsed] = useState(false);
const Sidebar: React.FC<{
  open: boolean;
  isDesktop: boolean;
  active: string;
  onSelect: (key: string) => void;
  onCloseMobile: () => void;
  dark: boolean;
  onToggleDark: () => void;
}> = ({
  open,
  isDesktop,
  active,
  onSelect,
  onCloseMobile,
  dark,
  onToggleDark,
}) => {
    const [collapsed, setCollapsed] = useState(false);
    const widthClass = isDesktop
      ? collapsed
        ? "w-20"
        : "w-64"
      : "w-[85vw] max-w-xs";

    const content = (
      <aside
        className={`flex h-full flex-col ${glassPanel} ${widthClass} p-3 transition-all duration-300`}
      >
        <div className="flex items-center justify-between mb-2">
          {/* Left side: logo + text */}
          <div className="flex items-center gap-2">
            {/* Logo from public folder */}
            <img
              src="/nowa.png" // or "/logo.svg"
              alt="Nowa AI Logo"
              className="h-21 w-42 object-contain"
            />
          </div>

          {/* Close button (mobile only) */}
          {!isDesktop && (
            <Button
              variant="ghost"
              size="icon"
              onClick={onCloseMobile}
              title="Close"
            >
              <PanelLeftClose />
            </Button>
          )}
        </div>


        <nav className="mt-2 space-y-1" aria-label="Primary">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const activeCls =
              active === item.key
                ? "bg-white/15 text-white border border-white/25"
                : "hover:bg-white/10 text-slate-800 dark:text-slate-200";

            return (
              <button
                key={item.key}
                onClick={() => {
                  onSelect(item.key);
                  window.location.hash = item.hash;
                  if (!isDesktop) onCloseMobile();
                }}
                className={
                  "w-full flex items-center rounded-xl px-3 py-2 text-sm transition-colors " +
                  (collapsed && isDesktop
                    ? "justify-center gap-0"
                    : "gap-3") +
                  " " +
                  activeCls
                }
                aria-label={item.label}
                aria-current={active === item.key ? "page" : undefined}
              >
                <Icon
                  className="h-5 w-5 opacity-90"
                  aria-hidden="true"
                />
                {!(collapsed && isDesktop) && (
                  <span className="truncate">{item.label}</span>
                )}
              </button>
            );
          })}
        </nav>

        <div className="mt-auto pt-3 border-t border-white/10 flex items-center justify-end gap-2">
          <button
            onClick={onToggleDark}
            className="inline-flex h-8 w-8 items-center justify-center rounded-xl border border-white/20 bg-black/10 hover:bg-black/25"
            aria-label={
              dark
                ? "Switch to light theme"
                : "Switch to dark theme"
            }
          >
            {dark ? (
              <Sun
                className="w-4 h-4 text-yellow-400"
                aria-hidden="true"
              />
            ) : (
              <Moon
                className="w-4 h-4 text-slate-200"
                aria-hidden="true"
              />
            )}
          </button>

          {isDesktop && (
            <button
              onClick={() => setCollapsed((c) => !c)}
              className="inline-flex h-8 w-8 items-center justify-center rounded-xl border border-white/20 bg-black/10 hover:bg-black/25"
              aria-label={
                collapsed
                  ? "Expand sidebar"
                  : "Collapse sidebar"
              }
            >
              {collapsed ? (
                <ChevronRight
                  className="w-4 h-4 text-emerald-400"
                  aria-hidden="true"
                />
              ) : (
                <ChevronLeft
                  className="w-4 h-4 text-emerald-400"
                  aria-hidden="true"
                />
              )}
            </button>
          )}
        </div>
      </aside>
    );
    if (isDesktop) {
      return (
        <div className={`sticky top-0 h-dvh ${open ? "block" : "hidden"} lg:block`}>
          {content}
        </div>
      );
    }

    return (
      <>
        {open && (
          <div
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
            onClick={onCloseMobile}
            aria-hidden="true"
          />
        )}

        <motion.div
          className="fixed inset-y-0 left-0 z-50"
          initial={{ x: "-100%" }}
          animate={{ x: open ? 0 : "-100%" }}
          transition={{ type: "spring", stiffness: 320, damping: 32 }}
        >
          {content}
        </motion.div>
      </>
    );
  };
// ========= Top-of-dashboard Ticker Strip =========

const STRIP_SYMBOLS: SymbolCode[] = [
  "BTC/USDT",
  "ETH/USDT",
  "SOL/USDT",
  "BNB/USDT",
];

const MiniTicker: React.FC<{ symbol: SymbolCode }> = ({ symbol }) => {
  const { price, change } = useTicker(symbol);

  const pct =
    typeof change === "number" && !Number.isNaN(change)
      ? change.toFixed(2)
      : "0.00";

  const dirClass =
    change > 0
      ? "text-emerald-400"
      : change < 0
        ? "text-rose-400"
        : "text-slate-400";

  return (
    <div
      className={
        "flex items-center justify-between rounded-2xl px-3 py-2 border text-xs sm:text-[11px] " +
        "bg-white/85 text-slate-900 border-slate-200 shadow-sm " +
        "dark:bg-slate-950/80 dark:text-slate-50 dark:border-slate-700/80"
      }
    >
      <div className="flex flex-col">
        <span className="text-[9px] uppercase tracking-wide text-slate-500 dark:text-slate-400">
          {symbol}
        </span>
        <span className="text-sm font-semibold leading-tight">
          {price
            ? price.toLocaleString("en-US", {
              maximumFractionDigits: 2,
            })
            : "--"}
        </span>
      </div>
      <span className={`${dirClass} text-[10px] font-semibold`}>
        {change > 0 ? "+" : ""}
        {pct}%
      </span>
    </div>
  );
};

const TickerStrip: React.FC = () => {
  return (
    <div className="px-3 md:px-6 lg:px-8 mt-1 mb-2">
      <div className="max-w-7xl mx-auto grid grid-cols-2 sm:grid-cols-4 gap-2">
        {STRIP_SYMBOLS.map((s) => (
          <MiniTicker key={s} symbol={s} />
        ))}
      </div>
    </div>
  );
};
// ========= AI Insight Strip (under header on Dashboard) =========

// ========= NOWA AI Insight Strip (branded version) =========

// ========= NOWA AI Insight Strip (branded + animated) =========

const InsightStrip: React.FC<{
  symbol: SymbolCode;
  mode: MarketMode;
}> = ({ symbol, mode }) => {
  const predict = usePredict(symbol);
  const metrics = useDashboardMetrics(symbol);
  const {
    latest,
    sentimentLabel: baseSentimentLabel,
  } = useMarketIntel(symbol);

  const cleanSymbol = String(symbol).replace(/-PERP|-OPT/, "");
  const direction = (predict?.direction || "HOLD").toUpperCase();
  const confidence = Math.round((predict?.price_confidence || 0.5) * 100);
  const regime = (predict?.regime || "NEUTRAL").toString().toUpperCase();
  const risk = (predict?.risk || "STABLE").toString().toUpperCase();

  // latest.sentiment is in [-1, 1]; map to 0–100 for display
  const sentimentScore = latest?.sentiment ?? 0;
  const sentimentPct = Math.round((sentimentScore + 1) * 50);

  const sentimentLabel =
    baseSentimentLabel === "Bullish"
      ? "RISK-ON"
      : baseSentimentLabel === "Bearish"
        ? "RISK-OFF"
        : "BALANCED";
  const signalAccent =
    direction === "LONG"
      ? "nowa-signal-pulse-long"
      : direction === "SHORT"
        ? "nowa-signal-pulse-short"
        : "nowa-signal-pulse-neutral";

  const Tag = ({
    title,
    value,
    accent = "",
  }: {
    title: string;
    value: React.ReactNode;
    accent?: string;
  }) => (
    <div
      className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-2xl border text-[10px] md:text-[11px] 
        bg-white/80 text-slate-900 border-slate-200 shadow-sm
        dark:bg-slate-950/90 dark:text-slate-50 dark:border-slate-700/70
        nowa-tag-hover ${accent}`}
    >
      <span className="uppercase tracking-wide text-[9px] opacity-70">
        {title}
      </span>
      <span className="font-semibold flex items-center gap-1">
        {value}
      </span>
    </div>
  );

  return (
    <div className="px-3 md:px-6 lg:px-8 mt-1 mb-2 select-none">
      <div className="max-w-7xl mx-auto flex flex-wrap gap-2 items-center">
        {/* 1️⃣ Active asset / mode */}
        {/* <Tag
          title="Active Market"
          value={
            <>
              {cleanSymbol}
              <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-slate-900/5 dark:bg-slate-50/5 text-slate-600 dark:text-slate-300">
                {mode.toUpperCase()}
              </span>
            </>
          }
        /> */}

        {/* 2️⃣ Nowa Bot Signal (pulsing glow) */}
        <Tag
          title="Nowa Bot Signal"
          accent={signalAccent}
          value={
            <>
              <span
                className={
                  direction === "LONG"
                    ? "text-emerald-400"
                    : direction === "SHORT"
                      ? "text-rose-400"
                      : "text-sky-400"
                }
              >
                {direction === "HOLD" ? "HOLD / NEUTRAL" : direction}
              </span>
              <span className="text-[9px] opacity-75">
                {confidence || 50}% confidence
              </span>
            </>
          }
        />

        {/* 3️⃣ AI Prediction */}
        <Tag
          title="AI Trend Read"
          value={
            regime === "BULLISH"
              ? "Uptrend bias"
              : regime === "BEARISH"
                ? "Downtrend risk"
                : "Range-bound / neutral"
          }
        />

        {/* 4️⃣ Market Sentiment */}
        <Tag
          title="Market Sentiment"
          value={
            <>
              <span
                className={
                  sentimentLabel === "RISK-ON"
                    ? "text-emerald-400"
                    : sentimentLabel === "RISK-OFF"
                      ? "text-rose-400"
                      : "text-sky-400"
                }
              >
                {sentimentLabel}
              </span>
              <span className="text-[9px] opacity-75">
                {sentimentPct}%
              </span>
            </>
          }
        />

        {/* 5️⃣ AI Risk */}
        <Tag
          title="AI Risk Level"
          value={
            <span
              className={
                risk === "HIGH"
                  ? "text-rose-400"
                  : risk === "ELEVATED"
                    ? "text-amber-400"
                    : "text-emerald-400"
              }
            >
              {risk}
            </span>
          }
        />

        {/* 6️⃣ Micro Alpha (funding + flow) */}
        <Tag
          title="Micro Alpha Inputs"
          value={
            <>
              <span className="opacity-80">
                Funding {(metrics?.funding_rate ?? 0).toFixed(4)}
              </span>
              <span className="w-px h-3 bg-slate-300/60 dark:bg-slate-600/80" />
              <span className="opacity-80">
                Flow {((metrics?.order_flow ?? 0) * 100).toFixed(1)}%
              </span>
            </>
          }
        />
      </div>
    </div>
  );
};


export const Shell: React.FC<
  React.PropsWithChildren<{
    symbol: SymbolCode;
    setSymbol: (s: SymbolCode) => void;
    mode: MarketMode;
    setMode: (m: MarketMode) => void;
    exchange: string;
    setExchange: (e: string) => void;
    tab: string;
    setTab: (v: string) => void;
  }>
> = ({ children, symbol, setSymbol, mode, setMode, exchange, setExchange, tab, setTab }) => {
  const isMobile = useIsMobile();
  const isDesktop = !isMobile;

  const [sidebarOpen, setSidebarOpen] = useState(isDesktop);
  useEffect(() => setSidebarOpen(isDesktop), [isDesktop]);
  const handleToggleSidebar = () => {
    setSidebarOpen((prev) => !prev);
  };
  const [onboardingDismissed, setOnboardingDismissed] =
    useLocalStorage<boolean>("nowa.onboarding.dismissed", false);
  const [dark, setDark] = useLocalStorage<boolean>("nowa.dark", true);
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  const pathname = usePathname?.();
  useEffect(() => {
    if (!isDesktop) setSidebarOpen(false);
  }, [pathname, isDesktop]);

  return (
    <div
      className="h-dvh w-full overflow-hidden text-slate-900 dark:text-white relative"
      style={{
        background:
          "radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,0.18), transparent 60%), radial-gradient(1200px 600px at 100% 10%, rgba(16,185,129,0.16), transparent 60%), linear-gradient(180deg, #f8fafc 0%, #e9edf5 60%, #e9edf5 100%)",
      }}
    >
      <div className="pointer-events-none absolute inset-0 hidden dark:block bg-[radial-gradient(900px_400px_at_0%_0%,rgba(99,102,241,0.12),transparent_60%),radial-gradient(800px_400px_at_100%_20%,rgba(16,185,129,0.12),transparent_60%),linear-gradient(180deg,#0b0f1a_0%,#0b0f1a_100%)]" />

      <div className="relative lg:flex h-full">
        <Sidebar
          open={sidebarOpen}
          isDesktop={isDesktop}
          active={tab}
          onSelect={setTab}
          onCloseMobile={() => setSidebarOpen(false)}
          dark={dark}
          onToggleDark={() => setDark(!dark)}
        />

        <main className="flex-1 h-dvh overflow-y-auto w-full">
          <header className="sticky top-0 z-20 w-full bg-transparent px-3 md:px-6 py-2">
            <NeonHeader
              symbol={symbol}
              mode={mode}
              exchange={exchange}
              onSetSymbol={setSymbol}
              onSetMode={setMode}
              onSetExchange={setExchange}
              isMobile={!isDesktop}
              onToggleSidebar={handleToggleSidebar}
            />
          </header>

          {/* Top tickers row only on Dashboard */}
          {tab === "dashboard" && (
            <InsightStrip symbol={symbol} mode={mode} />
          )}

          <div className="p-3 md:p-6 lg:p-8">
            <div className="max-w-7xl mx-auto">{children}</div>
          </div>
        </main>

      </div>
 <OnboardingModal
      open={!onboardingDismissed}
      onClose={() => setOnboardingDismissed(true)}
    />
      <Toaster />
    </div>
  );
};


const OnboardingModal: React.FC<{
  open: boolean;
  onClose: () => void;
}> = ({ open, onClose }) => {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/65 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="nowa-onboarding-title"
    >
      <div className="max-w-lg w-[90%] rounded-3xl p-6 bg-slate-950/95 text-slate-100 border border-slate-700/70 shadow-2xl">
        <h2
          id="nowa-onboarding-title"
          className="text-xl font-semibold mb-2"
        >
          Welcome to Nowa
        </h2>
        <p className="text-sm text-slate-300 mb-3">
          A quick guide so you can trust what you see:
        </p>
        <ul className="list-disc list-inside space-y-1 text-xs text-slate-300">
          <li>
            <strong>Nowa Bot Signal</strong> — AI-derived LONG / SHORT / NEUTRAL
            bias with confidence (pulsing glow shows importance).
          </li>
          <li>
            <strong>AI Trend Read</strong> — short-term regime:
            bullish, bearish, or range-bound.
          </li>
          <li>
            <strong>Micro Alpha Inputs</strong> — funding, order flow, and
            on-chain activity that influence the signal.
          </li>
          <li>
            Switch tabs from the left sidebar for Strategies, Backtests,
            Live Trades, and more.
          </li>
        </ul>
        <div className="flex justify-end gap-2 mt-4">
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-xs rounded-xl bg-slate-800 hover:bg-slate-700"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  );
};

