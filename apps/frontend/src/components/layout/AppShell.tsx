"use client";

import React, { useEffect, useState, useMemo } from "react";
import NeonHeader from "@/components/NeonHeader";
import type { MarketMode, SymbolCode } from "@/lib/api";
import { TrendingUp, TrendingDown, Minus, Info } from "lucide-react";

/* ---------- small helper ---------- */

function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}

/* ---------- shared UI tokens exported for tabs ---------- */

export const glassPanel =
  "bg-slate-900/60 border border-white/10 backdrop-blur-xl shadow-[0_18px_40px_rgba(15,23,42,0.45)] " +
  "dark:bg-slate-950/80 dark:border-slate-700/80";

export const faintText =
  "text-xs text-slate-500 dark:text-slate-400";

/* ---------- Card primitives (used in multiple tabs) ---------- */

interface CardProps extends React.HTMLAttributes<HTMLDivElement> { }

export const Card: React.FC<CardProps> = ({ className, ...props }) => (
  <div
    className={cn(
      "rounded-2xl border border-white/10 bg-slate-900/70 backdrop-blur-xl",
      "shadow-[0_18px_45px_rgba(15,23,42,0.55)]",
      "dark:bg-slate-950/80 dark:border-slate-800",
      className
    )}
    {...props}
  />
);

export const CardHeader: React.FC<CardProps> = ({
  className,
  ...props
}) => (
  <div
    className={cn(
      "px-4 pt-4 pb-2 flex flex-col gap-1 border-b border-white/5",
      "dark:border-slate-800",
      className
    )}
    {...props}
  />
);

export const CardTitle: React.FC<CardProps> = ({
  className,
  ...props
}) => (
  <h3
    className={cn(
      "text-sm font-semibold text-slate-100 tracking-tight",
      className
    )}
    {...props}
  />
);

export const CardDescription: React.FC<CardProps> = ({
  className,
  ...props
}) => (
  <p
    className={cn(
      "text-xs text-slate-500 dark:text-slate-400",
      className
    )}
    {...props}
  />
);

export const CardContent: React.FC<CardProps> = ({
  className,
  ...props
}) => (
  <div
    className={cn("px-4 pb-4 pt-2", className)}
    {...props}
  />
);

/* ---------- KPI tile ---------- */

interface KpiTileProps extends React.HTMLAttributes<HTMLDivElement> {
  label?: string;
  value?: React.ReactNode;
  hint?: React.ReactNode;
}

export const KpiTile: React.FC<KpiTileProps> = ({
  label,
  value,
  hint,
  className,
  ...props
}) => (
  <div
    className={cn(
      glassPanel,
      "rounded-xl p-3 flex flex-col gap-0.5",
      className
    )}
    {...props}
  >
    {label && (
      <div className={faintText}>{label}</div>
    )}
    {value && (
      <div className="text-sm font-semibold text-slate-50">
        {value}
      </div>
    )}
    {hint && (
      <div className="text-[10px] text-slate-500 dark:text-slate-400">
        {hint}
      </div>
    )}
  </div>
);
/* ---------- Tag primitive ---------- */

interface TagProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "emerald" | "rose" | "amber" | "blue";
}

export const Tag: React.FC<TagProps> = ({
  variant = "default",
  className,
  children,
  ...props
}) => {
  const variantClasses = {
    default: "bg-slate-800 text-slate-300 border-slate-700",
    emerald: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    rose:    "bg-rose-500/10 text-rose-400 border-rose-500/20",
    amber:   "bg-amber-500/10 text-amber-400 border-amber-500/20",
    blue:    "bg-blue-500/10 text-blue-400 border-blue-500/20",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-[10px] font-medium",
        variantClasses[variant],
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
};

/* ---------- Button primitive (used in SystemTab) ---------- */

type ButtonVariant = "primary" | "secondary" | "outline" | "ghost";

interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
}

export const Button: React.FC<ButtonProps> = ({
  variant = "primary",
  className,
  children,
  ...props
}) => {
  const variantClasses =
    variant === "primary"
      ? "bg-emerald-500 text-slate-900 hover:bg-emerald-400 border-transparent"
      : variant === "secondary"
        ? "bg-slate-800 text-slate-100 hover:bg-slate-700 border-slate-700"
        : variant === "outline"
          ? "bg-transparent text-slate-100 border-slate-600 hover:bg-slate-900/60"
          : "bg-transparent text-slate-300 border-transparent hover:bg-slate-900/50";

  return (
    <button
      className={cn(
        "inline-flex items-center justify-center rounded-xl border px-3 py-1.5 text-xs font-medium",
        "transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400",
        variantClasses,
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
};
/* ---------- Form primitives (Input, Label) ---------- */

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-9 w-full rounded-md border border-slate-800 bg-slate-950 px-3 py-1 text-sm shadow-sm transition-colors",
          "file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-slate-400",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-emerald-400 disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        ref={ref}
        {...props}
      />
    );
  }
);
Input.displayName = "Input";

export const Label = React.forwardRef<
  HTMLLabelElement,
  React.LabelHTMLAttributes<HTMLLabelElement>
>(({ className, ...props }, ref) => (
  <label
    ref={ref}
    className={cn(
      "text-xs font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 text-slate-300",
      className
    )}
    {...props}
  />
));
Label.displayName = "Label";

/* ---------- Badge primitive ---------- */

interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "secondary" | "destructive" | "outline";
}

export const Badge: React.FC<BadgeProps> = ({
  className,
  variant = "default",
  ...props
}) => {
  const variants = {
    default: "border-transparent bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20",
    secondary: "border-transparent bg-slate-800 text-slate-100 hover:bg-slate-700",
    destructive: "border-transparent bg-rose-500/10 text-rose-400 hover:bg-rose-500/20",
    outline: "text-slate-100 border-slate-700",
  };
  
  return (
    <div
      className={cn(
        "inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2",
        variants[variant],
        className
      )}
      {...props}
    />
  );
};
/* ---------- seeded RNG (for demo stats etc.) ---------- */

export function makeSeeded(seed: string): () => number {
  // FNV-1a style hash of the seed into a 32-bit int, then xorshift
  let h = 2166136261 >>> 0;

  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }

  return () => {
    // xorshift32
    h += h << 13;
    h ^= h >>> 7;
    h += h << 3;
    h ^= h >>> 17;
    h += h << 5;

    // Normalize to [0, 1)
    return (h >>> 0) / 4294967295;
  };
}

/* ---------- Data hooks for Dashboard & Intel (demo-friendly) ---------- */
/* These are lightweight demo implementations so your UI keeps working.
   They preserve the same shape DashboardTab expects. You can later swap
   them to call real backend endpoints if needed. */

type PredictResult = {
  direction: "up" | "down" | "flat";
  price_confidence: number; // 0–1
  volatility_pct: number;   // %
};

export function usePredict(symbol: SymbolCode): PredictResult {
  return useMemo(() => {
    const base = symbol.length;
    const direction: PredictResult["direction"] =
      base % 3 === 0 ? "up" : base % 3 === 1 ? "down" : "flat";
    const price_confidence = 0.55 + ((base % 10) / 100); // ~0.55–0.64
    const volatility_pct = 20 + (base % 20);             // ~20–39
    return { direction, price_confidence, volatility_pct };
  }, [symbol]);
}

type DashboardMetrics = {
  funding_rate: number;   // e.g. 0.0001 = 1 bps
  order_flow: number;     // -1..1, positive = bid heavy
  put_call_ratio: number; // options PCR
};

export function useDashboardMetrics(symbol: SymbolCode): DashboardMetrics {
  return useMemo(() => {
    const base = symbol.length;
    const funding_rate = ((base % 7) - 3) / 10_000;   // ~-30..+30 bps
    const order_flow = ((base % 10) - 5) / 10;        // -0.5..0.4
    const put_call_ratio = 0.8 + (base % 5) * 0.05;   // 0.8..1.0
    return { funding_rate, order_flow, put_call_ratio };
  }, [symbol]);
}

type MarketIntelData = {
  intel: {
    sentimentHistory: { t: number; score: number }[];
    onChainHistory: { t: number; v: number }[];
  };
  latest: { sentiment: number };
};

export function useMarketIntel(symbol: SymbolCode): MarketIntelData {
  return useMemo(() => {
    const base = symbol.length;
    const sentimentBase = ((base % 5) - 2) / 4; // ~ -0.5..0.5

    const sentimentHistory = Array.from({ length: 24 }, (_, i) => ({
      t: i,
      score: sentimentBase + Math.sin(i / 3) * 0.2,
    }));

    const onChainHistory = Array.from({ length: 24 }, (_, i) => ({
      t: i,
      v: 100 + i * 3 + (base % 10),
    }));

    return {
      intel: { sentimentHistory, onChainHistory },
      latest: { sentiment: sentimentBase },
    };
  }, [symbol]);
}

/* ---------- DirectionLabel & HelpTooltip ---------- */

interface DirectionLabelProps {
  value: number;
  ariaLabel?: string;
}

export const DirectionLabel: React.FC<DirectionLabelProps> = ({
  value,
  ariaLabel,
}) => {
  const isUp = value > 0;
  const isDown = value < 0;
  const Icon = isUp ? TrendingUp : isDown ? TrendingDown : Minus;
  const color =
    isUp ? "text-emerald-400" : isDown ? "text-rose-400" : "text-slate-400";

  return (
    <div
      className="mt-1 inline-flex items-center gap-1 rounded-full bg-slate-900/80 px-2 py-1 text-[10px] text-slate-300"
      aria-label={ariaLabel}
    >
      <Icon className={cn("w-3 h-3", color)} />
      <span className={color}>
        {isUp ? "Positive skew" : isDown ? "Negative skew" : "Flat"}
      </span>
    </div>
  );
};

interface HelpTooltipProps {
  label: string;
}

/* Simple inline helper; if you later want a real tooltip library you can swap it */
export const HelpTooltip: React.FC<HelpTooltipProps> = ({ label }) => (
  <span className="inline-flex items-center gap-1 text-[10px] text-slate-400">
    <Info className="w-3 h-3" />
    {label}
  </span>
);

/* ---------- localStorage hook (unchanged API) ---------- */

export function useLocalStorage<T>(
  key: string,
  defaultValue: T
): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(() => {
    if (typeof window === "undefined") return defaultValue;
    try {
      const stored = window.localStorage.getItem(key);
      return stored ? (JSON.parse(stored) as T) : defaultValue;
    } catch {
      return defaultValue;
    }
  });

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(
        key,
        JSON.stringify(value)
      );
    } catch {
      // ignore
    }
  }, [key, value]);

  return [value, setValue];
}

/* ---------- Shell layout (no sidebar, top header only) ---------- */

type ShellProps = {
  symbol: SymbolCode;
  setSymbol: (s: SymbolCode) => void;
  mode: MarketMode;
  setMode: (m: MarketMode) => void;
  exchange: string;
  setExchange: (e: string) => void;
  tab: string;
  setTab: (next: string) => void;
  children: React.ReactNode;
};

export const Shell: React.FC<ShellProps> = ({
  symbol,
  setSymbol,
  // kept for API compatibility if tabs use them internally
  mode,
  setMode,
  exchange,
  setExchange,
  tab,
  setTab,
  children,
}) => {
  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-50">
      {/* Top header with search, 4 nav items, live price card */}
      <header
        className="sticky top-0 z-40 px-3 md:px-4 lg:px-6 py-3
             bg-gradient-to-b from-slate-950/95 via-slate-950/92 to-slate-950/88
             backdrop-blur-xl shadow-[0_12px_30px_rgba(15,23,42,0.9)]"
      >
        <div className="relative">
          <NeonHeader
            symbol={symbol}
            onSetSymbol={setSymbol}
            activeTab={tab}
            onTabChange={setTab}
          />

          {/* Glass gradient underline under full header */}
          <div className="pointer-events-none absolute inset-x-0 -bottom-[2px] h-[2px]">
            <div className="w-full h-full bg-gradient-to-r from-transparent via-emerald-400/60 to-transparent blur-[2px] opacity-80" />
          </div>
        </div>
      </header>

      {/* Main content area */}
      <main className="flex-1 px-3 md:px-4 lg:px-6 pb-6 pt-4">
        <div className="w-full">{children}</div>
      </main>
    </div>
  );
};