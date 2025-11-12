// src/components/NeonHeader.tsx
"use client";

import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
  FormEvent,
} from "react";
import { motion } from "framer-motion";
import {
  Menu,
  Search,
  Store,
  TrendingUp,
  TrendingDown,
  Minus,
  Shield,
  Send,
} from "lucide-react";
import { useTheme } from "next-themes";
import API, {
  EXCHANGES,
  MarketMode,
  SymbolCode,
} from "@/lib/api";
import {
  LineChart,
  Line,
  ResponsiveContainer,
} from "recharts";
import { getIconSrcForSymbol } from "@/lib/cryptoIcons";
import {
  useAllowedExchanges,
  useWorkspaceLabel,
} from "@/lib/profile";

type Props = {
  symbol: SymbolCode;
  mode: MarketMode;
  exchange: string;
  onSetSymbol: (s: SymbolCode) => void;
  onSetMode: (m: MarketMode) => void;
  onSetExchange: (e: string) => void;
  isMobile?: boolean;
  onToggleSidebar?: () => void;
};

const NeonHeader: React.FC<Props> = ({
  symbol,
  mode,
  exchange,
  onSetSymbol,
  onSetMode,
  onSetExchange,
  isMobile = false,
  onToggleSidebar,
}) => {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";

  const allowedExchanges = useAllowedExchanges();
  const workspace = useWorkspaceLabel();

  // ---------- ticker + sparkline ----------
  const [ticker, setTicker] = useState({ price: 0, change: 0 });
  const [miniPrices, setMiniPrices] = useState<{ p: number }[]>([]);
  const [flash, setFlash] = useState<"up" | "down" | "none">("none");
  const lastPrice = useRef(0);

  useEffect(() => {
    let alive = true;

    (async () => {
      try {
        const d: any = await API.getTicker(symbol);
        if (!alive || !d) return;
        const price = Number(d.price) || 0;
        const change = Number(d.change) || 0;
        setTicker({ price, change });
      } catch {
        if (alive) setTicker({ price: 0, change: 0 });
      }
    })();

    const stop = API.streamTicker(symbol, (d: any) => {
      if (!alive || !d) return;
      const price = Number(d.price) || 0;
      const change =
        typeof d.change === "number" ? d.change : ticker.change;
      setTicker({ price, change });
    });

    return () => {
      alive = false;
      if (typeof stop === "function") stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol]);

  useEffect(() => {
    if (!ticker.price) return;

    setMiniPrices((prev) => {
      const next = [...prev, { p: ticker.price }];
      if (next.length > 80) next.shift();
      return next;
    });

    if (!lastPrice.current) {
      lastPrice.current = ticker.price;
      return;
    }

    const dir =
      ticker.price > lastPrice.current
        ? "up"
        : ticker.price < lastPrice.current
        ? "down"
        : "none";

    if (dir !== "none") {
      setFlash(dir);
      const id = setTimeout(() => setFlash("none"), 220);
      lastPrice.current = ticker.price;
      return () => clearTimeout(id);
    }
  }, [ticker.price]);

  // ---------- search + suggestions ----------
  const [q, setQ] = useState("");
  const [suggestions, setSuggestions] = useState<SymbolCode[]>([]);
  const [prices, setPrices] = useState<Record<string, number>>({});
  const [open, setOpen] = useState(false);

  const assets = useMemo(
    () =>
      API.TOP_ASSETS.map((a: any) =>
        API.formatSymbol(a, mode)
      ) as SymbolCode[],
    [mode]
  );

  useEffect(() => {
    const term = q.trim().toLowerCase();
    if (!term) {
      setSuggestions([]);
      setOpen(false);
      return;
    }
    const list = assets
      .filter((s) => s.toLowerCase().includes(term))
      .slice(0, 8);
    setSuggestions(list);
    setOpen(list.length > 0);
  }, [q, assets]);

  useEffect(() => {
    if (!open || suggestions.length === 0) return;
    let alive = true;

    (async () => {
      try {
        const out = await Promise.all(
          suggestions.map(async (s) => {
            try {
              const t: any = await API.getTicker(s);
              return [s, Number(t.price) || 0] as const;
            } catch {
              return [s, 0] as const;
            }
          })
        );
        if (!alive) return;
        const next: Record<string, number> = {};
        for (const [s, p] of out) next[s] = p;
        setPrices(next);
      } catch {
        if (!alive) return;
      }
    })();

    return () => {
      alive = false;
    };
  }, [open, suggestions]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const term = q.trim().toLowerCase();
    if (!term) return;
    const hit =
      suggestions[0] ||
      assets.find((s) => s.toLowerCase().includes(term));
    if (hit) onSetSymbol(hit);
    setQ("");
    setSuggestions([]);
    setOpen(false);
  };

  const pick = (s: SymbolCode) => {
    onSetSymbol(s);
    setQ("");
    setSuggestions([]);
    setOpen(false);
  };

  // ---------- styles ----------
  const priceColor =
    ticker.change > 0
      ? "text-emerald-500 dark:text-emerald-300"
      : ticker.change < 0
      ? "text-rose-500 dark:text-rose-300"
      : "text-slate-500 dark:text-slate-400";

  const sparkColor =
    ticker.change >= 0 ? "#22c55e" : "#fb7185";

  const desktopWrapper =
    "hidden md:flex w-full relative items-center gap-4 rounded-3xl border px-6 py-3 " +
    "border-border bg-[color-mix(in_oklch,var(--card)_96%,transparent)] backdrop-blur-xl " +
    "shadow-[0_14px_40px_rgba(15,23,42,0.06)] " +
    "dark:border-slate-800/70 dark:bg-slate-950/70";

  const searchWrapperDesktop =
    "relative flex-1 h-12 rounded-2xl flex items-center px-4 border " +
    "bg-white/90 border-border shadow-soft-sm " +
    "dark:bg-slate-950/92 dark:border-slate-700/70";

  const searchInputText =
    "w-full bg-transparent text-sm outline-none " +
    "text-slate-900 placeholder:text-slate-500 " +
    "dark:text-white dark:placeholder:text-slate-500";

  const modesWrapper =
    "flex h-12 items-center rounded-2xl px-1 " +
    "bg-[var(--muted)] border border-[var(--border)] shadow-soft-sm " +
    "dark:bg-white/5 dark:border-slate-700/60";

  const exchangeWrapper =
    "flex h-12 items-center gap-2 rounded-2xl px-3 border " +
    "bg-white/90 border-border shadow-soft-sm " +
    "dark:bg-slate-950/95 dark:border-slate-700/70";

  const priceBlock =
    "flex h-12 items-center gap-3 rounded-2xl px-4 border shadow-inner " +
    "bg-[var(--primary)]/6 border-[var(--primary)]/35 " +
    "dark:bg-slate-950/98 dark:border-slate-700/80";

  const suggestionListDesktop =
    "absolute left-0 right-0 top-[110%] z-30 max-h-64 overflow-y-auto rounded-2xl border shadow-2xl backdrop-blur-md text-[10px] " +
    "bg-white border-slate-200 " +
    "dark:bg-slate-950/98 dark:border-slate-700/70";

  const mobileWrapper =
    "md:hidden relative mt-2";
  const mobileInner =
    "relative flex flex-col gap-2 rounded-3xl border px-3.5 py-2.5 shadow-soft-lg " +
    "border-border bg-[color-mix(in_oklch,var(--card)_96%,transparent)] backdrop-blur-xl " +
    "dark:border-teal-500/15 dark:bg-slate-950/95";

  const mobileSearchWrapper =
    "flex items-center h-9 rounded-2xl px-3 border " +
    "bg-white/95 border-border shadow-soft-sm " +
    "dark:bg-slate-900/92 dark:border-slate-700";

  const mobileSearchInput =
    "w-full bg-transparent text-[11px] outline-none " +
    "text-slate-800 placeholder:text-slate-400 " +
    "dark:text-slate-100 dark:placeholder:text-slate-500";

  const suggestionListMobile =
    "absolute left-0 right-0 top-[105%] z-30 max-h-56 overflow-y-auto rounded-2xl border shadow-2xl backdrop-blur-md text-[9px] " +
    "bg-white border-slate-200 " +
    "dark:bg-slate-950/98 dark:border-slate-800";

  const suggestionButton =
    "flex w-full items-center justify-between px-3 py-1.5 transition-all hover:bg-emerald-500/8";

  // ---------- RENDER ----------
  return (
    <>
      {/* DESKTOP / TABLET */}
      <div className={desktopWrapper}>
        {/* Search */}
        <form
          onSubmit={handleSubmit}
          className={searchWrapperDesktop}
        >
          <Search className="w-4 h-4 text-slate-500 mr-2" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search market & hit Enter"
            className={searchInputText}
            onFocus={() => suggestions.length && setOpen(true)}
            onBlur={() => {
              setTimeout(() => setOpen(false), 120);
            }}
          />
          {open && suggestions.length > 0 && (
            <div className={suggestionListDesktop}>
              {suggestions.map((s) => {
                const iconSrc = getIconSrcForSymbol(s);
                const price =
                  prices[s] !== undefined
                    ? prices[s].toLocaleString("en-US", {
                        style: "currency",
                        currency: "USD",
                        maximumFractionDigits: 2,
                      })
                    : "--";
                return (
                  <button
                    key={s}
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => pick(s)}
                    className={suggestionButton}
                  >
                    <div className="flex items-center gap-2">
                      {iconSrc ? (
                        <img
                          src={iconSrc}
                          alt={s}
                          className="w-4 h-4 rounded-full"
                        />
                      ) : (
                        <div className="w-4 h-4 rounded-full bg-slate-800 text-[8px] flex items-center justify-center text-white">
                          {s[0]}
                        </div>
                      )}
                      <span>{s}</span>
                    </div>
                    <span className="text-slate-500">
                      {price}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </form>

        {/* Mode toggle */}
        <div className={modesWrapper}>
          {(["futures", "options", "spot"] as MarketMode[]).map(
            (m) => {
              const active = m === mode;
              return (
                <button
                  key={m}
                  onClick={() => onSetMode(m)}
                  className={
                    "mx-1 rounded-2xl px-3 py-1.5 text-sm font-medium capitalize transition-all " +
                    (active
                      ? isDark
                        ? "bg-emerald-500/25 text-white border border-emerald-400/60 shadow-inner"
                        : "bg-[var(--primary)] text-[var(--primary-foreground)] shadow-soft-sm"
                      : isDark
                      ? "text-slate-100/80 hover:bg-white/8"
                      : "text-slate-700 hover:bg-white/90 hover:text-slate-900")
                  }
                >
                  {m}
                </button>
              );
            }
          )}
        </div>

        {/* Exchange (filtered by onboarding) */}
        <div className={exchangeWrapper}>
          <Store className="w-4 h-4 text-slate-500" />
          <select
            value={
              allowedExchanges.includes(exchange)
                ? exchange
                : allowedExchanges[0] || EXCHANGES[0]
            }
            onChange={(e) => onSetExchange(e.target.value)}
            className="bg-transparent text-sm outline-none text-slate-800 dark:text-slate-100"
          >
            {allowedExchanges.map((ex) => (
              <option key={ex} value={ex}>
                {ex}
              </option>
            ))}
          </select>
          <span className="text-[9px] text-slate-500">
            venues from client onboarding
          </span>
        </div>

        {/* Price + sparkline */}
        <div
          className={
            priceBlock +
            (flash === "up"
              ? " ring-2 ring-emerald-500/40"
              : flash === "down"
              ? " ring-2 ring-rose-500/40"
              : "")
          }
        >
          <div className="flex flex-col leading-tight">
            <div className="text-[10px] text-slate-500">
              {symbol}
            </div>
            <div className="text-sm font-semibold text-slate-900 dark:text-slate-50">
              {ticker.price
                ? ticker.price.toLocaleString("en-US", {
                    style: "currency",
                    currency: "USD",
                    maximumFractionDigits: 2,
                  })
                : "--"}
            </div>
          </div>
          <div
            className={
              "flex items-center gap-1 text-[10px] " +
              priceColor
            }
          >
            {ticker.change > 0 && (
              <TrendingUp className="w-3 h-3" />
            )}
            {ticker.change < 0 && (
              <TrendingDown className="w-3 h-3" />
            )}
            {ticker.change === 0 && (
              <Minus className="w-3 h-3" />
            )}
            <span>
              {ticker.change > 0 ? "+" : ""}
              {ticker.change.toFixed(2)}%
            </span>
          </div>
          <div className="w-20 h-8">
            <ResponsiveContainer>
              <LineChart data={miniPrices}>
                <Line
                  type="monotone"
                  dataKey="p"
                  stroke={sparkColor}
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Workspace badge */}
        {workspace && (
          <div className="ml-auto flex flex-col items-end text-[9px]">
            <div className="inline-flex items-center gap-1 px-2 py-1 rounded-2xl bg-slate-900/90 text-slate-100 border border-indigo-500/40">
              <Shield className="w-3 h-3 text-indigo-400" />
              <span className="font-semibold">
                {workspace.name}
              </span>
            </div>
            <div className="text-slate-500 mt-0.5">
              {workspace.riskText}
            </div>
            <div className="flex items-center gap-2 text-slate-500">
              <span>
                Venues: {allowedExchanges.length}
              </span>
              <span className="flex items-center gap-1">
                <Send className="w-3 h-3" />
                {workspace.hasTelegram
                  ? "Telegram linked"
                  : "Telegram pending"}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* MOBILE */}
      <div className={mobileWrapper}>
        <div className={mobileInner}>
          <div className="flex items-center gap-2">
            {isMobile && onToggleSidebar && (
              <button
                onClick={onToggleSidebar}
                className="p-1.5 rounded-xl bg-slate-900/90 border border-slate-700 text-slate-100"
              >
                <Menu className="w-4 h-4" />
              </button>
            )}
            <form
              onSubmit={handleSubmit}
              className={mobileSearchWrapper}
            >
              <Search className="w-3 h-3 text-slate-500 mr-1.5" />
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Search market"
                className={mobileSearchInput}
                onFocus={() =>
                  suggestions.length && setOpen(true)
                }
                onBlur={() => {
                  setTimeout(() => setOpen(false), 120);
                }}
              />
              {open && suggestions.length > 0 && (
                <div className={suggestionListMobile}>
                  {suggestions.map((s) => {
                    const iconSrc = getIconSrcForSymbol(s);
                    const price =
                      prices[s] !== undefined
                        ? prices[s].toLocaleString("en-US", {
                            maximumFractionDigits: 2,
                          })
                        : "--";
                    return (
                      <button
                        key={s}
                        onMouseDown={(e) => e.preventDefault()}
                        onClick={() => pick(s)}
                        className={suggestionButton}
                      >
                        <div className="flex items-center gap-2">
                          {iconSrc ? (
                            <img
                              src={iconSrc}
                              alt={s}
                              className="w-3.5 h-3.5 rounded-full"
                            />
                          ) : (
                            <div className="w-3.5 h-3.5 rounded-full bg-slate-800 text-[7px] flex items-center justify-center text-white">
                              {s[0]}
                            </div>
                          )}
                          <span>{s}</span>
                        </div>
                        <span className="text-slate-500">
                          {price}
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </form>
          </div>

          {/* Modes */}
          <div className="flex gap-1 mt-1">
            {(["futures", "options", "spot"] as MarketMode[]).map(
              (m) => {
                const active = m === mode;
                return (
                  <button
                    key={m}
                    onClick={() => onSetMode(m)}
                    className={
                      "whitespace-nowrap rounded-2xl px-3 py-1 text-[10px] font-medium capitalize border transition-all " +
                      (active
                        ? "bg-[var(--primary)] text-[var(--primary-foreground)] border-[var(--primary)] dark:bg-emerald-500/25 dark:text-white dark:border-emerald-400/60"
                        : "bg-white text-slate-800 border-border shadow-soft-sm dark:bg-slate-900/95 dark:text-slate-200 dark:border-slate-700")
                    }
                  >
                    {m}
                  </button>
                );
              }
            )}
          </div>

          {/* Exchange + price */}
          <div className="mt-1 flex items-center gap-2">
            <select
              value={
                allowedExchanges.includes(exchange)
                  ? exchange
                  : allowedExchanges[0] || EXCHANGES[0]
              }
              onChange={(e) => onSetExchange(e.target.value)}
              className={
                "rounded-xl px-2 py-1 text-[9px] outline-none border " +
                "bg-white text-slate-800 border-border shadow-soft-sm " +
                "dark:bg-slate-900/95 dark:text-slate-100 dark:border-slate-700"
              }
            >
              {allowedExchanges.map((ex) => (
                <option key={ex} value={ex}>
                  {ex}
                </option>
              ))}
            </select>
            <div className="flex-1 flex items-center justify-between text-[9px]">
              <div className="flex flex-col">
                <span className="text-slate-500">
                  {symbol}
                </span>
                <span className="font-semibold text-slate-900 dark:text-slate-50">
                  {ticker.price
                    ? ticker.price.toLocaleString(
                        "en-US",
                        { maximumFractionDigits: 2 }
                      )
                    : "--"}
                </span>
              </div>
              <div className={priceColor + " flex items-center gap-1"}>
                {ticker.change > 0 && (
                  <TrendingUp className="w-3 h-3" />
                )}
                {ticker.change < 0 && (
                  <TrendingDown className="w-3 h-3" />
                )}
                {ticker.change === 0 && (
                  <Minus className="w-3 h-3" />
                )}
                <span>
                  {ticker.change > 0 ? "+" : ""}
                  {ticker.change.toFixed(2)}%
                </span>
              </div>
            </div>
          </div>

          {/* Workspace pill */}
          {workspace && (
            <div className="mt-1 flex items-center justify-between text-[8px] text-slate-500">
              <div className="flex items-center gap-1">
                <Shield className="w-3 h-3 text-indigo-400" />
                <span>{workspace.name}</span>
              </div>
              <div className="flex items-center gap-2">
                <span>{workspace.riskText}</span>
                <span className="flex items-center gap-1">
                  <Send className="w-3 h-3" />
                  {workspace.hasTelegram
                    ? "TG linked"
                    : "TG pending"}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default NeonHeader;
