// src/components/NeonHeader.tsx
"use client";

import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
  FormEvent,
} from "react";
import {
  Menu,
  Search,
  TrendingUp,
  TrendingDown,
  Minus,
} from "lucide-react";
import API, { MarketMode, SymbolCode } from "@/lib/api";
import {
  LineChart,
  Line,
  ResponsiveContainer,
} from "recharts";
import { getIconSrcForSymbol } from "@/lib/cryptoIcons";

type Props = {
  symbol: SymbolCode;
  onSetSymbol: (s: SymbolCode) => void;
  activeTab: string;
  onTabChange: (id: string) => void;
  isMobile?: boolean;
  onToggleSidebar?: () => void;
};

const DEFAULT_MODE: MarketMode = "futures";

const navItems = [
  { id: "dashboard", label: "Dashboard" },
  { id: "strategies", label: "Strategies" },
  { id: "intel", label: "Market Intel" },
  { id: "system", label: "System" },
] as const;

const NeonHeader: React.FC<Props> = ({
  symbol,
  onSetSymbol,
  activeTab,
  onTabChange,
  isMobile = false,
  onToggleSidebar,
}) => {
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
      const id = setTimeout(() => setFlash("none"), 260);
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
        API.formatSymbol(a, DEFAULT_MODE)
      ) as SymbolCode[],
    []
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
      ? "text-emerald-400"
      : ticker.change < 0
        ? "text-rose-400"
        : "text-slate-400";

  const sparkColor =
    ticker.change >= 0 ? "#22c55e" : "#fb7185";

  const desktopWrapper =
    "hidden md:flex w-full items-center gap-4 rounded-[26px] px-5 py-2.5 " +
    "border border-slate-800/70 bg-gradient-to-br from-slate-950/95 via-slate-950/92 to-slate-900/88 " +
    "shadow-[0_16px_50px_rgba(15,23,42,0.9)] backdrop-blur-2xl";

  const searchWrapperDesktop =
    "relative w-[80%] min-w-[340px] h-10 rounded-2xl flex items-center px-4 border transition-all " +
    "bg-slate-950/80 border-slate-800/80 hover:bg-slate-900/80 shadow-inner";


  const searchInputText =
    "w-full bg-transparent text-xs md:text-sm outline-none " +
    "text-slate-100 placeholder:text-slate-500";

  const suggestionListDesktop =
    "absolute left-0 right-0 top-[112%] z-50 max-h-64 overflow-y-auto rounded-2xl border shadow-2xl backdrop-blur-xl text-[10px] " +
    "bg-slate-950/98 border-slate-800";

  const priceBlock =
    "flex h-9 items-center gap-3 rounded-2xl px-4 border " +
    "bg-slate-950/85 border-slate-800/80 shadow-[0_0_0_1px_rgba(15,23,42,0.9)] " +
    "relative overflow-hidden";

  const mobileWrapper = "md:hidden relative mt-2";
  const mobileInner =
    "relative flex flex-col gap-2 rounded-3xl border px-3.5 py-2.5 shadow-[0_18px_40px_rgba(15,23,42,0.9)] " +
    "border-slate-900/80 bg-gradient-to-br from-slate-950/98 via-slate-950/96 to-slate-900/92 backdrop-blur-xl";

  const mobileSearchWrapper =
    "flex items-center h-9 rounded-2xl px-3 border " +
    "bg-slate-950/90 border-slate-800 shadow-inner";

  const mobileSearchInput =
    "w-full bg-transparent text-[11px] outline-none " +
    "text-slate-100 placeholder:text-slate-500";

  const suggestionListMobile =
    "absolute left-0 right-0 top-[105%] z-50 max-h-56 overflow-y-auto rounded-2xl border shadow-2xl backdrop-blur-xl text-[9px] " +
    "bg-slate-950/98 border-slate-800";

  const suggestionButton =
    "flex w-full items-center justify-between px-3 py-1.5 transition-all hover:bg-emerald-500/10";

  // ---------- RENDER ----------
  return (
    <>
      {/* DESKTOP / TABLET */}
      <div className={desktopWrapper}>
        {/* LEFT CLUSTER: NOWA + AI Pulse + SEARCH */}
        <div className="flex items-center gap-3 flex-[2] justify-end min-w-0">

          {/* NOWA logo + AI Pulse */}
          <div className="flex items-center gap-4 rounded-2xl px-5 py-3 bg-slate-950/95 border border-emerald-500/60 shadow-[0_0_42px_rgba(16,185,129,0.95)] select-none min-w-[220px]">

            <div className="relative">
              {/* pulse ring */}
              <span className="absolute inset-0 rounded-full bg-emerald-500/40 blur-md opacity-70 animate-ping" />
              <div className="relative flex items-center justify-center w-11 h-11 rounded-full bg-slate-950 border border-emerald-400/70 shadow-[0_0_22px_rgba(16,185,129,0.9)]">
                <img
                  src="/nowa.png"
                  alt="Nowa"
                  className="h-7 w-auto object-contain drop-shadow-[0_0_32px_rgba(16,185,129,1)]"
                />
              </div>

            </div>
            <div className="flex flex-col leading-tight">
              <span className="text-[11px] font-semibold tracking-wide text-slate-100">
                NOWA
              </span>
              <span className="flex items-center gap-1 text-[10px] text-emerald-300/95">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.9)]" />
                AI Pulse • Live
              </span>
            </div>
          </div>

          {/* Search */}
          <form
            onSubmit={handleSubmit}
            className={searchWrapperDesktop}
          >
            <Search className="w-4 h-4 text-slate-500 mr-2" />
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Type cryptocurrency"
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
                        <span className="text-slate-100">
                          {s}
                        </span>
                      </div>
                      <span className="text-slate-500">
                        {price}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
                  {/* AI Pulse line under search bar */}
      <div className="pointer-events-none absolute left-3 right-3 -bottom-[2px] h-[2px] overflow-hidden">
        <div className="pulse-line w-[140%] h-full bg-gradient-to-r from-emerald-400/0 via-emerald-400/80 to-emerald-400/0" />
      </div>

          </form>
        </div>

        {/* RIGHT CLUSTER: NAV + TICKER */}
        <div className="flex items-center gap-3 flex-[2] justify-end min-w-0">
          {/* Tabs */}
          <nav className="flex items-center gap-1.5 rounded-2xl px-1.5 py-0.5 bg-slate-950/70 border border-slate-800/80 shadow-inner">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onTabChange(item.id)}
                  className={
                    "px-3 py-1.5 rounded-2xl text-xs md:text-[13px] font-medium transition " +
                    (isActive
                      ? "bg-emerald-500 text-slate-950 shadow-[0_0_22px_rgba(16,185,129,0.85)]"
                      : "text-slate-200 hover:text-slate-50 hover:bg-slate-800/80")
                  }
                >
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Price + sparkline */}
          <div
            className={
              priceBlock +
              (flash === "up"
                ? " ring-1 ring-emerald-400 shadow-[0_0_30px_rgba(16,185,129,0.9)]"
                : flash === "down"
                  ? " ring-1 ring-red-400 shadow-[0_0_30px_rgba(248,113,113,0.9)]"
                  : "")
            }
          >
            {/* soft gradient glow */}
            <div className="pointer-events-none absolute inset-0 opacity-40">
              <div className="w-full h-full bg-gradient-to-r from-emerald-500/15 via-transparent to-rose-500/15" />
            </div>

            <div className="relative flex flex-col leading-tight">
              <div className="text-[10px] text-slate-400">
                {symbol}
              </div>
              <div className="text-sm font-semibold text-slate-50">
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
                "relative flex items-center gap-1 text-[10px] font-medium " +
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
            <div className="relative w-20 h-8">
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
        </div>
      </div>

      {/* MOBILE */}
      <div className={mobileWrapper}>
        <div className={mobileInner}>
          {/* row 1: menu + logo + search */}
          <div className="flex items-center gap-2">
            {isMobile && onToggleSidebar && (
              <button
                onClick={onToggleSidebar}
                className="p-1.5 rounded-xl bg-slate-950/90 border border-slate-800 text-slate-100"
              >
                <Menu className="w-4 h-4" />
              </button>
            )}
            <div className="flex items-center gap-2 rounded-2xl px-3 py-1.5 bg-slate-950/95 border border-emerald-500/60">
              <div className="relative">
                <span className="absolute inset-0 rounded-full bg-emerald-500/40 blur-md opacity-70 animate-ping" />
                <div className="relative flex items-center justify-center w-7 h-7 rounded-full bg-slate-950 border border-emerald-400/70">
                  <img
                    src="/nowa.png"
                    alt="Nowa"
                    className="h-4 w-auto object-contain"
                  />
                </div>
              </div>
              <span className="text-[11px] font-semibold text-slate-100">
                NOWA
              </span>
            </div>
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
                          <span className="text-slate-100">
                            {s}
                          </span>
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

          {/* row 2: tabs */}
          <div className="flex flex-wrap gap-1 mt-1">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onTabChange(item.id)}
                  className={
                    "whitespace-nowrap rounded-2xl px-3 py-1 text-[10px] font-medium border transition-all " +
                    (isActive
                      ? "bg-emerald-500 text-slate-950 border-emerald-500 shadow-[0_0_16px_rgba(16,185,129,0.85)]"
                      : "bg-slate-950/80 text-slate-200 border-slate-800 hover:bg-slate-900/90")
                  }
                >
                  {item.label}
                </button>
              );
            })}
          </div>

          {/* row 3: compact price */}
          <div className="mt-1 flex items-center justify-between text-[9px]">
            <div className="flex flex-col">
              <span className="text-slate-500">
                {symbol}
              </span>
              <span className="font-semibold text-slate-100">
                {ticker.price
                  ? ticker.price.toLocaleString("en-US", {
                    maximumFractionDigits: 2,
                  })
                  : "--"}
              </span>
            </div>
            <div
              className={
                priceColor + " flex items-center gap-1"
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
          </div>
        </div>
      </div>
    </>
  );
};

export default NeonHeader;
