// src/components/LiquidationsCard.tsx
"use client";

import React, { useMemo } from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "@/components/ui/card";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip as RTooltip,
  CartesianGrid,
  Legend,
} from "recharts";

/** ========= Demo-first Liquidations Card (Glass Theme) =========
 * - Matches MarsBotUI glass cards
 * - Transparent chart backgrounds
 * - Soft grid/ticks for dark theme
 * - Dummy data for client demo
 */

type LiqPoint = {
  t: number;
  longUsd: number;
  shortUsd: number;
  totalUsd: number;
};

type RecentLiq = {
  time: string;
  symbol: string;
  side: "LONG" | "SHORT";
  usd: number;
  venue: string;
};

export interface LiquidationsCardProps {
  symbol?: string;
  window?: "24h" | "7d";
  bucket?: "1h" | "30m";
}

/* ---------- Glass tokens (match your dashboard) ---------- */
const glassPanel =
  "backdrop-blur-xl bg-white/8 dark:bg-white/5 border border-white/15 shadow-[0_8px_30px_rgb(0,0,0,0.25)]";
const softText = "text-slate-300";
const faintText = "text-slate-400/70";

/* ---------- Format helpers ---------- */
const fmtUSD0 = (n: number) =>
  n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

/* ---------- Demo generator ---------- */
function makeMock(window: "24h" | "7d", bucket: "1h" | "30m"): { series: LiqPoint[]; recent: RecentLiq[] } {
  const buckets = window === "7d" ? (bucket === "1h" ? 7 * 24 : 7 * 48) : (bucket === "1h" ? 24 : 48);

  let base = 220_000;
  const series: LiqPoint[] = Array.from({ length: buckets }, (_, i) => {
    const noise = 0.95 + Math.random() * 0.12;
    base = Math.max(80_000, base * noise);

    const spikeChance = Math.random();
    const spike = spikeChance > 0.93 ? Math.random() * 900_000 : 0;

    const longUsd = Math.max(0, base * (0.35 + Math.random() * 0.5));
    const shortUsd = Math.max(0, base * (0.25 + Math.random() * 0.45));
    const totalUsd = longUsd + shortUsd + spike;

    return { t: i, longUsd, shortUsd, totalUsd };
  });

  const assets = ["BTC-PERP", "ETH-PERP", "SOL-PERP", "AVAX-PERP"];
  const venues = ["Binance", "Bybit", "OKX"];
  const now = Date.now();
  const stepMs = bucket === "1h" ? 3_600_000 : 1_800_000;

  const recent: RecentLiq[] = Array.from({ length: 8 }, () => {
    const side: RecentLiq["side"] = Math.random() > 0.5 ? "LONG" : "SHORT";
    return {
      time: new Date(now - Math.floor(Math.random() * 8) * stepMs).toISOString(),
      symbol: assets[Math.floor(Math.random() * assets.length)],
      side,
      usd: Math.floor(35_000 + Math.random() * 1_100_000),
      venue: venues[Math.floor(Math.random() * venues.length)],
    };
  }).sort((a, b) => (a.time < b.time ? 1 : -1));

  return { series, recent };
}

export default function LiquidationsCard({
  symbol = "BTC-PERP",
  window = "24h",
  bucket = "1h",
}: LiquidationsCardProps) {
  const { series, recent } = useMemo(() => makeMock(window, bucket), [window, bucket]);

  const longSum = series.reduce((s, p) => s + p.longUsd, 0);
  const shortSum = series.reduce((s, p) => s + p.shortUsd, 0);
  const totalSum = series.reduce((s, p) => s + p.totalUsd, 0);
  const maxSpike = Math.max(...series.map((p) => p.totalUsd));
  const longPct = totalSum ? longSum / totalSum : 0;
  const shortPct = totalSum ? shortSum / totalSum : 0;

  // Recharts styling for dark+glass
  const axisStroke = "rgba(226,232,240,0.55)"; // slate-200/55
  const gridStroke = "rgba(148,163,184,0.20)"; // slate-400/20
  const areaStroke = "#f59e0b";               // amber-500
  const barLong = "#10b981";                  // emerald-500
  const barShort = "#ef4444";                 // rose-500

  return (
    <Card className={glassPanel}>
      <CardHeader>
        <CardTitle className={softText}>Liquidations — {symbol} ({window})</CardTitle>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* KPIs */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <div className={`text-xs ${faintText}`}>Total (window)</div>
            <div className="text-2xl font-bold">{fmtUSD0(totalSum)}</div>
          </div>
          <div>
            <div className={`text-xs ${faintText}`}>Long Liq</div>
            <div className="text-2xl font-bold">{fmtUSD0(longSum)}</div>
            <div className={`text-xs ${faintText}`}>{(longPct * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div className={`text-xs ${faintText}`}>Short Liq</div>
            <div className="text-2xl font-bold">{fmtUSD0(shortSum)}</div>
            <div className={`text-xs ${faintText}`}>{(shortPct * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div className={`text-xs ${faintText}`}>Max Spike (bucket)</div>
            <div className="text-2xl font-bold">{fmtUSD0(maxSpike)}</div>
          </div>
        </div>

        {/* Charts row (transparent backgrounds) */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* Total liquidations area */}
          <div className="h-64 bg-transparent">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={series} style={{ background: "transparent" }} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
                <defs>
                  <linearGradient id="liqFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={areaStroke} stopOpacity={0.35} />
                    <stop offset="95%" stopColor={areaStroke} stopOpacity={0.04} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="t" tickLine={false} axisLine={false} stroke={axisStroke} tick={{ fill: axisStroke }} />
                <YAxis
                  stroke={axisStroke}
                  tick={{ fill: axisStroke }}
                  tickFormatter={(v) => (v >= 1_000_000 ? `${(v / 1_000_000).toFixed(1)}M` : `${(v / 1_000).toFixed(0)}k`)}
                />
                <CartesianGrid stroke={gridStroke} strokeDasharray="3 3" />
                <RTooltip
                  formatter={(v: number) => fmtUSD0(v)}
                  labelFormatter={(l) => `Bucket ${l}`}
                  contentStyle={{ background: "rgba(17,24,39,0.7)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 12 }}
                />
                <Area type="monotone" dataKey="totalUsd" stroke={areaStroke} fill="url(#liqFill)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Stacked long vs short */}
          <div className="h-64 bg-transparent">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={series} style={{ background: "transparent" }} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
                <XAxis dataKey="t" tickLine={false} axisLine={false} stroke={axisStroke} tick={{ fill: axisStroke }} />
                <YAxis
                  stroke={axisStroke}
                  tick={{ fill: axisStroke }}
                  tickFormatter={(v) => (v >= 1_000_000 ? `${(v / 1_000_000).toFixed(1)}M` : `${(v / 1_000).toFixed(0)}k`)}
                />
                <CartesianGrid stroke={gridStroke} strokeDasharray="3 3" />
                <Legend wrapperStyle={{ color: axisStroke }} />
                <RTooltip
                  formatter={(v: number) => fmtUSD0(v)}
                  labelFormatter={(l) => `Bucket ${l}`}
                  contentStyle={{ background: "rgba(17,24,39,0.7)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 12 }}
                />
                <Bar dataKey="longUsd" stackId="a" fill={barLong} />
                <Bar dataKey="shortUsd" stackId="a" fill={barShort} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Recent liquidation prints */}
        <div className="overflow-auto">
          <table className={`w-full text-sm ${softText}`}>
            <thead className="text-left">
              <tr className="border-b border-white/10">
                <th className="py-2 pr-3">Time</th>
                <th className="py-2 pr-3">Symbol</th>
                <th className="py-2 pr-3">Side</th>
                <th className="py-2 pr-3">Amount (USD)</th>
                <th className="py-2 pr-3">Venue</th>
              </tr>
            </thead>
            <tbody>
              {recent.map((r, i) => {
                const sideCls = r.side === "LONG" ? "text-emerald-400" : "text-rose-400";
                return (
                  <tr key={i} className="border-b border-white/5">
                    <td className="py-2 pr-3">{new Date(r.time).toLocaleTimeString()}</td>
                    <td className="py-2 pr-3 font-semibold">{r.symbol}</td>
                    <td className={`py-2 pr-3 font-semibold ${sideCls}`}>{r.side}</td>
                    <td className="py-2 pr-3 font-mono">{fmtUSD0(r.usd)}</td>
                    <td className="py-2 pr-3">{r.venue}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        
      </CardContent>
    </Card>
  );
}
