"use client";

import React, { useMemo } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ResponsiveContainer, AreaChart, Area, Tooltip, CartesianGrid, YAxis, XAxis } from "recharts";

const glassCard = "rounded-2xl backdrop-blur-xl bg-white/8 dark:bg-white/5 border border-white/15 shadow-[0_8px_30px_rgba(0,0,0,0.25)]";
const glassTile = "rounded-xl border border-white/15 bg-white/8 p-4";
const faintText = "text-slate-400/80";

type SeriesPoint = { t: number; v: number };
type Metric = { label: string; value: string; series: SeriesPoint[]; color: string };

function makeSeries(n = 40, base = 1000, noise = 0.15): SeriesPoint[] {
  let x = base;
  return Array.from({ length: n }, (_, i) => {
    x = x * (1 + (Math.random() - 0.5) * noise);
    return { t: i, v: Math.max(0, x) };
  });
}

export default function OnChainMetrics() {
  const metrics: Metric[] = useMemo(() => ([
    { label: "Active Addresses", value: "412k", series: makeSeries(60, 380_000, 0.08), color: "#06b6d4" },
    { label: "TX Count (24h)",  value: "1.8M", series: makeSeries(60, 1_600_000, 0.10), color: "#a78bfa" },
    { label: "Fees (24h)",      value: "$3.2M", series: makeSeries(60, 2_600_000, 0.12), color: "#f59e0b" },
    { label: "TVL",             value: "$8.9B", series: makeSeries(60, 7_500_000_000, 0.05), color: "#10b981" },
  ]), []);

  return (
    <Card className={glassCard}>
      <CardHeader><CardTitle>On-Chain Metrics</CardTitle></CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          {metrics.map((m, i) => (
            <div key={i} className={glassTile}>
              <div className="flex items-baseline justify-between">
                <div className="text-sm">{m.label}</div>
                <div className="text-xl font-bold">{m.value}</div>
              </div>
              <div className="h-24 mt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={m.series} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
                    <defs>
                      <linearGradient id={`oc-${i}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={m.color} stopOpacity={0.35}/>
                        <stop offset="95%" stopColor={m.color} stopOpacity={0.04}/>
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="t" hide />
                    <YAxis hide />
                    <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
                    <Tooltip contentStyle={{ background: "rgba(17,24,39,0.7)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 12 }}/>
                    <Area type="monotone" dataKey="v" stroke={m.color} fill={`url(#oc-${i})`} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          ))}
        </div>
        <div className={`mt-2 text-xs ${faintText}`}>* Demo tiles. Wire to onchain_collector.py.</div>
      </CardContent>
    </Card>
  );
}
