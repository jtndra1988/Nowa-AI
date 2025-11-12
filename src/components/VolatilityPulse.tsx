"use client";

import React, { useEffect, useMemo, useState } from "react";
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

type Pt = { t: number; iv: number; rv: number; hv: number };

function seed(n = 50) {
  let iv = 34, rv = 28, hv = 31;
  return Array.from({ length: n }, (_, i) => {
    iv = Math.max(8, iv + (Math.random() - 0.5) * 1.8);
    rv = Math.max(6, rv + (Math.random() - 0.5) * 1.2);
    hv = Math.max(7, hv + (Math.random() - 0.5) * 1.4);
    return { t: i, iv: +iv.toFixed(2), rv: +rv.toFixed(2), hv: +hv.toFixed(2) };
  });
}

const Badge = ({
  label,
  value,
  color,
}: {
  label: string
  value: number
  color: string
}) => (
  <div className="flex-1 min-w-[0] px-3 py-1.5 rounded-xl border border-white/10 bg-white/5 flex items-baseline gap-1">
    <span className="text-[10px] sm:text-xs opacity-70 truncate">
      {label}
    </span>
    <span
      className="text-sm sm:text-base font-semibold leading-none"
      style={{ color }}
    >
      {value.toFixed(2)}%
    </span>
  </div>
)


export default function VolatilityPulse({ start=0.26 }:{ start?: number }) {
  const [series, setSeries] = useState<Pt[]>(() => seed());
  const [idx, setIdx] = useState(start); // 0..1 (your Volatility Index normalized)

  useEffect(() => {
    const id = setInterval(() => {
      setSeries(prev => {
        const last = prev.at(-1)!;
        const next = {
          t: last.t + 1,
          iv: Math.max(8, last.iv + (Math.random() - 0.5) * 1.8),
          rv: Math.max(6, last.rv + (Math.random() - 0.5) * 1.2),
          hv: Math.max(7, last.hv + (Math.random() - 0.5) * 1.4),
        };
        return [...prev.slice(-49), next].map(p => ({ ...p, iv:+p.iv.toFixed(2), rv:+p.rv.toFixed(2), hv:+p.hv.toFixed(2) }));
      });
      setIdx(v => Math.min(1, Math.max(0, v + (Math.random() - 0.5) * 0.02)));
    }, 1400);
    return () => clearInterval(id);
  }, []);

  const iv = series.at(-1)?.iv ?? 0;
  const rv = series.at(-1)?.rv ?? 0;
  const hv = series.at(-1)?.hv ?? 0;

  const barPct = Math.round(idx * 100);

  return (
    <div className="space-y-4">
      {/* live badges */}
      <div className="flex flex-wrap gap-2">
        <Badge label="IV(30d)" color="#f59e0b" value={iv} />
        <Badge label="Realized(30d)" color="#06b6d4" value={rv} />
        <Badge label="Historical(1y)" color="#a78bfa" value={hv} />
      </div>

      {/* animated index bar */}
      <div className="rounded-xl border border-white/15 bg-white/5 p-3">
        <div className="flex items-center justify-between text-xs opacity-80 mb-2">
          <span>Volatility Index</span><span>{barPct}%</span>
        </div>
        <div className="h-3 rounded-full bg-white/10 overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{
              width: `${barPct}%`,
              background: `linear-gradient(90deg,#10b981,#f59e0b)`
            }}
          />
        </div>
      </div>

      {/* sparkline */}
      <div className="h-28">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={series} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="t" hide /><YAxis hide />
            <Tooltip contentStyle={{ background:"rgba(17,24,39,0.7)", border:"1px solid rgba(255,255,255,0.12)", borderRadius:12 }}/>
            <defs>
              <linearGradient id="ivg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#f59e0b" stopOpacity={0.35}/><stop offset="95%" stopColor="#f59e0b" stopOpacity={0.05}/></linearGradient>
              <linearGradient id="rvg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#06b6d4" stopOpacity={0.35}/><stop offset="95%" stopColor="#06b6d4" stopOpacity={0.05}/></linearGradient>
              <linearGradient id="hvg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#a78bfa" stopOpacity={0.35}/><stop offset="95%" stopColor="#a78bfa" stopOpacity={0.05}/></linearGradient>
            </defs>
            <Area type="monotone" dataKey="iv" stroke="#f59e0b" fill="url(#ivg)" />
            <Area type="monotone" dataKey="rv" stroke="#06b6d4" fill="url(#rvg)" />
            <Area type="monotone" dataKey="hv" stroke="#a78bfa" fill="url(#hvg)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
