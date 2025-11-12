"use client";

import React, { useEffect, useState } from "react";
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis } from "recharts";

type Tape = { t: string; side: "BUY"|"SELL"; px: number; sz: number };
type CvdPt = { i: number; cvd: number };

export default function OrderFlowStrip() {
  const [bidsPct, setBidsPct] = useState(0.62); // 0..1
  const [cvd, setCvd] = useState<CvdPt[]>([{ i: 0, cvd: 0 }]);
  const [tape, setTape] = useState<Tape[]>([]);

  useEffect(() => {
    const id = setInterval(() => {
      const buy = Math.random() > 0.5;
      const size = +(Math.random() * 2 + 0.2).toFixed(2);
      setBidsPct(p => {
        const next = Math.min(0.95, Math.max(0.05, p + (buy ? 0.015 : -0.015) + (Math.random() - 0.5) * 0.02));
        return +next.toFixed(3);
      });
      setCvd(prev => {
        const last = prev.at(-1)!.cvd;
        const next = { i: prev.at(-1)!.i + 1, cvd: +(last + (buy ? size : -size)).toFixed(2) };
        return [...prev.slice(-39), next];
      });
      setTape(prev => {
        const row: Tape = { t: new Date().toLocaleTimeString(), side: buy?"BUY":"SELL", px: +(100000 + (Math.random()-0.5)*80).toFixed(2), sz: size };
        return [row, ...prev].slice(0, 6);
      });
    }, 1000);
    return () => clearInterval(id);
  }, []);

  const asksPct = 1 - bidsPct;

  return (
    <div className="space-y-4">
      {/* animated bids/asks pill */}
      <div className="w-full h-8 bg-white/10 rounded-full overflow-hidden border border-white/15 relative">
        <div
          className="h-full bg-emerald-500/70 transition-all duration-500"
          style={{ width: `${bidsPct * 100}%` }}
        />
        <div className="absolute inset-0 flex justify-between items-center px-3 text-xs font-semibold">
          <span className="text-emerald-300">BIDS {(bidsPct*100).toFixed(0)}%</span>
          <span className="text-rose-300">ASKS {(asksPct*100).toFixed(0)}%</span>
        </div>
      </div>

      {/* micro CVD sparkline */}
      <div className="h-20 rounded-xl border border-white/15 bg-white/5 p-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={cvd}>
            <XAxis dataKey="i" hide /><YAxis hide />
            <Area type="monotone" dataKey="cvd" stroke="#10b981" fill="rgba(16,185,129,0.18)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* tiny tape */}
      <div className="rounded-xl border border-white/15 bg-white/5">
        <table className="w-full text-xs">
          <thead className="opacity-70">
            <tr>
              <th className="px-3 py-2 text-left">Time</th>
              <th className="px-3 py-2 text-left">Price</th>
              <th className="px-3 py-2 text-left">Size</th>
              <th className="px-3 py-2 text-left">Side</th>
            </tr>
          </thead>
          <tbody>
            {tape.map((r, i) => (
              <tr key={i} className="border-t border-white/10">
                <td className="px-3 py-1.5">{r.t}</td>
                <td className="px-3 py-1.5">{r.px.toLocaleString()}</td>
                <td className="px-3 py-1.5">{r.sz}</td>
                <td className={`px-3 py-1.5 font-semibold ${r.side==="BUY" ? "text-emerald-400":"text-rose-400"}`}>{r.side}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
