"use client";

import React, { useMemo } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ResponsiveContainer, ComposedChart, Area, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from "recharts";

const glassCard = "rounded-2xl backdrop-blur-xl bg-white/8 dark:bg-white/5 border border-white/15 shadow-[0_8px_30px_rgba(0,0,0,0.25)]";
const faintText = "text-slate-400/80";

type DevPoint = { day: string; commits: number; prs: number; issues: number };

function makeDevSeries(days = 30): DevPoint[] {
  const out: DevPoint[] = [];
  for (let i = 0; i < days; i++) {
    out.push({
      day: `${i+1}`,
      commits: Math.floor(10 + Math.random() * 40),
      prs: Math.floor(2 + Math.random() * 10),
      issues: Math.floor(1 + Math.random() * 8),
    });
  }
  return out;
}

export default function DeveloperActivity() {
  // Later: wire to github_collector.py aggregator
  const data = useMemo(() => makeDevSeries(30), []);
  return (
    <Card className={glassCard}>
      <CardHeader><CardTitle>Developer Activity (GitHub)</CardTitle></CardHeader>
      <CardContent className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="day" />
            <YAxis />
            <Tooltip contentStyle={{ background: "rgba(17,24,39,0.7)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 12 }}/>
            <Legend />
            <Area type="monotone" dataKey="commits" stroke="#06b6d4" fill="rgba(6,182,212,0.25)" />
            <Bar dataKey="prs" stackId="a" fill="#f59e0b" />
            <Bar dataKey="issues" stackId="a" fill="#a78bfa" />
          </ComposedChart>
        </ResponsiveContainer>
        <div className={`mt-2 text-xs ${faintText}`}>* Demo data. Wire to github_collector.py.</div>
      </CardContent>
    </Card>
  );
}
