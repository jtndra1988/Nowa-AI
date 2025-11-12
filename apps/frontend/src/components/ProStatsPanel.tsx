"use client";

import React, { useMemo } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

type Stat = { label: string; value: string; hint?: string };

// helper: pull numeric value out of strings like "2.41%", "Top 5%", "38 m"
function numeric(value: string): number {
  const n = parseFloat(value.replace(/[^0-9.-]/g, ""));
  return isNaN(n) ? 0 : n;
}

function statColor(label: string, valueStr: string): string {
  const value = numeric(valueStr);

  if (label.includes("ATR")) {
    return "from-amber-400/20 to-orange-500/10";
  }
  if (label.includes("Skew")) {
    return value < 0
      ? "from-sky-500/20 to-blue-500/10"
      : "from-rose-400/20 to-pink-500/10";
  }
  if (label.includes("Kurtosis")) {
    return "from-purple-400/20 to-indigo-500/10";
  }
  if (label.includes("Max Adverse")) {
    return "from-rose-400/20 to-red-500/10";
  }
  if (label.includes("Max Favorable")) {
    return "from-emerald-400/20 to-green-500/10";
  }
  if (label.includes("Avg Trade Duration")) {
    return "from-sky-400/15 to-indigo-500/10";
  }
  if (label.includes("Regime")) {
    return "from-cyan-400/20 to-sky-500/10";
  }
  if (label.includes("Liquidity")) {
    return "from-fuchsia-400/20 to-pink-500/10";
  }
  return "from-white/5 to-white/5";
}

export default function ProStatsPanel() {
  const glassCard =
    "rounded-2xl backdrop-blur-xl bg-white/8 dark:bg-white/5 border border-white/15 shadow-[0_8px_30px_rgba(0,0,0,0.25)]";

  const stats: Stat[] = useMemo(
    () => [
      { label: "ATR(14)", value: "2.41%" },
      { label: "Skew (30d)", value: "-0.12" },
      { label: "Kurtosis (30d)", value: "3.8" },
      { label: "Max Adverse Excursion", value: "1.9%" },
      { label: "Max Favorable Excursion", value: "4.2%" },
      { label: "Avg Trade Duration", value: "38 m" },
      { label: "Regime", value: "High-Vol Trend" },
      { label: "Liquidity Rank", value: "Top 5%" },
    ],
    []
  );

  return (
    <Card className={glassCard}>
      <CardHeader className="pb-3">
        <CardTitle className="text-base font-semibold">
          Advanced Statistics
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {stats.map((s) => (
            <div
              key={s.label}
              className={`
                rounded-xl p-3 backdrop-blur-xl border border-white/10
                bg-gradient-to-br ${statColor(s.label, s.value)}
                transition-all duration-300 hover:scale-[1.02]
              `}
            >
              <div className="text-[11px] opacity-70">{s.label}</div>
              <div className="text-lg font-semibold">{s.value}</div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
