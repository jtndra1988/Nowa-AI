"use client";

import React, { useMemo } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

const glassCard = "rounded-2xl backdrop-blur-xl bg-white/8 dark:bg-white/5 border border-white/15 shadow-[0_8px_30px_rgba(0,0,0,0.25)]";
const faintText = "text-slate-400/80";

type Matrix = { labels: string[]; values: number[][] }; // symmetric, diag 1.0

function makeMockMatrix(): Matrix {
  const labels = ["BTC","ETH","SOL","AVAX","BNB","ADA"];
  const n = labels.length;
  const values: number[][] = Array.from({length:n},()=>Array(n).fill(0));
  for (let i=0;i<n;i++){
    for (let j=i;j<n;j++){
      const v = i===j ? 1 : (Math.random()*2 - 1) * 0.8; // -0.8..0.8
      values[i][j]=v; values[j][i]=v;
    }
  }
  return { labels, values };
}

function cellColor(v: number): string {
  // map -1..+1 to color
  // negatives → rose, positives → emerald, near 0 → slate
  const a = Math.min(1, Math.abs(v));
  if (v > 0) return `rgba(16,185,129,${0.15 + a*0.55})`;    // emerald
  if (v < 0) return `rgba(244,63,94,${0.15 + a*0.55})`;     // rose
  return "rgba(148,163,184,0.25)";                           // slate
}

export default function CorrelationMatrix() {
  // Later: wire to cross_asset_corre_collector.py
  const { labels, values } = useMemo(makeMockMatrix, []);
  const n = labels.length;

  return (
    <Card className={glassCard}>
      <CardHeader><CardTitle>Correlation Matrix</CardTitle></CardHeader>
      <CardContent>
        <div className="overflow-auto">
          <div className="inline-grid" style={{ gridTemplateColumns: `repeat(${n+1}, minmax(64px,auto))`, gap: 6 }}>
            {/* top-left empty */}
            <div />
            {/* column labels */}
            {labels.map((l) => (
              <div key={`col-${l}`} className="text-xs text-center opacity-80">{l}</div>
            ))}
            {/* rows */}
            {labels.map((rowLabel, i) => (
              <React.Fragment key={`row-${rowLabel}`}>
                <div className="text-xs opacity-80 flex items-center">{rowLabel}</div>
                {values[i].map((v, j) => (
                  <div
                    key={`${i}-${j}`}
                    className="rounded-md border border-white/10 flex items-center justify-center text-xs font-mono"
                    style={{ background: cellColor(v), height: 40 }}
                    title={`ρ(${rowLabel}, ${labels[j]}) = ${v.toFixed(2)}`}
                  >
                    {v.toFixed(2)}
                  </div>
                ))}
              </React.Fragment>
            ))}
          </div>
        </div>
        <div className={`mt-2 text-xs ${faintText}`}>* Demo data. Wire to cross_asset_corre_collector.py.</div>
      </CardContent>
    </Card>
  );
}
