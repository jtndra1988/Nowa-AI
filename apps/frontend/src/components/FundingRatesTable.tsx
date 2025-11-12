"use client";

import React, { useMemo } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

const glassCard = "rounded-2xl backdrop-blur-xl bg-white/8 dark:bg-white/5 border border-white/15 shadow-[0_8px_30px_rgba(0,0,0,0.25)]";
const faintText = "text-slate-400/80";

type FundingRow = {
  exchange: string;
  symbol: string;
  funding: number;     // e.g. 0.012 => 0.012%
  nextIn: string;      // time to next funding (display)
  oiUsd: number;       // open interest (for context)
  skew: number;        // positive = long bias
};

function makeMock(): FundingRow[] {
  const rows: FundingRow[] = [
    { exchange: "Binance", symbol: "BTC-PERP", funding: 0.013, nextIn: "03:12", oiUsd: 3_200_000_000, skew: 0.22 },
    { exchange: "Bybit",   symbol: "ETH-PERP", funding: 0.008, nextIn: "03:12", oiUsd: 1_450_000_000, skew: 0.05 },
    { exchange: "OKX",     symbol: "SOL-PERP", funding: -0.004, nextIn: "03:12", oiUsd: 820_000_000,  skew: -0.11 },
    { exchange: "Binance", symbol: "AVAX-PERP", funding: 0.002, nextIn: "03:12", oiUsd: 410_000_000,  skew: 0.01 },
  ];
  return rows;
}

const pct = (n: number) => `${n.toFixed(3)}%`;
const usd0 = (n: number) => n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

export default function FundingRatesTable() {
  // Later: replace with SWR/React Query to /api/funding or your collector endpoint
  const rows = useMemo(makeMock, []);

  return (
    <Card className={glassCard}>
      <CardHeader><CardTitle>Funding Rates</CardTitle></CardHeader>
      <CardContent className="overflow-auto">
        <table className="w-full text-sm">
          <thead className="text-left">
            <tr className="border-b border-white/10">
              <th className="py-2 pr-3">Exchange</th>
              <th className="py-2 pr-3">Symbol</th>
              <th className="py-2 pr-3">Funding</th>
              <th className="py-2 pr-3">Next</th>
              <th className="py-2 pr-3">Open Interest</th>
              <th className="py-2 pr-3">Skew</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const fCls = r.funding >= 0 ? "text-emerald-300" : "text-rose-300";
              const badge = r.skew > 0 ? "bg-emerald-500/20 text-emerald-300" : (r.skew < 0 ? "bg-rose-500/20 text-rose-300" : "bg-white/10");
              return (
                <tr key={i} className="border-b border-white/5">
                  <td className="py-2 pr-3">{r.exchange}</td>
                  <td className="py-2 pr-3 font-semibold">{r.symbol}</td>
                  <td className={`py-2 pr-3 font-mono ${fCls}`}>{pct(r.funding)}</td>
                  <td className="py-2 pr-3">{r.nextIn}</td>
                  <td className="py-2 pr-3">{usd0(r.oiUsd)}</td>
                  <td className="py-2 pr-3">
                    <span className={`px-2 py-1 rounded-md text-xs ${badge}`}>
                      {r.skew > 0 ? `Long +${(r.skew*100).toFixed(0)}%` : r.skew < 0 ? `Short ${(r.skew*100).toFixed(0)}%` : "Neutral"}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div className={`mt-2 text-xs ${faintText}`}>* Demo data. Wire to funding_collector.py.</div>
      </CardContent>
    </Card>
  );
}
