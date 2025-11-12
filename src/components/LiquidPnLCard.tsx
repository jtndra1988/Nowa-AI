"use client";
import React from "react";

// 1. Define the props based on your MarsBotUI.tsx file
type Props = {
  side: "LONG" | "SHORT";
  size: number;
  riskPct: number;
  unrealizedPnl: number;
  entryPrice: number;
  markPrice: number;
};

// Helper style variables (copied from MarsBotUI.tsx)
const glassCard = "backdrop-blur-xl bg-white/8 dark:bg-white/5 border border-white/15 shadow-[0_8px_30px_rgb(0,0,0,0.25)] rounded-2xl";
const glassHover = "transition-transform duration-300 hover:-translate-y-0.5 hover:shadow-[0_12px_40px_rgba(0,0,0,0.35)]";
const faintText = "text-slate-500/80 dark:text-slate-400/80";

// 2. Create the component
export default function LiquidPnLCard({
  side,
  size,
  riskPct,
  unrealizedPnl,
  entryPrice,
  markPrice,
}: Props) {
  const isProfit = unrealizedPnl >= 0;
  const pnlColor = isProfit ? "text-emerald-400" : "text-rose-400";
  const sideColor = side === "LONG" ? "text-emerald-400" : "text-rose-400";

  return (
    // Use the Card styles directly on the root div
    <div className={`${glassCard} ${glassHover} p-4`}>
      {/* CardHeader equivalent */}
      <div className="pb-3 border-b border-white/10">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Current Position</h3>
      </div>
      
      {/* CardContent equivalent */}
      <div className="pt-4">
        {/* Top row: Side, Size, Risk */}
        <div className="flex justify-between items-start mb-4">
          <div>
            <div className={`text-xs ${faintText}`}>Side</div>
            <div className={`text-xl font-bold ${sideColor}`}>{side}</div>
          </div>
          <div>
            <div className={`text-xs ${faintText} text-right`}>Size</div>
            <div className="text-xl font-bold text-right">{size.toFixed(2)} ₿</div>
          </div>
          <div>
            <div className={`text-xs ${faintText} text-right`}>Risk</div>
            <div className="text-xl font-bold text-right">{riskPct.toFixed(0)}%</div>
          </div>
        </div>

        {/* PnL */}
        <div>
          <div className={`text-xs ${faintText}`}>Unrealized P&L</div>
          <div className={`text-3xl font-bold ${pnlColor} mb-3`}>
            {isProfit ? "+" : ""}${unrealizedPnl.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
        </div>

        {/* Entry / Mark */}
        <div className="flex justify-between text-sm">
          <div className={`${faintText}`}>Entry Price</div>
          <div>{entryPrice.toLocaleString()}</div>
        </div>
        <div className="flex justify-between text-sm">
          <div className={`${faintText}`}>Mark Price</div>
          <div>{markPrice.toLocaleString()}</div>
        </div>
      </div>
    </div>
  );
}