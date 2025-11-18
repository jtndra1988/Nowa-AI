// "use client";

// import React, { useEffect, useMemo, useState } from "react";
// import {
//   Card,
//   CardHeader,
//   CardTitle,
//   CardDescription,
//   CardContent,
//   KpiTile,
//   Badge,
//   glassPanel,
//   makeSeeded,
//   faintText,
// } from "../layout/AppShell";
// import type { SymbolCode, MarketMode } from "@/lib/api";
// import {
//   ResponsiveContainer,
//   LineChart,
//   Line,
//   XAxis,
//   YAxis,
//   Tooltip as RTooltip,
//   PieChart,
//   Pie,
//   Cell,
//   BarChart,
//   Bar,
// } from "recharts";
// import dynamic from "next/dynamic";

// const LiquidationsCard = dynamic(
//   () => import("../LiquidationsCard"),
//   { ssr: false }
// );

// type Props = {
//   symbol: SymbolCode;
//   mode: MarketMode;
//   exchange: string;
// };

// type EquityPoint = { t: number; v: number };

// type Pos = {
//   id: string;
//   symbol: string;
//   side: "LONG" | "SHORT";
//   size: number;
//   entry: number;
//   mark: number;
//   upnl: number;
//   rr: number;
//   lev: number;
// };

// const COLORS = ["#6366f1", "#22c55e", "#38bdf8", "#f97316"];

// const fmtUSD0 = (n: number) =>
//   n.toLocaleString("en-US", {
//     style: "currency",
//     currency: "USD",
//     maximumFractionDigits: 0,
//   });

// export const PortfolioTab: React.FC<Props> = ({
//   symbol,
//   mode,
//   exchange,
// }) => {
//   const seeded = useMemo(
//     () => makeSeeded(`portfolio-${symbol}-${mode}-${exchange}`),
//     [symbol, mode, exchange]
//   );

//   // ========== Equity curve (live-ish) ==========

//   const [equity, setEquity] = useState<EquityPoint[]>(() => {
//     let base = 118_000;
//     return Array.from({ length: 90 }, (_, i) => {
//       base = base * (1 + (seeded() - 0.48) * 0.003);
//       return { t: i + 1, v: Math.max(90_000, base) };
//     });
//   });

//   useEffect(() => {
//     const id = setInterval(() => {
//       setEquity(prev => {
//         if (!prev.length) return prev;
//         const last = prev[prev.length - 1];
//         const bump = 1 + (seeded() - 0.5) * 0.0025;
//         const nextV = Math.max(90_000, last.v * bump);
//         const next: EquityPoint = { t: last.t + 1, v: nextV };
//         const windowSize = 90;
//         return [...prev.slice(-windowSize + 1), next];
//       });
//     }, 2500);
//     return () => clearInterval(id);
//   }, [seeded]);

//   const latestEquity = equity.at(-1)?.v ?? 118_000;

//   // ========== Allocations (follow equity) ==========

//   const allocations = useMemo(
//     () => [
//       { asset: "BTC", value: latestEquity * 0.58 },
//       { asset: "ETH", value: latestEquity * 0.28 },
//       { asset: "USDC", value: latestEquity * 0.14 },
//     ],
//     [latestEquity]
//   );

//   // ========== Positions (marks + uPnL drift) ==========

//   const [positions, setPositions] = useState<Pos[]>(() => [
//     {
//       id: "p1",
//       symbol: "BTC-PERP",
//       side: "LONG",
//       size: 0.95,
//       entry: 100_800,
//       mark: 102_350,
//       upnl: 1473,
//       rr: 1.9,
//       lev: 2,
//     },
//     {
//       id: "p2",
//       symbol: "ETH-PERP",
//       side: "LONG",
//       size: 7.2,
//       entry: 3480,
//       mark: 3565,
//       upnl: 612,
//       rr: 1.4,
//       lev: 2,
//     },
//     {
//       id: "p3",
//       symbol: "SOL-PERP",
//       side: "SHORT",
//       size: 120,
//       entry: 142.5,
//       mark: 139.4,
//       upnl: 372,
//       rr: 1.2,
//       lev: 3,
//     },
//     {
//       id: "p4",
//       symbol: "AVAX-PERP",
//       side: "LONG",
//       size: 210,
//       entry: 35.8,
//       mark: 34.9,
//       upnl: -189,
//       rr: -0.6,
//       lev: 2,
//     },
//   ]);

//   useEffect(() => {
//     const id = setInterval(() => {
//       setPositions(prev =>
//         prev.map(p => {
//           const drift = 1 + (seeded() - 0.5) * 0.004;
//           const newMark = Math.max(0.5 * p.entry, p.mark * drift);
//           const pnl =
//             p.side === "LONG"
//               ? (newMark - p.entry) * p.size
//               : (p.entry - newMark) * p.size;
//           const notional = p.size * p.entry;
//           const rr =
//             notional > 0 ? pnl / (notional * 0.02) : p.rr;
//           return {
//             ...p,
//             mark: newMark,
//             upnl: pnl,
//             rr,
//           };
//         })
//       );
//     }, 2200);
//     return () => clearInterval(id);
//   }, [seeded]);

//   const uPnL = positions.reduce((s, p) => s + p.upnl, 0);
//   const dayPnL = (seeded() - 0.5) * 900;
//   const cash = latestEquity * 0.14;
//   const marginUsed = latestEquity - cash;
//   const estLev = Math.max(
//     1,
//     Math.min(4, 1 + (marginUsed / latestEquity) * 3)
//   );

//   const pnlDist = useMemo(() => {
//     const buckets = ["<-2%", "-2..-1%", "-1..0%", "0..1%", "1..2%", ">2%"];
//     return buckets.map(b => ({
//       bucket: b,
//       n: Math.floor(5 + seeded() * 20),
//     }));
//   }, [seeded]);

//   // Exposure breakdown for filler card (computed from positions)
//   const exposure = useMemo(() => {
//     let long = 0;
//     let short = 0;
//     positions.forEach(p => {
//       const notion = p.size * p.mark;
//       if (p.side === "LONG") long += notion;
//       else short += notion;
//     });
//     const gross = long + short || 1;
//     return [
//       { label: "Net", value: (long - short) / gross },
//       { label: "Long", value: long / gross },
//       { label: "Short", value: short / gross },
//     ];
//   }, [positions]);

//   // ========== Layout ==========

//   return (
//     <div className="flex flex-col gap-4">
//       {/* Top section: two columns */}
//       <div className="flex flex-col gap-4 lg:flex-row">
//         {/* Left column: 3 stacked cards (fills height) */}
//         <div className="flex-1 flex flex-col gap-4">
//           <Card className={glassPanel}>
//             <CardHeader>
//               <CardTitle>Portfolio Overview</CardTitle>
//               <CardDescription>
//                 Live-style mock for {symbol} · {mode} on {exchange}. Looks and behaves like
//                 production, but uses safe demo data.
//               </CardDescription>
//             </CardHeader>
//             <CardContent className="grid gap-4 md:grid-cols-4">
//               <KpiTile
//                 label="Equity"
//                 value={fmtUSD0(latestEquity)}
//                 hint="Simulated rolling curve"
//               />
//               <KpiTile
//                 label="Day PnL"
//                 value={
//                   <span
//                     className={
//                       dayPnL >= 0
//                         ? "text-emerald-300"
//                         : "text-rose-300"
//                     }
//                   >
//                     {dayPnL >= 0 ? "+" : "-"}
//                     {fmtUSD0(Math.abs(dayPnL))}
//                   </span>
//                 }
//               />
//               <KpiTile
//                 label="Unrealized PnL"
//                 value={
//                   <span
//                     className={
//                       uPnL >= 0
//                         ? "text-emerald-300"
//                         : "text-rose-300"
//                     }
//                   >
//                     {uPnL >= 0 ? "+" : "-"}
//                     {fmtUSD0(Math.abs(uPnL))}
//                   </span>
//                 }
//               />
//               <KpiTile
//                 label="Est. Leverage"
//                 value={`${estLev.toFixed(1)}x`}
//                 hint="Based on margin vs equity"
//               />
//             </CardContent>
//           </Card>

//           <Card className={glassPanel}>
//             <CardHeader>
//               <CardTitle>Equity Curve (Rolling)</CardTitle>
//               <CardDescription>
//                 Auto-updating 90-point window so your client sees motion.
//               </CardDescription>
//             </CardHeader>
//             <CardContent className="h-40">
//               <ResponsiveContainer>
//                 <LineChart data={equity}>
//                   <XAxis dataKey="t" hide />
//                   <YAxis hide />
//                   <RTooltip />
//                   <Line
//                     type="monotone"
//                     dataKey="v"
//                     dot={false}
//                     stroke="#22c55e"
//                     strokeWidth={1.6}
//                   />
//                 </LineChart>
//               </ResponsiveContainer>
//             </CardContent>
//           </Card>

//           {/* NEW: Exposure card to fill previous gap */}
//           <Card className={glassPanel}>
//             <CardHeader>
//               <CardTitle>Exposure Breakdown</CardTitle>
//               <CardDescription>
//                 Synthetic net / long / short exposure from demo positions.
//               </CardDescription>
//             </CardHeader>
//             <CardContent className="h-28">
//               <ResponsiveContainer>
//                 <BarChart
//                   data={exposure}
//                   margin={{ top: 4, right: 8, left: 0, bottom: 0 }}
//                 >
//                   <XAxis dataKey="label" tick={{ fontSize: 10 }} />
//                   <YAxis hide domain={[-1, 1]} />
//                   <RTooltip />
//                   <Bar dataKey="value" radius={6}>
//                     {exposure.map((e, i) => (
//                       <Cell
//                         key={e.label}
//                         fill={
//                           e.label === "Short"
//                             ? "#f97316"
//                             : e.label === "Net"
//                             ? "#38bdf8"
//                             : "#22c55e"
//                         }
//                       />
//                     ))}
//                   </Bar>
//                 </BarChart>
//               </ResponsiveContainer>
//             </CardContent>
//           </Card>
//         </div>

//         {/* Right column: Allocation + Positions + Risk */}
//         <div className="w-full lg:max-w-sm flex flex-col gap-4">
//           <Card className={glassPanel}>
//             <CardHeader>
//               <CardTitle>Allocation</CardTitle>
//               <CardDescription>
//                 Synthetic mix; updates with equity.
//               </CardDescription>
//             </CardHeader>
//             <CardContent className="flex flex-col items-center">
//               <div className="w-40 h-40">
//                 <ResponsiveContainer>
//                   <PieChart>
//                     <Pie
//                       data={allocations}
//                       dataKey="value"
//                       nameKey="asset"
//                       innerRadius={50}
//                       outerRadius={80}
//                       label={(d: any) =>
//                         `${d.name} ${(
//                           (d.value / latestEquity) *
//                           100
//                         ).toFixed(0)}%`
//                       }
//                     >
//                       {allocations.map((_, i) => (
//                         <Cell
//                           key={i}
//                           fill={COLORS[i % COLORS.length]}
//                         />
//                       ))}
//                     </Pie>
//                   </PieChart>
//                 </ResponsiveContainer>
//               </div>
//               <p className={`mt-1 text-xs ${faintText}`}>
//                 Real app will plug directly into balances, same layout.
//               </p>
//             </CardContent>
//           </Card>

//           <Card className={glassPanel}>
//             <CardHeader>
//               <CardTitle>Open Positions</CardTitle>
//             </CardHeader>
//             <CardContent className="max-h-40 overflow-auto text-xs">
//               <table className="w-full border-separate border-spacing-y-1">
//                 <thead className={faintText}>
//                   <tr>
//                     <th className="text-left">Symbol</th>
//                     <th className="text-left">Side</th>
//                     <th className="text-right">Size</th>
//                     <th className="text-right">Entry</th>
//                     <th className="text-right">Mark</th>
//                     <th className="text-right">uPnL</th>
//                     <th className="text-right">Lev</th>
//                   </tr>
//                 </thead>
//                 <tbody>
//                   {positions.map(p => {
//                     const pnlCls =
//                       p.upnl >= 0
//                         ? "text-emerald-300"
//                         : "text-rose-300";
//                     const sideCls =
//                       p.side === "LONG"
//                         ? "text-emerald-400"
//                         : "text-rose-400";
//                     return (
//                       <tr key={p.id} className="text-slate-200">
//                         <td>{p.symbol}</td>
//                         <td className={sideCls}>{p.side}</td>
//                         <td className="text-right">
//                           {p.size.toFixed(3)}
//                         </td>
//                         <td className="text-right">
//                           {fmtUSD0(p.entry)}
//                         </td>
//                         <td className="text-right">
//                           {fmtUSD0(p.mark)}
//                         </td>
//                         <td className={`text-right ${pnlCls}`}>
//                           {fmtUSD0(p.upnl)}
//                         </td>
//                         <td className="text-right">{p.lev}x</td>
//                       </tr>
//                     );
//                   })}
//                 </tbody>
//               </table>
//               <p className={`mt-1 text-[10px] ${faintText}`}>
//                 Execution disabled in demo – visuals only.
//               </p>
//             </CardContent>
//           </Card>

//           <Card className={glassPanel}>
//             <CardHeader>
//               <CardTitle>Risk Snapshot</CardTitle>
//             </CardHeader>
//             <CardContent className="space-y-1 text-[10px]">
//               <div>Concentration: BTC / ETH heavy with USDC buffer.</div>
//               <div>Simulated Max DD (90d): -12%</div>
//               <div>Hit-rate (simulated): 57%</div>
//               <div className="mt-1">
//                 <div className="mb-1">PnL distribution</div>
//                 <div className="grid grid-cols-3 gap-1">
//                   {pnlDist.map(b => (
//                     <Badge
//                       key={b.bucket}
//                       variant="muted"
//                       className="flex justify-between"
//                     >
//                       <span>{b.bucket}</span>
//                       <span>{b.n}</span>
//                     </Badge>
//                   ))}
//                 </div>
//               </div>
//             </CardContent>
//           </Card>
//         </div>
//       </div>

//       {/* Bottom: full-width liquidations */}
//       <div>
//         <LiquidationsCard />
//       </div>
//     </div>
//   );
// };

// export default PortfolioTab;
