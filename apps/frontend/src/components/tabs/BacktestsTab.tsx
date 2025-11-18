// "use client";

// import React, { useMemo } from "react";
// import {
//   Card,
//   CardHeader,
//   CardTitle,
//   CardDescription,
//   CardContent,
//   KpiTile,
//   glassPanel,
//   faintText,
//   Tag,
// } from "../layout/AppShell";
// import type { SymbolCode, MarketMode } from "@/lib/api";
// import {
//   ResponsiveContainer,
//   LineChart,
//   Line,
//   CartesianGrid,
//   XAxis,
//   YAxis,
//   Tooltip,
//   AreaChart,
//   Area,
// } from "recharts";

// type Props = {
//   symbol: SymbolCode;
//   mode: MarketMode;
//   exchange: string;
// };

// type CurvePoint = {
//   t: string;
//   mars: number;
//   benchmark: number;
//   dd?: number;
// };

// const buildMockCurve = (): CurvePoint[] => {
//   // Pure deterministic mock so it looks realistic but stable.
//   const pts: CurvePoint[] = [];
//   let mars = 100;
//   let bench = 100;
//   for (let i = 0; i < 90; i++) {
//     // pseudo regime: trending then chop then recovery
//     const driftMars =
//       i < 25 ? 0.25 : i < 55 ? -0.05 : 0.18;
//     const driftBench =
//       i < 25 ? 0.18 : i < 55 ? -0.12 : 0.09;
//     const volMars = 0.9;
//     const volBench = 1.2;

//     const shockMars =
//       ((Math.sin(i / 5) + Math.cos(i / 7)) / 60) *
//       volMars;
//     const shockBench =
//       ((Math.cos(i / 6) - Math.sin(i / 9)) / 55) *
//       volBench;

//     mars *= 1 + (driftMars + shockMars) / 100;
//     bench *= 1 + (driftBench + shockBench) / 100;

//     const peak = Math.max(
//       ...pts.map((p) => p.mars),
//       mars
//     );
//     const dd = (mars / peak - 1) * 100;

//     pts.push({
//       t: `D${i + 1}`,
//       mars: +mars.toFixed(2),
//       benchmark: +bench.toFixed(2),
//       dd,
//     });
//   }
//   return pts;
// };

// export const BacktestsTab: React.FC<Props> = ({
//   symbol,
//   mode,
//   exchange,
// }) => {
//   const curve = useMemo(buildMockCurve, []);
//   const last = curve[curve.length - 1];

//   const totalReturnMars = ((last.mars / 100 - 1) * 100).toFixed(1);
//   const totalReturnBench = (
//     (last.benchmark / 100 - 1) *
//     100
//   ).toFixed(1);

//   const maxDD = curve.reduce(
//     (m, p) => (p.dd! < m ? p.dd! : m),
//     0
//   );
//   const volMars = 18.4; // fixed nice demo numbers
//   const volBench = 27.9;
//   const sharpeMars = 2.05;
//   const sharpeBench = 0.86;

//   return (
//     <div className="flex flex-col gap-6">
//       {/* Header context */}
//       <div className="flex flex-wrap items-baseline gap-3">
//         <div className="text-xs text-slate-300">
//           Backtest profile for{" "}
//           <span className="text-indigo-300 font-medium">
//             {symbol}
//           </span>{" "}
//           in{" "}
//           <span className="text-sky-300">
//             {mode}
//           </span>{" "}
//           on{" "}
//           <span className="text-emerald-300">
//             {exchange}
//           </span>{" "}
//           (mocked but realistic to illustrate production behavior).
//         </div>
//         <Tag>2019–2024</Tag>
//         <Tag>Hourly bars</Tag>
//         <Tag>Transaction costs & slippage included</Tag>
//       </div>

//       {/* Top: Equity curve + stats */}
//       <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 items-stretch">
//         <Card className={`${glassPanel} xl:col-span-2`}>
//           <CardHeader>
//             <CardTitle className="text-sm">
//               Strategy vs Benchmark
//             </CardTitle>
//             <CardDescription className={faintText}>
//               Mars ensemble vs passive {symbol} exposure.
//             </CardDescription>
//           </CardHeader>
//           <CardContent className="h-64">
//             <ResponsiveContainer width="100%" height="100%">
//               <LineChart data={curve}>
//                 <CartesianGrid
//                   strokeDasharray="3 3"
//                   opacity={0.12}
//                 />
//                 <XAxis
//                   dataKey="t"
//                   tick={false}
//                   interval={8}
//                 />
//                 <YAxis />
//                 <Tooltip
//                   contentStyle={{
//                     background: "#020817",
//                     border:
//                       "1px solid rgba(129,140,248,0.4)",
//                     fontSize: 10,
//                   }}
//                 />
//                 <Line
//                   type="monotone"
//                   dataKey="mars"
//                   name="Mars Strategy"
//                   stroke="#6366f1"
//                   strokeWidth={1.6}
//                   dot={false}
//                 />
//                 <Line
//                   type="monotone"
//                   dataKey="benchmark"
//                   name="Benchmark"
//                   stroke="#475569"
//                   strokeWidth={1}
//                   dot={false}
//                   strokeDasharray="4 3"
//                 />
//               </LineChart>
//             </ResponsiveContainer>
//           </CardContent>
//         </Card>

//         <Card className={glassPanel}>
//           <CardHeader>
//             <CardTitle className="text-sm">
//               Performance Snapshot
//             </CardTitle>
//             <CardDescription className={faintText}>
//               Key metrics to show risk-adjusted edge.
//             </CardDescription>
//           </CardHeader>
//           <CardContent className="grid grid-cols-2 gap-3 text-xs">
//             <KpiTile
//               label="Mars Net Return"
//               value={`${totalReturnMars}%`}
//             />
//             <KpiTile
//               label="Benchmark Return"
//               value={`${totalReturnBench}%`}
//             />
//             <KpiTile
//               label="Mars Sharpe"
//               value={sharpeMars.toFixed(2)}
//             />
//             <KpiTile
//               label="Benchmark Sharpe"
//               value={sharpeBench.toFixed(2)}
//             />
//             <KpiTile
//               label="Mars Volatility"
//               value={`${volMars.toFixed(1)}%`}
//             />
//             <KpiTile
//               label="Benchmark Vol"
//               value={`${volBench.toFixed(1)}%`}
//             />
//             <KpiTile
//               label="Max Drawdown"
//               value={`${maxDD.toFixed(1)}%`}
//             />
//             <KpiTile
//               label="Win Rate (trades)"
//               value={`63.4%`}
//             />
//           </CardContent>
//         </Card>
//       </div>

//       {/* Middle: Drawdown + regime */}
//       <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
//         <Card className={glassPanel}>
//           <CardHeader>
//             <CardTitle className="text-sm">
//               Drawdown Curve (Mars)
//             </CardTitle>
//             <CardDescription className={faintText}>
//               Shows how the strategy behaves in stress periods.
//             </CardDescription>
//           </CardHeader>
//           <CardContent className="h-40">
//             <ResponsiveContainer width="100%" height="100%">
//               <AreaChart data={curve}>
//                 <CartesianGrid
//                   strokeDasharray="3 3"
//                   opacity={0.1}
//                 />
//                 <XAxis
//                   dataKey="t"
//                   tick={false}
//                   interval={10}
//                 />
//                 <YAxis
//                   tickFormatter={(v) => `${v}%`}
//                 />
//                 <Tooltip
//                   contentStyle={{
//                     background: "#020817",
//                     border:
//                       "1px solid rgba(248,113,113,0.45)",
//                     fontSize: 10,
//                   }}
//                   formatter={(v) => {
//   if (typeof v === "number") {
//     return `${v.toFixed(2)}%`;
//   }
//   // Fallback for other types (like string)
//   return `${v}%`;
// }}
//                 />
//                 <Area
//                   type="monotone"
//                   dataKey="dd"
//                   name="Drawdown"
//                   stroke="#f97316"
//                   fill="#f97316"
//                   fillOpacity={0.14}
//                 />
//               </AreaChart>
//             </ResponsiveContainer>
//           </CardContent>
//         </Card>

//         <Card className={glassPanel}>
//           <CardHeader>
//             <CardTitle className="text-sm">
//               Regime Response (Narrative)
//             </CardTitle>
//             <CardDescription className={faintText}>
//               Talking points for your client on how Mars adapts.
//             </CardDescription>
//           </CardHeader>
//           <CardContent className="text-[10px] text-slate-300 space-y-2">
//             <ul className="list-disc ml-4 space-y-1.5">
//               <li>
//                 <span className="text-emerald-300">
//                   Trend Regimes:
//                 </span>{" "}
//                 Mars lets winners run with tighter stop-outs,
//                 outperforming benchmark by{" "}
//                 <span className="text-emerald-300">
//                   ~12–18%
//                 </span>{" "}
//                 in major cycles.
//               </li>
//               <li>
//                 <span className="text-amber-300">
//                   Choppy / Mean-Revert:
//                 </span>{" "}
//                 position sizes decay automatically; net exposure
//                 compressed to preserve capital.
//               </li>
//               <li>
//                 <span className="text-sky-300">
//                   Stress Events:
//                 </span>{" "}
//                 on-chain outflows + perp stress + vol spikes trigger
//                 de-leveraging and temporary flat regimes.
//               </li>
//               <li>
//                 All logic is encapsulated in the same interface as
//                 your live tab: once backend is wired, these charts
//                 render from real runs without UI changes.
//               </li>
//             </ul>
//           </CardContent>
//         </Card>
//       </div>

//       {/* Bottom: Trade / risk breakdown */}
//       <Card className={glassPanel}>
//         <CardHeader>
//           <CardTitle className="text-sm">
//             Trade Distribution & Risk Controls
//           </CardTitle>
//           <CardDescription className={faintText}>
//             Use this section verbally to explain discipline & controls.
//           </CardDescription>
//         </CardHeader>
//         <CardContent className="grid grid-cols-1 lg:grid-cols-3 gap-4 text-[10px] text-slate-300">
//           <div>
//             <div className="font-semibold mb-1 text-slate-100">
//               Trade Distribution
//             </div>
//             <ul className="list-disc ml-4 space-y-1">
//               <li>~1,250 closed trades over test period</li>
//               <li>
//                 Avg holding: 4.8h (intraday to short swing)
//               </li>
//               <li>
//                 P&L driven by ~18% of trades (fat-tail winners)
//               </li>
//               <li>
//                 Stops enforced, no martingale / grid behavior
//               </li>
//             </ul>
//           </div>
//           <div>
//             <div className="font-semibold mb-1 text-slate-100">
//               Risk Framework
//             </div>
//             <ul className="list-disc ml-4 space-y-1">
//               <li>Dynamic per-trade risk caps (0.25–0.75%)</li>
//               <li>Global drawdown brake at -10%</li>
//               <li>
//                 Volatility- and liquidity-aware sizing
//               </li>
//               <li>
//                 Optional hard limits by venue / symbol
//               </li>
//             </ul>
//           </div>
//           <div>
//             <div className="font-semibold mb-1 text-slate-100">
//               How to Pitch This
//             </div>
//             <ul className="list-disc ml-4 space-y-1">
//               <li>
//                 “What you see here is generated client-side today,
//                 but wired to consume your real backtest API.”
//               </li>
//               <li>
//                 “Swapping from dummy to production data is a config
//                 change, not a redesign.”
//               </li>
//               <li>
//                 “All panels are live-reactive to symbol / venue
//                 context from the top bar.”
//               </li>
//             </ul>
//           </div>
//         </CardContent>
//       </Card>
//     </div>
//   );
// };

// export default BacktestsTab;
