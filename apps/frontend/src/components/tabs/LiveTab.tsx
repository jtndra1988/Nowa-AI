// "use client";

// import React, {
//   useEffect,
//   useMemo,
//   useRef,
//   useState,
// } from "react";
// import {
//   Card,
//   CardHeader,
//   CardTitle,
//   CardDescription,
//   CardContent,
//   Button,
//   Input,
//   Label,
//   Tag,
//   Badge,
//   useLocalStorage,
//   faintText,
// } from "../layout/AppShell";
// import API, {
//   type SymbolCode,
//   type MarketMode,
//   type Asset,
//   type OptionsChain,
// } from "@/lib/api";
// import { Switch } from "@/components/ui/switch";
// import { toast } from "sonner";

// /* ---------- Types / helpers ---------- */

// type OrderSide = "BUY" | "SELL";
// type OrderType = "Market" | "Limit";

// type MiniOrder = {
//   id: string;
//   t: string;
//   market: MarketMode;
//   symbol: SymbolCode;
//   side: OrderSide;
//   size: number;
//   price: number;
//   type: OrderType;
//   source: "MANUAL" | "AI";
//   pnl?: number;
// };

// const genId = () =>
//   (typeof crypto !== "undefined" && "randomUUID" in crypto
//     ? crypto.randomUUID()
//     : Math.random().toString(36).slice(2)) as string;

// /* ---------- Live Trades Feed ---------- */

// const LiveTradesFeed: React.FC<{
//   symbol: SymbolCode;
//   mode: MarketMode;
// }> = ({ symbol, mode }) => {
//   const [rows, setRows] = useState<any[]>([]);

//   useEffect(() => {
//     let alive = true;
//     setRows([]);

//     API.getLiveTrades(API.toAsset(symbol), mode)
//       .then((data) => {
//         if (alive && data) setRows(data);
//       })
//       .catch(() => {});

//     const stop = API.tickLiveTrades(
//       API.toAsset(symbol),
//       mode,
//       (updater) => {
//         if (!alive) return;
//         setRows((prev) => updater(prev));
//       }
//     );

//     return () => {
//       alive = false;
//       stop?.();
//     };
//   }, [symbol, mode]);

//   return (
//     <Card>
//       <CardHeader>
//         <CardTitle>Live Trades Feed</CardTitle>
//         <CardDescription>
//           Streaming trades for {symbol} · {mode}
//         </CardDescription>
//       </CardHeader>
//       <CardContent className="h-64 overflow-auto text-xs">
//         <div className="grid grid-cols-6 gap-2 mb-1 text-slate-400">
//           <div>Time</div>
//           <div>Side</div>
//           <div>Size</div>
//           <div>Price</div>
//           <div>Venue</div>
//           <div>Type</div>
//         </div>
//         {rows.map((r, i) => (
//           <div
//             key={i}
//             className="grid grid-cols-6 gap-2 py-1 border-b border-white/5"
//           >
//             <div>{r.ts}</div>
//             <div
//               className={
//                 r.side === "BUY" || r.side === "LONG"
//                   ? "text-emerald-400"
//                   : "text-rose-400"
//               }
//             >
//               {r.side}
//             </div>
//             <div>{r.size}</div>
//             <div>{r.price}</div>
//             <div>{r.exchange ?? "—"}</div>
//             <div>{r.type ?? "agg"}</div>
//           </div>
//         ))}
//       </CardContent>
//     </Card>
//   );
// };

// /* ---------- Options Ticket (for options mode) ---------- */

// const OptionsTicket: React.FC<{
//   asset: Asset;
//   chain: OptionsChain | null;
// }> = ({ asset }) => (
//   <Card className="mt-3 bg-slate-950/80 border border-white/12">
//     <CardContent className="p-3 text-xs">
//       <div className={faintText}>
//         Options ticket for {asset} is mocked. Plug your real options
//         chain + order form here for production.
//       </div>
//     </CardContent>
//   </Card>
// );

// /* ---------- Main LiveTab ---------- */

// export const LiveTab: React.FC<{
//   symbol: SymbolCode;
//   mode: MarketMode;
//   exchange: string;
// }> = ({ symbol, mode, exchange }) => {
//   const asset = useMemo<Asset>(() => API.toAsset(symbol), [symbol]);

//   const [ticker, setTicker] = useState<{ price: number } | null>(null);
//   const [pred, setPred] = useState<any>(null);
//   const [orders, setOrders] = useState<MiniOrder[]>([]);

//   // Trade ticket state
//   const [side, setSide] = useState<OrderSide>("BUY");
//   const [orderType, setOrderType] = useState<OrderType>("Market");
//   const [size, setSize] = useState("0.1");
//   const [limitPrice, setLimitPrice] = useState("");

//   // AI state
//   const [autoTrade, setAutoTrade] = useLocalStorage(
//     "mars.live.autotrade",
//     false
//   );
//   const [confCutoff, setConfCutoff] = useLocalStorage(
//     "mars.live.confCut",
//     0.62
//   );

//   // Streams
//   useEffect(() => {
//     setTicker(null);
//     const stop = API.streamTicker(symbol, (t) => setTicker({ price: t.price }));
//     return () => stop?.();
//   }, [symbol]);

//   useEffect(() => {
//     setPred(null);
//     const stop = API.streamPredict(asset, (p) => setPred(p));
//     return () => stop?.();
//   }, [asset]);

//   // Derived
//   const numericSize = useMemo(
//     () => parseFloat(size || "0") || 0,
//     [size]
//   );

//   const effectiveLimitPx = useMemo(() => {
//     if (orderType === "Market") {
//       return ticker?.price ?? 0;
//     }
//     const lp = parseFloat(limitPrice || "0");
//     return lp || ticker?.price || 0;
//   }, [orderType, limitPrice, ticker?.price]);

//   const ctaLabel = useMemo(() => {
//     const sz = numericSize || 0;
//     const dir = side === "BUY" ? "BUY" : "SELL";
//     return sz > 0 ? `${dir} ${sz} ${symbol}` : `${dir} ${symbol}`;
//   }, [side, numericSize, symbol]);

//   const pushOrder = (o: MiniOrder) => {
//     setOrders((prev) => [o, ...prev].slice(0, 200));
//   };

//   /* ----- Manual submit ----- */

//   const submitManual = () => {
//     if (!ticker?.price) return;
//     if (!numericSize) return;

//     const px =
//       orderType === "Market" ? ticker.price : effectiveLimitPx;
//     if (!px || px <= 0) return;

//     const ord: MiniOrder = {
//       id: genId(),
//       t: new Date().toLocaleTimeString(),
//       market: mode,
//       symbol,
//       side,
//       size: numericSize,
//       price: px,
//       type: orderType,
//       source: "MANUAL",
//     };
//     pushOrder(ord);

//     toast.success(
//       `Manual ${ord.type} ${ord.side} ${ord.size} ${ord.symbol} @ ${ord.price.toFixed(
//         2
//       )}`
//     );
//   };

//   /* ----- AI auto-trade ----- */

//   const aiAction: "LONG" | "SHORT" | "HOLD" = useMemo(() => {
//     if (!pred) return "HOLD";
//     const c = pred.price_confidence ?? 0;
//     if (c >= confCutoff && pred.direction === "up") return "LONG";
//     if (c >= confCutoff && pred.direction === "down") return "SHORT";
//     return "HOLD";
//   }, [pred, confCutoff]);

//   const lastOrderAt = useRef(0);
//   const lastSignal = useRef<"LONG" | "SHORT" | "HOLD">("HOLD");

//   useEffect(() => {
//     if (!autoTrade || !ticker?.price) return;
//     if (aiAction === "HOLD") {
//       lastSignal.current = "HOLD";
//       return;
//     }

//     const now = Date.now();
//     const cooldownMs = 8000;
//     const changed = aiAction !== lastSignal.current;
//     if (!changed || now - lastOrderAt.current < cooldownMs) return;

//     lastSignal.current = aiAction;
//     lastOrderAt.current = now;

//     const side: OrderSide = aiAction === "LONG" ? "BUY" : "SELL";
//     const aiSize =
//       numericSize > 0 ? Math.max(0.01, numericSize * 0.25) : 0.01;

//     let px = ticker.price;
//     let type: OrderType = orderType;

//     if (type === "Limit") {
//       // tiny improvement vs market (±10 bps)
//       const bump = aiAction === "LONG" ? -0.001 : 0.001;
//       px = ticker.price * (1 + bump);
//     }

//     const ord: MiniOrder = {
//       id: genId(),
//       t: new Date().toLocaleTimeString(),
//       market: mode,
//       symbol,
//       side,
//       size: aiSize,
//       price: px,
//       type,
//       source: "AI",
//     };
//     pushOrder(ord);

//     toast.success(
//       `AI ${ord.type} ${ord.side} ${ord.size} ${ord.symbol} @ ${ord.price.toFixed(
//         2
//       )}`,
//       {
//         description: `Confidence ${(
//           (pred?.price_confidence ?? 0) * 100
//         ).toFixed(1)}% ≥ cutoff ${(confCutoff * 100).toFixed(0)}%`,
//       }
//     );
//   }, [
//     autoTrade,
//     aiAction,
//     ticker?.price,
//     orderType,
//     numericSize,
//     mode,
//     symbol,
//     confCutoff,
//     pred?.price_confidence,
//   ]);

//   /* ----- Options chain (for options mode) ----- */

//   const [chain, setChain] = useState<OptionsChain | null>(null);

//   useEffect(() => {
//     if (mode !== "options") {
//       setChain(null);
//       return;
//     }
//     (async () => {
//       try {
//         const data = await API.getOptionsChain(asset);
//         setChain(data);
//       } catch (e) {
//         console.error("Options chain load failed", e);
//       }
//     })();
//   }, [mode, asset]);

//   /* ----- Layout ----- */

//   return (
//     <div className="flex flex-col gap-4">
//       {/* 3-column layout on xl: left 1/3 = trade ticket, right 2/3 = orders/feed */}
//       <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 items-start">
//         {/* Left column */}
//         <div className="xl:col-span-1 space-y-3">
//           <Card className="bg-slate-950/80 border border-white/12 shadow-[0_12px_60px_rgba(15,23,42,0.9)]">
//             <CardHeader>
//               <CardTitle>Trade Ticket</CardTitle>
//               <CardDescription>
//                 {symbol} · {mode} · {exchange}
//               </CardDescription>
//             </CardHeader>
//             <CardContent className="space-y-4">
//               {/* Buy / Sell */}
//               <div className="flex gap-2">
//                 <Button
//                   className={`flex-1 py-2 font-semibold rounded-xl transition-all duration-200 ${
//                     side === "BUY"
//                       ? "bg-emerald-500 hover:bg-emerald-400 text-white shadow-md shadow-emerald-700/30"
//                       : "bg-slate-800/80 hover:bg-slate-700/80 text-slate-300"
//                   }`}
//                   onClick={() => setSide("BUY")}
//                 >
//                   Buy
//                 </Button>
//                 <Button
//                   className={`flex-1 py-2 font-semibold rounded-xl transition-all duration-200 ${
//                     side === "SELL"
//                       ? "bg-rose-500 hover:bg-rose-400 text-white shadow-md shadow-rose-700/30"
//                       : "bg-slate-800/80 hover:bg-slate-700/80 text-slate-300"
//                   }`}
//                   onClick={() => setSide("SELL")}
//                 >
//                   Sell
//                 </Button>
//               </div>

//               {/* Market / Limit */}
//               <div className="flex gap-3 justify-center mt-2">
//                 <button
//                   className={`w-24 py-2 rounded-xl font-medium border transition-all duration-200 ${
//                     orderType === "Market"
//                       ? "bg-indigo-500 text-white border-indigo-400 shadow-md shadow-indigo-700/30"
//                       : "bg-slate-900/60 border-slate-700 text-slate-300 hover:bg-slate-800/80"
//                   }`}
//                   onClick={() => setOrderType("Market")}
//                 >
//                   Market
//                 </button>
//                 <button
//                   className={`w-24 py-2 rounded-xl font-medium border transition-all duration-200 ${
//                     orderType === "Limit"
//                       ? "bg-indigo-500 text-white border-indigo-400 shadow-md shadow-indigo-700/30"
//                       : "bg-slate-900/60 border-slate-700 text-slate-300 hover:bg-slate-800/80"
//                   }`}
//                   onClick={() => setOrderType("Limit")}
//                 >
//                   Limit
//                 </button>
//               </div>

//               {/* Size + Limit */}
//               <div className="grid grid-cols-2 gap-3 text-xs">
//                 <div>
//                   <Label>Size ({symbol})</Label>
//                   <Input
//                     value={size}
//                     onChange={(e) => setSize(e.target.value)}
//                     placeholder="0.1"
//                   />
//                 </div>
//                 <div>
//                   <Label>Limit Price (USDT)</Label>
//                   <Input
//                     disabled={orderType === "Market"}
//                     value={
//                       orderType === "Market"
//                         ? ticker?.price
//                           ? ticker.price.toFixed(2)
//                           : ""
//                         : limitPrice
//                     }
//                     onChange={(e) => setLimitPrice(e.target.value)}
//                     placeholder={
//                       ticker?.price ? ticker.price.toFixed(2) : "--"
//                     }
//                   />
//                 </div>
//               </div>

//               {/* CTA */}
//               <Button
//                 className="w-full mt-1 py-3 text-sm font-semibold bg-indigo-500 hover:bg-indigo-400 rounded-xl disabled:opacity-50"
//                 disabled={!ticker?.price || !numericSize}
//                 onClick={submitManual}
//               >
//                 {ctaLabel}
//               </Button>

//               {/* AI Section */}
//               <div className="border-t border-white/10 pt-4 mt-2 space-y-3">
//                 <div className="flex items-center justify-between gap-2 text-xs">
//                   <div>
//                     <div className="font-medium text-slate-100">
//                       AI Auto-Trade
//                     </div>
//                     <div className={faintText}>
//                       Fire small mock orders when confidence ≥ cutoff.
//                     </div>
//                   </div>
//                   <Switch
//                     checked={autoTrade}
//                     onCheckedChange={(v) => setAutoTrade(v)}
//                   />
//                 </div>

//                 <div className="space-y-1 text-xs">
//                   <Label>Confidence Cutoff</Label>
//                   <div className="flex items-center gap-3">
//                     <input
//                       type="range"
//                       min={50}
//                       max={95}
//                       step={1}
//                       value={confCutoff * 100}
//                       onChange={(e) =>
//                         setConfCutoff(
//                           Number(e.target.value) / 100
//                         )
//                       }
//                       className="w-full accent-indigo-400"
//                     />
//                     <span className="w-10 text-right">
//                       {(confCutoff * 100).toFixed(0)}%
//                     </span>
//                   </div>
//                 </div>

//                 <div className={faintText}>
//                   Uses model direction + cutoff to simulate{" "}
//                   {orderType === "Limit" ? "limit" : "market"} orders.
//                   Front-end only for this demo.
//                 </div>
//               </div>
//             </CardContent>
//           </Card>

//           {mode === "options" && (
//             <OptionsTicket asset={asset} chain={chain} />
//           )}
//         </div>

//         {/* Right column */}
//         <div className="xl:col-span-2 flex flex-col gap-4">
//           {/* Local orders blotter */}
//           <Card>
//             <CardHeader>
//               <CardTitle>Local Orders (Mock)</CardTitle>
//             </CardHeader>
//             <CardContent className="h-64 overflow-auto text-xs">
//               {!orders.length && (
//                 <div className={faintText}>
//                   No orders yet. Use the ticket or enable AI auto-trade.
//                 </div>
//               )}
//               {orders.map((o) => (
//                 <div
//                   key={o.id}
//                   className="grid grid-cols-8 gap-2 py-1 border-b border-white/5"
//                 >
//                   <div>{o.t}</div>
//                   <div>{o.symbol}</div>
//                   <div className="uppercase">
//                     {o.market}
//                   </div>
//                   <div
//                     className={
//                       o.side === "BUY"
//                         ? "text-emerald-400"
//                         : "text-rose-400"
//                     }
//                   >
//                     {o.side}
//                   </div>
//                   <div>{o.size}</div>
//                   <div>{o.price.toFixed(2)}</div>
//                   <div className="text-[10px]">
//                     {o.type}/{o.source}
//                   </div>
//                   <div>
//                     {o.pnl !== undefined
//                       ? o.pnl.toFixed(2)
//                       : "—"}
//                   </div>
//                 </div>
//               ))}
//             </CardContent>
//           </Card>

//           {/* Live trades feed */}
//           <LiveTradesFeed symbol={symbol} mode={mode} />
//         </div>
//       </div>
//     </div>
//   );
// };

// export default LiveTab;
