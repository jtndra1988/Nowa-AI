"use client";

import React, { useMemo, useState, useEffect } from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  glassPanel,
} from "../layout/AppShell";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
} from "recharts";
import type { SymbolCode, MarketMode } from "@/lib/api";
import {
  ArrowUpRight,
  ArrowDownRight,
  Activity,
  Target,
  ShieldAlert,
  TrendingUp,
  Zap,
  Users,
  Newspaper,
  Gauge,
  BrainCircuit,
  PlayCircle,
} from "lucide-react";

type Props = {
  symbol: SymbolCode;
  mode: MarketMode;
  exchange: string;
};

// --- 1. SIMULATION ENGINE (The "Dummy Brain") ---
const useDemoSimulation = (symbol: string) => {
  // Base state
  const [price, setPrice] = useState(86400);
  const [target, setTarget] = useState(87850);
  const [confidence, setConfidence] = useState(82);
  const [volatility, setVolatility] = useState(0.012);
  
  // Intel state
  const [sentiment, setSentiment] = useState(0.35); // Positive
  const [funding, setFunding] = useState(1.2);      // Longs paying shorts
  const [orderflow, setOrderflow] = useState(0.15); // Net buying

  const [history, setHistory] = useState<{ time: string; price: number }[]>([]);

  // Initialize fake history
  useEffect(() => {
    const initData = [];
    let base = 94800;
    for (let i = 0; i < 25; i++) {
      base += (Math.random() - 0.45) * 120;
      initData.push({
        time: `${10 + Math.floor(i/2)}:${(i % 2) * 30}`,
        price: base
      });
    }
    setHistory(initData);
  }, []);

  // The "Live Feed" Loop
  useEffect(() => {
    const interval = setInterval(() => {
      // 1. Wiggle Price
      setPrice(prev => prev + (Math.random() - 0.4) * 40);
      
      // 2. Wiggle Confidence & Volatility
      setConfidence(prev => Math.min(99, Math.max(40, prev + (Math.random() - 0.5) * 4)));
      setVolatility(prev => Math.max(0.005, prev + (Math.random() - 0.5) * 0.001));

      // 3. Update Intel occasionally
      if (Math.random() > 0.8) {
        setSentiment(prev => Math.max(-1, Math.min(1, prev + (Math.random() - 0.5) * 0.1)));
        setFunding(prev => prev + (Math.random() - 0.5) * 0.5);
        setOrderflow(prev => Math.max(-1, Math.min(1, prev + (Math.random() - 0.5) * 0.05)));
      }

      // 4. Add Chart Point (every few updates)
      if (Math.random() > 0.6) {
        setHistory(prev => {
          const lastPrice = prev[prev.length - 1].price;
          const newPrice = lastPrice + (Math.random() - 0.45) * 50;
          const now = new Date();
          const timeStr = `${now.getHours()}:${now.getMinutes().toString().padStart(2, '0')}`;
          
          const newHistory = [...prev.slice(1), { time: timeStr, price: newPrice }];
          return newHistory;
        });
      }
    }, 1500); // Updates every 1.5 seconds

    return () => clearInterval(interval);
  }, []);

  // Derive Direction from Target vs Price
  const direction = target > price ? "up" : target < price ? "down" : "flat";
  const rangeHigh = target * (1 + volatility);
  const rangeLow = price * (1 - volatility);

  return {
    currentPrice: price,
    targetPrice: target,
    rangeHigh,
    rangeLow,
    direction,
    confidence,
    volatility,
    sentiment,
    funding,
    orderflow,
    history
  };
};


// --- UX Helper: Dynamic Styles based on Trend ---
const getTrendStyles = (trend: "up" | "down" | "flat") => {
  if (trend === "up") {
    return {
      text: "text-emerald-400",
      bg: "bg-emerald-500/10",
      border: "border-emerald-500/40",
      shadow: "shadow-[0_0_40px_-10px_rgba(16,185,129,0.3)]",
      gradient: "from-emerald-500/20 to-emerald-900/5",
      iconBg: "bg-emerald-500/20",
      stroke: "#10b981",
    };
  }
  if (trend === "down") {
    return {
      text: "text-rose-400",
      bg: "bg-rose-500/10",
      border: "border-rose-500/40",
      shadow: "shadow-[0_0_40px_-10px_rgba(244,63,94,0.3)]",
      gradient: "from-rose-500/20 to-rose-900/5",
      iconBg: "bg-rose-500/20",
      stroke: "#f43f5e",
    };
  }
  return {
    text: "text-indigo-300",
    bg: "bg-indigo-500/5",
    border: "border-indigo-500/30",
    shadow: "shadow-[0_0_30px_-10px_rgba(99,102,241,0.2)]",
    gradient: "from-indigo-500/10 to-slate-900/5",
    iconBg: "bg-indigo-500/20",
    stroke: "#6366f1",
  };
};

export const DashboardTab: React.FC<Props> = ({ symbol, mode, exchange }) => {
  // 1. Connect the Simulation Hook
  const demo = useDemoSimulation(symbol);

  // 2. Map Data for UI
  const {
    currentPrice,
    targetPrice,
    rangeHigh,
    rangeLow,
    direction,
    confidence,
    volatility,
    sentiment,
    funding,
    orderflow,
    history
  } = demo;

  const forecastPct = ((targetPrice - currentPrice) / currentPrice) * 100;
  const trendStyles = getTrendStyles(direction as any);

  // 3. Chart Data Prep
  const chartData = useMemo(() => {
    const data = [...history];
    // Append Forecast Point
    data.push({
      time: "Forecast",
      price: targetPrice,
    });
    return data;
  }, [history, targetPrice]);

  return (
    <div className="w-full flex flex-col gap-6 text-slate-50 animate-in fade-in duration-700">
      
      {/* Demo Banner */}
      {/* <div className="w-full py-1.5 bg-indigo-500/10 border-b border-indigo-500/20 text-center text-[11px] font-mono text-indigo-300 uppercase tracking-[0.2em] flex items-center justify-center gap-2">
        <PlayCircle className="w-3 h-3 animate-pulse" />
        Live Demo Environment
      </div> */}

      {/* === HERO ROW === */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* 1. The "Oracle" Card (Forecast) */}
        <Card className={`lg:col-span-2 relative overflow-hidden border-2 transition-all duration-500 ${trendStyles.border} ${trendStyles.shadow} ${glassPanel}`}>
          {/* Ambient Background Glow */}
          <div className={`absolute inset-0 bg-gradient-to-br ${trendStyles.gradient} opacity-30`} />
          {/* Dynamic glow blob behind text */}
          <div className={`absolute -top-24 -right-24 w-96 h-96 rounded-full blur-[120px] opacity-20 pointer-events-none ${trendStyles.bg.replace('/10', '')}`} />

          <CardContent className="p-8 flex flex-col justify-between h-full relative z-10">
            <div>
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-950/50 border border-white/10 backdrop-blur-md text-xs font-bold uppercase tracking-wider text-slate-300 shadow-sm">
                    <BrainCircuit className="w-3.5 h-3.5 text-indigo-400" />
                    <span>AI Hourly Forecast</span>
                  </div>
                  {confidence > 70 && (
                    <span className="px-2.5 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-[10px] font-bold text-emerald-400 flex items-center gap-1 animate-pulse">
                      <Zap className="w-3 h-3" /> Strong Signal
                    </span>
                  )}
                </div>
                <div className="text-xs font-mono text-slate-400 flex items-center gap-2">
                  <span className="relative flex h-2 w-2">
                    <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${trendStyles.bg.replace('/10', '')}`}></span>
                    <span className={`relative inline-flex rounded-full h-2 w-2 ${trendStyles.text.replace('text-', 'bg-')}`}></span>
                  </span>
                  Live Model Output
                </div>
              </div>

              <div className="space-y-2">
                <h2 className="text-slate-400 text-sm font-medium uppercase tracking-wide pl-1">Target Price (1H)</h2>
                <div className="flex items-baseline gap-4 flex-wrap">
                  <span className="text-6xl md:text-7xl font-black tracking-tighter text-white drop-shadow-xl">
                    ${targetPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </span>
                  <div className={`flex items-center gap-1.5 px-4 py-2 rounded-xl border border-white/5 backdrop-blur-md text-xl font-bold shadow-inner ${trendStyles.bg} ${trendStyles.text}`}>
                    {direction === "up" ? <ArrowUpRight strokeWidth={3} /> : direction === "down" ? <ArrowDownRight strokeWidth={3} /> : <Activity strokeWidth={3} />}
                    {forecastPct > 0 ? "+" : ""}{forecastPct.toFixed(2)}%
                  </div>
                </div>
              </div>
              
              <div className="mt-6 p-4 rounded-xl bg-slate-950/40 border border-white/5 backdrop-blur-sm max-w-xl">
                <p className="text-slate-300 text-sm leading-relaxed">
                  <span className="text-indigo-400 font-semibold">AI Analysis:</span> The ensemble models predict 
                  <span className={`font-bold mx-1 ${trendStyles.text}`}>
                    {direction === "up" ? "UPWARD" : direction === "down" ? "DOWNWARD" : "SIDEWAYS"}
                  </span> 
                  movement. Current order flow and options positioning support a move toward 
                  <span className="text-white font-mono mx-1">${targetPrice.toLocaleString(undefined, {maximumFractionDigits: 0})}</span> 
                  within the next 60 minutes.
                </p>
              </div>
            </div>

            {/* Visual Range Bar */}
            <div className="mt-10 pt-6 border-t border-white/5">
              <div className="flex justify-between text-xs font-bold uppercase tracking-wider mb-3 text-slate-500">
                <span>Support (Low)</span>
                <span className="text-indigo-300">Expected 1H Range</span>
                <span>Resistance (High)</span>
              </div>
              
              <div className="relative h-4 bg-slate-900/80 rounded-full w-full overflow-hidden shadow-inner border border-white/5">
                {/* Range Zone */}
                <div className="absolute top-0 bottom-0 left-[15%] right-[15%] bg-gradient-to-r from-indigo-500/10 via-indigo-500/20 to-indigo-500/10 border-x border-indigo-500/20" />
                
                {/* Current Price Marker */}
                <div 
                  className="absolute top-0 bottom-0 w-0.5 bg-white/50 z-20 transition-all duration-1000 ease-in-out"
                  style={{ left: "50%" }}
                />
                
                {/* Target Dot (Animated) */}
                <div 
                  className={`absolute top-1/2 -translate-y-1/2 w-5 h-5 -ml-2.5 rounded-full border-2 border-white shadow-[0_0_15px_currentColor] z-30 transition-all duration-1000 ${trendStyles.text} ${trendStyles.bg}`}
                  style={{ left: direction === "up" ? "65%" : "35%" }}
                >
                   <div className="absolute inset-0 rounded-full animate-ping opacity-20 bg-current"></div>
                </div>
              </div>

              <div className="flex justify-between text-[11px] mt-2 font-mono font-medium text-slate-400">
                <span>${rangeLow.toLocaleString(undefined, {maximumFractionDigits:0})}</span>
                <span className="text-slate-500">Current: ${currentPrice.toLocaleString(undefined, {maximumFractionDigits:0})}</span>
                <span>${rangeHigh.toLocaleString(undefined, {maximumFractionDigits:0})}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 2. The "Trade Command" Column */}
        <div className="flex flex-col gap-4">
          {/* Action Badge */}
          <Card className={`relative overflow-hidden border ${trendStyles.border} shadow-2xl flex-1 flex flex-col justify-center items-center p-6 text-center group`}>
             <div className={`absolute inset-0 bg-gradient-to-b ${trendStyles.gradient} opacity-40`} />
             <div className="relative z-10">
                <div className="text-xs text-slate-400 uppercase tracking-[0.25em] font-bold mb-4">
                  Recommended Action
                </div>
                <div className={`text-5xl font-black tracking-tight drop-shadow-2xl scale-110 ${trendStyles.text}`}>
                  {direction === "up" ? "BUY" : direction === "down" ? "SELL" : "WAIT"}
                </div>
                
                {/* Stop Loss Chip */}
                <div className="mt-6 flex items-center justify-center gap-2">
                  <div className="px-4 py-2 rounded-lg bg-slate-950/60 border border-white/10 flex items-center gap-2 shadow-lg backdrop-blur-md">
                    <ShieldAlert className="w-4 h-4 text-amber-500" />
                    <div className="text-left">
                      <div className="text-[9px] text-slate-500 uppercase font-bold">Smart Stop Loss</div>
                      <div className="text-sm font-mono text-slate-200 font-bold">
                        ${(currentPrice * (direction === "up" ? 0.99 : 1.01)).toLocaleString(undefined, {maximumFractionDigits:0})}
                      </div>
                    </div>
                  </div>
                </div>
             </div>
          </Card>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 gap-3 flex-1">
             <Card className="bg-slate-900/80 border border-slate-800 p-4 flex flex-col justify-center items-center relative overflow-hidden">
                <div className="absolute inset-0 bg-indigo-500/5"></div>
                <Gauge className="w-6 h-6 text-indigo-400 mb-2 relative z-10" />
                <div className="text-2xl font-bold text-white relative z-10">{confidence.toFixed(0)}%</div>
                <div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider relative z-10">Confidence</div>
             </Card>

             <Card className="bg-slate-900/80 border border-slate-800 p-4 flex flex-col justify-center items-center relative overflow-hidden">
                <div className={`absolute inset-0 ${volatility > 0.01 ? 'bg-amber-500/10' : 'bg-slate-500/5'}`}></div>
                <Activity className={`w-6 h-6 mb-2 relative z-10 ${volatility > 0.01 ? 'text-amber-400' : 'text-slate-400'}`} />
                <div className={`text-2xl font-bold relative z-10 ${volatility > 0.01 ? 'text-amber-100' : 'text-white'}`}>
                  {(volatility * 100).toFixed(2)}%
                </div>
                <div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider relative z-10">Volatility</div>
             </Card>
          </div>
        </div>
      </div>

      {/* === MIDDLE ROW: LAYMAN INTEL === */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
         <IntelCard 
            title="News Vibe"
            value={sentiment > 0.2 ? "Positive" : sentiment < -0.2 ? "Negative" : "Neutral"}
            score={sentiment}
            desc="AI reading of global crypto news."
            icon={<Newspaper className="w-5 h-5" />}
            type="sentiment"
         />
         <IntelCard 
            title="Crowd Sentiment"
            value={funding > 1 ? "Greedy" : funding < -1 ? "Fearful" : "Balanced"}
            score={funding}
            desc="Are traders betting up or down?"
            icon={<Users className="w-5 h-5" />}
            type="funding"
         />
         <IntelCard 
            title="Buying Intensity"
            value={orderflow > 0.1 ? "Heavy Buying" : orderflow < -0.1 ? "Heavy Selling" : "Quiet"}
            score={orderflow}
            desc="Real-time volume pressure."
            icon={<TrendingUp className="w-5 h-5" />}
            type="orderflow"
         />
      </div>

      {/* === BOTTOM ROW: CHART === */}
      <Card className={`${glassPanel} bg-slate-900/80 border border-slate-800 shadow-2xl`}>
        <CardHeader className="pb-2 border-b border-white/5 flex flex-row items-center justify-between">
          <CardTitle className="text-sm font-medium text-slate-300 flex items-center gap-2">
            <Activity className="w-4 h-4 text-indigo-400" />
            Live Price Projection
          </CardTitle>
          <div className="flex gap-2">
             <span className="flex items-center gap-1 text-[10px] text-slate-500 uppercase font-bold">
               <span className="w-2 h-2 rounded-full bg-white"></span> Past
             </span>
             <span className="flex items-center gap-1 text-[10px] text-slate-500 uppercase font-bold">
               <span className={`w-2 h-2 rounded-full ${trendStyles.bg.replace('/10', '')}`}></span> Forecast
             </span>
          </div>
        </CardHeader>
        <CardContent className="h-[350px] w-full pt-4">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={trendStyles.stroke} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={trendStyles.stroke} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis 
                dataKey="time" 
                stroke="#475569" 
                fontSize={10} 
                tickLine={false} 
                axisLine={false} 
              />
              <YAxis 
                domain={['auto', 'auto']} 
                stroke="#475569" 
                fontSize={10} 
                tickLine={false} 
                axisLine={false} 
                tickFormatter={(val) => `$${val.toLocaleString(undefined, {maximumFractionDigits:0})}`}
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', color: '#f1f5f9', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)' }}
                itemStyle={{ color: '#e2e8f0' }}
                labelStyle={{ color: '#94a3b8', fontSize: '12px', marginBottom: '4px' }}
              />
              
              <ReferenceLine y={currentPrice} stroke="#64748b" strokeDasharray="3 3" label={{ position: 'insideBottomRight', value: 'NOW', fill: '#64748b', fontSize: 10, fontWeight: 'bold' }} />
              <ReferenceLine y={targetPrice} stroke={trendStyles.stroke} strokeWidth={1} label={{ position: 'insideTopRight', value: 'TARGET', fill: trendStyles.stroke, fontSize: 10, fontWeight: 'bold' }} />

              <Area 
                type="monotone" 
                dataKey="price" 
                stroke={trendStyles.stroke}
                strokeWidth={3}
                fillOpacity={1} 
                fill="url(#colorPrice)" 
                animationDuration={1500}
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

    </div>
  );
};

/* --- Intel Card Component with Dynamic Shadow Logic --- */
const IntelCard: React.FC<{ 
  title: string; 
  value: string; 
  score: number; 
  desc: string; 
  icon: any; 
  type: "sentiment" | "funding" | "orderflow"
}> = ({ title, value, score, desc, icon, type }) => {
  
  // Heuristic color logic with Shadow Boost
  let colorClass = "text-slate-400 bg-slate-500/5 border-slate-700";
  let shadowClass = "shadow-none";
  let iconBg = "bg-slate-800";
  
  if (type === "sentiment" || type === "orderflow") {
      if (score > 0.1) {
        colorClass = "text-emerald-400 bg-emerald-500/10 border-emerald-500/40";
        shadowClass = "shadow-[0_0_20px_-5px_rgba(16,185,129,0.2)]"; // Green Shadow
        iconBg = "bg-emerald-500/20";
      }
      else if (score < -0.1) {
        colorClass = "text-rose-400 bg-rose-500/10 border-rose-500/40";
        shadowClass = "shadow-[0_0_20px_-5px_rgba(244,63,94,0.2)]"; // Red Shadow
        iconBg = "bg-rose-500/20";
      }
  } else if (type === "funding") {
      // Funding > 1.5 is greedy (warning/amber), < -1.5 is fear (bullish reversal/green)
      if (score > 1.5) {
        colorClass = "text-amber-400 bg-amber-500/10 border-amber-500/40";
        shadowClass = "shadow-[0_0_20px_-5px_rgba(251,191,36,0.2)]"; // Amber Shadow
        iconBg = "bg-amber-500/20";
      }
      else if (score < -1.5) {
        colorClass = "text-emerald-400 bg-emerald-500/10 border-emerald-500/40";
        shadowClass = "shadow-[0_0_20px_-5px_rgba(16,185,129,0.2)]"; // Green Shadow
        iconBg = "bg-emerald-500/20";
      }
  }

  return (
    <Card className={`group border transition-all duration-500 hover:scale-[1.02] hover:-translate-y-1 ${colorClass.split(' ')[2]} ${colorClass.split(' ')[1]} ${shadowClass}`}>
      <CardContent className="p-5">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-xl border border-white/5 shadow-sm transition-colors ${iconBg}`}>
              {React.cloneElement(icon, { className: `w-5 h-5 ${colorClass.split(' ')[0]}` })}
            </div>
            <div>
              <div className="text-sm font-bold text-slate-200">{title}</div>
              <div className="text-[11px] text-slate-500 font-medium">{desc}</div>
            </div>
          </div>
        </div>
        
        <div className="flex items-end justify-between">
          <div className="text-xl font-bold text-white">{value}</div>
          <div className={`px-2.5 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wide border ${colorClass}`}>
            {type === "funding" ? `${score.toFixed(2)} bps` : type === "orderflow" ? `${score.toFixed(2)} tilt` : `${score.toFixed(2)} score`}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default DashboardTab;