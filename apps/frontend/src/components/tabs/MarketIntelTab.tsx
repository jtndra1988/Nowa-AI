"use client";

import React from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  Badge,
} from "../layout/AppShell";

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ReferenceLine,
  Cell,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Anchor,
  Zap,
  Info,
  Gauge,
  Target,
  Coins,
  ArrowRightLeft,
  BarChart3,
} from "lucide-react";
import type {
  SymbolCode,
  MarketMode,
  MarketIntel as MarketIntelResponse,
} from "@/lib/api";
import { getMarketIntel } from "@/lib/api";

// --- Types ---
type IntelSeriesPoint = {
  t?: any;
  time?: any;
  s?: number;
  score?: number;
  v?: number;
  value?: number;
  active?: number;
};

type CorItem = { name: string; val: number };

// --- Helpers ---
const safeNum = (v: any, d = 0) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
};

const last = <T,>(arr?: T[]): T | undefined =>
  arr && arr.length ? arr[arr.length - 1] : undefined;

// --- Sub-Components ---

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-900/90 border border-slate-700 p-3 rounded-lg shadow-xl backdrop-blur-md z-50">
        <p className="text-xs text-slate-400 mb-1">{label ?? "Data Point"}</p>
        {payload.map((p: any, i: number) => (
          <div key={i} className="flex items-center gap-2 text-sm font-medium">
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: p.color }}
            />
            <span className="text-slate-200">
              {p.name}: {p.value}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

const InsightBadge = ({
  type,
}: {
  type: "positive" | "negative" | "neutral" | "warning";
}) => {
  if (type === "positive")
    return (
      <Badge className="bg-green-500/20 text-green-400 border-green-500/50">
        FAVORABLE
      </Badge>
    );
  if (type === "negative")
    return (
      <Badge className="bg-red-500/20 text-red-400 border-red-500/50">
        NEGATIVE
      </Badge>
    );
  if (type === "warning")
    return (
      <Badge className="bg-orange-500/20 text-orange-400 border-orange-500/50">
        CAUTION
      </Badge>
    );
  return (
    <Badge className="bg-slate-500/20 text-slate-400 border-slate-500/50">
      STABLE
    </Badge>
  );
};

const InsightCard: React.FC<{
  title: string;
  icon: React.ReactNode;
  bullets: string[];
  signal: "positive" | "negative" | "neutral" | "warning";
  colorClass?: string;
}> = ({
  title,
  icon,
  bullets,
  signal,
  colorClass = "border-l-blue-500/50",
}) => (
  <Card className={`h-full border-l-4 ${colorClass} bg-slate-900/40`}>
    <CardHeader className="pb-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-slate-800/50 rounded-md text-slate-200">
            {icon}
          </div>
          <CardTitle className="text-base">{title}</CardTitle>
        </div>
        <InsightBadge type={signal} />
      </div>
    </CardHeader>
    <CardContent className="text-sm space-y-2">
      {bullets.map((b, i) => (
        <div key={i} className="flex items-start gap-2 text-slate-300">
          <span className="mt-1.5 w-1 h-1 rounded-full bg-slate-500 shrink-0" />
          <span>{b}</span>
        </div>
      ))}
    </CardContent>
  </Card>
);

const KpiCard = ({
  label,
  value,
  subValue,
  trend,
  color = "text-slate-100",
}: {
  label: string;
  value: string;
  subValue?: string;
  trend?: "up" | "down" | "neutral";
  color?: string;
}) => (
  <div className="bg-slate-900/50 border border-white/5 rounded-xl p-4 flex flex-col justify-between relative overflow-hidden group transition-all hover:border-white/10">
    <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
    <div className="z-10">
      <span className="text-xs uppercase tracking-wider text-slate-500 font-semibold">
        {label}
      </span>
      <div className={`text-2xl font-bold mt-1 ${color}`}>{value}</div>
      {subValue && (
        <div className="text-xs text-slate-400 mt-1">{subValue}</div>
      )}
    </div>
    {trend && (
      <div className="absolute top-4 right-4">
        {trend === "up" ? (
          <TrendingUp className="w-5 h-5 text-green-500" />
        ) : trend === "down" ? (
          <TrendingDown className="w-5 h-5 text-red-500" />
        ) : (
          <Activity className="w-5 h-5 text-slate-500" />
        )}
      </div>
    )}
  </div>
);

// --- Main Component ---

export const MarketIntelTab: React.FC<{
  symbol: SymbolCode;
  mode: MarketMode;
  exchange?: string;
}> = ({ symbol, mode, exchange = "Binance" }) => {
  const [intel, setIntel] = React.useState<MarketIntelResponse | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);

    // FIXED: api.ts only accepts symbol, removed 'mode' argument
    getMarketIntel(symbol)
      .then((data) => {
        if (!alive) return;
        setIntel(data);
      })
      .catch((err) => {
        console.error("Failed to load market intel", err);
        if (!alive) return;
        setIntel(null);
        setError(err?.message || "Failed to load intel");
      })
      .finally(() => {
        if (!alive) return;
        setLoading(false);
      });

    return () => {
      alive = false;
    };
  }, [symbol]);

  const pro: any = intel || {};

  // Clean the symbol for display (remove -PERP, -USD etc)
  const displaySymbol = symbol.split("-")[0];

  // Data Extraction
  const sentimentArr: IntelSeriesPoint[] = pro.sentimentHistory ?? [];
  const chainArr: IntelSeriesPoint[] =
    pro.onChainHistory ?? pro.onchainHistory ?? [];
  const cvdArr: IntelSeriesPoint[] = pro.cvdHistory ?? pro.cvd ?? [];

  // Latest Values & Safe Defaults
  const activeAddr = safeNum(
    chainArr.length
      ? chainArr[chainArr.length - 1].active ??
          chainArr[chainArr.length - 1].v
      : 0,
    0
  );
  const cvdDelta = safeNum(
    cvdArr.length
      ? cvdArr[cvdArr.length - 1].v ?? cvdArr[cvdArr.length - 1]
      : 0,
    0
  );
  const latestScore = safeNum(
    sentimentArr.length
      ? sentimentArr[sentimentArr.length - 1].score ??
          sentimentArr[sentimentArr.length - 1].s
      : 0,
    0
  );

  // Mock RSI Calculation (or extraction if API provided it later)
  const derivedRsi = 50 + latestScore * 20;

  // Logic & Derived State
  const regimeText =
    pro.radar?.regime ??
    (latestScore > 0.25
      ? "MOMENTUM"
      : latestScore < -0.25
      ? "MEAN-REVERT"
      : "BALANCED");

      const correlations: CorItem[] = Array.isArray(pro.correlations)
    ? pro.correlations
    : [];

  // Radar Chart Logic normalization (Spot centric)
  const trendScore =
    latestScore > 0 ? 75 + latestScore * 25 : 25 + latestScore * 25;
  const volumeScore = Math.min(Math.abs(cvdDelta / 1000), 100);
  const onChainScore = chainArr.length > 0 ? 80 : 40;
  const corrScore =
    correlations.length > 0
      ? Math.min(
          100,
          Math.round(
            (correlations.reduce((acc, c) => acc + Math.abs(c.val), 0) /
              correlations.length) *
              100
          )
        )
      : 50;

  const radarData = [
    { subject: "Sentiment", A: trendScore, fullMark: 100 },
    { subject: "Volume", A: volumeScore, fullMark: 100 },
    { subject: "On-Chain", A: onChainScore, fullMark: 100 },
    { subject: "Correlations", A: corrScore, fullMark: 100 },
    {
      subject: "Momentum",
      A: regimeText === "MOMENTUM" ? 85 : regimeText === "MEAN-REVERT" ? 55 : 65,
      fullMark: 100,
    },
  ];

  // --- Prediction Insights Generation ---

  // 1. Hourly Trend Outlook
  const trendBullets = [
    regimeText === "MOMENTUM"
      ? `${displaySymbol} is showing strong directional momentum.`
      : `${displaySymbol} is in a choppy/ranging zone.`,
    latestScore > 0.2
      ? "Social Sentiment is Bullish."
      : latestScore < -0.2
      ? "Social Sentiment is Bearish."
      : "Social Sentiment is Neutral.",
  ];
  let trendSignal: "positive" | "negative" | "neutral" | "warning" = "neutral";
  if (regimeText === "MOMENTUM") trendSignal = "positive";
  else if (regimeText === "BALANCED") trendSignal = "neutral";
  else trendSignal = "warning";

  // 2. Spot Market Condition
  const conditionBullets = [
    derivedRsi > 70
      ? `Market is Overbought (High RSI). Risk of pullback.`
      : derivedRsi < 30
      ? `Market is Oversold (Low RSI). Potential bounce.`
      : `RSI is neutral. Room for movement.`,
    cvdDelta > 0
      ? "Spot buyers are aggressive (Net Buy Volume)."
      : "Spot sellers are aggressive (Net Sell Volume).",
  ];
  let volSignal: "positive" | "negative" | "neutral" | "warning" = "neutral";
  if (derivedRsi > 75 || derivedRsi < 25) volSignal = "warning";
  else volSignal = "positive";

  // 3. Signal Confluence
  const confluenceBullets = [
    cvdDelta > 0 && latestScore > 0
      ? "Volume and Sentiment align Bullish (Strong Signal)."
      : cvdDelta < 0 && latestScore < 0
      ? "Volume and Sentiment align Bearish (Strong Signal)."
      : "Divergence: Volume and Sentiment disagree.",
    chainArr.length > 0
      ? `On-chain activity confirms trend strength.`
      : "Low on-chain activity; weak conviction.",
  ];
  let confSignal: "positive" | "negative" | "neutral" | "warning" = "neutral";
  const directionMatch =
    (cvdDelta > 0 && latestScore > 0) ||
    (cvdDelta < 0 && latestScore < 0);
  if (directionMatch) confSignal = "positive";
  else confSignal = "warning";

    const hasAnyData =
    sentimentArr.length > 0 ||
    chainArr.length > 0 ||
    cvdArr.length > 0 ||
    correlations.length > 0 ||
    !!(
      intel &&
      ((intel.ivHistory && intel.ivHistory.length > 0) ||
        (intel.fundingHistory && intel.fundingHistory.length > 0) ||
        (intel.oiHistory && intel.oiHistory.length > 0))
    );

  if (loading) {
    return (
      <Card className="border border-dashed border-slate-700 bg-slate-900/40">
        <CardHeader>
          <CardTitle className="text-sm">Loading Market Intel…</CardTitle>
          <CardDescription className="text-xs">
            Pulling sentiment, volume and on-chain context from the backend.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="border border-rose-500/40 bg-rose-950/40">
        <CardHeader>
          <CardTitle className="text-sm text-rose-200">
            Failed to load intel
          </CardTitle>
          <CardDescription className="text-xs text-rose-300/80">
            {error}
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }




  return (
    <div className="flex flex-col gap-6 pb-10 animate-in fade-in duration-500">
      {/* --- ASSET HEADER --- */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between bg-slate-900/50 p-4 rounded-xl border border-white/5 gap-4">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-blue-600/20 rounded-full flex items-center justify-center border border-blue-500/30 text-blue-400">
            <Coins className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              {displaySymbol} Market Intelligence
            </h2>
            <p className="text-sm text-slate-400 flex items-center gap-2">
              <ArrowRightLeft className="w-3 h-3" /> {exchange} Spot Market ·{" "}
              {mode.toUpperCase()}
            </p>
          </div>
        </div>

        {/* Quick Summary Text */}
        <div className="text-right hidden sm:block">
          <div className="text-xs text-slate-500 uppercase font-semibold">
            Primary Regime
          </div>
          <div
            className={`text-lg font-bold ${
              regimeText === "MOMENTUM"
                ? "text-green-400"
                : regimeText === "MEAN-REVERT"
                ? "text-amber-400"
                : "text-purple-400"
            }`}
          >
            {regimeText}
          </div>
        </div>
      </div>

      {/* --- KPI STATISTICS (SPOT ONLY) --- */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard
          label={`${displaySymbol} Sentiment`}
          value={latestScore.toFixed(2)}
          subValue="AI Social Score"
          trend={latestScore > 0 ? "up" : latestScore < 0 ? "down" : "neutral"}
          color={
            latestScore > 0
              ? "text-green-400"
              : latestScore < 0
              ? "text-red-400"
              : "text-slate-100"
          }
        />
        <KpiCard
          label="Spot Buying Pressure"
          value={cvdDelta.toFixed(0)}
          subValue="Net CVD (24h)"
          color={cvdDelta > 0 ? "text-green-400" : "text-red-400"}
          trend={cvdDelta > 0 ? "up" : cvdDelta < 0 ? "down" : "neutral"}
        />
        <KpiCard
          label="Relative Strength"
          value={derivedRsi.toFixed(0)}
          subValue={
            derivedRsi > 70
              ? "Overbought"
              : derivedRsi < 30
              ? "Oversold"
              : "Neutral"
          }
          trend={derivedRsi > 50 ? "up" : "down"}
        />
        <KpiCard
          label="Network Activity"
          value={activeAddr > 0 ? activeAddr.toLocaleString() : "Normal"}
          subValue="Active Addresses"
          trend="neutral"
        />
      </div>

      {/* --- MAIN CHARTS AREA --- */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* LEFT: Sentiment Chart */}
        <Card className="lg:col-span-2 border-slate-800 bg-slate-900/40 backdrop-blur">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-blue-500" />
              {displaySymbol} Sentiment Flow
            </CardTitle>
            <CardDescription>
              Correlation between AI Sentiment and price movement.
            </CardDescription>
          </CardHeader>
          <CardContent className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={sentimentArr.map((d, i) => ({
                  t: d.t ?? d.time ?? i,
                  s: safeNum(d.score ?? d.s, 0),
                }))}
              >
                <defs>
                  <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                    <stop
                      offset="5%"
                      stopColor="#3b82f6"
                      stopOpacity={0.3}
                    />
                    <stop
                      offset="95%"
                      stopColor="#3b82f6"
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="t" hide />
                <YAxis domain={[-1, 1]} hide />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={0} stroke="#475569" strokeDasharray="3 3" />
                <Area
                  type="monotone"
                  dataKey="s"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#colorScore)"
                  name="Sentiment Score"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* RIGHT: Spot Radar */}
        <Card className="border-slate-800 bg-slate-900/40 backdrop-blur">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="w-4 h-4 text-purple-500" />
              Spot Dynamics
            </CardTitle>
          </CardHeader>
          <CardContent className="h-[300px] flex items-center justify-center relative">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart
                cx="50%"
                cy="50%"
                outerRadius="70%"
                data={radarData}
              >
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis
                  dataKey="subject"
                  tick={{ fill: "#94a3b8", fontSize: 10 }}
                />
                <PolarRadiusAxis
                  angle={30}
                  domain={[0, 100]}
                  tick={false}
                  axisLine={false}
                />
                <Radar
                  name="Metric Score"
                  dataKey="A"
                  stroke="#8b5cf6"
                  strokeWidth={2}
                  fill="#8b5cf6"
                  fillOpacity={0.3}
                />
                <Tooltip content={<CustomTooltip />} />
              </RadarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* --- SECONDARY ROW: CORRELATIONS & ON-CHAIN --- */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* On-Chain Activity */}
        <Card className="lg:col-span-2 border-slate-800 bg-slate-900/40">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Anchor className="w-4 h-4 text-indigo-500" />
              On-Chain Activity
            </CardTitle>
          </CardHeader>
          <CardContent className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={chainArr.map((d, i) => ({
                  t: d.t ?? d.time ?? i,
                  v: safeNum(d.active ?? d.v ?? d.value, 0),
                }))}
              >
                <defs>
                  <linearGradient id="colorChain" x1="0" y1="0" x2="0" y2="1">
                    <stop
                      offset="5%"
                      stopColor="#6366f1"
                      stopOpacity={0.3}
                    />
                    <stop
                      offset="95%"
                      stopColor="#6366f1"
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="t" hide />
                <YAxis hide />
                <Tooltip content={<CustomTooltip />} />
                <Area
                  type="monotone"
                  dataKey="v"
                  stroke="#6366f1"
                  fill="url(#colorChain)"
                  strokeWidth={2}
                  name="Active Addresses"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Correlations */}
        <Card className="border-slate-800 bg-slate-900/40">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-amber-500" />
              Correlations
            </CardTitle>
          </CardHeader>
          <CardContent className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={correlations}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  horizontal
                  vertical={false}
                  stroke="#1e293b"
                />
                <XAxis type="number" hide domain={[-1, 1]} />
                <YAxis
                  dataKey="name"
                  type="category"
                  width={40}
                  tick={{ fill: "#94a3b8", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  cursor={{ fill: "#1e293b" }}
                  content={<CustomTooltip />}
                />
                <ReferenceLine x={0} stroke="#475569" />
                <Bar dataKey="val" radius={[0, 4, 4, 0]} barSize={12}>
                  {correlations.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.val > 0 ? "#10b981" : "#ef4444"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* --- PREDICTION INSIGHTS --- */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <InsightCard
          title={`${displaySymbol} Trend (1H)`}
          icon={<Gauge className="w-5 h-5" />}
          bullets={trendBullets}
          signal={trendSignal}
          colorClass="border-l-indigo-500"
        />
        <InsightCard
          title="Spot Conditions"
          icon={<BarChart3 className="w-5 h-5" />}
          bullets={conditionBullets}
          signal={volSignal}
          colorClass="border-l-amber-500"
        />
        <InsightCard
          title="Signal Confluence"
          icon={<Target className="w-5 h-5" />}
          bullets={confluenceBullets}
          signal={confSignal}
          colorClass="border-l-teal-500"
        />
      </div>

      {/* --- FOOTER NOTE --- */}
      <div className="flex items-start gap-2 p-4 bg-blue-900/20 border border-blue-800/50 rounded-lg text-xs text-blue-200">
        <Info className="w-4 h-4 mt-0.5 shrink-0" />
        <p>
          This intelligence panel analyzes {displaySymbol} spot conditions to
          provide context for the hourly price prediction model. Use it to
          understand whether the environment is momentum, mean-reversion, or
          mixed before acting on signals.
        </p>
      </div>

      {!hasAnyData && (
        <Card className="border border-dashed border-slate-700 bg-slate-900/40">
          <CardHeader>
            <CardTitle className="text-sm">
              Waiting for live intel…
            </CardTitle>
            <CardDescription className="text-xs">
              No historical intel received yet for this symbol. As soon as
              upstream collectors publish data, this panel will populate
              automatically.
            </CardDescription>
          </CardHeader>
        </Card>
      )}
    </div>
  );
};

export default MarketIntelTab;