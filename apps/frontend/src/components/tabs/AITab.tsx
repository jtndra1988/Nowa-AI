"use client";

import React, { useMemo, useState } from "react";
import { MarketMode, SymbolCode } from "@/lib/api";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  glassPanel,
  faintText,
  makeSeeded,
} from "../layout/AppShell";

type StrategiesTabProps = {
  symbol: SymbolCode;
  mode: MarketMode;
  exchange: string;
};

type BrainId =
  | "tft"
  | "tcn"
  | "xgb"
  | "decisionnet"
  | "options"
  | "macro";

type Brain = {
  id: BrainId;
  label: string;
  role: string;
  tagline: string;
  detail: string;
};

const BRAINS: Brain[] = [
  {
    id: "tft",
    label: "TFT – The Visionary",
    role: "Long-Term Trend & Context",
    tagline: "Spots long-term trends before the crowd.",
    detail:
      "Reads extended price history, volatility clusters, and macro context to detect real structural trends vs short-term noise.",
  },
  {
    id: "tcn",
    label: "TCN – The Reflex",
    role: "Short-Term Flow & Volatility",
    tagline: "Responds instantly to sharp movements.",
    detail:
      "Tracks short-term candles and microstructure to react to breakouts, fakeouts and intraday reversals in real time.",
  },
  {
    id: "xgb",
    label: "XGBoost – The Analyst",
    role: "Quant Logic & Sanity Check",
    tagline: "Applies data logic to every forecast.",
    detail:
      "Uses indicators, spreads, funding, basis and skew to validate or challenge the neural models and reduce false signals.",
  },
  {
    id: "options",
    label: "Options Expert – The Crowd Psychologist",
    role: "Volatility & Positioning",
    tagline: "Reads fear, greed & hedging flows.",
    detail:
      "Watches implied volatility, skew and term structure to see how professional traders are pricing risk and protection.",
  },
  {
    id: "macro",
    label: "Macro + On-Chain – The Economist",
    role: "Environment & Liquidity",
    tagline: "Understands the risk backdrop.",
    detail:
      "Monitors on-chain flows, liquidity, stablecoins, reserves and macro indices (DXY, SPX, rates) to tell Nowa when to size up or de-risk.",
  },
  {
    id: "decisionnet",
    label: "DecisionNet – The Judge",
    role: "Final Trade Gate",
    tagline: "Makes the final call after all agree.",
    detail:
      "Learns from historical outcomes. Only approves a Long / Short / Flat when the full panel and risk checks line up.",
  },
];

type Metric = {
  label: string;
  base: number;
  suffix?: string;
};

type Feature = { name: string; w: number };

type StrategyConfig = {
  id: BrainId;
  label: string;
  subtitle: string;
  latencyMs?: number;
  horizonMin?: number;
  windowDays?: number;
  lr?: number;
  riskCap?: number;
  regimes: string[];
  metrics?: Metric[];
  features?: Feature[];
  compact?: boolean; // for specialists / judge
};

type StrategyState = {
  id: BrainId;
  enabled: boolean;
  weight: number; // 0-100
};

const metricColor = (label: string, value: number): string => {
  if (label === "MaxDD") {
    if (value >= -10) return "text-emerald-300";
    if (value >= -16) return "text-sky-300";
    return "text-rose-300";
  }
  if (label === "Sharpe") {
    if (value >= 2) return "text-emerald-300";
    if (value >= 1.3) return "text-sky-300";
    return "text-amber-300";
  }
  if (["Precision", "Recall", "F1", "AUC"].includes(label)) {
    if (value >= 0.65) return "text-emerald-300";
    if (value >= 0.55) return "text-sky-300";
    return "text-amber-300";
  }
  return "text-slate-100";
};

const fmtMetric = (m: Metric, v: number) => {
  if (m.label === "MaxDD") return `${v.toFixed(1)}${m.suffix ?? "%"}`;
  if (m.label === "Sharpe") return v.toFixed(2);
  if (["Precision", "Recall", "F1", "AUC"].includes(m.label))
    return v.toFixed(2);
  return `${v.toFixed(2)}${m.suffix ?? ""}`;
};

const Toggle: React.FC<{ checked: boolean; onChange: (v: boolean) => void }> = ({
  checked,
  onChange,
}) => (
  <button
    type="button"
    onClick={() => onChange(!checked)}
    className={`relative inline-flex h-5 w-9 items-center rounded-full transition
      ${checked ? "bg-emerald-500/80" : "bg-slate-600"}`}
  >
    <span
      className={`inline-block h-4 w-4 transform rounded-full bg-slate-950 shadow transition
        ${checked ? "translate-x-4" : "translate-x-1"}`}
    />
  </button>
);

const WeightSlider: React.FC<{ value: number; onChange: (v: number) => void }> = ({
  value,
  onChange,
}) => (
  <input
    type="range"
    min={0}
    max={100}
    value={value}
    onChange={(e) => onChange(Number(e.target.value))}
    className="w-full accent-emerald-400 bg-transparent"
  />
);

const StrategiesTab: React.FC<StrategiesTabProps> = ({ symbol, mode, exchange }) => {
  const seeded = useMemo(
    () => makeSeeded(`ai-${symbol}-${mode}-${exchange}`),
    [symbol, mode, exchange]
  );

  const rand = () => seeded();

  const [selectedBrain, setSelectedBrain] = useState<BrainId>("tft");

  // configs for all six specialists
  const configs: StrategyConfig[] = useMemo(
    () => [
      {
        id: "tft",
        label: "TFT · Macro Trend & Carry",
        subtitle: "Understands deep trend & regime.",
        latencyMs: 11,
        horizonMin: 5,
        windowDays: 120,
        lr: 0.0008,
        riskCap: 0.6,
        regimes: ["trend", "range", "low-vol"],
        metrics: [
          { label: "Precision", base: 0.62 },
          { label: "Recall", base: 0.65 },
          { label: "F1", base: 0.67 },
          { label: "AUC", base: 0.8 },
          { label: "Sharpe", base: 2.19 },
          { label: "MaxDD", base: -13.0, suffix: "%" },
        ],
        features: [
          { name: "OrderBookImbalance(1m)", w: 0.28 },
          { name: "FundingRateDelta(8h)", w: 0.22 },
          { name: "OnChainFlows(24h)", w: 0.19 },
          { name: "RealizedVol(30m)", w: 0.18 },
          { name: "BTC.D / ETH.D", w: 0.13 },
        ],
      },
      {
        id: "tcn",
        label: "TCN · Short-term Flow & Reversion",
        subtitle: "Fast reflex on microstructure.",
        latencyMs: 7,
        horizonMin: 3,
        windowDays: 90,
        lr: 0.0001,
        riskCap: 0.5,
        regimes: ["trend", "range", "high-vol", "low-vol"],
        metrics: [
          { label: "Precision", base: 0.63 },
          { label: "Recall", base: 0.62 },
          { label: "F1", base: 0.65 },
          { label: "AUC", base: 0.77 },
          { label: "Sharpe", base: 1.65 },
          { label: "MaxDD", base: -16.0, suffix: "%" },
        ],
        features: [
          { name: "ATR(14)", w: 0.28 },
          { name: "RSI(7)", w: 0.23 },
          { name: "OBV", w: 0.20 },
          { name: "OI Change(5m)", w: 0.16 },
          { name: "CVD(1m)", w: 0.13 },
        ],
      },
      {
        id: "xgb",
        label: "XGB · Microstructure & Skew",
        subtitle: "Structured quant logic.",
        latencyMs: 2,
        horizonMin: 5,
        windowDays: 60,
        lr: 0.0003,
        riskCap: 0.4,
        regimes: ["range", "low-vol"],
        metrics: [
          { label: "Precision", base: 0.6 },
          { label: "Recall", base: 0.59 },
          { label: "F1", base: 0.59 },
          { label: "AUC", base: 0.73 },
          { label: "Sharpe", base: 1.44 },
          { label: "MaxDD", base: -17.0, suffix: "%" },
        ],
        features: [
          { name: "LagRet(1–6)", w: 0.33 },
          { name: "VWAP Dist", w: 0.25 },
          { name: "RollCorr(15m)", w: 0.19 },
          { name: "Skew(30m)", w: 0.12 },
          { name: "Depth Imb. 10 levels", w: 0.11 },
        ],
      },
      // Specialists / judge: compact cards with controls
      {
        id: "options",
        label: "Options Expert",
        subtitle: "Uses vol & skew to read crowd behaviour.",
        regimes: ["high-vol", "event-risk"],
        compact: true,
      },
      {
        id: "macro",
        label: "Macro + On-Chain",
        subtitle: "Controls exposure based on liquidity & risk regime.",
        regimes: ["risk-on", "risk-off", "stress"],
        compact: true,
      },
      {
        id: "decisionnet",
        label: "DecisionNet (Judge)",
        subtitle: "Final gate: needs alignment & risk conditions.",
        regimes: ["all"],
        compact: true,
      },
    ],
    []
  );

  const [strategies, setStrategies] = useState<StrategyState[]>(
    configs.map((c) => ({
      id: c.id,
      enabled: true,
      // base models full weight, specialists a bit lower by default
      weight:
        c.id === "options" || c.id === "macro" ? 60 : c.id === "decisionnet" ? 80 : 100,
    }))
  );

  const [ensembleMethod, setEnsembleMethod] = useState<
    "weighted" | "stacking" | "bayesian"
  >("bayesian");
  const [kFold, setKFold] = useState(5);
  const [walkForwardOOS, setWalkForwardOOS] = useState(30);
  const [autoTuneEnabled, setAutoTuneEnabled] = useState(true);
  const [cadenceHours, setCadenceHours] = useState(24);
  const [maxTrials, setMaxTrials] = useState(30);

  const updateStrategy = (id: BrainId, patch: Partial<StrategyState>) =>
    setStrategies((prev) =>
      prev.map((s) => (s.id === id ? { ...s, ...patch } : s))
    );

  const jitterMetric = (base: number, label: string): number => {
    const j =
      (rand() - 0.5) *
      (label === "Sharpe" ? 0.25 : label === "MaxDD" ? 1.2 : 0.03);
    return base + j;
  };

  const hybridEdge = useMemo(
    () => 0.7 + (rand() - 0.5) * 0.08,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [symbol, mode, exchange]
  );
  const alignmentScore = useMemo(
    () => 0.74 + (rand() - 0.5) * 0.06,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [symbol, mode, exchange]
  );

  const activeWeightSum = strategies
    .filter((s) => s.enabled)
    .reduce((sum, s) => sum + s.weight / 100, 0)
    .toFixed(2);

  return (
    <div className="relative flex flex-col gap-4 text-slate-50 overflow-x-hidden">
      {/* Background glow (clamped so no horizontal scroll) */}
      <div className="pointer-events-none absolute inset-0 -z-10 bg-gradient-to-b from-[#020817] via-black to-[#020817]" />
      <div className="pointer-events-none absolute right-0 top-10 h-32 w-32 rounded-full bg-emerald-500/10 blur-3xl" />
      <div className="pointer-events-none absolute left-0 bottom-10 h-32 w-32 rounded-full bg-sky-500/10 blur-3xl" />

      {/* HERO + PIPELINE + SNAPSHOT */}
      <div className="grid grid-cols-12 gap-4">
        {/* Left: hero & flow */}
        <section className="col-span-12 lg:col-span-8 rounded-2xl border border-emerald-500/15 bg-slate-950/70 backdrop-blur-2xl shadow-[0_0_40px_rgba(0,0,0,0.9)] p-4 relative overflow-hidden">
          <div className="pointer-events-none absolute inset-x-4 top-0 h-px bg-gradient-to-r from-transparent via-emerald-400/50 to-transparent" />
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1 min-w-0">
              <div className="text-[9px] uppercase tracking-[0.18em] text-emerald-400/90">
                Nowa AI Hybrid Engine
              </div>
              <h1 className="text-sm sm:text-base font-semibold mt-1 whitespace-nowrap overflow-hidden text-ellipsis">
                Six AI specialists, one clean decision.
              </h1>
              <p className="mt-2 text-[10px] text-slate-300">
                Nowa works like a focused trading team: each AI model has a
                role, and no trade is suggested until the team aligns and the
                Judge is satisfied.
              </p>
            </div>
            <div className="flex flex-col items-end gap-1 shrink-0">
              <div className="flex items-center gap-1 text-[9px] text-emerald-300">
                <span className="inline-flex h-2 w-2 rounded-full bg-emerald-400 animate-ping" />
                <span className="inline-flex h-2 w-2 rounded-full bg-emerald-300" />
                AI Logic (Demo)
              </div>
              <div className="px-2 py-1 rounded-full bg-slate-950/90 border border-slate-700/70 text-[9px]">
                {symbol} · {mode} · {exchange}
              </div>
            </div>
          </div>

          {/* Flow rail */}
          <div className="mt-3">
            <div className="text-[9px] text-slate-500 mb-1">
              Data → TFT → TCN → XGBoost → Options → Macro → DecisionNet → Final Signal
            </div>
            <div className="relative w-full h-6 flex items-center">
              <div className="absolute inset-x-0 mx-auto h-[2px] rounded-full bg-gradient-to-r from-slate-800 via-emerald-500/40 to-sky-500/40" />
              <div className="absolute inset-y-0 left-0 w-1/4 h-[2px] rounded-full bg-gradient-to-r from-transparent via-emerald-300 to-transparent animate-[pulse_1.8s_ease-in-out_infinite]" />
              <div className="relative w-full flex justify-between items-center px-1">
                {["TFT", "TCN", "XGB", "OPT", "MACRO", "JUDGE", "FINAL"].map(
                  (label, idx) => (
                    <div
                      key={label}
                      className="flex flex-col items-center gap-0.5"
                    >
                      <div
                        className={`w-2 h-2 rounded-full transition-all ${
                          idx === 6
                            ? "bg-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.9)]"
                            : "bg-slate-500/90"
                        }`}
                      />
                      <div className="text-[7px] text-slate-400">
                        {label}
                      </div>
                    </div>
                  )
                )}
              </div>
            </div>

            {/* Brain pills */}
            <div className="mt-2 flex flex-wrap gap-1">
              {BRAINS.map((b) => (
                <button
                  key={b.id}
                  type="button"
                  onMouseEnter={() => setSelectedBrain(b.id)}
                  onClick={() => setSelectedBrain(b.id)}
                  className="focus:outline-none"
                >
                  <div
                    className={`px-2 py-1 rounded-full text-[9px] border transition-all ${
                      selectedBrain === b.id
                        ? "border-emerald-400/80 bg-emerald-500/10 text-emerald-300 shadow-[0_0_10px_rgba(16,185,129,0.5)]"
                        : "border-slate-700/70 bg-slate-950/80 text-slate-300 hover:border-emerald-400/40 hover:text-emerald-200"
                    }`}
                  >
                    {b.label.split("–")[0].trim()}
                  </div>
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* Right: snapshot + selected brain */}
        <section className="col-span-12 lg:col-span-4 rounded-2xl border border-slate-700/60 bg-slate-950/85 backdrop-blur-2xl shadow-[0_0_30px_rgba(0,0,0,0.9)] p-3 flex flex-col gap-2">
          <div className="flex items-center justify-between gap-2">
            <div className="text-[9px] text-slate-400">
              Hybrid Conviction Snapshot
            </div>
            <div className="px-2 py-0.5 rounded-full bg-slate-950/95 border border-slate-700/70 text-[8px] text-slate-300">
              Demo only · No live orders
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="rounded-xl bg-slate-950 border border-emerald-500/35 px-2 py-2">
              <div className="text-[8px] text-slate-500 uppercase">
                Alignment Score
              </div>
              <div className="text-lg font-semibold text-emerald-400">
                {(alignmentScore * 100).toFixed(1)}%
              </div>
              <div className={`${faintText} text-[8px]`}>
                How strongly all six agree.
              </div>
              <div className="mt-1 h-1.5 rounded-full bg-slate-900 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-emerald-400 to-sky-400"
                  style={{ width: `${alignmentScore * 100}%` }}
                />
              </div>
            </div>
            <div className="rounded-xl bg-slate-950 border border-sky-500/30 px-2 py-2">
              <div className="text-[8px] text-slate-500 uppercase">
                Hybrid Edge (Simulated)
              </div>
              <div className="text-lg font-semibold text-sky-400">
                {(hybridEdge * 100).toFixed(1)}%
              </div>
              <div className={`${faintText} text-[8px]`}>
                Illustrative win-rate when aligned.
              </div>
              <div className="mt-1 h-1.5 rounded-full bg-slate-900 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-sky-400 to-emerald-300"
                  style={{ width: `${hybridEdge * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* Selected brain detail */}
          <div className="mt-1 rounded-xl bg-slate-950/95 border border-slate-700/70 px-2 py-2">
            <div className="text-[9px] text-emerald-300 font-semibold">
              {BRAINS.find((b) => b.id === selectedBrain)?.label}
            </div>
            <div className="text-[8px] text-sky-300">
              {BRAINS.find((b) => b.id === selectedBrain)?.role}
            </div>
            <div className="text-[8px] text-emerald-200 mt-0.5">
              {BRAINS.find((b) => b.id === selectedBrain)?.tagline}
            </div>
            <p className="text-[8px] text-slate-300 mt-0.5">
              {BRAINS.find((b) => b.id === selectedBrain)?.detail}
            </p>
          </div>
        </section>
      </div>

      {/* ENSEMBLE & VALIDATION */}
      <Card className={glassPanel}>
        <CardHeader>
          <CardTitle>Ensemble &amp; Validation</CardTitle>
          <CardDescription className="text-[10px]">
            These controls show how Nowa can blend all six models and validate
            them before going live. For the demo everything is local, but the
            structure mirrors your backend hybrid engine.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 lg:grid-cols-3">
            {/* Ensemble method */}
            <div className="space-y-2">
              <div className="text-xs font-medium text-slate-200">
                Ensemble Method
              </div>
              <div className="flex flex-col gap-2 text-[9px]">
                {[
                  {
                    id: "weighted" as const,
                    label: "Weighted",
                    desc: "Manual / learned weights per model.",
                  },
                  {
                    id: "stacking" as const,
                    label: "Stacking (Meta)",
                    desc: "Meta-learner on top of all experts.",
                  },
                  {
                    id: "bayesian" as const,
                    label: "Bayesian Model Avg.",
                    desc: "Weights based on true out-of-sample evidence.",
                  },
                ].map((m) => {
                  const active = ensembleMethod === m.id;
                  return (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => setEnsembleMethod(m.id)}
                      className={`w-full text-left px-3 py-2 rounded-xl border transition
                      ${
                        active
                          ? "bg-emerald-500/15 border-emerald-400/70 text-emerald-100 shadow-[0_0_18px_rgba(16,185,129,0.3)]"
                          : "bg-slate-950/80 border-slate-700/70 text-slate-300 hover:border-emerald-400/40"
                      }`}
                    >
                      <div className="text-[10px] font-semibold">
                        {m.label}
                      </div>
                      <div className={`${faintText} mt-0.5`}>{m.desc}</div>
                    </button>
                  );
                })}
              </div>
              <div className={`${faintText} text-[9px] mt-1`}>
                Active enabled weight sum: {activeWeightSum} (demo).
              </div>
            </div>

            {/* Cross-validation */}
            <div className="space-y-3">
              <div className="text-xs font-medium text-slate-200">
                Cross-Validation
              </div>
              <div className="text-[9px] text-slate-300">
                Time-ordered splits and walk-forward evaluation to avoid
                look-ahead and capture regime shifts.
              </div>
              <div>
                <div className="flex justify-between text-[9px]">
                  <span>K-Fold</span>
                  <span className="text-emerald-300 font-semibold">
                    {kFold}x
                  </span>
                </div>
                <input
                  type="range"
                  min={3}
                  max={8}
                  value={kFold}
                  onChange={(e) => setKFold(Number(e.target.value))}
                  className="w-full accent-emerald-400 bg-transparent"
                />
              </div>
              <div>
                <div className="flex justify-between text-[9px]">
                  <span>Walk-Forward OOS %</span>
                  <span className="text-emerald-300 font-semibold">
                    {walkForwardOOS}%
                  </span>
                </div>
                <input
                  type="range"
                  min={10}
                  max={50}
                  step={5}
                  value={walkForwardOOS}
                  onChange={(e) =>
                    setWalkForwardOOS(Number(e.target.value))
                  }
                  className="w-full accent-emerald-400 bg-transparent"
                />
              </div>
              <div className={`${faintText} text-[9px]`}>
                In real deployment, these guard against overfitting and promote
                stable configs.
              </div>
            </div>

            {/* Auto-tuning */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="text-xs font-medium text-slate-200">
                  Auto-Tuning
                </div>
                <Toggle
                  checked={autoTuneEnabled}
                  onChange={setAutoTuneEnabled}
                />
              </div>
              <div className="text-[9px] text-slate-300">
                Simulates periodic hyperparameter search and model refresh.
              </div>
              <div>
                <div className="flex justify-between text-[9px]">
                  <span>Cadence (hours)</span>
                  <span className="text-emerald-300 font-semibold">
                    {cadenceHours}h
                  </span>
                </div>
                <input
                  type="range"
                  min={6}
                  max={48}
                  step={6}
                  value={cadenceHours}
                  onChange={(e) =>
                    setCadenceHours(Number(e.target.value))
                  }
                  disabled={!autoTuneEnabled}
                  className="w-full accent-emerald-400 bg-transparent disabled:opacity-40"
                />
              </div>
              <div>
                <div className="flex justify-between text-[9px]">
                  <span>Max Trials / cycle</span>
                  <span className="text-emerald-300 font-semibold">
                    {maxTrials}
                  </span>
                </div>
                <input
                  type="range"
                  min={10}
                  max={50}
                  step={5}
                  value={maxTrials}
                  onChange={(e) =>
                    setMaxTrials(Number(e.target.value))
                  }
                  disabled={!autoTuneEnabled}
                  className="w-full accent-emerald-400 bg-transparent disabled:opacity-40"
                />
              </div>
              <div className={`${faintText} text-[9px]`}>
                In production, this would orchestrate retraining jobs against
                real Nowa data.
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* STRATEGY CARDS: all six models */}
      <div className="grid gap-4 xl:grid-cols-3">
        {configs.map((cfg) => {
          const st = strategies.find((s) => s.id === cfg.id)!;
          const metrics = (cfg.metrics || []).map((m) => ({
            meta: m,
            value: jitterMetric(m.base, m.label),
          }));
          const totalFeatureWeight =
            cfg.features?.reduce((s, f) => s + f.w, 0) || 1;

          const isCompact = !!cfg.compact;

          return (
            <Card
              key={cfg.id}
              className={`${glassPanel} bg-slate-950/90 border border-slate-800/80 shadow-[0_0_20px_rgba(0,0,0,0.8)] flex flex-col`}
            >
              <CardHeader className="flex flex-row items-start justify-between gap-3 pb-2">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 text-[8px]">
                    {cfg.latencyMs != null && (
                      <span className="px-2 py-0.5 rounded-full bg-slate-900 text-emerald-300">
                        {cfg.latencyMs}ms
                      </span>
                    )}
                    <span className="px-2 py-0.5 rounded-full bg-slate-900 text-emerald-300">
                      {st.enabled ? "Enabled" : "Disabled"}
                    </span>
                  </div>
                  <CardTitle className="mt-2 text-[12px] text-slate-50 truncate">
                    {cfg.label}
                  </CardTitle>
                  <CardDescription className="text-[8px] text-slate-300">
                    {cfg.subtitle}
                  </CardDescription>
                  {cfg.horizonMin && cfg.windowDays && (
                    <div className="text-[8px] text-slate-500 mt-0.5">
                      Horizon {cfg.horizonMin}m · Window {cfg.windowDays}d
                      {cfg.lr && ` · LR ${cfg.lr}`}
                      {cfg.riskCap && ` · Risk cap ${cfg.riskCap}%`}
                    </div>
                  )}
                </div>
                <div className="flex flex-col items-end gap-1">
                  <Toggle
                    checked={st.enabled}
                    onChange={(v) => updateStrategy(cfg.id, { enabled: v })}
                  />
                </div>
              </CardHeader>

              <CardContent className="flex flex-col gap-2 pt-0">
                {/* Weight control for all six */}
                <div>
                  <div className="flex items-center justify-between text-[8px]">
                    <span className={faintText}>Weight in ensemble</span>
                    <span className="text-emerald-300 font-semibold">
                      {st.weight}%
                    </span>
                  </div>
                  <WeightSlider
                    value={st.weight}
                    onChange={(w) => updateStrategy(cfg.id, { weight: w })}
                  />
                  <div className="mt-1 h-1.5 rounded-full bg-slate-900 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-emerald-400 via-sky-400 to-indigo-400"
                      style={{ width: `${Math.max(st.weight, 6)}%` }}
                    />
                  </div>
                </div>

                {/* Best regimes */}
                <div className="flex flex-wrap gap-1 text-[8px]">
                  <span className={faintText}>Key influence:</span>
                  {cfg.regimes.map((r) => (
                    <span
                      key={r}
                      className="px-2 py-0.5 rounded-full bg-slate-900 text-sky-300"
                    >
                      {r}
                    </span>
                  ))}
                </div>

                {/* Rich metrics only for core three */}
                {!isCompact && metrics.length > 0 && (
                  <div className="grid grid-cols-3 gap-2 text-[8px] mt-1">
                    {metrics.map(({ meta, value }) => (
                      <div
                        key={meta.label}
                        className="rounded-xl bg-slate-950/90 px-2 py-1.5 flex flex-col"
                      >
                        <div className={`${faintText}`}>
                          {meta.label}
                        </div>
                        <div
                          className={`text-xs font-semibold ${metricColor(
                            meta.label,
                            value
                          )}`}
                        >
                          {fmtMetric(meta, value)}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Feature importance only for core three */}
                {!isCompact && cfg.features && (
                  <div className="mt-1">
                    <div className="text-[8px] text-slate-300 mb-1">
                      Top Features (importance)
                    </div>
                    <div className="space-y-1">
                      {cfg.features.map((f) => {
                        const pct = (f.w / totalFeatureWeight) * 100;
                        return (
                          <div
                            key={f.name}
                            className="flex items-center gap-2 text-[8px]"
                          >
                            <div className="flex-1 truncate text-slate-400">
                              {f.name}
                            </div>
                            <div className="w-20 h-1.5 rounded-full bg-slate-900 overflow-hidden">
                              <div
                                className="h-full bg-emerald-400"
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                            <div className="w-6 text-right text-slate-400">
                              {pct.toFixed(0)}%
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Compact description for specialists & judge */}
                {isCompact && (
                  <div className="text-[8px] text-slate-300 mt-0.5">
                    This module adjusts conviction and sizing based on its
                    specialty. Higher weight = stronger say in the final
                    decision for its domain.
                  </div>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Footer actions */}
      <div className="flex justify-end gap-2 mt-1 text-[9px]">
        <button className="px-3 py-1.5 rounded-full bg-emerald-500/90 text-black font-medium shadow-lg shadow-emerald-500/30">
          Save Profile (Demo)
        </button>
        <button className="px-3 py-1.5 rounded-full bg-slate-900 text-slate-200 border border-slate-600/70">
          Load Last (Demo)
        </button>
      </div>
      <div className={`${faintText} text-[9px]`}>
        All visuals here run on dummy data for safety, but they map 1:1 to how
        Nowa’s real hybrid AI stack (TFT, TCN, XGBoost, Options Expert, Macro +
        On-Chain, DecisionNet) combines into a single, risk-aware trade signal.
      </div>
    </div>
  );
};

export { StrategiesTab };
export default StrategiesTab;
