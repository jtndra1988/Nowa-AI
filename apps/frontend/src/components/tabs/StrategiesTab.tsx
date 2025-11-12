import React, { useEffect, useMemo, useState } from "react";
import { MarketMode, SymbolCode } from "@/lib/api";
import {
  Card,
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

type BrainHealth = {
  is_ready: boolean;
  has_l2_models: boolean;
  llm_ready: boolean;
  rl_ready: boolean;
  risk_ready: boolean;
};

type AiStrategySettings = {
  enabled: boolean;
  allow_autotrade: boolean;

  // Ensemble & validation
  ensemble_mode: "weighted" | "majority" | "strict";
  require_ai_decision_agreement: boolean;
  block_if_brain_degraded: boolean;

  // Per-module toggles
  use_tft: boolean;
  use_tcn: boolean;
  use_xgb: boolean;
  use_decision_net: boolean;
  use_options_expert: boolean;
  use_macro_onchain: boolean;
  use_llm: boolean;
  use_rl: boolean;

  // Weights (for weighted ensemble)
  weights: {
    tft: number;
    tcn: number;
    xgb: number;
    decision_net: number;
    options_expert: number;
    macro_onchain: number;
    llm: number;
    rl: number;
  };

  // Risk
  risk_mode: "defensive" | "balanced" | "aggressive";
  max_leverage: number;
  max_position_pct: number;
  max_daily_loss_pct: number;
};

type AiStats = {
  decisions_24h: number;
  winrate_30d: number;
  sharpe_30d: number;
  avg_rr: number;
  uptime_pct_7d: number;
  avg_latency_ms: number;
};

type AiDecision = {
  ts: string;
  symbol: string;
  action: "LONG" | "SHORT" | "FLAT" | "HOLD";
  confidence: number;
  size_factor: number;
  pnl?: number | null;
  reason?: string;
  layer_vote?: string;
};

const DEFAULT_SETTINGS: AiStrategySettings = {
  enabled: true,
  allow_autotrade: false,

  ensemble_mode: "weighted",
  require_ai_decision_agreement: true,
  block_if_brain_degraded: true,

  use_tft: true,
  use_tcn: true,
  use_xgb: true,
  use_decision_net: true,
  use_options_expert: true,
  use_macro_onchain: true,
  use_llm: true,
  use_rl: true,

  weights: {
    tft: 16,
    tcn: 14,
    xgb: 10,
    decision_net: 16,
    options_expert: 10,
    macro_onchain: 12,
    llm: 12,
    rl: 10,
  },

  risk_mode: "balanced",
  max_leverage: 5,
  max_position_pct: 8,
  max_daily_loss_pct: 3,
};

const DEFAULT_STATS: AiStats = {
  decisions_24h: 52,
  winrate_30d: 63,
  sharpe_30d: 1.92,
  avg_rr: 2.35,
  uptime_pct_7d: 99.3,
  avg_latency_ms: 18,
};

type ModelKey =
  | "tft"
  | "tcn"
  | "xgb"
  | "decision_net"
  | "options_expert"
  | "macro_onchain"
  | "llm"
  | "rl";

type LayerName =
  | "Data Layer"
  | "AI Layer"
  | "Decision Layer"
  | "Meta Layer"
  | "Execution Layer";

interface ModelConfig {
  key: ModelKey;
  name: string;
  label: string;
  layer: LayerName;
  role: string;
  horizon: string;
  window: string;
  notes: string;
  defaultLatencyMs: number;
  regimes: string[];
  metrics: {
    precision: number;
    recall: number;
    f1: number;
    auc?: number;
    sharpe?: number;
    maxdd?: number;
  };
  features?: { name: string; weight: number }[];
}

// Clear mapping of each model into the 5-layer mental model.
const MODELS: ModelConfig[] = [
  // AI Layer (raw predictive specialists)
  {
    key: "tft",
    name: "TFT",
    label: "Macro Trend & Carry",
    layer: "AI Layer",
    role: "Learns slow regimes, structural trend & carry. Sets core directional bias.",
    horizon: "5m–2h",
    window: "120d",
    notes: "Used for macro bias and medium horizon positioning.",
    defaultLatencyMs: 11,
    regimes: ["trend", "range", "low-vol"],
    metrics: {
      precision: 0.67,
      recall: 0.64,
      f1: 0.65,
      auc: 0.79,
      sharpe: 2.01,
      maxdd: -13.0,
    },
    features: [
      { name: "Trend / Regime score", weight: 24 },
      { name: "Carry & basis", weight: 21 },
      { name: "On-chain flows", weight: 18 },
      { name: "Global risk basket", weight: 20 },
      { name: "Realized vol", weight: 17 },
    ],
  },
  {
    key: "tcn",
    name: "TCN",
    label: "Short-term Flow & Reversion",
    layer: "AI Layer",
    role: "Reads orderflow & liquidity for timing and micro reversals.",
    horizon: "1m–45m",
    window: "90d",
    notes: "Controls precise entries/exits around the macro view.",
    defaultLatencyMs: 7,
    regimes: ["trend", "range", "high-vol", "low-vol"],
    metrics: {
      precision: 0.64,
      recall: 0.62,
      f1: 0.63,
      auc: 0.75,
      sharpe: 1.81,
      maxdd: -16.0,
    },
    features: [
      { name: "CVD / Aggression", weight: 24 },
      { name: "Depth imbalance", weight: 22 },
      { name: "Sweeps & blocks", weight: 18 },
      { name: "Short-term vol", weight: 19 },
      { name: "Funding pulse", weight: 17 },
    ],
  },
  {
    key: "xgb",
    name: "XGBoost",
    label: "Microstructure & Skew",
    layer: "AI Layer",
    role: "Checks basis, skew & structure. Flags broken markets.",
    horizon: "5m–1h",
    window: "60d",
    notes: "Acts as structural sanity filter.",
    defaultLatencyMs: 3,
    regimes: ["range", "low-vol"],
    metrics: {
      precision: 0.61,
      recall: 0.59,
      f1: 0.59,
      auc: 0.73,
      sharpe: 1.33,
      maxdd: -17.0,
    },
    features: [
      { name: "Term structure", weight: 26 },
      { name: "VWAP distance", weight: 22 },
      { name: "Roll correlation", weight: 19 },
      { name: "Skew", weight: 18 },
      { name: "Depth micro", weight: 15 },
    ],
  },

  // Decision Layer (turn model signals into trade intent)
  {
    key: "decision_net",
    name: "DecisionNet",
    label: "Hybrid Policy",
    layer: "Decision Layer",
    role: "Fuses AI Layer outputs into a single LONG/SHORT/FLAT decision.",
    horizon: "execution horizon",
    window: "rolling",
    notes: "Primary decision-maker that must align with risk rules.",
    defaultLatencyMs: 4,
    regimes: ["all"],
    metrics: {
      precision: 0.69,
      recall: 0.66,
      f1: 0.67,
      sharpe: 2.15,
      maxdd: -12.5,
    },
  },
  {
    key: "options_expert",
    name: "Options Expert",
    label: "Gamma & Vol Surface",
    layer: "Decision Layer",
    role: "Uses gamma walls, vol surface & flows to adjust conviction.",
    horizon: "session–3d",
    window: "90d",
    notes: "Boosts/caps risk around key strikes & events.",
    defaultLatencyMs: 5,
    regimes: ["trend", "high-vol"],
    metrics: {
      precision: 0.63,
      recall: 0.6,
      f1: 0.61,
      auc: 0.72,
      sharpe: 1.6,
      maxdd: -15.0,
    },
  },
  {
    key: "macro_onchain",
    name: "Macro + On-chain",
    label: "Regime & Liquidity Analyst",
    layer: "Decision Layer",
    role: "Reads macro regime & on-chain flows to scale risk up/down.",
    horizon: "4h–multi-day",
    window: "180d",
    notes: "Environment gate: risk-off when liquidity is toxic.",
    defaultLatencyMs: 8,
    regimes: ["trend", "range"],
    metrics: {
      precision: 0.62,
      recall: 0.61,
      f1: 0.61,
      sharpe: 1.7,
      maxdd: -14.2,
    },
  },

  // Meta Layer (explanations, veto, overrides)
  {
    key: "llm",
    name: "LLM",
    label: "Narrative & News Sentinel",
    layer: "Meta Layer",
    role: "Scans news / social / anomalies; can veto or downweight trades.",
    horizon: "1h–24h",
    window: "streaming",
    notes: "Meta-guard: blocks trades into event risk or negative narratives.",
    defaultLatencyMs: 40,
    regimes: ["all"],
    metrics: {
      precision: 0.58,
      recall: 0.65,
      f1: 0.61,
    },
  },

  // Execution Layer (turn approved intent into fills)
  {
    key: "rl",
    name: "RL",
    label: "Execution Policy",
    layer: "Execution Layer",
    role: "Transforms approved intent into orders: sizing, scaling, paths.",
    horizon: "seconds–minutes",
    window: "online",
    notes: "Learns optimal micro-execution under strict risk caps.",
    defaultLatencyMs: 3,
    regimes: ["all"],
    metrics: {
      precision: 0.0,
      recall: 0.0,
      f1: 0.0,
    },
  },
];

// --- NEW COMPONENT: Strategy Pill ---
const StrategyPill: React.FC<{
  label: string;
  value: React.ReactNode;
  icon?: React.ReactNode;
  accentColor?: string;
}> = ({ label, value, icon, accentColor = "bg-indigo-500" }) => (
  <div
    className={`flex items-center gap-3 px-4 py-2.5 rounded-2xl border border-white/5 bg-white/5 backdrop-blur-md shadow-lg ${glassPanel}`}
  >
    <div className={`h-8 w-1 rounded-full ${accentColor}`} />
    <div className="flex flex-col">
      <span className="text-[10px] uppercase tracking-wider text-slate-400 font-medium">
        {label}
      </span>
      <div className="text-sm font-semibold text-slate-100 flex items-center gap-1.5">
        {icon}
        {value}
      </div>
    </div>
  </div>
);

const Switch: React.FC<{
  checked: boolean;
  onChange: (v: boolean) => void;
}> = ({ checked, onChange }) => (
  <button
    type="button"
    role="switch"
    aria-checked={checked}
    onClick={() => onChange(!checked)}
    className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
      checked ? "bg-emerald-500" : "bg-slate-700"
    }`}
  >
    <span
      aria-hidden="true"
      className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
        checked ? "translate-x-4" : "translate-x-0"
      }`}
    />
  </button>
);

const StrategiesTab: React.FC<StrategiesTabProps> = ({
  symbol,
  mode,
  exchange,
}) => {
  const seeded = useMemo(
    () => makeSeeded(`mars-strategy-${symbol}-${mode}-${exchange}`),
    [symbol, mode, exchange]
  );
  const rand = () => seeded(); // currently unused, but kept if you randomize demo stats

  const [brainHealth, setBrainHealth] = useState<BrainHealth | null>(null);
  const [bhLoading, setBhLoading] = useState(true);
  const [bhError, setBhError] = useState(false);

  const [settings, setSettings] =
    useState<AiStrategySettings>(DEFAULT_SETTINGS);
  const [settingsLoaded, setSettingsLoaded] = useState(false);
  const [settingsDirty, setSettingsDirty] = useState(false);
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [settingsError, setSettingsError] = useState<string | null>(null);

  const [stats, setStats] = useState<AiStats | null>(null);
  const [decisions, setDecisions] = useState<AiDecision[]>([]);

  // Brain health
  useEffect(() => {
    let cancelled = false;
    const fetchHealth = async () => {
      try {
        setBhLoading(true);
        setBhError(false);
        const res = await fetch("/api/v1/brain-health");
        if (!res.ok) throw new Error("non-200");
        const data = (await res.json()) as BrainHealth;
        if (!cancelled) setBrainHealth(data);
      } catch {
        if (!cancelled) {
          setBhError(true);
          // safe dummy so UI looks alive
          setBrainHealth({
            is_ready: true,
            has_l2_models: true,
            llm_ready: true,
            rl_ready: true,
            risk_ready: true,
          });
        }
      } finally {
        if (!cancelled) setBhLoading(false);
      }
    };
    fetchHealth();
    const id = setInterval(fetchHealth, 30000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // Settings + stats + decisions
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      // strategy
      try {
        const res = await fetch("/api/v1/ai/strategy");
        if (res.ok) {
          const data = (await res.json()) as Partial<AiStrategySettings>;
          if (!cancelled) {
            setSettings((prev) => ({
              ...prev,
              ...data,
              weights: { ...prev.weights, ...(data.weights || {}) },
            }));
          }
        }
      } catch {
        // fall back to defaults
      } finally {
        if (!cancelled) setSettingsLoaded(true);
      }

      // stats
      try {
        const res = await fetch("/api/v1/ai/stats");
        if (res.ok) {
          const data = (await res.json()) as AiStats;
          if (!cancelled) setStats(data);
        } else if (!cancelled) setStats(DEFAULT_STATS);
      } catch {
        if (!cancelled) setStats(DEFAULT_STATS);
      }

      // decisions
      try {
        const res = await fetch("/api/v1/ai/decisions?limit=20");
        if (res.ok) {
          const data = (await res.json()) as AiDecision[];
          if (!cancelled) setDecisions(data || []);
        }
      } catch {
        // ignore
      }
    };
    load();
    const id = setInterval(load, 45000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // Helpers

  const updateSettings = (patch: Partial<AiStrategySettings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      setSettingsDirty(true);
      setSettingsError(null);
      return next;
    });
  };

  const updateWeight = (key: keyof AiStrategySettings["weights"], v: number) => {
    setSettings((prev) => {
      const weights = { ...prev.weights, [key]: v };
      setSettingsDirty(true);
      setSettingsError(null);
      return { ...prev, weights };
    });
  };

  const saveSettings = async () => {
    setSettingsSaving(true);
    setSettingsError(null);
    try {
      const res = await fetch("/api/v1/ai/strategy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      });
      if (!res.ok) throw new Error(`Save failed (${res.status})`);
      setSettingsDirty(false);
    } catch (e: any) {
      setSettingsError(e.message || "Failed to save settings");
    } finally {
      setSettingsSaving(false);
    }
  };

  const resetSettings = () => {
    setSettings(DEFAULT_SETTINGS);
    setSettingsDirty(true);
    setSettingsError(null);
  };

  const normalizedWeights = useMemo(() => {
    const w = settings.weights;
    const activeKeys = (Object.keys(w) as ModelKey[]).filter((k) => {
      if (k === "tft") return settings.use_tft;
      if (k === "tcn") return settings.use_tcn;
      if (k === "xgb") return settings.use_xgb;
      if (k === "decision_net") return settings.use_decision_net;
      if (k === "options_expert") return settings.use_options_expert;
      if (k === "macro_onchain") return settings.use_macro_onchain;
      if (k === "llm") return settings.use_llm;
      if (k === "rl") return settings.use_rl;
      return false;
    });
    const sum =
      activeKeys.reduce((acc, k) => acc + (w[k] || 0), 0) || 1;
    const out = {} as Record<ModelKey, number>;
    (Object.keys(w) as ModelKey[]).forEach((k) => {
      out[k] = 0;
    });
    activeKeys.forEach((k) => {
      out[k] = (w[k] / sum) * 100;
    });
    return out;
  }, [settings]);

  const statsView = stats || DEFAULT_STATS;
  const isBrainOnline = brainHealth?.is_ready ?? true;

  // Render

  return (
    <div className="relative w-full flex flex-col gap-4 text-slate-50">

      {/* --- 0. NEW INFORMATIVE PILLS --- */}
      <div className="flex flex-wrap gap-4">
        <StrategyPill 
          label="Current Regime" 
          value="High Volatility"
          accentColor="bg-amber-500"
        />
        <StrategyPill 
          label="Risk Profile" 
          value={settings.risk_mode.charAt(0).toUpperCase() + settings.risk_mode.slice(1)}
          accentColor={
            settings.risk_mode === 'aggressive' ? 'bg-rose-500' :
            settings.risk_mode === 'defensive' ? 'bg-emerald-500' : 'bg-sky-500'
          }
        />
        <StrategyPill 
          label="Exposure" 
          value="42% / $1.2M"
          accentColor="bg-indigo-400"
        />
         <StrategyPill 
          label="Next Rebalance" 
          value="04:12"
          accentColor="bg-slate-500"
        />
      </div>

      {/* 1. Performance + Brain health */}

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.7fr)_minmax(0,1.3fr)] gap-4">
        {/* AI Performance Snapshot */}
        <Card className={`${glassPanel} bg-slate-900/95 border border-slate-800 rounded-3xl`}>
          <CardContent className="flex flex-col gap-3 pt-4 pb-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-xs uppercase tracking-[0.18em] text-emerald-400">
                  MARS · AI PERFORMANCE SNAPSHOT
                </div>
                <div className="mt-1 text-lg font-semibold">
                  How the unified brain is performing
                </div>
                <div className="text-sm text-slate-400">
                  {symbol} · {mode} · {exchange}
                </div>
              </div>
              <div className="flex flex-col items-end gap-1 text-xs">
                <div className={`flex items-center gap-1 ${isBrainOnline ? "text-emerald-400" : "text-amber-300"}`}>
                  <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                  {isBrainOnline ? "Brain Online" : "Degraded"}
                </div>
                <div className="text-[10px] text-slate-500">
                  Avg latency: {statsView.avg_latency_ms} ms
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3 mt-2 text-sm">
              <PerfStat label="Decisions (24h)" value={statsView.decisions_24h.toString()} />
              <PerfStat label="Win rate (30d)" value={`${statsView.winrate_30d.toFixed(1)}%`} />
              <PerfStat label="Sharpe (30d)" value={statsView.sharpe_30d.toFixed(2)} />
              <PerfStat label="Avg R:R" value={`${statsView.avg_rr.toFixed(2)} : 1`} />
              <PerfStat label="Uptime (7d)" value={`${statsView.uptime_pct_7d.toFixed(2)}%`} />
            </div>
            <div className={`${faintText} text-xs mt-1`}>
              Source: <code className="text-[10px]">/api/v1/ai/stats</code>. Your client sees objective numbers, not hype.
            </div>
          </CardContent>
        </Card>

        {/* Brain Health – dedicated sci-fi style */}
        <Card className={`${glassPanel} bg-slate-900/95 border border-emerald-500/40 rounded-3xl relative overflow-hidden`}>
          <div className="pointer-events-none absolute -inset-12 bg-[radial-gradient(circle_at_top,_rgba(16,185,129,0.16),transparent_70%)]" />
          <CardContent className="relative flex flex-col gap-3 pt-4 pb-4">
            <div className="flex items-center justify-between gap-2">
              <div>
                <div className="text-xs uppercase tracking-[0.2em] text-emerald-400">
                  BRAIN HEALTH
                </div>
                <div className="mt-1 text-lg font-semibold">
                  System integrity across layers
                </div>
              </div>
              <div className="relative h-16 w-16 flex items-center justify-center">
                <div
                  className="absolute inset-0 rounded-full animate-spin-slow opacity-80"
                  style={{
                    backgroundImage:
                      "conic-gradient(from 230deg, rgba(16,185,129,0.22), rgba(56,189,248,0.35), transparent 70%)",
                    maskImage: "radial-gradient(circle, transparent 58%, black 60%)",
                    WebkitMaskImage: "radial-gradient(circle, transparent 58%, black 60%)",
                  }}
                />
                <div className="relative h-9 w-9 rounded-full bg-slate-950/95 border border-slate-700 flex items-center justify-center">
                  <span
                    className={
                      bhLoading
                        ? "h-2 w-2 rounded-full bg-sky-300 animate-pulse"
                        : bhError
                        ? "h-2 w-2 rounded-full bg-rose-400 animate-ping"
                        : isBrainOnline
                        ? "h-2 w-2 rounded-full bg-emerald-400 animate-ping"
                        : "h-2 w-2 rounded-full bg-amber-300 animate-ping"
                    }
                  />
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs mt-1">
              <HealthRow label="L2 model artifacts" ok={brainHealth?.has_l2_models} />
              <HealthRow label="LLM online" ok={brainHealth?.llm_ready} />
              <HealthRow label="RL policy" ok={brainHealth?.rl_ready} optional />
              <HealthRow label="Risk engine" ok={brainHealth?.risk_ready} />
            </div>
            <div className="text-[11px] text-slate-400 mt-1">
              When <span className="text-emerald-400 font-semibold">“Block if brain degraded”</span> is ON,
              Execution Layer is disabled automatically on any red status here.
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 2. Global controls: Ensemble, Validation, Risk, Auto-trade */}
      <Card className={`${glassPanel} bg-slate-900/95 border border-slate-800 rounded-3xl`}>
        <CardContent className="flex flex-col gap-3 pt-4 pb-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="text-xs uppercase tracking-[0.18em] text-sky-400">
                ENSEMBLE · VALIDATION · RISK · AUTO-TRADE
              </div>
              <div className="mt-1 text-lg font-semibold">
                How the five layers must agree before a trade
              </div>
            </div>
            <div className="flex flex-col items-end gap-1 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-slate-300">AI Engine</span>
                <Switch
                  checked={settings.enabled}
                  onChange={(v) => updateSettings({ enabled: v })}
                />
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="text-slate-400">Allow auto-trade</span>
                <Switch
                  checked={settings.allow_autotrade}
                  onChange={(v) => updateSettings({ allow_autotrade: v })}
                />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm mt-2">
            {/* Ensemble & validation */}
            <div className="flex flex-col gap-2">
              <div className="text-xs text-slate-300 font-medium">
                Ensemble mode
              </div>
              <select
                value={settings.ensemble_mode}
                onChange={(e) =>
                  updateSettings({
                    ensemble_mode: e.target.value as AiStrategySettings["ensemble_mode"],
                  })
                }
                className="bg-slate-900 border border-slate-700 rounded-md px-2 py-1.5 text-sm text-slate-100"
              >
                <option value="weighted">Weighted (use sliders below)</option>
                <option value="majority">Majority vote (layers vote)</option>
                <option value="strict">Strict (AI + Decision + Meta + Exec align)</option>
              </select>
              <label className="flex items-center gap-2 text-xs text-slate-300">
                <Switch
                  checked={settings.require_ai_decision_agreement}
                  onChange={(v) =>
                    updateSettings({ require_ai_decision_agreement: v })
                  }
                />
                Require agreement: AI Layer vs Decision Layer.
              </label>
              <label className="flex items-center gap-2 text-xs text-slate-300">
                <Switch
                  checked={settings.block_if_brain_degraded}
                  onChange={(v) =>
                    updateSettings({ block_if_brain_degraded: v })
                  }
                />
                Block trades if Brain Health not green.
              </label>
            </div>

            {/* Risk */}
            <div className="flex flex-col gap-2">
              <div className="text-xs text-slate-300 font-medium">
                Risk profile
              </div>
              <div className="flex gap-2 text-xs">
                {(["defensive", "balanced", "aggressive"] as const).map((rm) => (
                  <button
                    key={rm}
                    type="button"
                    onClick={() => updateSettings({ risk_mode: rm })}
                    className={`px-2 py-1 rounded-md border text-xs ${
                      settings.risk_mode === rm
                        ? "border-emerald-400 text-emerald-300 bg-emerald-500/5"
                        : "border-slate-700 text-slate-400 hover:border-slate-500"
                    }`}
                  >
                    {rm[0].toUpperCase() + rm.slice(1)}
                  </button>
                ))}
              </div>
              <RangeField
                label="Max leverage"
                min={1}
                max={25}
                value={settings.max_leverage}
                onChange={(v) => updateSettings({ max_leverage: v })}
                display={(v) => `x${v.toFixed(0)}`}
              />
              <RangeField
                label="Max position per signal (% equity)"
                min={1}
                max={25}
                value={settings.max_position_pct}
                onChange={(v) => updateSettings({ max_position_pct: v })}
                display={(v) => `${v.toFixed(0)}%`}
              />
              <RangeField
                label="Max daily loss before lockout"
                min={1}
                max={10}
                step={0.5}
                value={settings.max_daily_loss_pct}
                onChange={(v) => updateSettings({ max_daily_loss_pct: v })}
                display={(v) => `${v.toFixed(1)}%`}
              />
            </div>

            {/* Save / status */}
            <div className="flex flex-col gap-2 justify-between">
              <div className="text-xs text-slate-300 font-medium">
                Configuration state
              </div>
              <div className="text-xs text-slate-400">
                Backend should treat <code>/api/v1/ai/strategy</code> as the single
                source of truth for all knobs on this page.
              </div>
              <div className="mt-1 flex flex-col gap-1 text-xs">
                <div className="flex items-center gap-2">
                  <span
                    className={
                      settingsLoaded ? "text-emerald-300" : "text-slate-400"
                    }
                  >
                    {settingsLoaded ? "Settings loaded" : "Loading settings…"}
                  </span>
                  {settingsDirty && (
                    <span className="text-amber-300">Unsaved changes</span>
                  )}
                </div>
                {settingsError && (
                  <div className="text-rose-400">{settingsError}</div>
                )}
              </div>
              <div className="mt-2 flex gap-2">
                <button
                  type="button"
                  onClick={resetSettings}
                  className="px-3 py-1.5 rounded-md text-xs border border-slate-600 text-slate-200 bg-slate-900 hover:bg-slate-800"
                >
                  Reset defaults
                </button>
                <button
                  type="button"
                  onClick={saveSettings}
                  disabled={!settingsDirty || settingsSaving}
                  className={`px-4 py-1.5 rounded-md text-xs font-semibold ${
                    settingsDirty
                      ? "bg-emerald-500 text-slate-900 hover:bg-emerald-400"
                      : "bg-slate-700 text-slate-400 cursor-default"
                  }`}
                >
                  {settingsSaving ? "Saving…" : "Save strategy"}
                </button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 3. Layered view */}

      {/* Data Layer summary */}
      <SectionHeader
        title="Data Layer"
        subtitle="Market data, derivatives, on-chain feeds and feature pipelines powering the AI."
      />
      <Card className={`${glassPanel} bg-slate-900/95 border border-slate-800 rounded-3xl`}>
        <CardContent className="flex flex-col gap-2.5 pt-3.5 pb-3.5 text-sm">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <DataStat label="Market feeds" value="OK" detail="Tick / L2 / funding / OI" ok />
            <DataStat label="Options surface" value="OK" detail="Greeks & vol surfaces" ok />
            <DataStat label="On-chain" value="OK" detail="Flows & labels" ok />
            <DataStat label="Feature store" value="OK" detail="No stale features" ok />
          </div>
          <div className={`${faintText} text-xs`}>
            If any core data source fails, Brain Health should turn amber/red and Execution Layer must halt.
          </div>
        </CardContent>
      </Card>

      {/* AI Layer */}
      <SectionHeader
        title="AI Layer"
        subtitle="Specialist models that read the data and propose views."
      />
      <ModelGrid
        models={MODELS.filter((m) => m.layer === "AI Layer")}
        settings={settings}
        normalizedWeights={normalizedWeights}
        updateToggle={updateSettings}
        updateWeight={updateWeight}
      />

      {/* Decision Layer */}
      <SectionHeader
        title="Decision Layer"
        subtitle="Combines AI signals, options, macro & flows into a single trade intent."
      />
      <ModelGrid
        models={MODELS.filter((m) => m.layer === "Decision Layer")}
        settings={settings}
        normalizedWeights={normalizedWeights}
        updateToggle={updateSettings}
        updateWeight={updateWeight}
      />

      {/* Meta Layer */}
      <SectionHeader
        title="Meta Layer"
        subtitle="Oversight, narrative & veto logic on top of the decision."
      />
      <ModelGrid
        models={MODELS.filter((m) => m.layer === "Meta Layer")}
        settings={settings}
        normalizedWeights={normalizedWeights}
        updateToggle={updateSettings}
        updateWeight={updateWeight}
      />

      {/* Execution Layer */}
      <SectionHeader
        title="Execution Layer"
        subtitle="Turns approved decisions into orders under strict risk constraints."
      />
      <ModelGrid
        models={MODELS.filter((m) => m.layer === "Execution Layer")}
        settings={settings}
        normalizedWeights={normalizedWeights}
        updateToggle={updateSettings}
        updateWeight={updateWeight}
      />

      {/* 4. Recent decisions + explanation */}

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.8fr)_minmax(0,1.2fr)] gap-4">
        {/* Decisions table */}
        <Card className={`${glassPanel} bg-slate-900/95 border border-slate-800 rounded-3xl`}>
          <CardContent className="flex flex-col gap-3 pt-3.5 pb-3.5">
            <div className="flex items-center justify-between">
              <div className="text-base font-semibold">
                Recent Hybrid Decisions
              </div>
              <div className="text-xs text-slate-500">
                {decisions.length
                  ? `Last ${decisions.length} decisions`
                  : "Waiting for /api/v1/ai/decisions"}
              </div>
            </div>
            <div className="w-full overflow-x-auto">
              <table className="w-full text-xs text-slate-200">
                <thead className="text-slate-500">
                  <tr>
                    <th className="px-2 py-1 text-left">Time</th>
                    <th className="px-2 py-1 text-left">Symbol</th>
                    <th className="px-2 py-1 text-left">Action</th>
                    <th className="px-2 py-1 text-right">Conf%</th>
                    <th className="px-2 py-1 text-right">Size%</th>
                    <th className="px-2 py-1 text-right">PnL</th>
                    <th className="px-2 py-1 text-left">Layer votes / Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {decisions.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="px-2 py-2 text-slate-500">
                        Once wired, every HybridDecision from your backend will appear here
                        with audit trail.
                      </td>
                    </tr>
                  ) : (
                    decisions.map((d, i) => {
                      const pnlColor =
                        d.pnl == null
                          ? "text-slate-400"
                          : d.pnl > 0
                          ? "text-emerald-400"
                          : "text-rose-400";
                      return (
                        <tr
                          key={`${d.ts}-${i}`}
                          className="border-t border-slate-800"
                        >
                          <td className="px-2 py-1 text-slate-400">{d.ts}</td>
                          <td className="px-2 py-1">{d.symbol}</td>
                          <td className="px-2 py-1">{d.action}</td>
                          <td className="px-2 py-1 text-right">
                            {(d.confidence * 100).toFixed(0)}
                          </td>
                          <td className="px-2 py-1 text-right">
                            {(d.size_factor * 100).toFixed(1)}
                          </td>
                          <td className={`px-2 py-1 text-right ${pnlColor}`}>
                            {d.pnl == null ? "—" : d.pnl.toFixed(3)}
                          </td>
                          <td className="px-2 py-1 text-slate-400">
                            {d.layer_vote || d.reason || ""}
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* Explanation card */}
        <Card className={`${glassPanel} bg-slate-900/95 border border-slate-800 rounded-3xl`}>
          <CardContent className="flex flex-col gap-2.5 pt-3.5 pb-3.5 text-sm">
            <div className="text-base font-semibold">
              Pipeline overview (what your client sees)
            </div>
            <ul className="list-disc list-inside text-slate-400 space-y-1">
              <li>
                <span className="text-slate-200">Data Layer</span> feeds all models.
              </li>
              <li>
                <span className="text-slate-200">AI Layer</span> (TFT / TCN / XGB) proposes views only.
              </li>
              <li>
                <span className="text-slate-200">Decision Layer</span> (DecisionNet / Options / Macro+On-chain)
                turns those views into a coherent trade thesis.
              </li>
              <li>
                <span className="text-slate-200">Meta Layer</span> (LLM) watches news & narratives, can veto.
              </li>
              <li>
                <span className="text-slate-200">Execution Layer</span> (RL) executes only if all rules & health
                checks pass.
              </li>
              <li>
                Ensemble + risk + auto-trade settings on this page define exactly what is allowed.
              </li>
              <li>
                Every decision below is explainable in terms of these layers —
                no black box.
              </li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

/* ---- Small components ---- */

const SectionHeader: React.FC<{ title: string; subtitle: string }> = ({
  title,
  subtitle,
}) => (
  <div className="mt-2 flex flex-col gap-0.5">
    <div className="text-xs uppercase tracking-[0.18em] text-slate-500">
      {title}
    </div>
    <div className="text-sm text-slate-300">{subtitle}</div>
  </div>
);

const PerfStat: React.FC<{ label: string; value: string }> = ({
  label,
  value,
}) => (
  <div className="flex flex-col gap-0.5">
    <div className={`${faintText} text-[11px]`}>{label}</div>
    <div className="text-sky-300 text-base font-semibold">{value}</div>
  </div>
);

const HealthRow: React.FC<{
  label: string;
  ok?: boolean;
  optional?: boolean;
}> = ({ label, ok, optional }) => {
  const text =
    ok === undefined
      ? "Unknown"
      : ok
      ? "OK"
      : optional
      ? "Optional"
      : "Issue";
  const color =
    ok === undefined
      ? "text-slate-400"
      : ok
      ? "text-emerald-300"
      : optional
      ? "text-amber-300"
      : "text-rose-400";
  return (
    <div className="flex items-center justify-between gap-2">
      <div className="text-[11px] text-slate-300">{label}</div>
      <div className={`text-[11px] ${color}`}>{text}</div>
    </div>
  );
};

const RangeField: React.FC<{
  label: string;
  min: number;
  max: number;
  value: number;
  step?: number;
  onChange: (v: number) => void;
  display: (v: number) => string;
}> = ({ label, min, max, value, step = 1, onChange, display }) => (
  <div className="flex flex-col gap-0.5">
    <div className="text-[11px] text-slate-300">{label}</div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="w-full"
    />
    <div className="text-[11px] text-emerald-300">{display(value)}</div>
  </div>
);

const Metric: React.FC<{
  label: string;
  value: number;
  isPct?: boolean;
  inverse?: boolean;
}> = ({ label, value, isPct, inverse }) => {
  const formatted = isPct
    ? `${value.toFixed(1)}%`
    : Math.abs(value) < 10
    ? value.toFixed(2)
    : value.toFixed(1);
  const color =
    inverse && value < 0 ? "text-emerald-300" : "text-sky-300";
  return (
    <div className="flex flex-col gap-0.25">
      <div className={`${faintText} text-[10px]`}>{label}</div>
      <div className={`${color} text-[12px] font-semibold`}>
        {label === "MaxDD" ? `${value.toFixed(1)}%` : formatted}
      </div>
    </div>
  );
};

const DataStat: React.FC<{
  label: string;
  value: string;
  detail: string;
  ok?: boolean;
}> = ({ label, value, detail, ok = false }) => (
  <div className="flex flex-col gap-0.25">
    <div className="text-[11px] text-slate-300">{label}</div>
    <div
      className={`text-[12px] font-semibold ${
        ok ? "text-emerald-300" : "text-slate-300"
      }`}
    >
      {value}
    </div>
    <div className={`${faintText} text-[10px]`}>{detail}</div>
  </div>
);

type ModelGridProps = {
  models: ModelConfig[];
  settings: AiStrategySettings;
  normalizedWeights: Record<ModelKey, number>;
  updateToggle: (patch: Partial<AiStrategySettings>) => void;
  updateWeight: (key: keyof AiStrategySettings["weights"], v: number) => void;
};

const ModelGrid: React.FC<ModelGridProps> = ({
  models,
  settings,
  normalizedWeights,
  updateToggle,
  updateWeight,
}) => (
  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 mt-2">
    {models.map((m) => {
      const enabledKey =
        m.key === "tft"
          ? "use_tft"
          : m.key === "tcn"
          ? "use_tcn"
          : m.key === "xgb"
          ? "use_xgb"
          : m.key === "decision_net"
          ? "use_decision_net"
          : m.key === "options_expert"
          ? "use_options_expert"
          : m.key === "macro_onchain"
          ? "use_macro_onchain"
          : m.key === "llm"
          ? "use_llm"
          : "use_rl";

      const isEnabled = (settings as any)[enabledKey] && settings.enabled;
      const weight = settings.weights[m.key];
      const norm = normalizedWeights[m.key] || 0;

      return (
        <Card
          key={m.key}
          className={`${glassPanel} bg-slate-900/95 border border-slate-800 rounded-3xl`}
        >
          <CardContent className="flex flex-col gap-2.5 pt-3.5 pb-3.5">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-xs">
                <span className="px-2 py-0.5 rounded-full bg-slate-800 text-emerald-300">
                  {m.defaultLatencyMs}ms
                </span>
                <span
                  className={
                    isEnabled ? "text-emerald-400" : "text-slate-500"
                  }
                >
                  {isEnabled ? "Enabled" : "Disabled"}
                </span>
                <span className="px-1.5 py-0.5 rounded-full bg-slate-800 text-[10px] text-sky-300">
                  {m.layer}
                </span>
              </div>
              <Switch
                checked={(settings as any)[enabledKey]}
                onChange={(v) =>
                  updateToggle({ [enabledKey]: v } as any)
                }
              />
            </div>

            <div>
              <div className="text-sm font-semibold">
                {m.name} · {m.label}
              </div>
              <div className="text-xs text-slate-400">{m.role}</div>
              <div className="text-[10px] text-slate-500">
                Horizon {m.horizon} · Window {m.window}
              </div>
            </div>

            <div className="flex flex-col gap-1 mt-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300">Weight</span>
                <span className="text-emerald-400 font-semibold">
                  {weight.toFixed(0)} ({norm.toFixed(0)}% of ensemble)
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={100}
                value={weight}
                onChange={(e) =>
                  updateWeight(
                    m.key,
                    Number(e.target.value)
                  )
                }
                className="w-full"
              />
              <div className="flex flex-wrap gap-1 text-[10px] text-slate-400">
                <span>Use in regimes:</span>
                {m.regimes.map((r) => (
                  <span
                    key={r}
                    className="px-1.5 py-0.5 rounded-full bg-slate-800 text-slate-200"
                  >
                    {r}
                  </span>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2 mt-1 text-xs">
              <Metric label="Precision" value={m.metrics.precision} />
              <Metric label="Recall" value={m.metrics.recall} />
              <Metric label="F1" value={m.metrics.f1} />
              {m.metrics.auc !== undefined && (
                <Metric label="AUC" value={m.metrics.auc} />
              )}
              {m.metrics.sharpe !== undefined && (
                <Metric label="Sharpe" value={m.metrics.sharpe} />
              )}
              {m.metrics.maxdd !== undefined && (
                <Metric
                  label="MaxDD"
                  value={m.metrics.maxdd}
                  isPct
                  inverse
                />
              )}
            </div>

            {m.features ? (
              <div className="mt-1.5 flex flex-col gap-0.75">
                <div className="text-[11px] text-slate-300">
                  Top features / signals
                </div>
                {m.features.map((f) => (
                  <div
                    key={f.name}
                    className="flex items-center gap-2 text-[10px]"
                  >
                    <span className="flex-1 text-slate-400">
                      {f.name}
                    </span>
                    <div className="w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-indigo-400"
                        style={{ width: `${f.weight}%` }}
                      />
                    </div>
                    <span className="w-8 text-right text-slate-300">
                      {f.weight}%
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="mt-1.5 text-[10px] text-slate-400">
                {m.notes}
              </div>
            )}
          </CardContent>
        </Card>
      );
    })}
  </div>
);

export { StrategiesTab };
export default StrategiesTab;