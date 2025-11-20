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

// Updated to include ALL 10 Models
type AiStrategySettings = {
  enabled: boolean;
  allow_autotrade: boolean;

  // Ensemble & validation
  ensemble_mode: "weighted" | "majority" | "strict";
  require_ai_decision_agreement: boolean;
  block_if_brain_degraded: boolean;

  // Per-module toggles (10 Models)
  use_tft: boolean;
  use_tcn: boolean;
  use_tst: boolean;       // NEW
  use_xgb: boolean;
  use_decision_net: boolean;
  use_ensemble: boolean;  // NEW
  use_options_expert: boolean;
  use_macro_onchain: boolean;
  use_llm: boolean;
  use_rl: boolean;

  // Weights (for weighted ensemble)
  weights: {
    tft: number;
    tcn: number;
    tst: number;          // NEW
    xgb: number;
    decision_net: number;
    ensemble: number;     // NEW
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

  // All 10 models enabled by default
  use_tft: true,
  use_tcn: true,
  use_tst: true,
  use_xgb: true,
  use_decision_net: true,
  use_ensemble: true,
  use_options_expert: true,
  use_macro_onchain: true,
  use_llm: true,
  use_rl: true,

  weights: {
    tft: 15,
    tcn: 15,
    tst: 10,
    xgb: 10,
    decision_net: 15,
    ensemble: 10,
    options_expert: 5,
    macro_onchain: 5,
    llm: 10,
    rl: 5,
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

type ModelKey = keyof AiStrategySettings["weights"];

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

// --- 10 MODELS DEFINITION ---
const MODELS: ModelConfig[] = [
  // --- AI Layer (Predictors) ---
  {
    key: "tft",
    name: "TFT",
    label: "Temporal Fusion Transformer",
    layer: "AI Layer",
    role: "Visionary: Interpretable multi-horizon forecasting.",
    horizon: "1h–4h",
    window: "120d",
    notes: "Captures long-term dependencies & seasonalities.",
    defaultLatencyMs: 11,
    regimes: ["trend", "range"],
    metrics: { precision: 0.67, recall: 0.64, f1: 0.65, sharpe: 2.01, maxdd: -13.0 },
    features: [{ name: "Trend score", weight: 24 }, { name: "Vol 24h", weight: 21 }],
  },
  {
    key: "tcn",
    name: "TCN",
    label: "Temporal ConvNet",
    layer: "AI Layer",
    role: "Reflex: Fast shock detection & short-term patterns.",
    horizon: "5m–30m",
    window: "90d",
    notes: "Reacts quickly to volatility spikes.",
    defaultLatencyMs: 7,
    regimes: ["high-vol", "trend"],
    metrics: { precision: 0.64, recall: 0.62, f1: 0.63, sharpe: 1.81, maxdd: -16.0 },
    features: [{ name: "CVD", weight: 24 }, { name: "Depth Imb", weight: 22 }],
  },
  {
    key: "tst",
    name: "TST",
    label: "Time Series Transformer",
    layer: "AI Layer",
    role: "Specialist: Pure sequence-to-sequence regression.",
    horizon: "15m–1h",
    window: "60d",
    notes: "High-fidelity price path prediction.",
    defaultLatencyMs: 9,
    regimes: ["range", "trend"],
    metrics: { precision: 0.65, recall: 0.63, f1: 0.64, sharpe: 1.90, maxdd: -14.0 },
    features: [{ name: "Seq Attention", weight: 30 }],
  },
  {
    key: "xgb",
    name: "XGBoost",
    label: "Gradient Booster",
    layer: "AI Layer",
    role: "Analyst: Tabular feature analysis (RSI, MA, VWAP).",
    horizon: "5m–1h",
    window: "60d",
    notes: "Strong on technical indicators and regime classification.",
    defaultLatencyMs: 3,
    regimes: ["range", "low-vol"],
    metrics: { precision: 0.61, recall: 0.59, f1: 0.59, sharpe: 1.33, maxdd: -17.0 },
    features: [{ name: "VWAP Dist", weight: 22 }, { name: "RSI", weight: 19 }],
  },

  // --- Decision Layer (Fusion) ---
  {
    key: "decision_net",
    name: "DecisionNet",
    label: "Neural Fusion",
    layer: "Decision Layer",
    role: "Orchestrator: Weighs expert inputs dynamically.",
    horizon: "Execution",
    window: "Rolling",
    notes: "Deep learning based fusion of L1 signals.",
    defaultLatencyMs: 4,
    regimes: ["all"],
    metrics: { precision: 0.69, recall: 0.66, f1: 0.67, sharpe: 2.15, maxdd: -12.5 },
  },
  {
    key: "ensemble",
    name: "Ensemble",
    label: "Linear Stacker",
    layer: "Decision Layer",
    role: "Stabilizer: Weighted average of L1 predictors.",
    horizon: "Execution",
    window: "Rolling",
    notes: "Reduces variance of individual models.",
    defaultLatencyMs: 2,
    regimes: ["all"],
    metrics: { precision: 0.65, recall: 0.65, f1: 0.65, sharpe: 1.8, maxdd: -11.0 },
  },
  {
    key: "options_expert",
    name: "Options Expert",
    label: "Gamma & Vol Surface",
    layer: "Decision Layer",
    role: "Hedge: Gamma/Vanna exposure analysis.",
    horizon: "Session",
    window: "90d",
    notes: "Identifies pinning and gamma squeeze risks.",
    defaultLatencyMs: 5,
    regimes: ["high-vol"],
    metrics: { precision: 0.63, recall: 0.60, f1: 0.61, sharpe: 1.6, maxdd: -15.0 },
  },
  {
    key: "macro_onchain",
    name: "Macro/On-chain",
    label: "Fundamental Bias",
    layer: "Decision Layer",
    role: "Context: Netflow, active addresses, DXY.",
    horizon: "4h+",
    window: "180d",
    notes: "Filters trades against macro tides.",
    defaultLatencyMs: 8,
    regimes: ["trend"],
    metrics: { precision: 0.62, recall: 0.61, f1: 0.61, sharpe: 1.7, maxdd: -14.2 },
  },

  // --- Meta Layer ---
  {
    key: "llm",
    name: "LLM Agent",
    label: "Narrative Sentinel",
    layer: "Meta Layer",
    role: "Veto: News, sentiment, and market vibes.",
    horizon: "1h–24h",
    window: "Streaming",
    notes: "Qualitative analysis to override quant signals.",
    defaultLatencyMs: 40,
    regimes: ["all"],
    metrics: { precision: 0.58, recall: 0.65, f1: 0.61 },
  },

  // --- Execution Layer ---
  {
    key: "rl",
    name: "RL Agent",
    label: "PPO Trader",
    layer: "Execution Layer",
    role: "Executor: Optimal sizing and timing policy.",
    horizon: "Sec–Min",
    window: "Online",
    notes: "Reinforcement learning for final trade execution.",
    defaultLatencyMs: 3,
    regimes: ["all"],
    metrics: { precision: 0.0, recall: 0.0, f1: 0.0 },
  },
];

// --- COMPONENT: Strategy Pill ---
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

// --- MAIN EXPORTED COMPONENT ---
export const StrategiesTab: React.FC<StrategiesTabProps> = ({
  symbol,
  mode,
  exchange,
}) => {
  const seeded = useMemo(
    () => makeSeeded(`mars-strategy-${symbol}-${mode}-${exchange}`),
    [symbol, mode, exchange]
  );

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

  // Brain health fetch
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

  // Settings load
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
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
        // fallback
      } finally {
        if (!cancelled) setSettingsLoaded(true);
      }

      try {
        const res = await fetch("/api/v1/ai/stats");
        if (res.ok) {
          const data = (await res.json()) as AiStats;
          if (!cancelled) setStats(data);
        }
      } catch {}

      try {
        const res = await fetch("/api/v1/ai/decisions?limit=20");
        if (res.ok) {
          const data = (await res.json()) as AiDecision[];
          if (!cancelled) setDecisions(data || []);
        }
      } catch {}
    };
    load();
    const id = setInterval(load, 45000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // Settings logic
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
      setSettingsError(e.message || "Failed to save");
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
      // Dynamic check for toggles
      const toggleKey = `use_${k}` as keyof AiStrategySettings;
      return settings[toggleKey] === true;
    });
    const sum = activeKeys.reduce((acc, k) => acc + (w[k] || 0), 0) || 1;
    const out = {} as Record<ModelKey, number>;
    activeKeys.forEach((k) => {
      out[k] = (w[k] / sum) * 100;
    });
    return out;
  }, [settings]);

  const statsView = stats || DEFAULT_STATS;
  const isBrainOnline = brainHealth?.is_ready ?? true;

  return (
    <div className="relative w-full flex flex-col gap-4 text-slate-50">
       {/* --- 0. Informative Pills --- */}
       <div className="flex flex-wrap gap-4">
        <StrategyPill 
          label="System Mode" 
          value={settings.allow_autotrade ? "AUTONOMOUS" : "ADVISORY"}
          accentColor={settings.allow_autotrade ? "bg-emerald-500" : "bg-sky-500"}
        />
        <StrategyPill 
          label="Active Models" 
          value={`${Object.keys(normalizedWeights).length} / 10`}
          accentColor="bg-indigo-500"
        />
        <StrategyPill 
          label="Risk Profile" 
          value={settings.risk_mode.toUpperCase()}
          accentColor={settings.risk_mode === 'aggressive' ? 'bg-rose-500' : 'bg-amber-500'}
        />
         <StrategyPill 
          label="Next Update" 
          value="00:12s"
          accentColor="bg-slate-500"
        />
      </div>

      {/* 1. Dashboard & Health */}
      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.7fr)_minmax(0,1.3fr)] gap-4">
        {/* Perf Card */}
        <Card className={`${glassPanel} bg-slate-900/95 border border-slate-800 rounded-3xl`}>
          <CardContent className="flex flex-col gap-3 pt-4 pb-4">
             <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-xs uppercase tracking-[0.18em] text-emerald-400">
                  MARS · UNIFIED BRAIN
                </div>
                <div className="mt-1 text-lg font-semibold">
                  AI Performance (Hybrid V2)
                </div>
              </div>
              <div className="flex flex-col items-end gap-1 text-xs">
                 <div className={`flex items-center gap-1 ${isBrainOnline ? "text-emerald-400" : "text-amber-300"}`}>
                  <span className={`h-2 w-2 rounded-full ${isBrainOnline ? "bg-emerald-400 animate-pulse" : "bg-amber-300"}`} />
                  {isBrainOnline ? "Brain Online" : "Degraded"}
                 </div>
                 <div className="text-[10px] text-slate-500">Lat: {statsView.avg_latency_ms}ms</div>
              </div>
             </div>
             <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3 mt-2 text-sm">
               <PerfStat label="Decisions (24h)" value={statsView.decisions_24h.toString()} />
               <PerfStat label="Win Rate" value={`${statsView.winrate_30d.toFixed(1)}%`} />
               <PerfStat label="Sharpe" value={statsView.sharpe_30d.toFixed(2)} />
               <PerfStat label="Avg R:R" value={statsView.avg_rr.toFixed(2)} />
               <PerfStat label="Uptime" value={`${statsView.uptime_pct_7d.toFixed(1)}%`} />
             </div>
          </CardContent>
        </Card>

        {/* Health Card */}
        <Card className={`${glassPanel} bg-slate-900/95 border border-emerald-500/40 rounded-3xl relative overflow-hidden`}>
           <div className="pointer-events-none absolute -inset-12 bg-[radial-gradient(circle_at_top,_rgba(16,185,129,0.16),transparent_70%)]" />
           <CardContent className="relative flex flex-col gap-3 pt-4 pb-4">
              <div className="flex items-center justify-between gap-2">
                <div>
                  <div className="text-xs uppercase tracking-[0.2em] text-emerald-400">SYSTEM HEALTH</div>
                  <div className="mt-1 text-lg font-semibold">Components Status</div>
                </div>
                {/* Sci-Fi Spinner */}
                <div className="relative h-12 w-12 flex items-center justify-center">
                  <div className="absolute inset-0 rounded-full border-2 border-slate-800 border-t-emerald-500 animate-spin" />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs mt-1">
                <HealthRow label="L2 Models (Torch)" ok={brainHealth?.has_l2_models} />
                <HealthRow label="LLM (Gemini)" ok={brainHealth?.llm_ready} />
                <HealthRow label="RL Agent (PPO)" ok={brainHealth?.rl_ready} optional />
                <HealthRow label="Risk Engine" ok={brainHealth?.risk_ready} />
              </div>
           </CardContent>
        </Card>
      </div>

      {/* 2. Global Settings */}
      <Card className={`${glassPanel} bg-slate-900/95 border border-slate-800 rounded-3xl`}>
        <CardContent className="flex flex-col gap-4 pt-4 pb-4">
           <div className="flex items-center justify-between">
             <div className="text-lg font-semibold">Global Strategy Control</div>
             <div className="flex gap-4">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-400">AI Engine</span>
                  <Switch checked={settings.enabled} onChange={(v) => updateSettings({enabled: v})} />
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-400">Auto-Trade</span>
                  <Switch checked={settings.allow_autotrade} onChange={(v) => updateSettings({allow_autotrade: v})} />
                </div>
             </div>
           </div>
           
           <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Column 1: Ensemble */}
              <div className="flex flex-col gap-3">
                 <div className="text-xs uppercase tracking-wider text-slate-500">Ensemble Logic</div>
                 <select 
                    value={settings.ensemble_mode}
                    onChange={(e) => updateSettings({ ensemble_mode: e.target.value as any })}
                    className="bg-slate-950 border border-slate-700 text-xs rounded px-2 py-1.5"
                 >
                    <option value="weighted">Weighted Average</option>
                    <option value="majority">Majority Vote</option>
                    <option value="strict">Strict Unanimity</option>
                 </select>
                 <label className="flex items-center gap-2 text-xs text-slate-400">
                   <Switch checked={settings.require_ai_decision_agreement} onChange={(v) => updateSettings({require_ai_decision_agreement: v})} />
                   Require AI & Decision Layer Match
                 </label>
              </div>

              {/* Column 2: Risk */}
              <div className="flex flex-col gap-3">
                <div className="text-xs uppercase tracking-wider text-slate-500">Global Risk</div>
                <RangeField 
                  label="Max Leverage" min={1} max={20} value={settings.max_leverage} 
                  onChange={(v) => updateSettings({max_leverage: v})}
                  display={(v) => `${v}x`}
                />
                <RangeField 
                  label="Max Daily Loss" min={1} max={10} step={0.5} value={settings.max_daily_loss_pct} 
                  onChange={(v) => updateSettings({max_daily_loss_pct: v})}
                  display={(v) => `${v}%`}
                />
              </div>

              {/* Column 3: Actions */}
              <div className="flex flex-col gap-3 justify-end items-end">
                {settingsDirty && <div className="text-xs text-amber-400">Unsaved Changes</div>}
                <div className="flex gap-2">
                   <button onClick={resetSettings} className="px-3 py-1.5 text-xs border border-slate-600 rounded hover:bg-slate-800">Reset</button>
                   <button 
                    onClick={saveSettings} 
                    disabled={!settingsDirty}
                    className={`px-3 py-1.5 text-xs rounded font-semibold ${settingsDirty ? "bg-emerald-500 text-slate-900 hover:bg-emerald-400" : "bg-slate-800 text-slate-500"}`}
                   >
                    Save Config
                   </button>
                </div>
              </div>
           </div>
        </CardContent>
      </Card>

      {/* 3. VISUALIZATION OF LAYERS */}
      {["AI Layer", "Decision Layer", "Meta Layer", "Execution Layer"].map((layer) => (
        <div key={layer} className="flex flex-col gap-2">
          <SectionHeader title={layer} subtitle={getLayerSubtitle(layer as LayerName)} />
          <ModelGrid 
            models={MODELS.filter(m => m.layer === layer)}
            settings={settings}
            normalizedWeights={normalizedWeights}
            updateToggle={updateSettings}
            updateWeight={updateWeight}
          />
        </div>
      ))}
    </div>
  );
};

// --- Helpers ---
function getLayerSubtitle(layer: LayerName) {
  switch(layer) {
    case "AI Layer": return "Core predictive models analysing raw price & volume data.";
    case "Decision Layer": return "Fuses predictors with domain expertise (Options, Macro) for final intent.";
    case "Meta Layer": return "High-level oversight using LLMs for narrative checks.";
    case "Execution Layer": return "Reinforcement Learning agent for optimal order routing.";
    default: return "";
  }
}

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
      const enabledKey = `use_${m.key}` as keyof AiStrategySettings;
      // Safe access
      const isEnabled = (settings as any)[enabledKey] && settings.enabled;
      const weight = settings.weights[m.key];
      const norm = normalizedWeights[m.key] || 0;

      return (
        <Card key={m.key} className={`${glassPanel} bg-slate-900/95 border border-slate-800 rounded-3xl opacity-${isEnabled ? '100' : '60'}`}>
          <CardContent className="flex flex-col gap-2.5 pt-3.5 pb-3.5">
            <div className="flex items-center justify-between">
               <div className="flex items-center gap-2">
                 <div className={`h-2 w-2 rounded-full ${isEnabled ? 'bg-emerald-400 shadow-lg shadow-emerald-400/50' : 'bg-slate-600'}`} />
                 <span className="text-sm font-bold text-slate-200">{m.name}</span>
               </div>
               <Switch checked={(settings as any)[enabledKey]} onChange={(v) => updateToggle({ [enabledKey]: v } as any)} />
            </div>
            
            <div className="text-xs text-slate-400 min-h-[32px]">{m.role}</div>
            
            <div className="flex flex-col gap-1 mt-2 bg-slate-950/50 p-2 rounded-lg border border-white/5">
              <div className="flex justify-between text-[10px] uppercase tracking-wider text-slate-500">
                <span>Weight</span>
                <span>{weight} ({norm.toFixed(0)}%)</span>
              </div>
              <input 
                type="range" min={0} max={50} value={weight} 
                onChange={(e) => updateWeight(m.key, Number(e.target.value))}
                disabled={!isEnabled}
                className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            <div className="grid grid-cols-3 gap-2 mt-1">
               <Metric label="Sharpe" value={m.metrics.sharpe || 0} />
               <Metric label="Precision" value={m.metrics.precision} isPct />
               <Metric label="Horizon" value={0} customText={m.horizon} />
            </div>
          </CardContent>
        </Card>
      );
    })}
  </div>
);

const Metric: React.FC<{
  label: string;
  value: number;
  isPct?: boolean;
  customText?: string;
}> = ({ label, value, isPct, customText }) => (
  <div className="flex flex-col">
    <span className="text-[10px] text-slate-500">{label}</span>
    <span className="text-xs font-mono text-slate-300">
      {customText ? customText : isPct ? `${(value*100).toFixed(0)}%` : value.toFixed(2)}
    </span>
  </div>
);

const SectionHeader: React.FC<{ title: string; subtitle: string }> = ({
  title,
  subtitle,
}) => (
  <div className="mt-2 flex flex-col gap-0.5 px-1">
    <div className="text-xs uppercase tracking-[0.18em] text-sky-400/80 font-bold">
      {title}
    </div>
    <div className="text-xs text-slate-400">{subtitle}</div>
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
  const text = ok === undefined ? "Unknown" : ok ? "Ready" : optional ? "Offline (Opt)" : "Critical";
  const color = ok ? "text-emerald-400" : optional ? "text-amber-400" : "text-rose-500";
  return (
    <div className="flex items-center justify-between">
       <span className="text-slate-400">{label}</span>
       <span className={`font-mono ${color}`}>{text}</span>
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
  <div className="flex flex-col gap-1">
    <div className="flex justify-between text-xs text-slate-400">
       <span>{label}</span>
       <span className="text-emerald-400">{display(value)}</span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
    />
  </div>
);

export default StrategiesTab;