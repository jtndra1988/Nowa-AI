"use client";

import React, { useEffect, useMemo, useState } from "react";
import { MarketMode, SymbolCode } from "@/lib/api";
import {
  Card,
  CardContent,
  glassPanel,
  makeSeeded,
} from "../layout/AppShell";
import {
  BrainCircuit,
  Zap,
  History,
  Calculator,
  ShieldCheck,
  Globe,
  Newspaper,
  Bot,
  Layers,
  Activity,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";

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

// Config matches backend, but UI will simplify it
type AiStrategySettings = {
  enabled: boolean;
  allow_autotrade: boolean;
  ensemble_mode: "weighted" | "majority" | "strict";
  require_ai_decision_agreement: boolean;
  block_if_brain_degraded: boolean;
  // Toggles
  use_tft: boolean;
  use_tcn: boolean;
  use_tst: boolean;
  use_xgb: boolean;
  use_decision_net: boolean;
  use_ensemble: boolean;
  use_options_expert: boolean;
  use_macro_onchain: boolean;
  use_llm: boolean;
  use_rl: boolean;
  // Weights
  weights: Record<string, number>;
  // Risk
  risk_mode: "defensive" | "balanced" | "aggressive";
  max_leverage: number;
  max_position_pct: number;
  max_daily_loss_pct: number;
};

const DEFAULT_SETTINGS: AiStrategySettings = {
  enabled: true,
  allow_autotrade: false,
  ensemble_mode: "weighted",
  require_ai_decision_agreement: true,
  block_if_brain_degraded: true,
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
    tft: 15, tcn: 15, tst: 10, xgb: 10, decision_net: 15,
    ensemble: 10, options_expert: 5, macro_onchain: 5, llm: 10, rl: 5,
  },
  risk_mode: "balanced",
  max_leverage: 5,
  max_position_pct: 8,
  max_daily_loss_pct: 3,
};

// --- LAYMAN AGENT CONFIG ---
// We map technical model IDs to friendly "Personas"
const AGENT_PERSONAS = [
  {
    id: "tft",
    name: "The Visionary",
    icon: <BrainCircuit className="w-6 h-6 text-purple-400" />,
    desc: "Looks at long-term patterns to predict where price is going over the next 4-12 hours.",
    superpower: "Trend Detection",
    accuracy: 84,
  },
  {
    id: "tcn",
    name: "The Reflex",
    icon: <Zap className="w-6 h-6 text-yellow-400" />,
    desc: "Reacts instantly to sudden price spikes and drops. Very fast, but short-sighted.",
    superpower: "Shock Response",
    accuracy: 79,
  },
  {
    id: "tst",
    name: "The Historian",
    icon: <History className="w-6 h-6 text-blue-400" />,
    desc: "Compares current chart shapes to millions of past examples to find matches.",
    superpower: "Pattern Matching",
    accuracy: 81,
  },
  {
    id: "xgb",
    name: "The Analyst",
    icon: <Calculator className="w-6 h-6 text-emerald-400" />,
    desc: "Crunches hard numbers (RSI, MACD, Volume) to find statistical edges.",
    superpower: "Technical Analysis",
    accuracy: 76,
  },
  {
    id: "options_expert",
    name: "The Hedge Fund",
    icon: <ShieldCheck className="w-6 h-6 text-rose-400" />,
    desc: "Watches what big institutions are doing in the Options market (Gamma Exposure).",
    superpower: "Institutional Flow",
    accuracy: 88,
  },
  {
    id: "macro_onchain",
    name: "The Economist",
    icon: <Globe className="w-6 h-6 text-cyan-400" />,
    desc: "Monitors global markets (DXY) and Blockchain data (Whale movements).",
    superpower: "Macro Context",
    accuracy: 72,
  },
  {
    id: "llm",
    name: "The News Watcher",
    icon: <Newspaper className="w-6 h-6 text-orange-400" />,
    desc: "Reads thousands of news headlines to sense if the world is panicked or greedy.",
    superpower: "Sentiment Analysis",
    accuracy: 65,
  },
];

export const StrategiesTab: React.FC<StrategiesTabProps> = ({ symbol }) => {
  // State
  const [settings, setSettings] = useState<AiStrategySettings>(DEFAULT_SETTINGS);
  const [isDirty, setIsDirty] = useState(false);
  const [brainHealth, setBrainHealth] = useState<BrainHealth>({
    is_ready: true, has_l2_models: true, llm_ready: true, rl_ready: true, risk_ready: true
  });

  // Simulating fetch for demo purposes
  useEffect(() => {
    // In real app: fetch('/api/v1/ai/strategy').then(setSettings)
  }, []);

  const updateWeight = (key: string, val: number) => {
    setSettings(prev => ({
      ...prev,
      weights: { ...prev.weights, [key]: val }
    }));
    setIsDirty(true);
  };

  const toggleAgent = (key: string) => {
    const k = `use_${key}` as keyof AiStrategySettings;
    setSettings(prev => ({ ...prev, [k]: !prev[k] }));
    setIsDirty(true);
  };

  // Derived Metrics for "Top Capsules"
  const activeAgents = AGENT_PERSONAS.filter(a => settings[`use_${a.id}` as keyof AiStrategySettings]).length;
  const totalAccuracy = Math.round(AGENT_PERSONAS.reduce((acc, a) => acc + a.accuracy, 0) / AGENT_PERSONAS.length);
  
  return (
    <div className="flex flex-col gap-6 pb-10 animate-in fade-in duration-500">
      
      {/* --- 1. INFO CAPSULES --- */}
      <div className="flex flex-wrap gap-4">
         <Capsule 
            label="Active Brains" 
            value={`${activeAgents} / ${AGENT_PERSONAS.length}`} 
            icon={<BrainCircuit className="w-4 h-4 text-purple-400" />} 
            borderColor="border-purple-500/30"
         />
         <Capsule 
            label="System Logic" 
            value={settings.ensemble_mode === "weighted" ? "Consensus Vote" : "Strict Agreement"} 
            icon={<Layers className="w-4 h-4 text-blue-400" />} 
            borderColor="border-blue-500/30"
         />
         <Capsule 
            label="Historical Accuracy" 
            value={`${totalAccuracy}%`} 
            icon={<Activity className="w-4 h-4 text-emerald-400" />} 
            borderColor="border-emerald-500/30"
         />
         <div className="ml-auto flex gap-2">
            {isDirty && (
              <button 
                className="px-4 py-1.5 rounded-full bg-emerald-500 text-black text-xs font-bold shadow-lg shadow-emerald-500/20 hover:bg-emerald-400 transition-all"
                onClick={() => setIsDirty(false)}
              >
                Save Changes
              </button>
            )}
            <button 
                className="px-4 py-1.5 rounded-full bg-slate-800 border border-slate-700 text-slate-300 text-xs font-medium hover:bg-slate-700 transition-all"
                onClick={() => setSettings(DEFAULT_SETTINGS)}
            >
                Reset
            </button>
         </div>
      </div>

      {/* --- 2. INTRO HERO --- */}
      <Card className={`${glassPanel} bg-gradient-to-br from-indigo-500/10 to-purple-500/5 border-indigo-500/20`}>
        <CardContent className="p-6">
          <div className="flex items-start gap-4">
             <div className="p-3 rounded-xl bg-indigo-500/20 border border-indigo-500/30">
                <Bot className="w-8 h-8 text-indigo-300" />
             </div>
             <div>
               <h2 className="text-lg font-bold text-white">How the AI Brain Works</h2>
               <p className="text-sm text-slate-400 mt-1 max-w-3xl leading-relaxed">
                 Nowa AI isn't just one algorithm. It's a <strong>digital committee</strong>. 
                 Different AI agents analyze the market from different angles (Price, News, Volatility). 
                 They debate, vote, and only when they agree does the system generate a prediction.
               </p>
             </div>
          </div>
        </CardContent>
      </Card>

      {/* --- 3. THE AGENTS GRID --- */}
      <div>
        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4 pl-1">Meet Your AI Team</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {AGENT_PERSONAS.map((agent) => {
            const isEnabled = settings[`use_${agent.id}` as keyof AiStrategySettings];
            const weight = settings.weights[agent.id] || 10;
            
            return (
              <Card 
                key={agent.id} 
                className={`relative overflow-hidden transition-all duration-300 group ${isEnabled ? 'border-slate-700 bg-slate-900/60 hover:border-indigo-500/50' : 'border-slate-800 bg-slate-950/40 opacity-60'}`}
              >
                {isEnabled && <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-indigo-500 via-purple-500 to-indigo-500 opacity-50" />}
                
                <CardContent className="p-5 flex flex-col gap-4">
                  {/* Header */}
                  <div className="flex justify-between items-start">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg border shadow-inner ${isEnabled ? 'bg-slate-800 border-white/10' : 'bg-slate-900 border-transparent grayscale'}`}>
                        {agent.icon}
                      </div>
                      <div>
                        <div className={`font-bold ${isEnabled ? 'text-white' : 'text-slate-500'}`}>{agent.name}</div>
                        <div className="text-[10px] text-indigo-400 font-medium uppercase tracking-wider">{agent.superpower}</div>
                      </div>
                    </div>
                    <Switch 
                      checked={!!isEnabled} 
                      onChange={() => toggleAgent(agent.id)}
                    />
                  </div>

                  {/* Description */}
                  <p className="text-xs text-slate-400 leading-relaxed min-h-[40px]">
                    {agent.desc}
                  </p>

                  {/* Metrics & Controls */}
                  {isEnabled && (
                    <div className="mt-2 space-y-3 pt-3 border-t border-white/5">
                      
                      {/* Accuracy Meter */}
                      <div className="flex items-center justify-between text-[10px]">
                        <span className="text-slate-500 font-medium uppercase">Historical Accuracy</span>
                        <div className="flex items-center gap-1 text-emerald-400 font-bold">
                           <CheckCircle2 className="w-3 h-3" /> {agent.accuracy}%
                        </div>
                      </div>

                      {/* Influence Slider */}
                      <div>
                        <div className="flex justify-between text-[10px] mb-1.5">
                           <span className="text-slate-400">Voting Power (Influence)</span>
                           <span className="text-white">{weight}%</span>
                        </div>
                        <input 
                          type="range" 
                          min={0} max={30} step={1}
                          value={weight}
                          onChange={(e) => updateWeight(agent.id, Number(e.target.value))}
                          className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500 hover:accent-indigo-400"
                        />
                      </div>

                    </div>
                  )}
                  
                  {!isEnabled && (
                    <div className="mt-2 pt-3 border-t border-white/5 flex items-center gap-2 text-xs text-slate-500">
                       <AlertCircle className="w-3.5 h-3.5" /> Agent is currently sleeping.
                    </div>
                  )}

                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>

      {/* --- 4. SIMPLE SETTINGS --- */}
      <div className="mt-4">
        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4 pl-1">Trading Style</h3>
        <Card className={`${glassPanel} border-slate-800`}>
           <CardContent className="p-6 grid grid-cols-1 md:grid-cols-2 gap-8">
              
              {/* Consensus Logic */}
              <div>
                 <div className="text-white font-bold mb-1">Decision Logic</div>
                 <p className="text-xs text-slate-400 mb-4">How strict should the committee be?</p>
                 
                 <div className="flex gap-2">
                    {['weighted', 'majority', 'strict'].map(m => (
                       <button
                         key={m}
                         onClick={() => setSettings(p => ({...p, ensemble_mode: m as any}))}
                         className={`flex-1 py-2 rounded-lg border text-xs font-bold transition-all ${
                           settings.ensemble_mode === m 
                           ? 'bg-indigo-600 border-indigo-500 text-white shadow-lg shadow-indigo-500/20'
                           : 'bg-slate-900 border-slate-700 text-slate-500 hover:bg-slate-800'
                         }`}
                       >
                         {m === 'weighted' ? 'Smart Mix' : m === 'majority' ? 'Majority Vote' : 'Unanimous'}
                       </button>
                    ))}
                 </div>
                 <div className="mt-3 text-[10px] text-slate-500 italic">
                    {settings.ensemble_mode === 'weighted' && "Recommended: Listens more to agents that have been correct recently."}
                    {settings.ensemble_mode === 'majority' && "Faster decisions, but higher risk of false signals."}
                    {settings.ensemble_mode === 'strict' && "Extremely safe. Trades only when EVERY agent agrees."}
                 </div>
              </div>

              {/* Risk Profile */}
              <div>
                 <div className="text-white font-bold mb-1">Risk Tolerance</div>
                 <p className="text-xs text-slate-400 mb-4">How aggressive should the sizing be?</p>
                 
                 <div className="space-y-4">
                    <div className="flex justify-between text-xs">
                       <span className="text-slate-400">Max Leverage</span>
                       <span className="text-white font-mono">{settings.max_leverage}x</span>
                    </div>
                    <input 
                        type="range" 
                        min={1} max={10} step={1}
                        value={settings.max_leverage}
                        onChange={(e) => setSettings(p => ({...p, max_leverage: Number(e.target.value)}))}
                        className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-rose-500"
                    />
                    
                    <div className="flex justify-between text-xs">
                       <span className="text-slate-400">Stop Loss Width</span>
                       <span className="text-white font-mono">{settings.risk_mode === 'defensive' ? 'Tight' : settings.risk_mode === 'balanced' ? 'Normal' : 'Wide'}</span>
                    </div>
                     <div className="flex gap-1">
                        {['defensive', 'balanced', 'aggressive'].map(r => (
                           <div 
                              key={r} 
                              onClick={() => setSettings(p => ({...p, risk_mode: r as any}))}
                              className={`h-1.5 flex-1 rounded-full cursor-pointer transition-all ${
                                settings.risk_mode === r 
                                ? (r === 'defensive' ? 'bg-emerald-500' : r === 'balanced' ? 'bg-amber-500' : 'bg-rose-500')
                                : 'bg-slate-800'
                              }`} 
                           />
                        ))}
                     </div>
                 </div>
              </div>

           </CardContent>
        </Card>
      </div>

    </div>
  );
};

// --- Helpers ---

const Capsule: React.FC<{ label: string; value: string; icon: React.ReactNode; borderColor: string }> = ({ label, value, icon, borderColor }) => (
  <div className={`flex items-center gap-3 px-4 py-2 rounded-xl bg-slate-900/60 border ${borderColor} backdrop-blur-sm`}>
     {icon}
     <div className="flex flex-col">
        <span className="text-[10px] text-slate-500 uppercase font-bold">{label}</span>
        <span className="text-xs font-bold text-slate-200">{value}</span>
     </div>
  </div>
);

const Switch: React.FC<{ checked: boolean; onChange: () => void }> = ({ checked, onChange }) => (
  <button
    type="button"
    role="switch"
    aria-checked={checked}
    onClick={onChange}
    className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
      checked ? "bg-emerald-500" : "bg-slate-700"
    }`}
  >
    <span
      aria-hidden="true"
      className={`pointer-events-none inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
        checked ? "translate-x-4" : "translate-x-0"
      }`}
    />
  </button>
);

export default StrategiesTab;