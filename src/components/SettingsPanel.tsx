"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  Card, CardHeader, CardTitle, CardContent,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";

import {
  Tabs, TabsList, TabsTrigger, TabsContent,
} from "@/components/ui/tabs";
import {
  Select, SelectTrigger, SelectValue, SelectContent, SelectItem,
} from "@/components/ui/select";

/* === Glass tokens (aligned to MarsBotUI) === */
const glassPanel =
  "backdrop-blur-xl bg-white/8 dark:bg-white/5 border border-white/15";
const glassCard =
  "rounded-2xl " + glassPanel + " shadow-[0_8px_30px_rgba(0,0,0,0.25)]";
const glassHover =
  "transition-transform duration-300 hover:-translate-y-0.5 hover:shadow-[0_12px_40px_rgba(0,0,0,0.35)]";
const faintText = "text-slate-400/80";
const softText = "text-slate-200";

/* === Types/Defaults (same schema) === */
type Mode = "futures" | "options" | "spot";
type PositionSizing = "fixed" | "kelly" | "vol-target";
type TimeInForce = "IOC" | "FOK" | "GTC";
type OrderType = "limit" | "market" | "postOnly";
type Ensemble = "weighted" | "stacking" | "bayesian";

type ModelCfg = { enabled: boolean; weight: number };
type ModelsMap = Record<string, ModelCfg>;

export type Settings = {
  active_symbols: string[];
  exchanges: string[];
  mode: Mode;
  base_currency: "USD" | "USDT" | "USDC";
  max_leverage: number;
  max_concurrent: number;
  risk_per_trade_pct: number;
  daily_loss_cap_pct: number;
  kill_switch_drawdown_pct: number;
  position_sizing: PositionSizing;
  slippage_bps: number;
  fee_bps: number;
  time_in_force: TimeInForce;
  order_type: OrderType;
  hedge_enabled: boolean;
  stop_type: "atr" | "fixed" | "vol";
  stop_atr_mult: number;
  take_profit_pct: number;
  trailing_stop_pct: number;
  ensemble_method: Ensemble;
  models: ModelsMap;
  autotune_enabled: boolean;
  autotune_hours: number;
  autotune_trials: number;
  halt_on_vol_spike: boolean;
  max_spread_bps: number;
  blocklist_symbols: string[];
  telegram_enabled: boolean;
  telegram_chat_id: string;
  alert_thresholds: { drawdown_pct: number; pnl_pct: number };
};

const DEFAULTS: Settings = {
  active_symbols: ["BTC-PERP", "ETH-PERP", "SOL-PERP"],
  exchanges: ["Binance", "Bybit"],
  mode: "futures",
  base_currency: "USDT",
  max_leverage: 3,
  max_concurrent: 4,
  risk_per_trade_pct: 1.0,
  daily_loss_cap_pct: 5.0,
  kill_switch_drawdown_pct: 15.0,
  position_sizing: "vol-target",
  slippage_bps: 2,
  fee_bps: 2,
  time_in_force: "GTC",
  order_type: "limit",
  hedge_enabled: true,
  stop_type: "atr",
  stop_atr_mult: 2.0,
  take_profit_pct: 1.2,
  trailing_stop_pct: 0.6,
  ensemble_method: "weighted",
  models: { TFT: { enabled: true, weight: 0.5 }, TCN: { enabled: true, weight: 0.3 }, XGB: { enabled: true, weight: 0.2 } },
  autotune_enabled: false,
  autotune_hours: 12,
  autotune_trials: 20,
  halt_on_vol_spike: true,
  max_spread_bps: 20,
  blocklist_symbols: [],
  telegram_enabled: false,
  telegram_chat_id: "",
  alert_thresholds: { drawdown_pct: 8, pnl_pct: 3 },
};

const LS_KEY = "nowory.settings.v1";

/* === utils/validation === */
const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n));
const toNum = (v: string, fallback = 0) => (Number.isFinite(Number(v)) ? Number(v) : fallback);

function loadFromLocalStorage(): Settings {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return DEFAULTS;
    return { ...DEFAULTS, ...JSON.parse(raw) };
  } catch {
    return DEFAULTS;
  }
}
function saveToLocalStorage(s: Settings) {
  localStorage.setItem(LS_KEY, JSON.stringify(s));
}
type Errors = Partial<Record<keyof Settings, string>> & Record<string, string>;
function validate(s: Settings): Errors {
  const e: Errors = {};
  if (!s.active_symbols.length) e.active_symbols = "Add at least one symbol.";
  if (!s.exchanges.length) e.exchanges = "Select at least one exchange.";
  if (s.max_leverage < 1 || s.max_leverage > 25) e.max_leverage = "1–25 allowed.";
  if (s.risk_per_trade_pct <= 0 || s.risk_per_trade_pct > 5) e.risk_per_trade_pct = "0–5% allowed.";
  if (s.daily_loss_cap_pct < 0 || s.daily_loss_cap_pct > 20) e.daily_loss_cap_pct = "0–20% allowed.";
  if (s.kill_switch_drawdown_pct < 0 || s.kill_switch_drawdown_pct > 50) e.kill_switch_drawdown_pct = "0–50% allowed.";
  if (s.stop_type === "atr" && (s.stop_atr_mult <= 0 || s.stop_atr_mult > 5)) e.stop_atr_mult = "0–5 allowed.";
  const w = Object.values(s.models).reduce((acc, m) => acc + (m.enabled ? m.weight : 0), 0);
  if (w <= 0) e.models = "At least one model must be enabled.";
  return e;
}

/* === small glass “chip editor” === */
function ChipInput({
  label, value, onChange, placeholder, error,
}: { label: string; value: string[]; onChange: (v: string[]) => void; placeholder?: string; error?: string }) {
  const [text, setText] = useState("");
  const add = (tok: string) => {
    const t = tok.trim();
    if (!t) return;
    if (!value.includes(t)) onChange([...value, t]);
    setText("");
  };
  return (
    <div>
      <Label className="mb-2 block">{label}</Label>
      <div className="flex flex-wrap items-center gap-2 mb-2">
        {value.map((v) => (
          <span key={v} className={`px-2 py-1 rounded-lg border border-white/20 bg-white/10 text-xs ${softText}`}>
            {v}
            <button
              className="ml-2 opacity-70 hover:opacity-100"
              onClick={() => onChange(value.filter((x) => x !== v))}
              type="button"
              aria-label={`Remove ${v}`}
            >
              ×
            </button>
          </span>
        ))}
      </div>
      <Input
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === ",") {
            e.preventDefault();
            add(text);
          }
        }}
        placeholder={placeholder ?? "Type and press Enter"}
        className="h-11"
      />
      {error && <div className="mt-1 text-xs text-rose-400">{error}</div>}
    </div>
  );
}

/* === glass sub-tile wrapper === */
const Tile: React.FC<React.PropsWithChildren<{ className?: string }>> = ({ children, className = "" }) => (
  <div className={`${glassPanel} rounded-xl p-4 ${className}`}>{children}</div>
);

/* === main panel === */
export function SettingsPanel({ embedded = false }: { embedded?: boolean }) {
  const [settings, setSettings] = useState<Settings>(DEFAULTS);
  const [tab, setTab] = useState("global");
  const [saving, setSaving] = useState(false);
  const errors = useMemo(() => validate(settings), [settings]);

  useEffect(() => { setSettings(loadFromLocalStorage()); }, []);

  const save = async () => {
    const errs = validate(settings);
    if (Object.keys(errs).length) return alert("Fix validation errors before saving.");
    setSaving(true);
    await new Promise((r) => setTimeout(r, 350)); // demo
    saveToLocalStorage(settings);
    setSaving(false);
    alert("Settings saved (demo).");
  };

  const resetDefaults = () => setSettings(DEFAULTS);

  const PanelBody = (
    <div className="space-y-6 pb-24">
      {/* Fancy segmented tabs to match shell */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className={`text-xs ${faintText}`}>Strategy Settings (demo — localStorage)</div>
      </div>

      <Tabs value={tab} onValueChange={setTab} className="w-full">
        <TabsList
  className={
    `w-full rounded-xl border border-white/15 ${glassPanel}
     flex gap-1 overflow-x-auto px-1 py-1
     md:grid md:grid-cols-6 md:overflow-visible`
  }
>
  {["global","risk","execution","stops","models","controls"].map((k) => (
    <TabsTrigger
      key={k}
      value={k}
      className={
        `rounded-lg px-3 py-1.5 whitespace-nowrap text-xs md:text-[11px]
         data-[state=active]:bg-white/20 data-[state=active]:text-white
         hover:bg-white/10`
      }
    >
      {k === "stops" ? "Stops/Targets" :
       k === "models" ? "Signals/Models" :
       k === "controls" ? "Controls & Alerts" :
       k.charAt(0).toUpperCase()+k.slice(1)}
    </TabsTrigger>
  ))}
</TabsList>


        {/* GLOBAL */}
        <TabsContent value="global">
          <Card className={`${glassCard} ${glassHover}`}>
            <CardHeader><CardTitle className={softText}>Global</CardTitle></CardHeader>
            <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Tile>
                <ChipInput
                  label="Active Symbols"
                  value={settings.active_symbols}
                  onChange={(v) => setSettings({ ...settings, active_symbols: v })}
                  placeholder="e.g. BTC-PERP, ETH-PERP"
                  error={errors.active_symbols}
                />
              </Tile>
              <Tile>
                <ChipInput
                  label="Exchanges"
                  value={settings.exchanges}
                  onChange={(v) => setSettings({ ...settings, exchanges: v })}
                  placeholder="e.g. Binance, Bybit"
                  error={errors.exchanges}
                />
              </Tile>
              <Tile>
                <Label>Mode</Label>
                <Select value={settings.mode} onValueChange={(v: Mode) => setSettings({ ...settings, mode: v })}>
                  <SelectTrigger className="mt-2"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="futures">Futures</SelectItem>
                    <SelectItem value="options">Options</SelectItem>
                    <SelectItem value="spot">Spot</SelectItem>
                  </SelectContent>
                </Select>
              </Tile>
              <Tile>
                <Label>Base Currency</Label>
                <Select
                  value={settings.base_currency}
                  onValueChange={(v: "USD" | "USDT" | "USDC") => setSettings({ ...settings, base_currency: v })}
                >
                  <SelectTrigger className="mt-2"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="USD">USD</SelectItem>
                    <SelectItem value="USDT">USDT</SelectItem>
                    <SelectItem value="USDC">USDC</SelectItem>
                  </SelectContent>
                </Select>
              </Tile>
            </CardContent>
          </Card>
        </TabsContent>

        {/* RISK */}
        <TabsContent value="risk">
          <Card className={`${glassCard} ${glassHover}`}>
            <CardHeader><CardTitle className={softText}>Risk</CardTitle></CardHeader>
            <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Tile>
                <Label>Max Leverage</Label>
                <Input className="mt-2" type="number" min={1} max={25}
                  value={settings.max_leverage}
                  onChange={(e) => setSettings({ ...settings, max_leverage: clamp(toNum(e.target.value, 3), 1, 25) })}
                />
                {errors.max_leverage && <div className="text-xs text-rose-400 mt-1">{errors.max_leverage}</div>}
              </Tile>

              <Tile>
                <Label>Max Concurrent Positions</Label>
                <Input className="mt-2" type="number" min={1} max={20}
                  value={settings.max_concurrent}
                  onChange={(e) => setSettings({ ...settings, max_concurrent: clamp(toNum(e.target.value, 4), 1, 20) })}
                />
              </Tile>

              <Tile>
                <Label>Risk / Trade (%)</Label>
                <div className="mt-3">
                  <Slider value={[settings.risk_per_trade_pct]} min={0.1} max={5} step={0.1}
                    onValueChange={([v]) => setSettings({ ...settings, risk_per_trade_pct: v })}
                  />
                  <div className="mt-1 text-sm">{settings.risk_per_trade_pct.toFixed(1)}%</div>
                  {errors.risk_per_trade_pct && <div className="text-xs text-rose-400 mt-1">{errors.risk_per_trade_pct}</div>}
                </div>
              </Tile>

              <Tile>
                <Label>Daily Loss Cap (%)</Label>
                <Input className="mt-2" type="number" min={0} max={20} step={0.1}
                  value={settings.daily_loss_cap_pct}
                  onChange={(e) => setSettings({ ...settings, daily_loss_cap_pct: clamp(toNum(e.target.value, 5), 0, 20) })}
                />
              </Tile>

              <Tile>
                <Label>Kill Switch Drawdown (%)</Label>
                <Input className="mt-2" type="number" min={0} max={50} step={0.1}
                  value={settings.kill_switch_drawdown_pct}
                  onChange={(e) => setSettings({ ...settings, kill_switch_drawdown_pct: clamp(toNum(e.target.value, 15), 0, 50) })}
                />
              </Tile>

              <Tile>
                <Label>Position Sizing</Label>
                <Select value={settings.position_sizing} onValueChange={(v: PositionSizing) => setSettings({ ...settings, position_sizing: v })}>
                  <SelectTrigger className="mt-2"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="fixed">Fixed</SelectItem>
                    <SelectItem value="kelly">Kelly</SelectItem>
                    <SelectItem value="vol-target">Vol Target</SelectItem>
                  </SelectContent>
                </Select>
              </Tile>
            </CardContent>
          </Card>
        </TabsContent>

        {/* EXECUTION */}
        <TabsContent value="execution">
          <Card className={`${glassCard} ${glassHover}`}>
            <CardHeader><CardTitle className={softText}>Execution</CardTitle></CardHeader>
            <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Tile>
                <Label>Slippage (bps)</Label>
                <Input className="mt-2" type="number" min={0} max={100}
                  value={settings.slippage_bps}
                  onChange={(e) => setSettings({ ...settings, slippage_bps: clamp(toNum(e.target.value, 2), 0, 100) })}
                />
              </Tile>
              <Tile>
                <Label>Fee (bps)</Label>
                <Input className="mt-2" type="number" min={0} max={30}
                  value={settings.fee_bps}
                  onChange={(e) => setSettings({ ...settings, fee_bps: clamp(toNum(e.target.value, 2), 0, 30) })}
                />
              </Tile>
              <Tile>
                <Label>Time in Force</Label>
                <Select value={settings.time_in_force} onValueChange={(v: TimeInForce) => setSettings({ ...settings, time_in_force: v })}>
                  <SelectTrigger className="mt-2"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="IOC">IOC</SelectItem>
                    <SelectItem value="FOK">FOK</SelectItem>
                    <SelectItem value="GTC">GTC</SelectItem>
                  </SelectContent>
                </Select>
              </Tile>
              <Tile>
                <Label>Order Type</Label>
                <Select value={settings.order_type} onValueChange={(v: OrderType) => setSettings({ ...settings, order_type: v })}>
                  <SelectTrigger className="mt-2"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="limit">Limit</SelectItem>
                    <SelectItem value="market">Market</SelectItem>
                    <SelectItem value="postOnly">Post Only</SelectItem>
                  </SelectContent>
                </Select>
              </Tile>
              <Tile className="flex items-center gap-3">
                <Switch checked={settings.hedge_enabled} onCheckedChange={(v) => setSettings({ ...settings, hedge_enabled: v })} />
                <Label>Enable Hedging</Label>
              </Tile>
            </CardContent>
          </Card>
        </TabsContent>

        {/* STOPS */}
        <TabsContent value="stops">
          <Card className={`${glassCard} ${glassHover}`}>
            <CardHeader><CardTitle className={softText}>Stops & Targets</CardTitle></CardHeader>
            <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Tile>
                <Label>Stop Type</Label>
                <Select value={settings.stop_type} onValueChange={(v: "atr" | "fixed" | "vol") => setSettings({ ...settings, stop_type: v })}>
                  <SelectTrigger className="mt-2"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="atr">ATR</SelectItem>
                    <SelectItem value="fixed">Fixed %</SelectItem>
                    <SelectItem value="vol">Volatility</SelectItem>
                  </SelectContent>
                </Select>
              </Tile>
              <Tile>
                <Label>Stop ATR Multiplier</Label>
                <Input className="mt-2" type="number" step={0.1} min={0.1} max={5}
                  value={settings.stop_atr_mult}
                  onChange={(e) => setSettings({ ...settings, stop_atr_mult: clamp(toNum(e.target.value, 2), 0.1, 5) })}
                />
              </Tile>
              <Tile>
                <Label>Take Profit (%)</Label>
                <Input className="mt-2" type="number" step={0.1} min={0} max={10}
                  value={settings.take_profit_pct}
                  onChange={(e) => setSettings({ ...settings, take_profit_pct: clamp(toNum(e.target.value, 1.2), 0, 10) })}
                />
              </Tile>
              <Tile>
                <Label>Trailing Stop (%)</Label>
                <Input className="mt-2" type="number" step={0.1} min={0} max={10}
                  value={settings.trailing_stop_pct}
                  onChange={(e) => setSettings({ ...settings, trailing_stop_pct: clamp(toNum(e.target.value, 0.6), 0, 10) })}
                />
              </Tile>
            </CardContent>
          </Card>
        </TabsContent>

        {/* MODELS */}
        <TabsContent value="models">
          <Card className={`${glassCard} ${glassHover}`}>
            <CardHeader><CardTitle className={softText}>Signals & Models</CardTitle></CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Tile>
                  <Label>Ensemble Method</Label>
                  <Select value={settings.ensemble_method} onValueChange={(v: Ensemble) => setSettings({ ...settings, ensemble_method: v })}>
                    <SelectTrigger className="mt-2"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="weighted">Weighted</SelectItem>
                      <SelectItem value="stacking">Stacking</SelectItem>
                      <SelectItem value="bayesian">Bayesian</SelectItem>
                    </SelectContent>
                  </Select>
                </Tile>
                <Tile className="grid grid-cols-3 gap-4">
                  {Object.entries(settings.models).map(([name, cfg]) => (
                    <div key={name} className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold">{name}</span>
                        <Switch checked={cfg.enabled}
                          onCheckedChange={(v) => setSettings({ ...settings, models: { ...settings.models, [name]: { ...cfg, enabled: v } } })}
                        />
                      </div>
                      <Label>Weight</Label>
                      <Input type="number" step={0.05} min={0} max={1}
                        value={cfg.weight}
                        onChange={(e) => setSettings({
                          ...settings,
                          models: { ...settings.models, [name]: { ...cfg, weight: clamp(toNum(e.target.value, cfg.weight), 0, 1) } },
                        })}
                      />
                    </div>
                  ))}
                </Tile>
              </div>
              {validate(settings).models && <div className="text-xs text-rose-400">{validate(settings).models}</div>}
              <Tile className="grid grid-cols-3 gap-6">
                <div>
                  <Label>Auto-Tune</Label>
                  <div className="mt-2 flex items-center gap-3">
                    <Switch checked={settings.autotune_enabled}
                            onCheckedChange={(v) => setSettings({ ...settings, autotune_enabled: v })} />
                    <span>Enable periodic search</span>
                  </div>
                </div>
                <div>
                  <Label>Auto-Tune Hours</Label>
                  <Input className="mt-2" type="number" min={1} max={168}
                         value={settings.autotune_hours}
                         onChange={(e) => setSettings({ ...settings, autotune_hours: clamp(toNum(e.target.value, 12), 1, 168) })}/>
                </div>
                <div>
                  <Label>Trials / Cycle</Label>
                  <Input className="mt-2" type="number" min={1} max={500}
                         value={settings.autotune_trials}
                         onChange={(e) => setSettings({ ...settings, autotune_trials: clamp(toNum(e.target.value, 20), 1, 500) })}/>
                </div>
              </Tile>
            </CardContent>
          </Card>
        </TabsContent>

        {/* CONTROLS */}
        <TabsContent value="controls">
          <Card className={`${glassCard} ${glassHover}`}>
            <CardHeader><CardTitle className={softText}>Controls & Alerts</CardTitle></CardHeader>
            <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Tile className="flex items-center gap-3">
                <Switch checked={settings.halt_on_vol_spike}
                        onCheckedChange={(v) => setSettings({ ...settings, halt_on_vol_spike: v })}/>
                <Label>Halt on Volatility Spike</Label>
              </Tile>
              <Tile>
                <Label>Max Spread (bps)</Label>
                <Input className="mt-2" type="number" min={0} max={200}
                       value={settings.max_spread_bps}
                       onChange={(e) => setSettings({ ...settings, max_spread_bps: clamp(toNum(e.target.value, 20), 0, 200) })}/>
              </Tile>
              <Tile className="md:col-span-2">
                <ChipInput
                  label="Blocklist Symbols"
                  value={settings.blocklist_symbols}
                  onChange={(v) => setSettings({ ...settings, blocklist_symbols: v })}
                  placeholder="e.g. XRP-PERP"
                />
              </Tile>
              <Tile className="flex items-center gap-3">
                <Switch checked={settings.telegram_enabled}
                        onCheckedChange={(v) => setSettings({ ...settings, telegram_enabled: v })}/>
                <Label>Telegram Alerts</Label>
              </Tile>
              <Tile>
                <Label>Telegram Chat ID</Label>
                <Input className="mt-2" placeholder="123456789"
                       value={settings.telegram_chat_id}
                       onChange={(e) => setSettings({ ...settings, telegram_chat_id: e.target.value })}/>
              </Tile>
              <Tile className="md:col-span-2">
                <Label>Alert Thresholds</Label>
                <div className="grid grid-cols-2 gap-3 mt-2">
                  <div>
                    <Label className="text-xs">Drawdown %</Label>
                    <Input className="mt-1" type="number" min={0} max={100}
                           value={settings.alert_thresholds.drawdown_pct}
                           onChange={(e) => setSettings({ ...settings, alert_thresholds: { ...settings.alert_thresholds, drawdown_pct: clamp(toNum(e.target.value, 8), 0, 100) } })}/>
                  </div>
                  <div>
                    <Label className="text-xs">PnL %</Label>
                    <Input className="mt-1" type="number" min={0} max={100}
                           value={settings.alert_thresholds.pnl_pct}
                           onChange={(e) => setSettings({ ...settings, alert_thresholds: { ...settings.alert_thresholds, pnl_pct: clamp(toNum(e.target.value, 3), 0, 100) } })}/>
                  </div>
                </div>
              </Tile>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Sticky action bar */}
      <div className={`sticky bottom-3 z-10`}>
        <div className={`rounded-xl ${glassPanel} border border-white/15 p-3 flex flex-wrap gap-2 justify-end`}>
          <Button variant="outline" onClick={resetDefaults}>Reset</Button>
          <Button variant="outline" onClick={() => {
            const blob = new Blob([JSON.stringify(settings, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a"); a.href = url; a.download = "settings.json"; a.click(); URL.revokeObjectURL(url);
          }}>Export JSON</Button>
          <label className="inline-flex">
            <input hidden type="file" accept="application/json"
                   onChange={(e) => {
                     const f = e.target.files?.[0]; if (!f) return;
                     const fr = new FileReader(); fr.onload = () => {
                       try { setSettings({ ...DEFAULTS, ...JSON.parse(String(fr.result || "{}")) }); }
                       catch { alert("Invalid JSON"); }
                     }; fr.readAsText(f);
                   }} />
            <span className="px-4 py-2 rounded-md border border-white/20 cursor-pointer">Import JSON</span>
          </label>
          <Button onClick={save} disabled={saving}>{saving ? "Saving…" : "Save"}</Button>
        </div>
      </div>
    </div>
  );

  if (embedded) return PanelBody;
  return (
    <Card className={`${glassCard} ${glassHover}`}>
      <CardHeader><CardTitle className={softText}>Strategy Settings</CardTitle></CardHeader>
      <CardContent>{PanelBody}</CardContent>
    </Card>
  );
}
