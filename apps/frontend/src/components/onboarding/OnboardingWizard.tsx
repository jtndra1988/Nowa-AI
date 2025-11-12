// src/components/onboarding/OnboardingWizard.tsx
"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ShieldAlert,
  Gauge,
  Server,
  Send,
  CheckCircle2,
  Link2,
  ArrowRight,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { EXCHANGES } from "@/lib/api";
import { useClientProfile } from "@/lib/profile";

const steps = ["Risk", "Exchanges", "Telegram", "Review"] as const;
type Step = (typeof steps)[number];

const OnboardingWizard: React.FC = () => {
  const { profile, saveProfile, patchProfile } = useClientProfile();
  const [active, setActive] = useState<Step>("Risk");

  if (!profile) return null;
  if (profile.isOnboarded) return null;

  const updateProfile = (patch: Partial<typeof profile>) => {
    saveProfile({ ...profile, ...patch });
  };

  const setRisk = (level: "conservative" | "balanced" | "aggressive") => {
    const presets = {
      conservative: { maxDailyDrawdown: 2, maxPerTradeRisk: 0.5, leverageCap: 2 },
      balanced: { maxDailyDrawdown: 5, maxPerTradeRisk: 1, leverageCap: 5 },
      aggressive: { maxDailyDrawdown: 10, maxPerTradeRisk: 2, leverageCap: 10 },
    }[level];

    updateProfile({
      risk: {
        level,
        ...presets,
        allowShorting: level !== "conservative",
        allowPerps: true,
        allowOptions: level === "aggressive",
      },
    });
  };

  const toggleExchange = (ex: string) => {
    const existing = profile.exchanges || [];
    const idx = existing.findIndex((e) => e.exchange === ex);
    let next = existing;
    if (idx === -1) {
      next = [
        ...existing,
        {
          id: `${ex}-${Date.now()}`,
          exchange: ex,
          label: `${ex} (not connected)`,
          hasApiKey: false,
          readOnly: true,
        },
      ];
    } else {
      next = existing.filter((e) => e.exchange !== ex);
    }
    updateProfile({ exchanges: next });
  };

  const markTelegramConnected = () => {
    updateProfile({
      telegram: {
        ...(profile.telegram || {}),
        isConnected: true,
      },
    });
  };

  const complete = () => {
    saveProfile({
      ...profile,
      isOnboarded: true,
    });
  };

  return (
    <div className="fixed inset-0 z-40 bg-black/70 backdrop-blur-xl flex items-center justify-center">
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-4xl rounded-3xl bg-slate-950/95 border border-indigo-500/25 shadow-[0_24px_110px_rgba(15,23,42,0.95)] p-6 md:p-8 space-y-5"
      >
        {/* Stepper */}
        <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400">
          {steps.map((s) => {
            const isActive = s === active;
            const isDone = steps.indexOf(s) < steps.indexOf(active);
            return (
              <button
                key={s}
                onClick={() => setActive(s)}
                className={cn(
                  "px-3 py-1.5 rounded-2xl flex items-center gap-1.5 border transition-all",
                  isActive
                    ? "border-indigo-400/80 bg-indigo-500/10 text-indigo-200"
                    : isDone
                    ? "border-emerald-500/40 bg-emerald-500/5 text-emerald-300"
                    : "border-white/10 hover:bg-white/5"
                )}
              >
                {isDone && <CheckCircle2 className="w-3 h-3" />}
                {s}
              </button>
            );
          })}
          <span className="ml-auto text-[10px] text-slate-500">
            Workspace: <span className="text-slate-200">{profile.name}</span> • {profile.email}
          </span>
        </div>

        <AnimatePresence mode="wait">
          {/* Risk Step */}
          {active === "Risk" && (
            <motion.div
              key="risk"
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -16 }}
              className="grid md:grid-cols-4 gap-4 items-stretch"
            >
              <div className="md:col-span-1 space-y-2 text-xs text-slate-400">
                <div className="flex items-center gap-2">
                  <Gauge className="w-4 h-4 text-indigo-400" />
                  <h2 className="text-sm font-semibold text-slate-100">Risk profile</h2>
                </div>
                <p>
                  These caps bind all models. Nowa won&apos;t exceed these limits even if signals are strong.
                </p>
              </div>
              <div className="md:col-span-3 grid md:grid-cols-3 gap-3">
                {[
                  ["conservative", "Max DD 2%, risk 0.5%, lev 2x"],
                  ["balanced", "Max DD 5%, risk 1%, lev 5x"],
                  ["aggressive", "Max DD 10%, risk 2%, lev 10x"],
                ].map(([level, desc]) => (
                  <button
                    key={level}
                    onClick={() => setRisk(level as any)}
                    className={cn(
                      "h-full rounded-2xl border px-3 py-3 text-left text-xs space-y-1.5 bg-white/2 hover:bg-white/6 transition-all",
                      profile.risk.level === level
                        ? "border-indigo-400/90 shadow-[0_0_24px_rgba(79,70,229,0.7)] text-indigo-100"
                        : "border-white/10 text-slate-300"
                    )}
                  >
                    <div className="font-semibold capitalize">{level}</div>
                    <div className="text-[10px] text-slate-400">{desc}</div>
                  </button>
                ))}
                {/* Fine-tune */}
                <div className="md:col-span-3 mt-1 grid grid-cols-3 gap-2 text-[10px] text-slate-400">
                  <div>
                    Max daily drawdown (%)
                    <input
                      type="number"
                      min={1}
                      max={30}
                      value={profile.risk.maxDailyDrawdown}
                      onChange={(e) =>
                        updateProfile({
                          risk: {
                            ...profile.risk,
                            maxDailyDrawdown: Number(e.target.value || 0),
                            level: "custom",
                          },
                        })
                      }
                      className="mt-0.5 w-full rounded-xl bg-slate-900/70 border border-white/10 px-2 py-1"
                    />
                  </div>
                  <div>
                    Max risk per trade (%)
                    <input
                      type="number"
                      min={0.1}
                      max={10}
                      step={0.1}
                      value={profile.risk.maxPerTradeRisk}
                      onChange={(e) =>
                        updateProfile({
                          risk: {
                            ...profile.risk,
                            maxPerTradeRisk: Number(e.target.value || 0),
                            level: "custom",
                          },
                        })
                      }
                      className="mt-0.5 w-full rounded-xl bg-slate-900/70 border border-white/10 px-2 py-1"
                    />
                  </div>
                  <div>
                    Leverage cap (x)
                    <input
                      type="number"
                      min={1}
                      max={100}
                      value={profile.risk.leverageCap}
                      onChange={(e) =>
                        updateProfile({
                          risk: {
                            ...profile.risk,
                            leverageCap: Number(e.target.value || 1),
                            level: "custom",
                          },
                        })
                      }
                      className="mt-0.5 w-full rounded-xl bg-slate-900/70 border border-white/10 px-2 py-1"
                    />
                  </div>
                </div>
              </div>
              <div className="md:col-span-4 flex justify-end">
                <button
                  onClick={() => setActive("Exchanges")}
                  className="px-4 py-2 rounded-2xl bg-indigo-500/90 text-xs text-white flex items-center gap-1 hover:bg-indigo-400"
                >
                  Next: Exchanges
                  <Server className="w-3 h-3" />
                </button>
              </div>
            </motion.div>
          )}

          {/* Exchanges Step */}
          {active === "Exchanges" && (
            <motion.div
              key="exchanges"
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -16 }}
              className="space-y-4"
            >
              <div className="flex items-center gap-2 text-sm text-slate-100">
                <Server className="w-4 h-4 text-indigo-400" />
                Select execution venues
              </div>
              <p className="text-[10px] text-slate-400">
                Choose where this client is allowed to trade. API keys are submitted to backend only (design note); this
                UI is safe for demos.
              </p>
              <div className="grid md:grid-cols-5 gap-3 text-xs">
                {EXCHANGES.map((ex) => {
                  const enabled = (profile.exchanges || []).some((e) => e.exchange === ex);
                  return (
                    <button
                      key={ex}
                      onClick={() => toggleExchange(ex)}
                      className={cn(
                        "rounded-2xl border px-3 py-2 flex flex-col gap-1 hover:bg-white/5 transition-all",
                        enabled
                          ? "border-emerald-400/80 text-emerald-200 shadow-[0_0_18px_rgba(45,212,191,0.55)]"
                          : "border-white/10 text-slate-300"
                      )}
                    >
                      <span className="font-semibold text-[11px]">{ex}</span>
                      <span className="text-[9px] text-slate-500">
                        {enabled ? "Enabled for routing" : "Click to enable"}
                      </span>
                    </button>
                  );
                })}
              </div>
              <div className="rounded-2xl border border-yellow-500/40 bg-yellow-500/5 px-3 py-2 flex gap-2 items-start text-[9px] text-yellow-200">
                <ShieldAlert className="w-3 h-3 mt-0.5" />
                <p>
                  Implementation note: call your backend like <code>/api/client/exchanges</code> with encrypted API
                  credentials. Never keep secrets in localStorage.
                </p>
              </div>
              <div className="flex justify-between text-[10px]">
                <button
                  onClick={() => setActive("Risk")}
                  className="px-3 py-1.5 rounded-2xl border border-white/15 text-slate-300 hover:bg-white/5"
                >
                  Back
                </button>
                <button
                  onClick={() => setActive("Telegram")}
                  className="px-4 py-2 rounded-2xl bg-indigo-500/90 text-white flex items-center gap-1 hover:bg-indigo-400"
                >
                  Next: Telegram routing
                  <Send className="w-3 h-3" />
                </button>
              </div>
            </motion.div>
          )}

          {/* Telegram Step */}
          {active === "Telegram" && (
            <motion.div
              key="telegram"
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -16 }}
              className="space-y-4"
            >
              <div className="flex items-center gap-2 text-sm text-slate-100">
                <Send className="w-4 h-4 text-sky-400" />
                Connect Telegram routing
              </div>
              <p className="text-[10px] text-slate-400">
                Map execution alerts & risk events to a Telegram user/channel for this client.
              </p>
              <div className="grid md:grid-cols-3 gap-3 text-[10px]">
                <div className="md:col-span-2 space-y-2">
                  <label className="text-slate-300">Telegram handle or channel</label>
                  <input
                    placeholder="@yourdesk or https://t.me/your_channel"
                    value={profile.telegram?.handle ?? ""}
                    onChange={(e) =>
                      updateProfile({
                        telegram: {
                          ...(profile.telegram || { isConnected: false }),
                          handle: e.target.value,
                        },
                      })
                    }
                    className="w-full rounded-2xl bg-slate-900/70 border border-white/10 px-3 py-2"
                  />
                  <p className="text-[9px] text-slate-500">
                    In real flow: click &ldquo;Generate link&rdquo; → backend returns a secure deep link for your bot.
                  </p>
                  <button
                    type="button"
                    onClick={markTelegramConnected}
                    className="mt-1 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-2xl bg-sky-500/80 text-[10px] text-white hover:bg-sky-400"
                  >
                    Simulate Telegram connection
                    <Link2 className="w-3 h-3" />
                  </button>
                </div>
                <div className="space-y-1.5 rounded-2xl border border-white/10 bg-slate-900/80 p-3">
                  <div className="font-semibold text-slate-100 text-[10px]">Example routing</div>
                  <ul className="list-disc list-inside space-y-0.5 text-[9px] text-slate-400">
                    <li>Hybrid model conviction &gt; 70% → PM channel</li>
                    <li>Auto-hedge / circuit breaker → Risk channel</li>
                    <li>Order fills / cancels → Exec log channel</li>
                  </ul>
                </div>
              </div>
              <div className="flex justify-between text-[10px]">
                <button
                  onClick={() => setActive("Exchanges")}
                  className="px-3 py-1.5 rounded-2xl border border-white/15 text-slate-300 hover:bg-white/5"
                >
                  Back
                </button>
                <button
                  onClick={() => setActive("Review")}
                  className="px-4 py-2 rounded-2xl bg-indigo-500/90 text-white flex items-center gap-1 hover:bg-indigo-400"
                >
                  Next: Review & Launch
                  <CheckCircle2 className="w-3 h-3" />
                </button>
              </div>
            </motion.div>
          )}

          {/* Review Step */}
          {active === "Review" && (
            <motion.div
              key="review"
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -16 }}
              className="space-y-4 text-[10px]"
            >
              <div className="flex items-center gap-2 text-sm text-slate-100">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                Review & confirm client workspace
              </div>
              <div className="grid md:grid-cols-3 gap-3">
                <div className="rounded-2xl border border-white/10 p-3 space-y-1.5">
                  <div className="font-semibold text-slate-200 text-[11px]">Identity</div>
                  <div className="text-slate-400">{profile.name}</div>
                  <div className="text-slate-500">{profile.email}</div>
                </div>
                <div className="rounded-2xl border border-white/10 p-3 space-y-1.5">
                  <div className="font-semibold text-slate-200 text-[11px]">Risk</div>
                  <div className="text-slate-400 capitalize">
                    Mode: {profile.risk.level}
                  </div>
                  <div className="text-slate-500">
                    Max DD {profile.risk.maxDailyDrawdown}% • Trade risk{" "}
                    {profile.risk.maxPerTradeRisk}% • Lev x{profile.risk.leverageCap}
                  </div>
                </div>
                <div className="rounded-2xl border border-white/10 p-3 space-y-1.5">
                  <div className="font-semibold text-slate-200 text-[11px]">Routing</div>
                  <div className="text-slate-400">
                    Exchanges: {(profile.exchanges || []).length} selected
                  </div>
                  <div className="text-slate-500">
                    Telegram: {profile.telegram?.isConnected ? "linked (demo)" : "not linked"}
                  </div>
                </div>
              </div>
              <div className="flex justify-between">
                <button
                  onClick={() => setActive("Telegram")}
                  className="px-3 py-1.5 rounded-2xl border border-white/15 text-slate-300 hover:bg-white/5"
                >
                  Back
                </button>
                <button
                  onClick={complete}
                  className="px-4 py-2 rounded-2xl bg-emerald-500/90 text-white flex items-center gap-1 hover:bg-emerald-400"
                >
                  Finish & open Nowa UI
                  <ArrowRight className="w-3 h-3" />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
};

export default OnboardingWizard;
