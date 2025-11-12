// src/components/auth/LoginPanel.tsx
"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Lock, ShieldCheck, ArrowRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { useSession, useClientProfile, ClientProfile } from "@/lib/profile";
import { EXCHANGES } from "@/lib/api";

const LoginPanel: React.FC = () => {
  const { session, login } = useSession();
  const { profile, saveProfile } = useClientProfile();
  const [name, setName] = useState(session?.name ?? "");
  const [email, setEmail] = useState(session?.email ?? "");
  const [loading, setLoading] = useState(false);

  // If already fully onboarded, don't show the login screen.
  if (session && profile?.isOnboarded) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) return;
    setLoading(true);

    const baseProfile: ClientProfile =
      profile ?? {
        id: (typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : Math.random().toString(36).slice(2)),
        name: name || email.split("@")[0],
        email,
        createdAt: new Date().toISOString(),
        company: "",
        risk: {
          level: "balanced",
          maxDailyDrawdown: 5,
          maxPerTradeRisk: 1,
          leverageCap: 5,
          allowShorting: true,
          allowPerps: true,
          allowOptions: false,
        },
        exchanges: EXCHANGES.map((ex) => ({
          id: `${ex}-demo`,
          exchange: ex,
          label: `${ex} (not connected)`,
          hasApiKey: false,
          readOnly: true,
        })),
        telegram: {
          isConnected: false,
        },
        isOnboarded: false,
      };

    login({ email, name: baseProfile.name });
    saveProfile(baseProfile);
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[radial-gradient(circle_at_top,_#040816,_#020817)] text-slate-100">
      <motion.div
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md rounded-3xl border border-white/10 bg-white/5 backdrop-blur-2xl shadow-[0_18px_80px_rgba(15,23,42,0.8)] p-8 space-y-6"
      >
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-2xl bg-indigo-500/90 flex items-center justify-center shadow-[0_0_28px_rgba(129,140,248,0.9)]">
            <Lock className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-semibold tracking-tight">
              Sign in to <span className="text-indigo-400">Nowa Control</span>
            </h1>
            <p className="text-xs text-slate-400">
              Each client gets an isolated workspace: risk limits, exchanges, Telegram routes.
            </p>
          </div>
        </div>

        <form className="space-y-4" onSubmit={handleSubmit}>
          <div className="space-y-1">
            <label className="text-xs text-slate-300">Workspace / Desk Name</label>
            <input
              className="w-full rounded-2xl bg-slate-900/60 border border-white/10 px-3 py-2.5 text-sm outline-none focus:ring-2 focus:ring-indigo-500/70"
              placeholder="e.g. Falcon Capital"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs text-slate-300">Work Email</label>
            <input
              type="email"
              required
              className="w-full rounded-2xl bg-slate-900/60 border border-white/10 px-3 py-2.5 text-sm outline-none focus:ring-2 focus:ring-indigo-500/70"
              placeholder="you@fund.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <p className="text-[10px] text-slate-500 flex items-center gap-1">
              <ShieldCheck className="w-3 h-3" />
              In production this would be OTP / SSO. No API keys are stored in the browser.
            </p>
          </div>

          <button
            type="submit"
            disabled={loading}
            className={cn(
              "w-full mt-2 inline-flex items-center justify-center gap-2 rounded-2xl bg-indigo-500/90 hover:bg-indigo-400/95",
              "text-sm font-medium text-white py-2.5 transition-all shadow-[0_14px_40px_rgba(79,70,229,0.65)]",
              loading && "opacity-60 cursor-wait"
            )}
          >
            Continue to onboarding
            <ArrowRight className="w-4 h-4" />
          </button>
        </form>

        <div className="pt-2 border-t border-white/5 text-[10px] text-slate-500 space-y-1">
          <p>Next: Risk profile → Exchange connections → Telegram setup → Live Nowa UI.</p>
        </div>
      </motion.div>
    </div>
  );
};

export default LoginPanel;
