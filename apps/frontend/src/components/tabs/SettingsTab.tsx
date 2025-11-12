// src/components/tabs/SettingsTab.tsx
"use client";

import React from "react";
import dynamic from "next/dynamic";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "../layout/AppShell";
import { useClientProfile } from "@/lib/profile";

const SettingsPanel = dynamic(
  () =>
    import("@/components/SettingsPanel").then(
      (m) => m.SettingsPanel
    ),
  { ssr: false }
);

export const SettingsTab: React.FC = () => {
  const { profile } = useClientProfile();

  return (
    <div className="grid gap-4 lg:grid-cols-[2.1fr,1.4fr]">
      {/* Existing global settings */}
      <Card className="h-full">
        <CardHeader>
          <CardTitle>Platform Settings</CardTitle>
        </CardHeader>
        <CardContent>
          <SettingsPanel />
        </CardContent>
      </Card>

      {/* Client onboarding snapshot */}
      <div className="space-y-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">
              Client Workspace
            </CardTitle>
          </CardHeader>
          <CardContent className="text-[10px] space-y-1.5">
            {profile ? (
              <>
                <div className="font-semibold text-slate-100">
                  {profile.name}
                </div>
                <div className="text-slate-400">
                  {profile.email}
                </div>
                <div className="text-slate-500">
                  Created:&nbsp;
                  {new Date(
                    profile.createdAt
                  ).toLocaleString()}
                </div>
                <div className="text-slate-500">
                  Status:&nbsp;
                  {profile.isOnboarded
                    ? "Onboarding complete"
                    : "Pending onboarding"}
                </div>
              </>
            ) : (
              <div className="text-slate-500">
                No active client. Login & complete
                onboarding to bind this UI to a desk.
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">
              Risk Guardrails
            </CardTitle>
          </CardHeader>
          <CardContent className="text-[10px] space-y-1">
            {profile ? (
              <>
                <div className="text-slate-300">
                  Mode:{" "}
                  <span className="capitalize">
                    {profile.risk.level}
                  </span>
                </div>
                <div className="text-slate-400">
                  Max daily drawdown:{" "}
                  {profile.risk.maxDailyDrawdown}%
                </div>
                <div className="text-slate-400">
                  Max per trade risk:{" "}
                  {profile.risk.maxPerTradeRisk}%
                </div>
                <div className="text-slate-400">
                  Leverage cap: x
                  {profile.risk.leverageCap}
                </div>
                <div className="text-slate-500">
                  Shorting:{" "}
                  {profile.risk.allowShorting
                    ? "allowed"
                    : "blocked"}
                  {" • "}Perps:{" "}
                  {profile.risk.allowPerps
                    ? "allowed"
                    : "blocked"}
                  {" • "}Options:{" "}
                  {profile.risk.allowOptions
                    ? "allowed"
                    : "blocked"}
                </div>
                <div className="text-[9px] text-emerald-400/90">
                  Every AI decision, route & ticket in this
                  UI is expected to respect these caps (for
                  demo: UI-level only).
                </div>
              </>
            ) : (
              <div className="text-slate-500">
                Risk limits appear here once a client is
                onboarded.
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">
              Connectivity Snapshot
            </CardTitle>
          </CardHeader>
          <CardContent className="text-[10px] space-y-1.5">
            {profile ? (
              <>
                <div className="text-slate-300">
                  Exchanges enabled:{" "}
                  {profile.exchanges?.length || 0}
                </div>
                <div className="flex flex-wrap gap-1 text-slate-400">
                  {profile.exchanges?.map((ex) => (
                    <span
                      key={ex.id}
                      className="px-2 py-0.5 rounded-xl bg-slate-900/90 border border-slate-700/80"
                    >
                      {ex.exchange}
                      {ex.readOnly && " • read-only"}
                    </span>
                  ))}
                  {!profile.exchanges?.length && (
                    <span className="text-slate-500">
                      No venues selected yet.
                    </span>
                  )}
                </div>
                <div className="text-slate-300">
                  Telegram:{" "}
                  {profile.telegram?.isConnected
                    ? `Linked (${profile.telegram.handle || "bot"})`
                    : "Not linked"}
                </div>
                <div className="text-[9px] text-slate-500">
                  In production, this maps to your backend
                  client record (KYC/KYB + encrypted API
                  keys). Here it&apos;s a visual contract for
                  your demo.
                </div>
              </>
            ) : (
              <div className="text-slate-500">
                Once a client completes onboarding, their
                venues & Telegram routes are summarized here.
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default SettingsTab;
