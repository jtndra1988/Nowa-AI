// src/lib/profile.ts
"use client";

import { useEffect, useMemo, useState } from "react";
import { EXCHANGES } from "./api";

export type RiskLevel = "conservative" | "balanced" | "aggressive" | "custom";

export interface RiskProfile {
  level: RiskLevel;
  maxDailyDrawdown: number; // %
  maxPerTradeRisk: number;  // %
  leverageCap: number;      // x
  allowShorting: boolean;
  allowPerps: boolean;
  allowOptions: boolean;
}

export interface ExchangeConnection {
  id: string;
  exchange: string;
  label: string;
  hasApiKey: boolean; // UI/demo flag only
  readOnly: boolean;  // true = safe demo / paper
}

export interface TelegramConfig {
  handle?: string;
  chatId?: string;
  isConnected: boolean;
}

export interface ClientProfile {
  id: string;
  name: string;
  email: string;
  company?: string;
  createdAt: string;
  risk: RiskProfile;
  exchanges: ExchangeConnection[];
  telegram: TelegramConfig;
  isOnboarded: boolean;
}

export interface Session {
  email: string;
  name?: string;
}

const PROFILE_KEY = "mars.clientProfile";
const SESSION_KEY = "mars.session";

function safeParse<T>(raw: string | null): T | null {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

/* ========== Session (simple login) ========== */

export function useSession() {
  const [session, setSession] = useState<Session | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    setSession(safeParse<Session>(window.localStorage.getItem(SESSION_KEY)));
  }, []);

  const login = (payload: Session) => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(SESSION_KEY, JSON.stringify(payload));
    setSession(payload);
  };

  const logout = () => {
    if (typeof window === "undefined") return;
    window.localStorage.removeItem(SESSION_KEY);
    window.localStorage.removeItem(PROFILE_KEY);
    setSession(null);
  };

  return { session, login, logout };
}

/* ========== Client Profile (risk + venues + telegram) ========== */

export function useClientProfile() {
  const [profile, setProfile] = useState<ClientProfile | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    setProfile(safeParse<ClientProfile>(window.localStorage.getItem(PROFILE_KEY)));
  }, []);

  const saveProfile = (next: ClientProfile) => {
    if (typeof window === "undefined") return;
    setProfile(next);
    window.localStorage.setItem(PROFILE_KEY, JSON.stringify(next));
  };

  const patchProfile = (patch: Partial<ClientProfile>) => {
    if (!profile) return;
    const next = { ...profile, ...patch };
    saveProfile(next);
  };

  return { profile, saveProfile, patchProfile };
}

/* ========== Derived helpers ========== */

export function useAllowedExchanges(): string[] {
  const { profile } = useClientProfile();

  return useMemo(() => {
    // Fix: Return [...EXCHANGES] to create a mutable copy
    if (!profile?.isOnboarded || !profile.exchanges?.length) return [...EXCHANGES];
    
    const enabled = profile.exchanges
      .filter((e) => !!e && !!e.exchange)
      .map((e) => e.exchange);
    const unique = Array.from(new Set(enabled));
    const filtered = EXCHANGES.filter((ex) => unique.includes(ex));
    
    // Fix: Return [...EXCHANGES] in the fallback case as well
    return filtered.length ? filtered : [...EXCHANGES];
  }, [profile]);
}

export function useWorkspaceLabel() {
  const { profile } = useClientProfile();

  if (!profile) return null;

  const risk = profile.risk;
  const riskText =
    risk.level === "custom"
      ? `Custom • Max DD ${risk.maxDailyDrawdown}%`
      : `${risk.level[0].toUpperCase()}${risk.level.slice(1)} • Max DD ${risk.maxDailyDrawdown}%`;

  return {
    name: profile.name,
    email: profile.email,
    riskText,
    isOnboarded: profile.isOnboarded,
    hasTelegram: !!profile.telegram?.isConnected,
  };
}
