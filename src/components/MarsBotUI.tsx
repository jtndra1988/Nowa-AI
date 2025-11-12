// src/components/MarsBotUI.tsx
"use client";

import React, { useEffect, useState } from "react";
import { EXCHANGES, MarketMode, SymbolCode } from "@/lib/api";
import { Shell, useLocalStorage } from "./layout/AppShell";

import { DashboardTab } from "./tabs/DashboardTab";
import { StrategiesTab } from "./tabs/StrategiesTab";
import { BacktestsTab } from "./tabs/BacktestsTab";
import { LiveTab } from "./tabs/LiveTab";
import { PortfolioTab } from "./tabs/PortfolioTab";
import { MarketIntelTab } from "./tabs/MarketIntelTab";
import { SystemTab } from "./tabs/SystemTab";
import { SettingsTab } from "./tabs/SettingsTab";

const VALID_TABS = [
  "dashboard",
  "ai",
  "strategies",
  "backtests",
  "live",
  "portfolio",
  "intel",
  "system",
  "settings",
] as const;

type TabKey = (typeof VALID_TABS)[number];

const MarsBotUI: React.FC = () => {
  // Core shared state (synced with localStorage so demo feels persistent)
  const [tab, setTab] = useState<TabKey>("dashboard");
  const [symbol, setSymbol] = useLocalStorage<SymbolCode>(
    "nowa.symbol",
    "BTC-PERP"
  );
  const [mode, setMode] = useLocalStorage<MarketMode>(
    "nowa.mode",
    "futures"
  );
  const [exchange, setExchange] = useLocalStorage<string>(
    "nowa.exchange",
    EXCHANGES[0]
  );

  // Keep URL hash (#dashboard, #strategies, etc.) in sync with selected tab
  useEffect(() => {
    const applyHash = () => {
      const raw = (window.location.hash || "#dashboard").replace("#", "");
      const key = VALID_TABS.includes(raw as TabKey)
        ? (raw as TabKey)
        : "dashboard";
      setTab(key);
    };

    applyHash();
    window.addEventListener("hashchange", applyHash);
    return () => window.removeEventListener("hashchange", applyHash);
  }, []);

  const handleTabChange = (next: string) => {
    if (!VALID_TABS.includes(next as TabKey)) return;
    const key = next as TabKey;
    setTab(key);
    // update hash so deep links & sidebar remain consistent
    if (typeof window !== "undefined") {
      window.location.hash = `#${key}`;
    }
  };

  const renderTab = () => {
    switch (tab) {
      case "dashboard":
        return (
          <DashboardTab
            symbol={symbol}
            mode={mode}
            exchange={exchange}
          />
        );
      case "strategies":
        // Config + explanation of TFT / TCN / XGB ensemble (already in your tab)
        return (
          <StrategiesTab
            symbol={symbol}
            mode={mode}
            exchange={exchange}
          />
        );

      case "backtests":
        // Uses your mock / static backtest-style UI
        return (
          <BacktestsTab
            symbol={symbol}
            mode={mode}
            exchange={exchange}
          />
        );

      case "live":
        // Reads mock live trades via API.tickLiveTrades
        return (
          <LiveTab
            symbol={symbol}
            mode={mode}
            exchange={exchange}
          />
        );

      case "portfolio":
        return (
          <PortfolioTab
            symbol={symbol}
            mode={mode}
            exchange={exchange}
          />
        );

      case "intel":
        // On-chain, sentiment, dev activity (mocked but structured)
        return (
          <MarketIntelTab
            symbol={symbol}
            mode={mode}
            exchange={exchange}
          />
        );

      case "system":
        // Uses API.MOCK + fake infra status
        return <SystemTab />;

      case "settings":
        // General app settings (already handled inside SettingsPanel)
        return <SettingsTab />;

      default:
        return (
          <DashboardTab
            symbol={symbol}
            mode={mode}
            exchange={exchange}
          />
        );
    }
  };

  return (
    <Shell
      symbol={symbol}
      setSymbol={setSymbol}
      mode={mode}
      setMode={setMode}
      exchange={exchange}
      setExchange={setExchange}
      tab={tab}
      setTab={handleTabChange}
    >
      {renderTab()}
    </Shell>
  );
};

export default MarsBotUI;
