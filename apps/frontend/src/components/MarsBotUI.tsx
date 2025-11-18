"use client";

import React, { useEffect, useState } from "react";
import { EXCHANGES, MarketMode, SymbolCode } from "@/lib/api";
import { Shell, useLocalStorage } from "./layout/AppShell";

import { DashboardTab } from "./tabs/DashboardTab";
import { StrategiesTab } from "./tabs/StrategiesTab";
import { MarketIntelTab } from "./tabs/MarketIntelTab";
import { SystemTab } from "./tabs/SystemTab";

const VALID_TABS = ["dashboard", "strategies", "intel", "system"] as const;
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

  // Keep URL hash (#dashboard, #strategies, #intel, #system) in sync with selected tab
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
    // update hash so deep links remain consistent
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
        return (
          <StrategiesTab
            symbol={symbol}
            mode={mode}
            exchange={exchange}
          />
        );
      case "intel":
        return (
          <MarketIntelTab
            symbol={symbol}
            mode={mode}
            exchange={exchange}
          />
        );
      case "system":
        return <SystemTab />;
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
