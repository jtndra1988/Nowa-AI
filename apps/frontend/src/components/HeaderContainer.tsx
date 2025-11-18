"use client";

import { useState } from "react";
import NeonHeader from "@/components/NeonHeader";
import type { SymbolCode } from "@/lib/api";

export default function HeaderContainer() {
  // Single source of truth for currently selected symbol in the header
  const [symbol, setSymbol] = useState<SymbolCode>("BTC-PERP" as SymbolCode);
  
  // FIX: Add state for the active tab
  // You might need to change "dashboard" to whatever your default tab ID is (e.g., "trade", "analysis", etc.)
  const [activeTab, setActiveTab] = useState<string>("dashboard"); 

  return (
    <NeonHeader
      symbol={symbol}
      onSetSymbol={setSymbol}
      // FIX: Pass the missing props required by NeonHeader
      activeTab={activeTab}
      onTabChange={setActiveTab}
    />
  );
}