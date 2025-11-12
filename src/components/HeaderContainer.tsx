"use client";
import { useState } from "react";
import NeonHeader from "@/components/NeonHeader";

// If you can import the real Props type, prefer:
//   import type { Props as NeonHeaderProps } from "@/components/NeonHeader";
// and remove the fallback below.

// --- Fallback props shape (adjust if your real NeonHeader differs) ---
type NeonHeaderProps = {
  symbol: string;
  mode: "futures" | "options" | "spot"; // CHANGED to lowercase
  exchange: string;
  onSetSymbol: (s: string) => void;
  onSetMode: (m: "futures" | "options" | "spot") => void; // CHANGED to lowercase
  onSetExchange: (ex: string) => void;

  // If your NeonHeader also expects search text, price pair, etc.,
  // keep these optional here and pass stubs:
  onOpenSidebar?: () => void;
  search?: string;
  onSearch?: (q: string) => void;
};

export default function HeaderContainer() {
  const [symbol, setSymbol] = useState<string>("BTC-PERP");
  const [mode, setMode] = useState<"futures" | "options" | "spot">("futures"); // CHANGED to lowercase
  const [exchange, setExchange] = useState<string>("Binance");

  // Optional extras (safe stubs if your header supports them)
  const [search, setSearch] = useState("");
  const onOpenSidebar = () => {
    // No-op; AppShell already injects a hamburger on mobile.
    // If your NeonHeader shows its own button, you can wire it to a prop later.
  };

  const props: NeonHeaderProps = {
    symbol,
    mode,
    exchange,
    onSetSymbol: setSymbol,
    onSetMode: setMode,
    onSetExchange: setExchange,
    onOpenSidebar,
    search,
    onSearch: setSearch,
  };

  // If TypeScript complains due to extra/missing props, you can temporarily do:
  // return <NeonHeader {...(props as any)} />;  // until you align exact types.
  return <NeonHeader {...props} />;
}