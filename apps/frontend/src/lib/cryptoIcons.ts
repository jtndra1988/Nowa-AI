// src/lib/cryptoIcons.ts
// Dynamically resolves icons from unpkg CDN.
// No local SVG files required.

export function getIconSrcForSymbol(symbol: string): string {
  // 1. Normalize: "BTC-PERP" -> "btc"
  const base = symbol.split(/[-/]/)[0].toLowerCase();

  // 2. Return the remote URL for the icon
  return `https://unpkg.com/cryptocurrency-icons@0.18.1/svg/color/${base}.svg`;
}