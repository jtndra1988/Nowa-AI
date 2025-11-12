// src/lib/cryptoIcons.ts
// Dynamically resolves icons from `cryptocurrency-icons` if they exist.
// Works for BTC, ETH, etc. Missing tokens fall back to undefined.

export function getIconSrcForSymbol(symbol: string): string | undefined {
  // Normalize: "BTC/USDT", "btc-perp" -> "btc"
  const base = symbol.split(/[-/]/)[0].toLowerCase();

  try {
    // Prefer color icons
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    return require(`cryptocurrency-icons/svg/color/${base}.svg`).default as string;
  } catch {
    try {
      // Fallback to black icon variant if color missing
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      return require(`cryptocurrency-icons/svg/black/${base}.svg`).default as string;
    } catch {
      // No icon found in the package
      return undefined;
    }
  }
}
