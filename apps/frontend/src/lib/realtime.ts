// src/lib/realtime.ts
import { Ticker, SymbolCode } from "./api";

const WS_BASE =
  process.env.NEXT_PUBLIC_WS_BASE || process.env.NEXT_PUBLIC_API_BASE?.replace(/^http/, "ws");

/** Generic WS helper with auto-cleanup & silent failure fallback. */
function connect(path: string, onMessage: (data: any) => void): () => void {
  if (!WS_BASE || typeof window === "undefined") {
    // no WS config; caller should fall back to polling
    return () => {};
  }

  const url = `${WS_BASE}${path}`;
  let ws: WebSocket | null = null;
  let closed = false;

  const open = () => {
    if (closed) return;
    ws = new WebSocket(url);

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        onMessage(data);
      } catch {
        // ignore malformed
      }
    };

    ws.onclose = () => {
      if (!closed) {
        // light reconnect
        setTimeout(open, 1500);
      }
    };
  };

  open();

  return () => {
    closed = true;
    ws?.close();
  };
}

/** Subscribe to live ticker; falls back to polling if WS not set. */
export function subscribeTicker(
  symbol: SymbolCode,
  onUpdate: (t: Ticker) => void
): () => void {
  if (WS_BASE && typeof window !== "undefined") {
    return connect(`/ws/ticker?symbol=${encodeURIComponent(symbol)}`, (msg) => {
      if (msg?.symbol && typeof msg.price === "number") {
        onUpdate(msg as Ticker);
      }
    });
  }

  // Fallback: small polling loop (already "industry standard" enough for demo)
  let stop = false;
  const loop = async () => {
    const { default: API } = await import("./api");
    while (!stop) {
      try {
        const t = await API.getTicker(symbol);
        onUpdate(t);
      } catch {
        // ignore
      }
      await new Promise((r) => setTimeout(r, 1500));
    }
  };
  loop();
  return () => {
    stop = true;
  };
}
