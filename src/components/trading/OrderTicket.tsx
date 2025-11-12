// src/components/trading/OrderTicket.tsx
"use client";

import React, { useState } from "react";
import API, { type MarketMode, type SymbolCode, type OrderType, type OrderSide } from "@/lib/api";
import { toast } from "sonner";

type Props = {
  symbol: SymbolCode;
  mode: MarketMode;
  exchange: string;
};

export const OrderTicket: React.FC<Props> = ({ symbol, mode, exchange }) => {
  const [side, setSide] = useState<OrderSide>("BUY");
  const [type, setType] = useState<OrderType>("market");
  const [size, setSize] = useState<number>(0.01);
  const [price, setPrice] = useState<number | "">("");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async () => {
    if (!size || size <= 0) {
      toast.error("Enter valid size");
      return;
    }
    if (type === "limit" && (!price || Number(price) <= 0)) {
      toast.error("Enter valid limit price");
      return;
    }
    setSubmitting(true);
    try {
      const order = await API.placeOrder({
        symbol,
        market: mode,
        side,
        type,
        size: Number(size),
        price: type === "limit" ? Number(price) : undefined,
        exchange: exchange as any,
      });
      toast.success("Order submitted", {
        description: `${order.side} ${order.size} ${order.symbol} @ ${order.price} (${order.exchange ?? "AI route"})`,
      });
    } catch (e: any) {
      toast.error("Order failed", {
        description: e?.message || "Check connection / backend",
      });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-3 flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-xs font-semibold text-slate-300">
          Manual Control Ticket
        </div>
        <div className="text-[10px] text-slate-400">
          Executed safely via backend; keys never touch frontend.
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-[11px]">
        <div>
          <div className="text-slate-400 mb-1">Side</div>
          <div className="flex gap-1">
            {(["BUY", "SELL"] as OrderSide[]).map((s) => (
              <button
                key={s}
                onClick={() => setSide(s)}
                className={`flex-1 py-1.5 rounded-xl text-[11px] ${
                  side === s
                    ? s === "BUY"
                      ? "bg-emerald-500/20 text-emerald-300"
                      : "bg-rose-500/20 text-rose-300"
                    : "bg-white/5 text-slate-300"
                }`}
                aria-pressed={side === s}
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="text-slate-400 mb-1">Type</div>
          <div className="flex gap-1">
            {(["market", "limit"] as OrderType[]).map((t) => (
              <button
                key={t}
                onClick={() => setType(t)}
                className={`flex-1 py-1.5 rounded-xl text-[11px] capitalize ${
                  type === t
                    ? "bg-indigo-500/25 text-indigo-200"
                    : "bg-white/5 text-slate-300"
                }`}
                aria-pressed={type === t}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="text-slate-400 mb-1">
            Size <span className="text-slate-500">(contracts / units)</span>
          </div>
          <input
            type="number"
            value={size}
            onChange={(e) => setSize(Number(e.target.value))}
            className="w-full rounded-xl bg-slate-900/60 border border-white/10 px-2 py-1 text-[11px] outline-none focus:border-indigo-400/70"
            min={0}
          />
        </div>

        {type === "limit" && (
          <div>
            <div className="text-slate-400 mb-1">Limit price (USDT)</div>
            <input
              type="number"
              value={price}
              onChange={(e) => setPrice(e.target.value === "" ? "" : Number(e.target.value))}
              className="w-full rounded-xl bg-slate-900/60 border border-white/10 px-2 py-1 text-[11px] outline-none focus:border-indigo-400/70"
              min={0}
            />
          </div>
        )}
      </div>

      <button
        onClick={handleSubmit}
        disabled={submitting}
        className="mt-1 w-full py-2 rounded-xl bg-indigo-500/90 hover:bg-indigo-400 text-xs font-semibold text-white shadow-[0_10px_40px_rgba(79,70,229,0.55)] transition-transform hover:-translate-y-0.5 disabled:opacity-60"
      >
        {submitting ? "Submitting..." : "Execute via NOWA Backend"}
      </button>

      <p className="mt-1 text-[9px] text-slate-500">
        If backend is unreachable, the ticket fails **safe** (no client-side
        exchange calls).
      </p>
    </div>
  );
};
