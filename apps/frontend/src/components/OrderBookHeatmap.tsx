"use client";
import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import API, { SymbolCode } from "@/lib/api"; // <-- 1. Import SymbolCode

// 2. Define types for the order book
type PriceLevel = [number, number]; // [price, size]
type OrderBook = {
  bids: PriceLevel[];
  asks: PriceLevel[];
};

// +++ ADD THIS MOCK FUNCTION +++
// This generates plausible-looking mock data
function buildMockBook(): OrderBook {
  const bids: PriceLevel[] = [];
  const asks: PriceLevel[] = [];
  let price = 69420 + (Math.random() * 20 - 10); // Mock center price

  let bidPrice = price;
  for (let i = 0; i < 10; i++) {
    bidPrice -= (Math.random() * 5);
    bids.push([Math.round(bidPrice * 10) / 10, 0.5 + Math.random() * 3]);
  }
  
  let askPrice = price;
  for (let i = 0; i < 10; i++) {
    askPrice += (Math.random() * 5);
    asks.push([Math.round(askPrice * 10) / 10, 0.5 + Math.random() * 3]);
  }
  return { bids, asks };
}


export default function OrderBookHeatmap({ symbol }: { symbol: SymbolCode }) { // <-- 3. Fix prop type
  const [book, setBook] = useState<OrderBook>({ bids: [], asks: [] }); // <-- 4. Fix useState type

  // +++ REPLACE THE OLD useEffect WITH THIS ONE +++
  useEffect(() => {
    let active = true;

    // Set initial mock data
    setBook(buildMockBook());

    // Simulate the stream with new mock data every 2.5 seconds
    const id = setInterval(() => {
      if (active) {
        setBook(buildMockBook());
      }
    }, 2500);

    return () => {
      active = false;
      clearInterval(id); // Clear the interval on unmount
    };
  }, [symbol]); // Re-run if symbol changes

  // 5. Fix function parameter types
  const row = (side: PriceLevel[], [price, size]: PriceLevel, i: number) => {
    // Calculate max size *only* for the visible slice
    const max = Math.max(...side.slice(0, 10).map((r: PriceLevel) => r[1]));
    const intensity = size / max;

    return (
      <motion.div
        key={price} // Use price as key for better stability
        initial={{ opacity: 0.4 }}
        animate={{ opacity: 0.6 + intensity * 0.4 }}
        transition={{ duration: 0.25 }}
        className={`flex justify-between text-xs px-2 py-1 font-mono ${
          side === book.bids ? "text-green-300" : "text-red-300"
        }`}
        style={{
          background: side === book.bids
            ? `rgba(16,185,129,${intensity * 0.4})`
            : `rgba(244,63,94,${intensity * 0.4})`,
        }}
      >
        <span>{price.toLocaleString('en-US', { minimumFractionDigits: 2 })}</span>
        <span>{size.toFixed(2)}</span>
      </motion.div>
    );
  };

  return (
    <div className="rounded-xl bg-white/5 border border-white/10 overflow-hidden">
      <div className="text-xs text-center py-1 opacity-70">Order Book</div>
      <div className="grid grid-cols-2 divide-x divide-white/10">
        <div>{book.bids.slice(0, 10).map((r, i) => row(book.bids, r, i))}</div>
        <div>{book.asks.slice(0, 10).map((r, i) => row(book.asks, r, i))}</div>
      </div>
    </div>
  );
}