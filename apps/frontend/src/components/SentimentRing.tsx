"use client";

import React from "react";

type Props = {
  /** sentiment value in [0,1]; caller can normalize however they like */
  value: number;
};

export default function SentimentRing({ value }: Props) {
  const safe = Number.isFinite(value)
    ? Math.max(0, Math.min(1, value))
    : 0.5;

  const pct = Math.round(safe * 100);

  const r = 52;
  const strokeWidth = 10;
  const circumference = 2 * Math.PI * r;
  const offset = circumference * (1 - safe);

  const color =
    safe > 0.66
      ? "#22c55e" // strong risk-on
      : safe > 0.55
      ? "#4ade80"
      : safe < 0.34
      ? "#fb7185" // strong risk-off
      : safe < 0.45
      ? "#f97316"
      : "#38bdf8";

  const label =
    safe > 0.66
      ? "Risk-On"
      : safe > 0.55
      ? "Tilt On"
      : safe < 0.34
      ? "Risk-Off"
      : safe < 0.45
      ? "Tilt Off"
      : "Neutral";

  return (
    <div className="relative flex items-center justify-center">
      <svg
        width={140}
        height={140}
        className="transition-all duration-700"
      >
        {/* background track */}
        <circle
          cx={70}
          cy={70}
          r={r}
          fill="transparent"
          stroke="rgba(148,163,253,0.14)"
          strokeWidth={strokeWidth}
        />
        {/* progress arc */}
        <circle
          cx={70}
          cy={70}
          r={r}
          fill="transparent"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={`${circumference} ${circumference}`}
          strokeDashoffset={offset}
          className="transition-all duration-700 ease-out"
          style={{
            transformOrigin: "50% 50%",
            transform: "rotate(-90deg)",
            filter: `drop-shadow(0 0 14px ${color}66)`,
          }}
        />
      </svg>

      {/* center label */}
      <div className="absolute flex flex-col items-center justify-center">
        <div className="text-sm font-semibold text-slate-50">
          {pct}%
        </div>
        <div className="text-[9px] text-slate-400">{label}</div>
      </div>
    </div>
  );
}
