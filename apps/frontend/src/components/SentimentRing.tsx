"use client";

import React from "react";

interface SentimentRingProps {
  score: number; // 0 to 100
  size?: number;
  strokeWidth?: number;
  trend?: "up" | "down" | "flat";
}

export default function SentimentRing({
  score,
  size = 50,
  strokeWidth = 4,
  trend = "flat",
}: SentimentRingProps) {
  // 1. Clamp score to 0-100
  const value = Math.min(100, Math.max(0, isNaN(score) ? 0 : score));

  // 2. Circle Geometry
  const center = size / 2;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  
  // 3. The "Cut" Logic (Dash Offset)
  // offset = circumference - (percent * circumference)
  const offset = circumference - (value / 100) * circumference;

  // 4. Colors based on Trend or Score
  let color = "text-indigo-400";
  if (trend === "up") color = "text-emerald-400";
  if (trend === "down") color = "text-rose-400";
  if (trend === "flat" && value > 60) color = "text-emerald-400"; // Fallback high score
  if (trend === "flat" && value < 40) color = "text-rose-400";    // Fallback low score

  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <svg
        width={size}
        height={size}
        className="transform -rotate-90 transition-all duration-700 ease-out"
      >
        {/* Background Track */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="transparent"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-slate-800"
        />
        
        {/* Progress Arc */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="transparent"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          className={`transition-all duration-1000 ease-out ${color} drop-shadow-[0_0_4px_rgba(0,0,0,0.5)]`}
          style={{
            strokeDasharray: circumference,
            strokeDashoffset: offset,
          }}
        />
      </svg>

      {/* Center Text */}
      <div className={`absolute inset-0 flex items-center justify-center text-[11px] font-bold ${color}`}>
        {Math.round(value)}%
      </div>
    </div>
  );
}