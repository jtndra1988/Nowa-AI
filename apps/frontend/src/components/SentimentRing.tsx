"use client";

import React from "react";

interface SentimentRingProps {
  score: number; // 0 to 100
  size?: number;
  strokeWidth?: number;
}

export default function SentimentRing({
  score,
  size = 56,
  strokeWidth = 4,
}: SentimentRingProps) {
  // Calculate circle parameters
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (score / 100) * circumference;

  // Determine color based on score (matching your dashboard logic)
  const getColor = (s: number) => {
    if (s >= 60) return "text-emerald-400";
    if (s <= 40) return "text-rose-400";
    return "text-sky-400";
  };

  const colorClass = getColor(score);

  return (
    <div
      className="relative flex items-center justify-center"
      style={{ width: size, height: size }}
    >
      {/* SVG container */}
      <svg
        width={size}
        height={size}
        className="transform -rotate-90 transition-all duration-500"
      >
        {/* Background track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="transparent"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-slate-800"
        />
        {/* Progress arc */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="transparent"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className={`transition-all duration-700 ease-out ${colorClass}`}
        />
      </svg>
      
      {/* Optional center percentage (small) */}
      <div className={`absolute text-[10px] font-semibold ${colorClass}`}>
        {Math.round(score)}
      </div>
    </div>
  );
}