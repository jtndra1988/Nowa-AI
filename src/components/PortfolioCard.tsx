"use client";

import React from 'react';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
} from 'recharts';

// --- Mock Data Interfaces ---
// Based on what your backend's portfolio/state.py might track

interface Allocation {
  [key: string]: any; // <-- FIX: Added index signature for recharts compatibility
  asset: string;
  value: number; // in USD
  percentage: number; // 0.0 - 1.0
}

interface Position {
  id: string;
  symbol: string; // e.g., "BTCUSDT"
  side: 'LONG' | 'SHORT';
  size: number; // in base asset (e.g., 1.5 BTC)
  entryPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  pnlPercent: number; // 0.0 - 1.0
}

interface PortfolioState {
  totalValue: number; // Total portfolio value in USD
  realizedPnl: number;
  unrealizedPnl: number;
  allocations: Allocation[];
  positions: Position[];
}

// --- Mock Data ---
// This data simulates the state from your backend
const mockPortfolio: PortfolioState = {
  totalValue: 125320.75,
  realizedPnl: 15200.0,
  unrealizedPnl: 3020.5,
  allocations: [
    { asset: 'BTC', value: 50120.3, percentage: 0.4 },
    { asset: 'ETH', value: 37596.22, percentage: 0.3 },
    { asset: 'SOL', value: 25064.15, percentage: 0.2 },
    { asset: 'USDC', value: 12532.08, percentage: 0.1 },
  ],
  positions: [
    {
      id: '1',
      symbol: 'BTCUSDT',
      side: 'LONG',
      size: 0.75,
      entryPrice: 65000.0,
      currentPrice: 66827.06,
      unrealizedPnl: 1370.29,
      pnlPercent: 0.028,
    },
    {
      id: '2',
      symbol: 'ETHUSDT',
      side: 'LONG',
      size: 10.0,
      entryPrice: 3600.0,
      currentPrice: 3759.62,
      unrealizedPnl: 1596.22,
      pnlPercent: 0.044,
    },
    {
      id: '3',
      symbol: 'SOLUSDT',
      side: 'SHORT',
      size: 50.0,
      entryPrice: 170.0,
      currentPrice: 169.1,
      unrealizedPnl: 45.0,
      pnlPercent: 0.005,
    },
    {
      id: '4',
      symbol: 'AVAXUSDT',
      side: 'LONG',
      size: 100.0,
      entryPrice: 35.0,
      currentPrice: 34.9,
      unrealizedPnl: -10.0,
      pnlPercent: -0.002,
    },
  ],
};

// --- Helper Functions ---
const fmtUSD = (v: number) =>
  `$${Intl.NumberFormat('en-US', {
    notation: 'compact',
    maximumFractionDigits: 2,
  }).format(v)}`;

const fmtPercent = (v: number) =>
  `${(v * 100).toFixed(1)}%`;

// Colors for the pie chart
const PIE_COLORS = ['#34d399', '#60a5fa', '#c084fc', '#a3a3a3'];

// Glass theme styles
const glassPanel =
  "rounded-2xl backdrop-blur-xl bg-white/8 dark:bg-white/5 border border-white/15 shadow-[0_8px_30px_rgba(0,0,0,0.25)]";
const softText = "text-slate-200";

// --- Main Component ---
export default function PortfolioCard() {
  // In a real app, you'd fetch this data
  const [portfolio, setPortfolio] =
    React.useState<PortfolioState>(mockPortfolio);
  const { totalValue, allocations, positions } = portfolio;

  return (
    <Card className={`${glassPanel} col-span-1 md:col-span-2`}>
      <CardHeader>
        <CardTitle className={softText}>Portfolio</CardTitle>
        <div className="text-3xl font-bold text-white">${totalValue.toLocaleString('en-US', { maximumFractionDigits: 2 })}</div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="allocations">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="allocations">Allocations</TabsTrigger>
            <TabsTrigger value="positions">Positions ({positions.length})</TabsTrigger>
          </TabsList>
          
          {/* Allocations Tab */}
          <TabsContent value="allocations" className="h-[250px] w-full pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={allocations}
                  dataKey="value"
                  nameKey="asset"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#8884d8"
                >
                  {allocations.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value: number, name: string) => [fmtUSD(value), name]}
                  contentStyle={{
                    backgroundColor: "rgba(10, 20, 30, 0.9)",
                    borderColor: "rgba(255, 255, 255, 0.2)",
                    borderRadius: "8px",
                  }}
                  labelStyle={{ color: '#fff' }}
                />
                <Legend
                  formatter={(value, entry: any) => { // <-- FIX: Cast entry to 'any' to bypass incorrect TS type
                    // `value` is the asset name (e.g., "BTC")
                    // `entry.payload` is the original data object (Allocation)
                    const dataObject = entry.payload;
                    return (
                      <span style={{ color: '#fff' }}>
                        {value}: {fmtPercent(dataObject?.percentage)}
                      </span>
                    );
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </TabsContent>
          
          {/* Positions Tab */}
          <TabsContent value="positions" className="h-[250px] w-full pt-4 overflow-y-auto space-y-2">
            {positions.map((pos) => {
              const posSideColor = pos.side === 'LONG' ? 'text-emerald-400' : 'text-rose-400';
              const posPnlColor = pos.unrealizedPnl >= 0 ? 'text-emerald-400' : 'text-rose-400';

              return (
                <div key={pos.id} className="text-sm p-3 bg-muted/50 rounded-lg">
                  <div className="flex justify-between items-center mb-1">
                    <div className="font-bold text-base text-white">
                      {pos.symbol}
                      <span className={`ml-2 text-xs font-medium ${posSideColor}`}>
                        {pos.side}
                      </span>
                    </div>
                    <div className={`font-mono font-semibold ${posPnlColor}`}>
                      {pos.unrealizedPnl.toFixed(2)} USD
                    </div>
                  </div>
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <div>
                      Size: <span className="text-foreground">{pos.size}</span>
                    </div>
                    <div>
                      Entry: <span className="text-foreground">${pos.entryPrice}</span>
                    </div>
                    <div>
                      Mark: <span className="text-foreground">${pos.currentPrice}</span>
                    </div>
                    <div className={`${posPnlColor}`}>
                      ({(pos.pnlPercent * 100).toFixed(2)}%)
                    </div>
                  </div>
                </div>
              );
            })}
            {/* THIS is where the extra <div> was, I've removed it. */}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}