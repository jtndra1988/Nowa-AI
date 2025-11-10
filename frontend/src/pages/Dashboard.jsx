import React, { useState } from 'react';

// --- Mock Data ---
// This is fake, static data we'll use to build the UI.
// Later, we will replace this with live API calls.

// Mock data for the main prediction widget
const mockPrediction = {
  price_prediction: 0.005, // A slightly bullish signal
  volatility_prediction: 0.2, // Medium volatility
  feature_importance: {
    "Price Action": 0.6,
    "Order Book": 0.3,
    "Sentiment": 0.1,
  },
};

// Mock data for the user's risk settings
const mockRiskSettings = {
  max_position_usd: 10000,
  confidence_cutoff: 0.65,
  max_drawdown_pct: 10,
};

// Mock data for the user's live portfolio
const mockPortfolio = [
  { id: 1, symbol: 'BTC/USDT', size: 0.5, entry_price: 60000, pnl: 1500.25, type: 'long' },
  { id: 2, symbol: 'ETH/USDT', size: 10, entry_price: 4000, pnl: -200.50, type: 'long' },
];
// --- End Mock Data ---


// --- Re-usable Card Component ---
const Card = ({ title, children, className = "" }) => (
  <div className={`bg-gray-800 shadow-lg rounded-xl p-6 ${className}`}>
    <h2 className="text-xl font-semibold text-white mb-4">{title}</h2>
    {children}
  </div>
);

// --- Dashboard Page ---
export default function Dashboard() {
  const [isBotActive, setIsBotActive] = useState(false);

  // Helper to format the "Why?" chart data
  const importanceData = Object.entries(mockPrediction.feature_importance)
    .sort(([, a], [, b]) => b - a);

  // Helper to determine signal strength
  const signal = mockPrediction.price_prediction > 0 ? "BULLISH" : "BEARISH";
  const signalColor = signal === "BULLISH" ? "text-green-400" : "text-red-400";
  
  // Helper to determine volatility
  const volatility = mockPrediction.volatility_prediction > 0.3 ? "HIGH" : (mockPrediction.volatility_prediction > 0.15 ? "MEDIUM" : "LOW");
  const volColor = volatility === "HIGH" ? "text-red-400" : (volatility === "MEDIUM" ? "text-yellow-400" : "text-green-400");


  return (
    <div className="p-6 bg-gray-900 text-gray-200 min-h-screen">
      <h1 className="text-3xl font-bold text-white mb-6">MARS Bot Dashboard</h1>

      {/* Main Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* === Left Column (Main Content) === */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* --- Bot Control Panel --- */}
          <Card title="Bot Control">
            <div className="flex items-center justify-between">
              <span className={`text-lg font-medium ${isBotActive ? 'text-green-400' : 'text-red-400'}`}>
                {isBotActive ? "BOT IS ACTIVE" : "BOT IS INACTIVE"}
              </span>
              <button
                onClick={() => setIsBotActive(!isBotActive)}
                className={`px-6 py-2 rounded-lg font-bold text-white transition-all ${isBotActive ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}`}
              >
                {isBotActive ? "DEACTIVATE" : "ACTIVATE BOT"}
              </button>
            </div>
            <p className="text-sm text-gray-400 mt-3">
              When active, the bot will automatically execute trades based on the live signal and your risk settings.
            </p>
          </Card>

          {/* --- Live Portfolio --- */}
          <Card title="Live Portfolio">
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b border-gray-700 text-gray-400 text-sm">
                    <th className="py-2">Symbol</th>
                    <th className="py-2">Size</th>
                    <th className="py-2">Entry Price</th>
                    <th className="py-2">Type</th>
                    <th className="py-2">Unrealized P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {mockPortfolio.map((pos) => (
                    <tr key={pos.id} className="border-b border-gray-600">
                      <td className="py-3 font-medium text-white">{pos.symbol}</td>
                      <td className="py-3">{pos.size}</td>
                      <td className="py-3">${pos.entry_price.toLocaleString()}</td>
                      <td className={`py-3 font-medium ${pos.type === 'long' ? 'text-green-400' : 'text-red-400'}`}>
                        {pos.type.toUpperCase()}
                      </td>
                      <td className={`py-3 font-medium ${pos.pnl > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        ${pos.pnl.toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

        </div>

        {/* === Right Column (Sidebar) === */}
        <div className="lg:col-span-1 space-y-6">

          {/* --- Live Signal --- */}
          <Card title="Live Signal (BTC/USDT)">
            <div className="text-center">
              <div className={`text-5xl font-bold ${signalColor} mb-2`}>
                {signal}
              </div>
              <div className="text-lg text-gray-300">
                Confidence: {Math.abs(mockPrediction.price_prediction * 100).toFixed(1)}%
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-gray-700 text-center">
              <div className="text-sm text-gray-400 mb-1">VOLATILITY FORECAST</div>
              <div className={`text-2xl font-bold ${volColor}`}>
                {volatility}
              </div>
            </div>
          </Card>

          {/* --- Interpretability "Why?" Chart --- */}
          <Card title="Signal Interpretability">
            <p className="text-sm text-gray-400 mb-4">
              The "Why?" chart shows which feature groups are driving the current signal.
            </p>
            {/* This is a MOCKUP of a donut chart. We can add a library like Recharts later. */}
            <div className="space-y-3">
              {importanceData.map(([name, value]) => (
                <div key={name}>
                  <div className="flex justify-between text-sm font-medium text-gray-300 mb-1">
                    <span>{name}</span>
                    <span>{(value * 100).toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2.5">
                    <div 
                      className="bg-blue-500 h-2.5 rounded-full" 
                      style={{ width: `${value * 100}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* --- Risk Settings --- */}
          <Card title="Risk Settings">
            <form className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Max Position (USD)
                </label>
                <input 
                  type="number" 
                  defaultValue={mockRiskSettings.max_position_usd}
                  className="w-full bg-gray-700 border border-gray-600 rounded-md p-2 text-white"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Confidence Cutoff (Trades only above this %)
                </label>
                <input 
                  type="number"
                  step="0.01"
                  defaultValue={mockRiskSettings.confidence_cutoff}
                  className="w-full bg-gray-700 border border-gray-600 rounded-md p-2 text-white"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Max Drawdown (%)
                </label>
                <input 
                  type="number" 
                  defaultValue={mockRiskSettings.max_drawdown_pct}
                  className="w-full bg-gray-700 border border-gray-600 rounded-md p-2 text-white"
                />
              </div>
              <button 
                type="button" 
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg"
              >
                Save Settings
              </button>
            </form>
          </Card>

        </div>
      </div>
    </div>
  );
}