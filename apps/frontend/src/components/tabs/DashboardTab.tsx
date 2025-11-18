"use client";

import React, { useMemo } from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  KpiTile,
  glassPanel,
  faintText,
  usePredict,
  useDashboardMetrics,
  useMarketIntel,
  DirectionLabel,
  HelpTooltip,
} from "../layout/AppShell";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";
import SentimentRing from "../SentimentRing";
import type { SymbolCode, MarketMode } from "@/lib/api";

type Props = {
  symbol: SymbolCode;
  mode: MarketMode;
  exchange: string;
};

export const DashboardTab: React.FC<Props> = ({
  symbol,
  mode,
  exchange,
}) => {
  const predict = usePredict(symbol);
  const metrics = useDashboardMetrics(symbol);
  const { intel, latest } = useMarketIntel(symbol);

  // ---------- Core signal logic (hourly) ----------
  const dir = predict?.direction ?? "flat";

  const conf =
    typeof predict?.price_confidence === "number"
      ? predict.price_confidence
      : 0.6;
  const confClamped = Math.max(0, Math.min(1, conf));
  const confPct = confClamped * 100;

  const sideLabel =
    dir === "up"
      ? "Upside bias"
      : dir === "down"
      ? "Downside bias"
      : "Flat / Neutral";

  const botSignal = useMemo(() => {
    if (confClamped < 0.55 || dir === "flat") return "HOLD";
    if (dir === "up") return confClamped > 0.72 ? "STRONG BUY" : "BUY";
    if (dir === "down") return confClamped > 0.72 ? "STRONG SELL" : "SELL";
    return "HOLD";
  }, [dir, confClamped]);

  const botColor =
    botSignal.includes("BUY")
      ? "text-emerald-300"
      : botSignal.includes("SELL")
      ? "text-rose-300"
      : "text-slate-100";

  const biasLabel =
    dir === "up"
      ? "LONG BIAS"
      : dir === "down"
      ? "SHORT BIAS"
      : "NEUTRAL";

  const biasColor =
    dir === "up"
      ? "text-emerald-400"
      : dir === "down"
      ? "text-rose-400"
      : "text-slate-300";

  // ---------- Hourly price & range ----------
  const basePrice =
    typeof (metrics as any)?.last_price === "number"
      ? (metrics as any).last_price
      : typeof (predict as any)?.spot_price === "number"
      ? (predict as any).spot_price
      : 1000;

  const vol24 =
    typeof predict?.volatility_pct === "number"
      ? predict.volatility_pct
      : 32;

  const nextHourMovePct =
    typeof (predict as any)?.expected_move_1h_pct === "number"
      ? (predict as any).expected_move_1h_pct
      : vol24 / 24;

  const hourlyRangePct = Math.max(
    0.3,
    typeof (predict as any)?.expected_range_1h_pct === "number"
      ? (predict as any).expected_range_1h_pct
      : vol24 / Math.sqrt(24)
  );

  const expectedUpPrice = basePrice * (1 + hourlyRangePct / 100);
  const expectedDownPrice = basePrice * (1 - hourlyRangePct / 100);

  const hourlySeries = useMemo(() => {
    const steps = 6;
    const data: { t: string; price: number }[] = [];
    const directionSign = dir === "up" ? 1 : dir === "down" ? -1 : 0;
    const perStepDriftPct =
      (directionSign * Math.abs(nextHourMovePct)) / steps || 0;

    let price = basePrice;
    for (let i = 0; i <= steps; i++) {
      data.push({
        t: i === 0 ? "Now" : `${i}h`,
        price,
      });
      price = price * (1 + perStepDriftPct / 100);
    }
    return data;
  }, [basePrice, dir, nextHourMovePct]);

  // ---------- Market sentiment (hourly context) ----------
  const rawSent =
    typeof latest?.sentiment === "number" ? latest.sentiment : 0;
  // latest.sentiment ~ -0.5..0.5 -> 0..100
  const sentimentScore = Math.round((rawSent + 0.5) * 100);
  const sentimentClamped = Math.max(0, Math.min(100, sentimentScore));

  const sentimentLabel =
    sentimentClamped > 60
      ? "Risk-On / Bullish"
      : sentimentClamped < 40
      ? "Risk-Off / Bearish"
      : "Balanced / Neutral";

  const sentimentColor =
    sentimentClamped > 60
      ? "text-emerald-300"
      : sentimentClamped < 40
      ? "text-rose-300"
      : "text-sky-300";

  const sentimentHistory = intel?.sentimentHistory ?? [];

  // ---------- Flow & options metrics ----------
  const fundingBps =
    typeof metrics.funding_rate === "number"
      ? metrics.funding_rate * 10_000
      : 0;
  const orderFlow = metrics.order_flow ?? 0;
  const putCall = metrics.put_call_ratio ?? 1;

  return (
    <div className="flex flex-col gap-4">
      {/* =================== TOP ROW =================== */}
      <div className="grid gap-4 xl:grid-cols-3">
        {/* Hourly Signal / Bot Call */}
        <Card className={`${glassPanel} xl:col-span-2`}>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div>
                  <CardTitle>Hourly Signal</CardTitle>
                  <CardDescription>
                    Next 1 hour bias & AI conviction.
                  </CardDescription>
                </div>
                {/* FIX: Passed text as label prop instead of children */}
                <HelpTooltip label="Phase 1 focuses only on hourly predictions. Nowa estimates direction, conviction and a plausible 1h range – it does not place trades yet." />
              </div>
              <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-300 text-[10px]">
                Live · Hourly
              </span>
            </div>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            <div className="space-y-1">
              <div
                className="text-sm"
                aria-label={`Nowa hourly signal indicates ${sideLabel.toLowerCase()} with ${confPct.toFixed(
                  1
                )} percent confidence`}
              >
                <span className={`font-semibold ${biasColor}`}>
                  {biasLabel}
                </span>
                <span className="ml-1 text-xs text-slate-400">
                  ({confPct.toFixed(1)}% confidence)
                </span>
              </div>
              <div className={`text-3xl font-semibold ${botColor}`}>
                {botSignal}
              </div>
              <p className={`${faintText} text-[11px]`}>
                This view is scoped to the next hour only. Longer
                horizons, execution and portfolio logic will come
                in later phases. Use this panel to understand how
                the AI reacts to market regimes, not to mirror-trade it.
              </p>
            </div>

            <div className="grid grid-cols-3 gap-3 text-[11px] mt-2">
              <div>
                <div className={faintText}>Next 1h move</div>
                <div
                  className={
                    nextHourMovePct >= 0
                      ? "text-emerald-300 font-semibold"
                      : "text-rose-300 font-semibold"
                  }
                >
                  {nextHourMovePct >= 0 ? "+" : ""}
                  {nextHourMovePct.toFixed(2)}%
                </div>
              </div>
              <div>
                <div className={faintText}>1h range (±)</div>
                <div className="text-sky-300 font-semibold">
                  ±{hourlyRangePct.toFixed(2)}%
                </div>
              </div>
              <div>
                <div className={faintText}>Last price (demo)</div>
                <div className="text-slate-100 font-semibold">
                  {basePrice.toLocaleString("en-US", {
                    style: "currency",
                    currency: "USD",
                    maximumFractionDigits: 2,
                  })}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Market Sentiment (hourly context) */}
        <Card className={glassPanel}>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Market Sentiment (1h)</CardTitle>
                <CardDescription>
                  Risk-on / risk-off tone feeding the signal.
                </CardDescription>
              </div>
              {/* FIX: Passed text as label prop instead of children */}
              <HelpTooltip label="Synthetic sentiment built from demo inputs. In production this would be wired to real funding, positioning and on-chain feeds." />
            </div>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="flex items-center gap-4">
              <SentimentRing score={sentimentClamped} />
              <div className="space-y-1 text-xs">
                <div
                  className={
                    sentimentColor + " font-semibold text-sm"
                  }
                >
                  {sentimentLabel}
                </div>
                <div className="text-slate-400 text-[11px]">
                  Sentiment score is normalised from -1..+1 into
                  0-100.
                </div>
                <div className="flex items-center gap-1 text-[11px]">
                  <span className="text-slate-400">Score:</span>
                  <span className="text-slate-100 font-semibold">
                    {sentimentClamped}/100
                  </span>
                </div>
              </div>
            </div>
            <div className="h-16">
              <ResponsiveContainer>
                <AreaChart data={sentimentHistory}>
                  <CartesianGrid
                    stroke="rgba(148,163,253,0.12)"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="t"
                    tick={{ fontSize: 10, fill: "#9CA3AF" }}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: "#9CA3AF" }}
                    tickFormatter={(v) => v.toFixed(1)}
                    domain={[-1, 1]}
                  />
                  <Tooltip />
                  <Area
                    type="monotone"
                    dataKey="score"
                    stroke="#22c55e"
                    fill="rgba(34,197,94,0.12)"
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* =================== MIDDLE ROW =================== */}
      <div className="grid gap-4 xl:grid-cols-3">
        {/* Hourly Price Path */}
        <Card className={`${glassPanel} xl:col-span-2`}>
          <CardHeader>
            <CardTitle>Hourly Price Path</CardTitle>
            <CardDescription>
              Synthetic path for the next few hours based on the
              hourly signal.
            </CardDescription>
          </CardHeader>
          <CardContent className="h-60">
            <ResponsiveContainer>
              <AreaChart data={hourlySeries}>
                <CartesianGrid
                  stroke="rgba(148,163,253,0.12)"
                  vertical={false}
                />
                <XAxis
                  dataKey="t"
                  tick={{ fontSize: 11, fill: "#9CA3AF" }}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: "#9CA3AF" }}
                  tickFormatter={(v) =>
                    v.toLocaleString("en-US", {
                      maximumFractionDigits: 0,
                    })
                  }
                />
                <Tooltip
                  formatter={(value: any) =>
                    (value as number).toLocaleString("en-US", {
                      style: "currency",
                      currency: "USD",
                      maximumFractionDigits: 2,
                    })
                  }
                />
                <Area
                  type="monotone"
                  dataKey="price"
                  stroke="#22c55e"
                  fill="rgba(34,197,94,0.20)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
            <p className={`${faintText} text-[9px] mt-1`}>
              Curve is illustrative: it uses the hourly bias and
              volatility to sketch a plausible path, not a precise
              tick-by-tick forecast.
            </p>
          </CardContent>
        </Card>

        {/* Flow & Volatility Snapshot */}
        <Card className={glassPanel}>
          <CardHeader>
            <CardTitle>Flow & Volatility (1h lens)</CardTitle>
            <CardDescription>
              What the model sees in funding, flow and options.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 text-[11px]">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className={faintText}>Funding bias</span>
                <DirectionLabel
                  value={fundingBps}
                  ariaLabel="Funding bias in basis points"
                />
              </div>
              <p className={faintText}>
                {Math.abs(fundingBps).toFixed(1)} bps ·
                {fundingBps > 0
                  ? " Longs paying shorts (risk-on)."
                  : fundingBps < 0
                  ? " Shorts paying longs (hedging demand)."
                  : " Funding is balanced at the moment."}
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className={faintText}>Orderflow tilt (1h)</span>
                <DirectionLabel
                  value={orderFlow}
                  ariaLabel="Relative bid vs ask pressure"
                />
              </div>
              <p className={faintText}>
                {orderFlow > 0
                  ? "Bids dominant · buyers more aggressive into the book."
                  : orderFlow < 0
                  ? "Asks dominant · supply is heavier on the tape."
                  : "Flow looks balanced over the last hour."}
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className={faintText}>Put/Call ratio</span>
                <span className="font-semibold text-slate-100">
                  {putCall.toFixed(2)}
                </span>
              </div>
              <p className={faintText}>
                Values below 1.0 tend to be more bullish, above 1.0
                more hedged.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* =================== BOTTOM KPIs (HOURLY) =================== */}
      <div className="grid gap-4 md:grid-cols-3 xl:grid-cols-6">
        <KpiTile
          label="Signal Confidence (1h)"
          value={`${confPct.toFixed(1)}%`}
          hint="Model conviction specifically for the next 1 hour."
        />
        <KpiTile
          label="Next 1h Move (Δ%)"
          value={`${nextHourMovePct >= 0 ? "+" : ""}${nextHourMovePct.toFixed(
            2
          )}%`}
          hint="Expected directional move over the next hour."
        />
        <KpiTile
          label="Expected 1h Range (±%)"
          value={`±${hourlyRangePct.toFixed(2)}%`}
          hint="Typical range the price might trade within in the next hour."
        />
        <KpiTile
          label="Market Sentiment Score"
          value={`${sentimentClamped}/100`}
          hint="Aggregated risk-on / risk-off tone from sentiment inputs."
        />
        <KpiTile
          label="Funding Bias (bps)"
          value={fundingBps.toFixed(1)}
          hint="Positive = longs paying shorts · negative = shorts paying longs."
        />
        <KpiTile
          label="Orderflow Tilt"
          value={orderFlow.toFixed(2)}
          hint=">0 means bid-heavy flow, <0 ask-heavy."
        />
      </div>
    </div>
  );
};

export default DashboardTab;