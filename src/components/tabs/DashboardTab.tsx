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

  // ---- Series prep ----
  const sentimentSeries = useMemo(
    () =>
      (intel.sentimentHistory ?? []).map((d: any, i: number) => ({
        t: d.t ?? d.time ?? i,
        s: typeof d.score === "number" ? d.score : 0,
      })),
    [intel.sentimentHistory]
  );

  const onchainSeries = useMemo(
    () =>
      (intel.onChainHistory ?? []).map((d: any, i: number) => {
        const raw =
          typeof d.active === "number"
            ? d.active
            : typeof d.v === "number"
              ? d.v
              : 0;
        return { t: d.t ?? d.time ?? i, v: raw };
      }),
    [intel.onChainHistory]
  );

  // ---- Core signal logic ----
  const dir = predict?.direction ?? "flat";
  const sideLabel =
    dir === "up" ? "LONG BIAS" :
      dir === "down" ? "SHORT BIAS" :
        "NEUTRAL";
  const conf =
    typeof predict?.price_confidence === "number"
      ? predict.price_confidence
      : 0.6;
  const confClamped = Math.max(0, Math.min(1, conf));
  const confPct = confClamped * 100;

  const botSignal = useMemo(() => {
    if (confClamped < 0.55 || dir === "flat") return "HOLD";
    if (dir === "up") return confClamped > 0.72 ? "STRONG BUY" : "BUY";
    if (dir === "down")
      return confClamped > 0.72 ? "STRONG SELL" : "SELL";
    return "HOLD";
  }, [dir, confClamped]);

  const botColor =
    botSignal.includes("BUY")
      ? "text-emerald-400"
      : botSignal.includes("SELL")
        ? "text-rose-400"
        : "text-slate-300";

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

  // Metrics
  const vol24 =
    typeof predict?.volatility_pct === "number"
      ? predict.volatility_pct
      : 32;
  const latestIv = Math.max(18, Math.min(120, vol24 * 1.2));

  const fundingBps =
    typeof metrics.funding_rate === "number"
      ? metrics.funding_rate * 10_000
      : 0;

  const orderflow =
    typeof metrics.order_flow === "number"
      ? {
        bids: (1 + metrics.order_flow) / 2,
        asks: (1 - metrics.order_flow) / 2,
      }
      : { bids: 0.5, asks: 0.5 };

  const putCall =
    typeof metrics.put_call_ratio === "number"
      ? metrics.put_call_ratio
      : 0.92;

  // Sentiment ring input: latest.sentiment is approx [-1,1]
  const sentimentRaw =
    typeof latest.sentiment === "number" ? latest.sentiment : 0;
  const sentimentValue = Math.max(
    0,
    Math.min(1, (sentimentRaw + 1) / 2)
  );
  const sentimentLabel =
    sentimentRaw > 0.25
      ? "Risk-On"
      : sentimentRaw < -0.25
        ? "Risk-Off"
        : "Neutral";

  return (
    <div className="flex flex-col gap-4">
      {/* =================== TOP ROW =================== */}
      <div className="grid gap-4 xl:grid-cols-3">


        {/* Live Signal / Bot Call */}
        <Card className={glassPanel}>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Live Signal</CardTitle>
                <CardDescription>
                  Real-time feel using mock ensemble output.
                </CardDescription>
              </div>
              <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-300 text-[10px]">
                Streaming
              </span>
            </div>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div
              className="text-sm"
              aria-label={`Nowa Bot Signal indicates ${sideLabel.toLowerCase()} with ${confPct.toFixed(
                1
              )} percent confidence`}
            >
              <span className="font-semibold">{sideLabel}</span>
              <span className="ml-1 text-xs text-slate-400">
                ({confPct.toFixed(1)}% conf.)
              </span>
            </div>
            <div className={`text-3xl font-semibold ${botColor}`}>
              {botSignal}
            </div>
            <div className="grid grid-cols-3 gap-3 text-[11px] mt-2">
              <div>
                <div className={faintText}>Bias</div>
                <div className={`${biasColor} font-semibold`}>
                  {biasLabel}
                </div>
              </div>
              <div>
                <div className={faintText}>
                  Horizon (scalp/swing)
                </div>
                <div className="text-slate-100 font-semibold">
                  5–30m
                </div>
              </div>
              <div>
                <div className={faintText}>Win Prob (demo)</div>
                <div className="text-sky-300 font-semibold">
                  {Math.max(
                    52,
                    Math.min(78, confPct)
                  ).toFixed(1)}
                  %
                </div>
              </div>
            </div>
            <p
              className={`${faintText} text-[10px] leading-snug`}
            >
              Values are synthetic but structurally identical to your
              production response schema, so going live is a
              configuration change, not a redesign.
            </p>
          </CardContent>
        </Card>

        {/* Market Sentiment with glowing ring & number */}
        <Card className={glassPanel}>
          <CardHeader>
            <CardTitle>Market Sentiment</CardTitle>
            <CardDescription>
              Aggregated risk-on / risk-off view
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col items-center justify-center gap-2">
            <SentimentRing value={sentimentValue} />
            <div className="text-xs text-slate-300">
              {sentimentLabel} regime from on-chain, funding &
              order-flow composites.
            </div>
          </CardContent>
        </Card>
        {/* Mars AI Summary */}
        <Card className={glassPanel}>
          <CardHeader>
            <CardTitle>Nowa AI Summary</CardTitle>
            <CardDescription>
              {symbol} · {mode} · {exchange}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <p className={faintText}>
              Ensemble models currently lean{" "}
              <span className={biasColor}>{biasLabel}</span> with
              confidence{" "}
              <span className="text-sky-300">
                {confPct.toFixed(1)}%
              </span>
              .
            </p>
            <p className={faintText}>
              <span className="font-semibold text-slate-200">
                Nowa Bot Signal:
              </span>{" "}
              <span className={`font-semibold ${botColor}`}>
                {botSignal}
              </span>{" "}
            </p>
            <p className={faintText}>
              Implied volatility around{" "}
              <span className="text-indigo-300">
                {latestIv.toFixed(1)}%
              </span>{" "}
              and funding at{" "}
              <span
                className={
                  fundingBps > 0
                    ? "text-emerald-300"
                    : fundingBps < 0
                      ? "text-rose-300"
                      : "text-slate-300"
                }
              >
                {fundingBps.toFixed(2)} bps
              </span>{" "}
              set the risk tone for directional and carry trades.
            </p>
          </CardContent>
        </Card>
      </div>

      {/* =================== MIDDLE ROW =================== */}
      <div className="grid gap-4 xl:grid-cols-3">
        {/* Order Book Imbalance */}
        <Card className={glassPanel}>
          <CardHeader>
            <CardTitle>Order Book Imbalance</CardTitle>
            <CardDescription>
              Relative stacked liquidity across bids and asks.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex justify-between text-[10px] text-slate-400">
              <span>Bids</span>
              <span>Asks</span>
            </div>
            <div className="w-full h-3 rounded-full bg-slate-900 overflow-hidden flex">
              <div
                className="bg-emerald-400/80"
                style={{ width: `${orderflow.bids * 100}%` }}
              />
              <div
                className="bg-rose-500/80"
                style={{ width: `${orderflow.asks * 100}%` }}
              />
            </div>
            <div className="flex justify-between text-[10px] text-slate-400">
              <span>
                {(orderflow.bids * 100).toFixed(1)}% bid depth
              </span>
              <span>
                {(orderflow.asks * 100).toFixed(1)}% ask depth
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Funding & IV Snapshot */}
        <Card className={glassPanel}>
          <CardHeader>
            <CardTitle>Funding &amp; IV Snapshot</CardTitle>
            <CardDescription>
              Perp skew vs. volatility for this asset.
            </CardDescription>
            <DirectionLabel
              value={fundingBps}
              ariaLabel={`Perpetual funding rate ${fundingBps.toFixed(1)} basis points`}
            />
          </CardHeader>
          <CardContent className="h-40">
            <ResponsiveContainer>
              <AreaChart
                data={[
                  { t: "Funding (bps)", v: fundingBps },
                  { t: "30d IV (%)", v: latestIv },
                ]}
              >
                <CartesianGrid
                  stroke="rgba(148,163,253,0.14)"
                  vertical={false}
                />
                <XAxis dataKey="t" />
                <YAxis hide />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey="v"
                  stroke="#38bdf8"
                  fill="rgba(56,189,248,0.25)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* On-chain & Dev Pulse */}
        <Card className={glassPanel}>
          <CardHeader>
            <CardTitle>On-chain &amp; Dev Pulse</CardTitle>
            <CardDescription>
              Activity + builder flow trend.
            </CardDescription>
          </CardHeader>
          <CardContent className="h-40">
            <ResponsiveContainer>
              <AreaChart data={onchainSeries}>
                <CartesianGrid
                  stroke="rgba(148,163,253,0.12)"
                  vertical={false}
                />
                <XAxis dataKey="t" hide />
                <YAxis hide />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey="v"
                  stroke="#6366f1"
                  fill="rgba(79,70,229,0.28)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* =================== BOTTOM KPIs =================== */}
      <div className="grid gap-4 md:grid-cols-4">
        <KpiTile
          label="AI Confidence"
          value={`${confPct.toFixed(1)}%`}
          hint="Primary model conviction."
        />
        <KpiTile
          label="Funding Skew"
          value={`${fundingBps.toFixed(2)} bps`}
          hint="Demo funding from metrics."
        />
        <KpiTile
          label="Put/Call Ratio"
          value={putCall.toFixed(2)}
          hint="Options sentiment."
        />
        <KpiTile
          label="Volatility (30d proxy)"
          value={`${latestIv.toFixed(1)}%`}
          hint="From model volatility."
        />
      </div>
    </div>
  );
};

export default DashboardTab;
