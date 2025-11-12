"use client";

import React, { useMemo } from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  KpiTile,
  Badge,
  glassPanel,
  faintText,
} from "../layout/AppShell";
import { useMarketIntel } from "../layout/AppShell";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";
import type {
  SymbolCode,
  MarketMode,
} from "@/lib/api";

type IntelSeriesPoint = {
  t?: any;
  time?: any;
  s?: number;
  score?: number;
  v?: number;
  value?: number;
  active?: number;
  iv?: number;
  funding?: number;
};

type CorItem = { name: string; val: number };

const safeNum = (v: any, d = 0) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
};

const last = <T,>(
  arr?: T[]
): T | undefined =>
  arr && arr.length
    ? arr[arr.length - 1]
    : undefined;

const Spark: React.FC<{
  pts?: { v: number }[];
}> = ({ pts }) => {
  if (!pts || pts.length < 2)
    return null;
  const min = Math.min(
    ...pts.map((p) => p.v)
  );
  const max = Math.max(
    ...pts.map((p) => p.v)
  );
  const rng = max - min || 1;
  const d = pts
    .map((p, i) => {
      const x =
        (i / (pts.length - 1)) *
        100;
      const y =
        100 -
        ((p.v - min) / rng) * 100;
      return `${
        i === 0 ? "M" : "L"
      }${x},${y}`;
    })
    .join(" ");
  return (
    <svg
      viewBox="0 0 100 100"
      className="w-full h-6"
      preserveAspectRatio="none"
    >
      <path
        d={d}
        fill="none"
        stroke="currentColor"
        strokeWidth={1}
      />
    </svg>
  );
};

const PlaybookCard: React.FC<{
  title: string;
  bullets: string[];
}> = ({ title, bullets }) => (
  <Card>
    <CardHeader className="pb-2">
      <CardTitle className="text-sm">
        {title}
      </CardTitle>
    </CardHeader>
    <CardContent className="text-xs space-y-1">
      {bullets.map((b, i) => (
        <div
          key={i}
          className={faintText}
        >
          • {b}
        </div>
      ))}
    </CardContent>
  </Card>
);

const ExplainCard: React.FC = () => (
  <Card>
    <CardHeader>
      <CardTitle>
        What it means
      </CardTitle>
      <CardDescription>
        Human-readable explainer
      </CardDescription>
    </CardHeader>
    <CardContent className="text-xs space-y-1">
      <p>
        <b>Funding Skew</b> shows
        who pays whom on
        perpetuals. Positive =
        longs paying (crowded
        longs); negative =
        shorts paying.
      </p>
      <p>
        <b>30d IV</b> is options’
        view of future vol. High:
        favor credit; Low: favor
        debit.
      </p>
      <p>
        <b>Open Interest</b>{" "}
        tracks outstanding
        futures/options; rising
        with price = trend
        confirmation.
      </p>
      <p>
        <b>CVD</b> tracks net buy
        vs sell pressure; big
        divergences matter.
      </p>
      <p>
        <b>Regime</b> (momentum vs
        mean-revert) guides
        breakout vs fade.
      </p>
    </CardContent>
  </Card>
);

const RiskRadar: React.FC<{
  regime: string;
  trend: string;
  liquidity: string;
  warning: string;
}> = ({
  regime,
  trend,
  liquidity,
  warning,
}) => (
  <Card>
    <CardHeader>
      <CardTitle>
        Risk Radar
      </CardTitle>
    </CardHeader>
    <CardContent className="grid grid-cols-2 gap-2 text-xs">
      <div>
        <div
          className={
            faintText
          }
        >
          Trend
        </div>
        <div>{trend}</div>
      </div>
      <div>
        <div
          className={
            faintText
          }
        >
          Liquidity
        </div>
        <div>{liquidity}</div>
      </div>
      <div>
        <div
          className={
            faintText
          }
        >
          Regime
        </div>
        <div>{regime}</div>
      </div>
      <div>
        <div
          className={
            faintText
          }
        >
          Warning
        </div>
        <div>{warning}</div>
      </div>
    </CardContent>
  </Card>
);

export const MarketIntelTab: React.FC<{
  symbol: SymbolCode;
  mode: MarketMode;
  exchange: string;
}> = ({ symbol, mode }) => {
  const { intel } = useMarketIntel(symbol);
  const pro: any = intel || {};

  const sentimentArr: IntelSeriesPoint[] =
    pro.sentimentHistory ?? [];
  const chainArr: IntelSeriesPoint[] =
    pro.onChainHistory ??
    pro.onchainHistory ??
    [];
  const ivArr: IntelSeriesPoint[] =
    pro.ivHistory ?? [];
  const fundArr: IntelSeriesPoint[] =
    pro.fundingHistory ??
    pro.fundingSkew ??
    [];
  const oiArr: IntelSeriesPoint[] =
    pro.oiHistory ??
    pro.openInterest ??
    [];
  const cvdArr: IntelSeriesPoint[] =
    pro.cvdHistory ??
    pro.cvd ??
    [];

  const fundingSkew = safeNum(
    last(fundArr)?.v ??
      last(fundArr)?.funding,
    0
  );
  const iv30 = safeNum(
    last(ivArr)?.iv ??
      last(ivArr)?.v ??
      last(ivArr)?.value,
    0.55
  );
  const oi = safeNum(
    last(oiArr)?.v ?? last(oiArr),
    0
  );
  const cvdDelta = safeNum(
    last(cvdArr)?.v ?? last(cvdArr),
    0
  );
  const latestScore = safeNum(
    last(sentimentArr)?.score ??
      last(sentimentArr)?.s,
    0
  );

  const regimeText =
    pro.radar?.regime ??
    (latestScore > 0.25
      ? "MOMENTUM"
      : latestScore < -0.25
      ? "MEAN-REVERT"
      : "BALANCED");

  const crowdingLabel =
    pro.radar?.crowding ??
    (fundingSkew > 0.0001
      ? "Longs crowded"
      : "Normal");

  const correlations: CorItem[] =
    Array.isArray(pro.correlations)
      ? pro.correlations
      : [
          { name: "BTC.D", val: 0.42 },
          { name: "ETH.D", val: 0.35 },
          { name: "SPX", val: 0.18 },
          { name: "DXY", val: -0.27 },
          { name: "VIX", val: -0.31 },
        ];

  const sparkFunding = (fundArr ?? [])
    .slice(-30)
    .map((x: any) => ({
      v: safeNum(
        x.v ?? x.funding,
        0
      ),
    }));
  const sparkIv = (ivArr ?? [])
    .slice(-30)
    .map((x: any) => ({
      v: safeNum(
        x.iv ??
          x.v ??
          x.value,
        0
      ),
    }));
  const sparkOi = (oiArr ?? [])
    .slice(-30)
    .map((x: any) => ({
      v: safeNum(
        x.v ?? x,
        0
      ),
    }));
  const sparkCvd = (cvdArr ?? [])
    .slice(-30)
    .map((x: any) => ({
      v: safeNum(
        x.v ?? x,
        0
      ),
    }));

  const futuresBullets = [
    regimeText === "MOMENTUM"
      ? "Favor trend-following entries; trail stops below swing lows."
      : regimeText ===
        "MEAN-REVERT"
      ? "Fade extremes; quick exits; mean-revert focus."
      : "Trade light until clearer bias.",
    fundingSkew > 0
      ? "Funding positive: watch long crowding; reduce leverage."
      : fundingSkew < 0
      ? "Funding negative: watch short squeezes; partial TP."
      : "Funding neutral: normal sizing.",
  ];

  const ivPct = iv30 * 100;

  const optionsBullets = [
    ivPct > 80
      ? "IV high: favor credit strategies / iron condors."
      : ivPct < 40
      ? "IV low: favor debit structures / calendars."
      : "IV mid: balance between credit and debit spreads.",
    regimeText === "MOMENTUM"
      ? "Directional call/put spreads with trailing exits."
      : regimeText ===
        "MEAN-REVERT"
      ? "Broken wings or calendars around mean."
      : "Neutral flies/condors near expected range.",
  ];

  const spotBullets = [
    regimeText === "MOMENTUM"
      ? "DCA on dips, avoid late chasing."
      : regimeText ===
        "MEAN-REVERT"
      ? "Add at support, trim into rips."
      : "Modest DCA until clarity.",
    chainArr.length
      ? "Scale if on-chain activity confirms trend."
      : "Size conservatively until on-chain confirms.",
  ];

  const trend =
    pro.radar?.trend ??
    (latestScore > 0
      ? "UP"
      : "DOWN");
  const liquidity =
    pro.radar?.liquidity ??
    (chainArr.length
      ? "NORMAL"
      : "LOW");
  const warning =
    pro.radar?.warning ??
    (crowdingLabel !==
    "Normal"
      ? crowdingLabel
      : "—");

  return (
    <div className="flex flex-col gap-6">
      {/* KPI GRID */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiTile
          label="Funding Skew"
          value={`${(
            fundingSkew *
            1e4
          ).toFixed(2)} bps`}
         // spark={sparkFunding}
        />
        <KpiTile
          label="30d IV"
          value={`${ivPct.toFixed(
            1
          )}%`}
          //spark={sparkIv}
        />
        <KpiTile
          label="Open Interest"
          value={oi.toLocaleString()}
          //spark={sparkOi}
        />
        <KpiTile
          label="CVD Δ"
          value={cvdDelta.toFixed(
            0
          )}
         // spark={sparkCvd}
        />
      </div>

      {/* NEON CHARTS */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle>
              Sentiment
            </CardTitle>
          </CardHeader>
          <CardContent className="h-56">
            <ResponsiveContainer
              width="100%"
              height="100%"
            >
              <LineChart
                data={sentimentArr.map(
                  (d, i) => ({
                    t:
                      d.t ??
                      d.time ??
                      i,
                    s: safeNum(
                      d.score ??
                        d.s,
                      0
                    ),
                  })
                )}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  opacity={0.08}
                />
                <XAxis
                  dataKey="t"
                  hide
                />
                <YAxis
                  domain={[
                    -1,
                    1,
                  ]}
                  hide
                />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="s"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>
              On-Chain
              Activity
            </CardTitle>
          </CardHeader>
          <CardContent className="h-56">
            <ResponsiveContainer
              width="100%"
              height="100%"
            >
              <AreaChart
                data={chainArr.map(
                  (d, i) => ({
                    t:
                      d.t ??
                      d.time ??
                      i,
                    v: safeNum(
                      d.active ??
                        d.v ??
                        d.value,
                      0
                    ),
                  })
                )}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  opacity={0.08}
                />
                <XAxis
                  dataKey="t"
                  hide
                />
                <YAxis hide />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey="v"
                  strokeWidth={1}
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Correlations + Radar */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>
              Cross-Market
              Correlations
            </CardTitle>
          </CardHeader>
          <CardContent className="h-40">
            <ResponsiveContainer
              width="100%"
              height="100%"
            >
              <BarChart
                data={correlations.map(
                  (c) => ({
                    name: c.name,
                    v: safeNum(
                      c.val,
                      0
                    ),
                  })
                )}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  opacity={0.08}
                />
                <XAxis
                  dataKey="name"
                />
                <YAxis />
                <Tooltip />
                <Bar
                  dataKey="v"
                  radius={6}
                />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <RiskRadar
          regime={regimeText}
          trend={trend}
          liquidity={`${liquidity}`}
          warning={`${warning}`}
        />
      </div>

      {/* Playbooks */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <PlaybookCard
          title="Futures Playbook"
          bullets={futuresBullets}
        />
        <PlaybookCard
          title="Options Playbook"
          bullets={optionsBullets}
        />
        <PlaybookCard
          title="Spot Playbook"
          bullets={spotBullets}
        />
      </div>

      {/* Explainer */}
      <ExplainCard />
    </div>
  );
};

export default MarketIntelTab;
