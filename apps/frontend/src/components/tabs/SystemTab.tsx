"use client";

import React, {
  useEffect,
  useState,
  useCallback,
  useRef,
} from "react";
import { toast } from "sonner";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  glassPanel,
  faintText,
  Button,
} from "../layout/AppShell";
import API, { SystemStatus, ServiceStatus } from "@/lib/api";
import {
  Activity,
  Cpu,
  Zap,
  Server,
  Brain,
  AlertTriangle,
  Database,
  CloudLightning,
  Cloud,
  Globe2,
  Sparkles,
} from "lucide-react";
const RL_MODE = (process.env.NEXT_PUBLIC_RL_MODE ?? "standby") as
  | "standby"
  | "live";

type ConnectionState = "connecting" | "live" | "degraded" | "down";

type StatusCardProps = {
  label: string;
  value: string;
  color: string;
  icon?: React.ComponentType<React.SVGProps<SVGSVGElement>>;
};

const StatusCard: React.FC<StatusCardProps> = ({
  label,
  value,
  color,
  icon: Icon,
}) => (
  <div
    className={`${glassPanel} rounded-xl p-3 flex items-center justify-between`}
  >
    <div>
      <div className={`text-xs ${faintText}`}>{label}</div>
      <div className={`text-sm font-semibold ${color}`}>{value}</div>
    </div>
    {Icon && <Icon className={`w-4 h-4 opacity-60 ${color}`} />}
  </div>
);

function serviceColor(status: ServiceStatus | undefined): string {
  switch (status) {
    case "ok":
      return "text-emerald-300";
    case "degraded":
      return "text-amber-300";
    case "down":
      return "text-rose-400";
    default:
      return "text-slate-500";
  }
}

function serviceLabel(status: ServiceStatus | undefined): string {
  switch (status) {
    case "ok":
      return "Healthy";
    case "degraded":
      return "Degraded";
    case "down":
      return "Down";
    default:
      return "Unknown";
  }
}

export const SystemTab: React.FC = () => {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [connState, setConnState] = useState<ConnectionState>("connecting");
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pollTimerRef = useRef<number | null>(null);
  const backoffRef = useRef<number>(5000);
  const stoppedRef = useRef<boolean>(false);
  const sseUnsubRef = useRef<null | (() => void)>(null);

  const clearPollTimer = () => {
    if (pollTimerRef.current !== null) {
      window.clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  };

  const schedulePoll = useCallback(
    (delay: number) => {
      if (stoppedRef.current) return;
      clearPollTimer();
      pollTimerRef.current = window.setTimeout(() => {
        void fetchStatus();
      }, delay);
    },
    [] // eslint-disable-line react-hooks/exhaustive-deps
  );

  const applyConnectionState = (s: SystemStatus) => {
    if (s.status === "DOWN") setConnState("down");
    else if (s.status === "DEGRADED") setConnState("degraded");
    else setConnState("live");
  };

  const updateFromSnapshot = useCallback((snap: SystemStatus) => {
    setStatus(snap);
    applyConnectionState(snap);
    if (snap.timestamp) {
      setLastUpdated(new Date(snap.timestamp));
    } else {
      setLastUpdated(new Date());
    }
    backoffRef.current = 5000;
    setError(null);
  }, []);

  // ---- Polling fallback ----

  const fetchStatus = useCallback(
    async (opts?: { manual?: boolean }) => {
      try {
        if (opts?.manual) {
          clearPollTimer();
          backoffRef.current = 5000;
        }
        const data = await API.getSystemStatus();
        updateFromSnapshot(data);

        const nextDelay =
          data.status === "OK"
            ? 3000
            : data.status === "DEGRADED"
            ? 7000
            : 12000;
        schedulePoll(nextDelay);
      } catch (e: any) {
        console.error("Failed to fetch system status", e);
        setError(e?.message || "Unable to reach backend (polling)");
        backoffRef.current = Math.min(backoffRef.current * 1.7, 30000);
        setConnState("down");
        schedulePoll(backoffRef.current);
      }
    },
    [schedulePoll, updateFromSnapshot]
  );

  const startPollingFallback = useCallback(() => {
    console.warn("[SystemTab] Falling back to polling");
    toast.warning("Realtime stream unavailable, using fallback polling");
    clearPollTimer();
    void fetchStatus({ manual: true });
  }, [fetchStatus]);

  // ---- SSE subscription ----

  const startSSE = useCallback(() => {
    if (sseUnsubRef.current) {
      sseUnsubRef.current();
      sseUnsubRef.current = null;
    }

    setConnState("connecting");

    sseUnsubRef.current = API.subscribeSystemStream(
      (snap) => {
        updateFromSnapshot(snap);
      },
      (err) => {
        console.error("[SystemTab] SSE error", err);
        setError("Realtime stream error, switching to polling fallback");
        setConnState("degraded");

        if (sseUnsubRef.current) {
          sseUnsubRef.current();
          sseUnsubRef.current = null;
        }

        startPollingFallback();
      }
    );
  }, [startPollingFallback, updateFromSnapshot]);

  // ---- Lifecycle / visibility ----

  useEffect(() => {
    stoppedRef.current = false;

    const handleVisibility = () => {
      if (document.visibilityState === "hidden") {
        clearPollTimer();
      } else {
        void fetchStatus({ manual: true });
      }
    };

    document.addEventListener("visibilitychange", handleVisibility);
    startSSE();

    return () => {
      stoppedRef.current = true;
      clearPollTimer();
      document.removeEventListener("visibilitychange", handleVisibility);
      if (sseUnsubRef.current) {
        sseUnsubRef.current();
        sseUnsubRef.current = null;
      }
    };
  }, [fetchStatus, startSSE]);

  const s: SystemStatus = status || {
    status: "DOWN",
    api_latency: 0,
    brain: {
      ready: false,
      l2_model: false,
      llm_engine: false,
      rl_agent: false,
    },
    resources: {
      cpu_load: 0,
      ram_usage: 0,
      gpu_util: 0,
    },
    services: {},
  };

  const services = s.services || {};
  const db = services.db;
  const redis = services.redis;
  const celery = services.celery;
  const binance = services.binance;
  const llm = services.llm_provider;

  const connLabel =
    connState === "live"
      ? "LIVE"
      : connState === "degraded"
      ? "DEGRADED"
      : connState === "connecting"
      ? "CONNECTING"
      : "DOWN";

  const connColor =
    connState === "live"
      ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
      : connState === "degraded"
      ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
      : connState === "connecting"
      ? "bg-sky-500/20 text-sky-300 border-sky-500/40"
      : "bg-rose-500/20 text-rose-300 border-rose-500/40";

  const connDot =
    connState === "live"
      ? "bg-emerald-400"
      : connState === "degraded"
      ? "bg-amber-400"
      : connState === "connecting"
      ? "bg-sky-400"
      : "bg-rose-400";

  return (
    <div className="grid gap-6">
      <Card>
        <CardHeader className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-400" />
              System Status
              <span
                className={`ml-2 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] border ${connColor}`}
              >
                <span
                  className={`inline-block h-1.5 w-1.5 rounded-full ${connDot} animate-pulse`}
                />
                {connLabel}
              </span>
            </CardTitle>
            <CardDescription>
              Real-time telemetry from the AI Engine &amp; Backend
              Infrastructure.
            </CardDescription>
          </div>

          <div className="flex flex-col items-start gap-1 text-[11px] text-slate-400 md:items-end">
            <div>
              Latency (polling):{" "}
              <span
                className={
                  s.api_latency < 250
                    ? "text-emerald-300"
                    : s.api_latency < 800
                    ? "text-amber-300"
                    : "text-rose-300"
                }
              >
                {s.api_latency} ms
              </span>
            </div>
            <div>
              Last update:{" "}
              {lastUpdated
                ? lastUpdated.toLocaleTimeString()
                : "waiting for first sample..."}
            </div>
          </div>
        </CardHeader>

        <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Core Health */}
          <StatusCard
            label="Backend API"
            value={s.status === "OK" ? "HEALTHY" : s.status}
            color={s.status === "OK" ? "text-emerald-400" : "text-rose-400"}
            icon={Server}
          />
          <StatusCard
            label="AI Brain Status"
            value={s.brain.ready ? "ONLINE" : "OFFLINE"}
            color={s.brain.ready ? "text-emerald-400" : "text-rose-400"}
            icon={Brain}
          />
          <StatusCard
            label="L2 Ensemble"
            value={s.brain.l2_model ? "Active" : "Inactive"}
            color={s.brain.l2_model ? "text-blue-400" : "text-slate-500"}
          />
          <StatusCard
            label="LLM Engine"
            value={s.brain.llm_engine ? "Connected" : "Disconnected"}
            color={s.brain.llm_engine ? "text-indigo-300" : "text-slate-500"}
          />

          {/* Resources */}
          <StatusCard
            label="GPU Utilization"
            value={`${s.resources.gpu_util.toFixed(0)}%`}
            color="text-purple-400"
            icon={Cpu}
          />
          <StatusCard
            label="CPU Load"
            value={`${s.resources.cpu_load.toFixed(0)}%`}
            color="text-amber-300"
          />
          <StatusCard
            label="Memory (RAM)"
            value={`${s.resources.ram_usage.toFixed(0)}%`}
            color="text-sky-300"
          />
          {/* RL Agent – treated as optional (standby vs live) */}
{(() => {
  const rlLabel =
    RL_MODE === "standby"
      ? "Standby"
      : s.brain.rl_agent
      ? "Online"
      : "Down";

  const rlColor =
    RL_MODE === "standby"
      ? "text-slate-500"
      : s.brain.rl_agent
      ? "text-emerald-300"
      : "text-rose-400";

  return (
    <StatusCard
      label="RL Agent"
      value={rlLabel}
      color={rlColor}
    />
  );
})()}


          {/* Per-service health */}
          <StatusCard
            label="Database"
            value={serviceLabel(db?.status)}
            color={serviceColor(db?.status)}
            icon={Database}
          />
          <StatusCard
            label="Redis"
            value={serviceLabel(redis?.status)}
            color={serviceColor(redis?.status)}
            icon={Cloud}
          />
          <StatusCard
            label="Celery Workers"
            value={serviceLabel(celery?.status)}
            color={serviceColor(celery?.status)}
            icon={CloudLightning}
          />
          <StatusCard
            label="Price Connectivity"
            value={serviceLabel(binance?.status)}
            color={serviceColor(binance?.status)}
            icon={Globe2}
          />
          <StatusCard
            label="LLM Provider"
            value={serviceLabel(llm?.status)}
            color={serviceColor(llm?.status)}
            icon={Sparkles}
          />

          {/* Actions + Error */}
          <div className="col-span-2 md:col-span-4 flex flex-col gap-2 mt-2">
            <div className="flex flex-wrap gap-2">
              <Button
                variant="secondary"
                onClick={() => {
                  toast.success("Health check requested");
                  void fetchStatus({ manual: true });
                }}
              >
                Refresh Status
              </Button>
              <Button
                variant="outline"
                disabled={!s.brain.ready}
                onClick={() => {
                  toast.info(
                    "Model reload request sent (simulated – wire to backend when ready)"
                  );
                }}
              >
                Reload Models
              </Button>
            </div>

            {error && (
              <div className="flex items-center gap-2 text-xs text-amber-300">
                <AlertTriangle className="w-3 h-3" />
                <span>{error}</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SystemTab;
