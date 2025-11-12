"use client";

import React, { useEffect, useState } from "react";
import { toast } from "sonner";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  KpiTile,
  glassPanel,
  faintText,
  Button,
} from "../layout/AppShell";
import { ShieldCheck, DatabaseZap } from "lucide-react";
import API from "@/lib/api";

type Health = {
  workers: number;
  latency: number;
  gpu: number;
  cpu: number;
  ram: number;
  queue: number;
  status: "OK" | "DEGRADED" | "DOWN";
};

export const SystemTab: React.FC = () => {
  const [gpu, setGpu] = useState(42);
  const [cpu, setCpu] = useState(18);
  const [ram, setRam] = useState(62);
  const [queue, setQueue] = useState(3);
  const [orders, setOrders] = useState(1);
  const [latency, setLatency] = useState(122);
  const [backend, setBackend] = useState<"Up" | "Degraded" | "Down">("Up");
  const [workers, setWorkers] = useState("Healthy");
  const [modelVer] = useState("v2.7.3-rl");
  const isMock = (API as any)?.MOCK === true;

  useEffect(() => {
    const id = setInterval(() => {
      setGpu((v) => Math.min(97, Math.max(1, v + (Math.random() * 8 - 4))));
      setCpu((v) => Math.min(95, Math.max(1, v + (Math.random() * 6 - 3))));
      setRam((v) => Math.min(99, Math.max(1, v + (Math.random() * 4 - 2))));
      setQueue((v) => Math.max(0, v + (Math.random() * 2 - 1)));
      setOrders((v) => Math.max(0, v + (Math.random() * 2 - 1)));
      setLatency((v) => Math.max(45, v + (Math.random() * 15 - 8)));
      setBackend(Math.random() > 0.95 ? "Degraded" : "Up");
    }, 3000);
    return () => clearInterval(id);
  }, []);

  const StatusCard = ({ label, value, color }: any) => (
    <div className={`${glassPanel} rounded-xl p-3`}>
      <div className={`text-xs ${faintText}`}>{label}</div>
      <div className={`text-sm font-semibold ${color}`}>{value}</div>
    </div>
  );

  return (
    <div className="grid gap-6">
      <Card>
        <CardHeader>
          <CardTitle>System Status</CardTitle>
          <CardDescription>AI engine health, compute, infra & services</CardDescription>
        </CardHeader>

        <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatusCard label="Workers" value={workers} color="text-emerald-400" />
          <StatusCard label="Inference Latency" value={`${latency} ms`} color="text-indigo-400" />
          <StatusCard
            label="Backend API"
            value={backend}
            color={backend === "Up" ? "text-emerald-400" : "text-amber-400"}
          />
          <StatusCard label="Mode" value={isMock ? "Mock" : "Live"} color="text-blue-300" />

          <StatusCard label="GPU Utilization" value={`${gpu.toFixed(0)}%`} color="text-purple-400" />
          <StatusCard label="CPU Load" value={`${cpu.toFixed(0)}%`} color="text-amber-300" />
          <StatusCard label="Memory Use" value={`${ram.toFixed(0)}%`} color="text-sky-300" />
          <StatusCard label="Queue Depth" value={`${queue.toFixed(0)}`} color="text-pink-300" />
          <StatusCard label="Open Orders" value={orders.toFixed(0)} color="text-teal-300" />
          <StatusCard label="Model Version" value={modelVer} color="text-indigo-300" />

          <div className="col-span-2 flex gap-2">
            <Button variant="secondary" onClick={() => { toast.success("Model Reload Triggered"); }}>
              Reload Model
            </Button>
            <Button variant="outline" onClick={() => { toast.success("Worker Restart Command Sent"); }}>
              Restart Workers
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SystemTab;
