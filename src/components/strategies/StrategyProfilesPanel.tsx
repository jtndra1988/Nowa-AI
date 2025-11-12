// src/components/strategies/StrategyProfilesPanel.tsx
"use client";

import React, { useState } from "react";

export type StrategyConfig = Record<string, any>;

type Profile = {
  name: string;
  config: StrategyConfig;
};

type Props = {
  currentConfig: StrategyConfig;
  onApply: (cfg: StrategyConfig) => void;
};

const LS_KEY = "nowa.strategy.profiles.v1";

function loadProfiles(): Profile[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(LS_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as Profile[];
  } catch {
    return [];
  }
}

function saveProfiles(list: Profile[]) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(LS_KEY, JSON.stringify(list));
}

export const StrategyProfilesPanel: React.FC<Props> = ({
  currentConfig,
  onApply,
}) => {
  const [profiles, setProfiles] = useState<Profile[]>(() => loadProfiles());
  const [name, setName] = useState("");
  const [exportJson, setExportJson] = useState("");
  const [importJson, setImportJson] = useState("");

  const sync = (next: Profile[]) => {
    setProfiles(next);
    saveProfiles(next);
  };

  const handleSave = () => {
    if (!name.trim()) return;
    const next = [
      ...profiles.filter((p) => p.name !== name.trim()),
      { name: name.trim(), config: currentConfig },
    ];
    sync(next);
    setName("");
  };

  const handleLoad = (p: Profile) => {
    onApply(p.config);
  };

  const handleDelete = (n: string) => {
    sync(profiles.filter((p) => p.name !== n));
  };

  const handleExport = () => {
    setExportJson(JSON.stringify(profiles, null, 2));
  };

  const handleImport = () => {
    try {
      const parsed = JSON.parse(importJson);
      if (Array.isArray(parsed)) {
        sync(parsed as Profile[]);
        setImportJson("");
      }
    } catch {
      // ignore; you can show toast here
    }
  };

  return (
    <div className="mt-4 rounded-2xl border border-white/10 bg-white/5 p-3 flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-xs font-semibold text-indigo-300">
          Strategy Profiles
        </div>
        <div className="text-[9px] text-slate-500">
          Save / load named AI configs. Share via JSON.
        </div>
      </div>

      <div className="flex gap-2 items-center text-[10px]">
        <input
          placeholder="Profile name, e.g. 'Aggressive BTC Swing'"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="flex-1 rounded-xl bg-slate-950/70 border border-white/10 px-2 py-1 text-[10px] outline-none focus:border-indigo-400/70"
        />
        <button
          onClick={handleSave}
          className="px-3 py-1.5 rounded-xl bg-indigo-500/90 text-[10px] text-white"
        >
          Save
        </button>
      </div>

      <div className="flex flex-wrap gap-1 mt-1">
        {profiles.map((p) => (
          <div
            key={p.name}
            className="flex items-center gap-1 px-2 py-1 rounded-xl bg-white/5 text-[9px]"
          >
            <button
              onClick={() => handleLoad(p)}
              className="underline decoration-indigo-400/70 underline-offset-2"
            >
              {p.name}
            </button>
            <button
              onClick={() => handleDelete(p.name)}
              aria-label={`Delete profile ${p.name}`}
              className="text-slate-500 hover:text-rose-400"
            >
              ✕
            </button>
          </div>
        ))}
        {profiles.length === 0 && (
          <span className="text-[9px] text-slate-500">
            No profiles yet. Save your current setup.
          </span>
        )}
      </div>

      <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-2 text-[9px]">
        <div>
          <button
            onClick={handleExport}
            className="mb-1 px-2 py-1 rounded-xl bg-white/5 border border-white/10"
          >
            Export JSON
          </button>
          <textarea
            value={exportJson}
            readOnly
            rows={4}
            className="w-full rounded-xl bg-slate-950/70 border border-white/10 px-2 py-1"
          />
        </div>
        <div>
          <div className="flex items-center justify-between mb-1">
            <span>Import JSON</span>
            <button
              onClick={handleImport}
              className="px-2 py-1 rounded-xl bg-indigo-500/80 text-white"
            >
              Import
            </button>
          </div>
          <textarea
            value={importJson}
            onChange={(e) => setImportJson(e.target.value)}
            rows={4}
            className="w-full rounded-xl bg-slate-950/70 border border-white/10 px-2 py-1"
          />
        </div>
      </div>
    </div>
  );
};
