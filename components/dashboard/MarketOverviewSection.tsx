"use client";

import { useEffect, useState } from "react";
import { fetchMarketOverview } from "@/lib/api/prices";
import type { MarketIndexItem } from "@/lib/api/prices";

function MarketSparkline({ color, isUp }: { color: string; isUp: boolean }) {
  const path = isUp
    ? "M0 34 C20 38 28 22 46 26 C62 30 70 14 88 18 C108 22 114 10 132 12 C146 14 150 8 160 10"
    : "M0 12 C18 10 28 26 44 22 C60 18 72 34 90 30 C108 26 118 42 134 38 C148 34 152 44 160 40";
  return (
    <svg viewBox="0 0 160 54" className="h-9 w-20" aria-hidden="true">
      <path d={path} fill="none" stroke={color} strokeWidth="4" strokeLinecap="round" />
    </svg>
  );
}

function getMarketStatus(): { label: string; isOpen: boolean } {
  // KST = UTC+9
  const now = new Date();
  const kstHour = (now.getUTCHours() + 9) % 24;
  const kstMin  = now.getUTCMinutes();
  const totalMin = kstHour * 60 + kstMin;
  const day = (now.getUTCDay() + (now.getUTCHours() >= 15 ? 1 : 0)) % 7; // 주말 체크용 근사

  const isWeekend = now.getDay() === 0 || now.getDay() === 6;
  if (isWeekend) return { label: "주말 휴장", isOpen: false };
  if (totalMin < 9 * 60)           return { label: "장 개장 전", isOpen: false };
  if (totalMin <= 15 * 60 + 30)    return { label: "장중",       isOpen: true  };
  return                                  { label: "장 닫힘",     isOpen: false };
}

export default function MarketOverviewSection() {
  const [indices, setIndices] = useState<MarketIndexItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [marketStatus, setMarketStatus] = useState(getMarketStatus);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const timer = setInterval(() => setMarketStatus(getMarketStatus()), 60_000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    fetchMarketOverview()
      .then((data) => { setIndices(data); setError(null); })
      .catch((err: unknown) => {
        setIndices([]);
        setError(err instanceof Error ? err.message : "시장 개요를 불러오지 못했습니다.");
      })
      .finally(() => setLoading(false));
  }, []);

  return (
    <article className="rounded-2xl bg-white p-4 shadow-[0_20px_60px_rgba(15,23,42,0.06)]">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className={`h-2.5 w-2.5 rounded-full ${
            marketStatus.isOpen
              ? "bg-emerald-400 shadow-[0_0_0_4px_rgba(52,211,153,0.14)]"
              : "bg-slate-300"
          } ${loading ? "animate-pulse" : ""}`} />
          <h2 className="text-lg font-black text-slate-950">국내 정규장</h2>
        </div>
        <span className={`rounded-full px-3 py-1 text-xs font-black ${
          marketStatus.isOpen
            ? "bg-emerald-50 text-emerald-600"
            : "bg-slate-100 text-slate-400"
        }`}>
          {marketStatus.label}
        </span>
      </div>

      {error ? (
        <div className="rounded-xl border border-rose-100 bg-rose-50 px-4 py-3 text-sm font-bold text-rose-600">
          {error}
        </div>
      ) : null}

      <div className="grid gap-3 md:grid-cols-3">
        {indices.map((index) => (
          <div
            key={index.name}
            className="rounded-xl border border-slate-100 bg-slate-50/70 p-3"
          >
            <div className="flex items-start justify-between gap-2">
              <div>
                <p className="text-sm font-bold text-slate-500">{index.name}</p>
                <p className="mt-1 text-2xl font-black text-slate-950">{index.value}</p>
                <p className={`mt-0.5 text-sm font-black ${index.isUp ? "text-rose-500" : "text-blue-500"}`}>
                  {index.change} ({index.percent})
                </p>
              </div>
              <MarketSparkline color={index.color} isUp={index.isUp} />
            </div>
          </div>
        ))}
        {!loading && !error && indices.length === 0 ? (
          <div className="rounded-xl border border-slate-100 bg-slate-50 px-4 py-8 text-center text-sm font-bold text-slate-400 md:col-span-3">
            수신된 시장 데이터가 없습니다.
          </div>
        ) : null}
      </div>
    </article>
  );
}
