"use client";

import { useState } from "react";
import { createPortal } from "react-dom";
import Link from "next/link";
import {
  FiActivity,
  FiExternalLink,
  FiFileText,
  FiTrendingUp,
  FiX,
} from "react-icons/fi";
import type { ReasonKey, ReportStock } from "@/lib/ai-report/types";
import {
  getPredictHref,
  getSignalBadgeClass,
  getSourceHref,
  getSourceLabel,
} from "./reportHelpers";

const reasonCards: { key: ReasonKey; label: string; icon: typeof FiFileText }[] =
  [
    { key: "news", label: "뉴스", icon: FiFileText },
    { key: "disclosure", label: "공시", icon: FiFileText },
    { key: "flow", label: "재료/수급", icon: FiTrendingUp },
  ];

const chartRanges = [
  { key: "1D", label: "1일", count: 31, weight: 0.8 },
  { key: "1W", label: "1주", count: 44, weight: 1 },
  { key: "3M", label: "3달", count: 52, weight: 1.12 },
  { key: "1Y", label: "1년", count: 60, weight: 1.24 },
  { key: "5Y", label: "5년", count: 68, weight: 1.38 },
  { key: "ALL", label: "전체", count: 76, weight: 1.52 },
] as const;

type ChartRangeKey = (typeof chartRanges)[number]["key"];

function makeCandleSeries(isDown: boolean, rangeKey: ChartRangeKey) {
  const range =
    chartRanges.find((item) => item.key === rangeKey) ?? chartRanges[0];
  const xStep = 500 / Math.max(range.count - 1, 1);
  const base = isDown ? 172 : 198;
  const direction = isDown ? 1 : -1;
  const rangeWeight = range.weight;

  return Array.from({ length: range.count }, (_, index) => {
    const x = Math.round(index * xStep);
    const trend = direction * index * 2.2 * rangeWeight;
    const wave = Math.sin(index * 0.82) * 18 + Math.cos(index * 0.33) * 9;
    const impulse = index > range.count * 0.78 ? direction * -34 : 0;
    const open = Math.max(30, Math.min(226, base + trend + wave + impulse));
    const closeShift =
      Math.sin(index * 1.35 + (isDown ? 0.4 : 1.1)) * 18 + direction * 2;
    const close = Math.max(28, Math.min(230, open + closeShift));
    const high = Math.max(16, Math.min(open, close) - (10 + (index % 5) * 4));
    const low = Math.min(244, Math.max(open, close) + (12 + (index % 4) * 5));

    return { x, open, high, low, close };
  });
}

function DetailChart({ stock }: { stock: ReportStock }) {
  const [chartRange, setChartRange] = useState<ChartRangeKey>("1D");
  const isDown = stock.change.startsWith("-");
  const candles = makeCandleSeries(isDown, chartRange);

  return (
    <div className="overflow-hidden rounded-2xl bg-white ring-1 ring-slate-100">
      <div className="flex flex-wrap items-center justify-end gap-1 border-b border-slate-100 bg-slate-50/70 px-3 py-2">
        {chartRanges.map((range) => (
          <button
            key={range.key}
            type="button"
            onClick={() => setChartRange(range.key)}
            className={`h-8 rounded-lg px-3 text-xs font-black transition ${
              chartRange === range.key
                ? "bg-slate-200 text-slate-700 shadow-sm"
                : "text-slate-400 hover:bg-white hover:text-slate-700"
            }`}
          >
            {range.label}
          </button>
        ))}
      </div>
      <div className="relative h-64 overflow-hidden bg-white">
        <svg
          viewBox="0 0 500 267"
          preserveAspectRatio="none"
          className="absolute inset-0 h-full w-full"
          aria-hidden="true"
        >
          <rect width="500" height="267" fill="#ffffff" />
          {[0, 92, 227, 362, 497].map((x) => (
            <line
              key={x}
              x1={x}
              x2={x}
              y1="0"
              y2="267"
              stroke="#e9eef5"
              strokeWidth="1"
            />
          ))}
          {[34, 84, 132, 180, 228].map((y) => (
            <line
              key={y}
              x1="0"
              x2="500"
              y1={y}
              y2={y}
              stroke="#e9eef5"
              strokeWidth="1"
            />
          ))}
          <line
            x1="0"
            x2="500"
            y1="120"
            y2="120"
            stroke="#f43f5e"
            strokeDasharray="2 3"
            strokeWidth="1.2"
          />
          {candles.map((candle) => {
            const rising = candle.close < candle.open;
            const color = rising ? "#e8294f" : "#2459d6";
            const bodyY = Math.min(candle.open, candle.close);
            const bodyHeight = Math.max(
              Math.abs(candle.close - candle.open),
              2,
            );
            const candleWidth =
              chartRange === "1D"
                ? 11
                : chartRange === "1W"
                  ? 8
                  : chartRange === "3M"
                    ? 7
                    : 5;

            return (
              <g key={candle.x}>
                <line
                  x1={candle.x}
                  x2={candle.x}
                  y1={candle.high}
                  y2={candle.low}
                  stroke={color}
                  strokeWidth="1.2"
                />
                <rect
                  x={candle.x - candleWidth / 2}
                  y={bodyY}
                  width={candleWidth}
                  height={bodyHeight}
                  fill={color}
                />
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

type ReportDetailModalProps = {
  stock: ReportStock;
  onClose: () => void;
};

export default function ReportDetailModal({
  stock,
  onClose,
}: ReportDetailModalProps) {
  if (typeof document === "undefined") return null;

  return createPortal(
    <div className="fixed inset-0 z-[100] flex min-h-dvh items-start justify-center overflow-y-auto bg-slate-950/35 p-4 py-6 backdrop-blur-sm sm:items-center sm:p-6">
      <button
        type="button"
        aria-label="상세 패널 닫기"
        className="absolute inset-0 cursor-default"
        onClick={onClose}
      />
      <aside className="relative w-full max-w-[920px] overflow-hidden rounded-2xl bg-white shadow-[0_28px_90px_rgba(15,23,42,0.28)] animate-[modalSwoopIn_0.36s_cubic-bezier(0.16,1,0.3,1)]">
        <div className="ai-report-detail-scroll max-h-[calc(100vh-32px)] overflow-y-auto sm:max-h-[calc(100vh-48px)]">
          <div className="p-5 sm:p-6 lg:p-8">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <p className="text-sm font-bold text-[#5267ff]">
                  Stock Detail
                </p>
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <h2 className="break-keep text-3xl font-black text-slate-950">
                    {stock.name}
                  </h2>
                  <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-black text-slate-500">
                    {stock.code}
                  </span>
                  <span
                    className={`rounded-full px-2.5 py-1 text-xs font-black ${getSignalBadgeClass(
                      stock.signal,
                    )}`}
                  >
                    {stock.signal}
                  </span>
                </div>
              </div>
              <button
                type="button"
                onClick={onClose}
                title="닫기"
                className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-slate-400 transition hover:bg-slate-100 hover:text-slate-950"
              >
                <FiX className="h-5 w-5" />
              </button>
            </div>

            <div className="mt-6">
              <DetailChart stock={stock} />
            </div>

            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              <div className="rounded-2xl bg-[#eef2ff] p-4 ring-1 ring-[#dfe5ff]">
                <p className="text-xs font-bold text-[#5267ff]">추천 점수</p>
                <p className="mt-2 text-3xl font-black text-slate-950">
                  {stock.score}
                </p>
              </div>
              <div className="rounded-2xl bg-slate-50 p-4">
                <p className="text-xs font-bold text-slate-400">현재가</p>
                <p className="mt-2 text-lg font-black text-slate-950">
                  {stock.price}
                </p>
              </div>
              <div className="rounded-2xl bg-slate-50 p-4">
                <p className="text-xs font-bold text-slate-400">등락률</p>
                <p
                  className={
                    "mt-2 text-lg font-black " +
                    (stock.change.startsWith("-")
                      ? "text-blue-500"
                      : "text-rose-500")
                  }
                >
                  {stock.change}
                </p>
              </div>
            </div>

            <section className="mt-5 rounded-2xl bg-[#f4f7ff] p-5">
              <p className="text-xs font-black text-[#5267ff]">핵심 요약</p>
              <p className="mt-3 text-lg font-black leading-8 text-slate-900">
                {stock.summary}
              </p>
            </section>

            <section className="mt-5 space-y-3">
              <h3 className="text-lg font-black text-slate-950">핵심 데이터</h3>
              {reasonCards.map((item) => {
                const Icon = item.icon;
                const reason = stock.reasons[item.key];

                return (
                  <article
                    key={item.key}
                    className="rounded-2xl border border-slate-100 p-4"
                  >
                    <p className="flex items-center gap-2 text-sm font-black text-slate-700">
                      <Icon className="h-4 w-4 text-[#5267ff]" />
                      {item.label}
                    </p>
                    <p className="mt-2 text-sm font-bold leading-6 text-slate-700">
                      {reason.summary}
                    </p>
                    <p className="mt-3 text-sm leading-6 text-slate-500">
                      {reason.details}
                    </p>
                    <div className="mt-3 flex flex-wrap items-center gap-1.5">
                      {reason.sources.map((source) => {
                        const href = getSourceHref(source);

                        return (
                          <div
                            key={`${getSourceLabel(source)}-${href ?? "text"}`}
                            className="flex items-center gap-1.5 rounded-full bg-slate-100 px-2 py-1"
                          >
                            <span className="text-[11px] font-bold text-slate-500">
                              {getSourceLabel(source)}
                            </span>
                            {href ? (
                              <a
                                href={href}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex h-5 w-5 items-center justify-center rounded-full text-slate-400 transition hover:bg-white hover:text-[#5267ff]"
                                aria-label={`${getSourceLabel(source)} 링크 열기`}
                              >
                                <FiExternalLink className="h-3.5 w-3.5" />
                              </a>
                            ) : null}
                          </div>
                        );
                      })}
                    </div>
                  </article>
                );
              })}
            </section>
          </div>

          <div className="sticky bottom-0 border-t border-slate-100 bg-white/95 px-5 py-4 backdrop-blur sm:px-6 lg:px-8">
            <Link
              href={getPredictHref(stock)}
              className="flex h-12 w-full items-center justify-center gap-2 rounded-xl border border-indigo-100 bg-gradient-to-r from-indigo-50 via-fuchsia-50 to-rose-50 text-sm font-black shadow-sm transition hover:border-indigo-200 hover:from-indigo-100 hover:via-fuchsia-100 hover:to-rose-100"
            >
              <FiActivity className="h-4 w-4 text-[#5267ff]" />
              <span className="bg-gradient-to-r from-[#5267ff] via-fuchsia-500 to-rose-500 bg-clip-text text-transparent">
                AI 매매 확률 보기
              </span>
            </Link>
          </div>
        </div>
      </aside>
    </div>,
    document.body,
  );
}
