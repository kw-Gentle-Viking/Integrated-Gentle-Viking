"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  FiBookOpen,
  FiClock,
  FiExternalLink,
  FiFileText,
  FiTrendingUp,
} from "react-icons/fi";
import { loadCandlesFromCsv } from "@/lib/chart/parseStockCsv";
import type { CandleType } from "@/lib/chart/types";
import {
  mockMarketIndices,
  mockTodayNews,
  mockDisclosures,
  mockPopularStocks,
  mockRecentStocks,
} from "@/lib/dashboard/mock";
import CandleChart from "@/components/chart/CandleChart";

const chartRanges = [
  { key: "1D", label: "1일", points: 80 },
  { key: "1W", label: "1주", points: 160 },
  { key: "1M", label: "1개월", points: 320 },
  { key: "1Y", label: "1년", points: 720 },
];

function MarketSparkline({ color, isUp }: { color: string; isUp: boolean }) {
  const path = isUp
    ? "M0 34 C20 38 28 22 46 26 C62 30 70 14 88 18 C108 22 114 10 132 12 C146 14 150 8 160 10"
    : "M0 12 C18 10 28 26 44 22 C60 18 72 34 90 30 C108 26 118 42 134 38 C148 34 152 44 160 40";

  return (
    <svg viewBox="0 0 160 54" className="h-9 w-20" aria-hidden="true">
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth="4"
        strokeLinecap="round"
      />
    </svg>
  );
}

export default function DashboardPage() {
  const router = useRouter();
  const [candles, setCandles] = useState<CandleType[]>([]);
  const [selectedStock, setSelectedStock] = useState(mockPopularStocks[0]);
  const [chartRange, setChartRange] = useState(chartRanges[0]);

  useEffect(() => {
    loadCandlesFromCsv("/005930.csv")
      .then((data) => setCandles(data))
      .catch(() => setCandles([]));
  }, []);

  const selectedMeta = useMemo(
    () => ({
      name: selectedStock.name,
      price: selectedStock.price,
      change: selectedStock.change,
      isUp: selectedStock.isUp,
    }),
    [selectedStock],
  );
  const visibleCandles = useMemo(
    () => candles.slice(-chartRange.points),
    [candles, chartRange],
  );

  return (
    <div className="space-y-4">
      <section className="grid items-stretch gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
        <article className="rounded-2xl bg-white p-4 shadow-[0_20px_60px_rgba(15,23,42,0.06)]">
          <div className="mb-4 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 shadow-[0_0_0_4px_rgba(52,211,153,0.14)]" />
              <h2 className="text-lg font-black text-slate-950">국내 정규장</h2>
            </div>
            <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-black text-emerald-600">
              장중
            </span>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            {mockMarketIndices.map((index) => (
              <div
                key={index.name}
                className="rounded-xl border border-slate-100 bg-slate-50/70 p-3"
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="text-sm font-bold text-slate-500">
                      {index.name}
                    </p>
                    <p className="mt-1 text-2xl font-black text-slate-950">
                      {index.value}
                    </p>
                    <p
                      className={`mt-0.5 text-sm font-black ${index.isUp ? "text-rose-500" : "text-blue-500"}`}
                    >
                      {index.change} ({index.percent})
                    </p>
                  </div>
                  <MarketSparkline color={index.color} isUp={index.isUp} />
                </div>
              </div>
            ))}
          </div>
        </article>

        <Link
          href="/mypage?tab=assets"
          className="block rounded-2xl bg-slate-950 p-4 text-white shadow-[0_20px_60px_rgba(15,23,42,0.18)] transition hover:-translate-y-0.5 hover:bg-slate-900 hover:shadow-[0_28px_90px_rgba(15,23,42,0.24)]"
        >
          <p className="text-sm font-bold text-slate-300">내 자산</p>
          <p className="mt-2 text-2xl font-black text-white">139만 7,380원</p>
          <p className="mt-2 text-sm font-bold text-emerald-300">+2.37% 오늘</p>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <div className="rounded-xl bg-white/10 p-2.5 ring-1 ring-white/10">
              <p className="text-xs text-slate-400">수익</p>
              <p className="mt-1 font-black text-white">+29,770원</p>
            </div>
            <div className="rounded-xl bg-white/10 p-2.5 ring-1 ring-white/10">
              <p className="text-xs text-slate-400">현금</p>
              <p className="mt-1 font-black text-white">86,539원</p>
            </div>
          </div>
        </Link>
      </section>

      <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
        <article
          onClick={() => router.push("/market")}
          className="cursor-pointer rounded-2xl bg-white p-4 shadow-[0_20px_60px_rgba(15,23,42,0.06)] transition hover:shadow-[0_24px_70px_rgba(82,103,255,0.12)]"
        >
          <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
            <div>
              <p className="text-sm font-bold text-[#5267ff]">실시간 차트</p>
              <h2 className="mt-1 text-xl font-black text-slate-950">
                {selectedMeta.name}
              </h2>
            </div>
            <div className="text-right">
              <p className="text-lg font-black text-slate-950">
                {selectedMeta.price}
              </p>
              <p
                className={`text-sm font-black ${selectedMeta.isUp ? "text-rose-500" : "text-blue-500"}`}
              >
                {selectedMeta.change}
              </p>
            </div>
          </div>
          {visibleCandles.length > 0 ? (
            <div className="overflow-hidden rounded-2xl border border-slate-100">
              <CandleChart candles={visibleCandles} height={316} />
            </div>
          ) : (
            <div className="flex h-[316px] items-center justify-center rounded-2xl border border-slate-100 bg-slate-50 text-sm font-bold text-slate-400">
              차트를 불러오는 중입니다.
            </div>
          )}
          <div className="mt-3 grid grid-cols-4 rounded-2xl bg-slate-100 p-1">
            {chartRanges.map((range) => (
              <button
                key={range.key}
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setChartRange(range);
                }}
                className={`h-10 rounded-xl text-sm font-black transition ${
                  chartRange.key === range.key
                    ? "bg-white text-slate-950 shadow-sm"
                    : "text-slate-400 hover:text-slate-700"
                }`}
              >
                {range.label}
              </button>
            ))}
          </div>
        </article>

        <article
          onClick={() => router.push("/market")}
          className="cursor-pointer rounded-2xl bg-white p-4 shadow-[0_20px_60px_rgba(15,23,42,0.06)] transition hover:shadow-[0_24px_70px_rgba(82,103,255,0.12)]"
        >
          <div className="mb-3 flex items-center justify-between">
            <div>
              <h2 className="mt-1 text-lg font-black text-slate-950">
                실시간 종목 랭킹
              </h2>
            </div>
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#eef2ff] text-[#5267ff]">
              <FiTrendingUp className="h-4 w-4" />
            </span>
          </div>

          <div className="space-y-2">
            {mockPopularStocks.slice(0, 6).map((stock) => {
              const isSelected = selectedStock?.name === stock.name;

              return (
                <button
                  key={stock.rank}
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedStock(stock);
                  }}
                  className={`flex w-full items-center gap-2.5 rounded-lg border p-2.5 text-left transition ${
                    isSelected
                      ? "border-[#5267ff]/40 bg-[#eef2ff]"
                      : "border-slate-100 hover:bg-slate-50"
                  }`}
                >
                  <span
                    className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-xs font-black ${stock.rank <= 3 ? "bg-[#5267ff] text-white" : "bg-slate-100 text-slate-500"}`}
                  >
                    {stock.rank}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-black text-slate-950">
                      {stock.name}
                    </p>
                    <p className="text-xs font-bold text-slate-400">
                      {stock.volume}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs font-black text-slate-950">
                      {stock.price}
                    </p>
                    <p
                      className={`text-xs font-black ${stock.isUp ? "text-rose-500" : "text-blue-500"}`}
                    >
                      {stock.change}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>
        </article>
      </section>

      <section className="grid gap-4 xl:grid-cols-3">
        <article className="rounded-2xl bg-[#0f1f3d] p-4 text-white shadow-[0_20px_60px_rgba(15,31,61,0.16)]">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <h2 className="mt-1 text-base font-black text-white">
                오늘의 뉴스
              </h2>
            </div>
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-white/10 text-sky-200 ring-1 ring-white/10">
              <FiBookOpen className="h-4 w-4" />
            </span>
          </div>

          <div className="space-y-2">
            {mockTodayNews.slice(0, 3).map((news) => (
              <div
                key={news.title}
                className="rounded-lg bg-white/10 px-2.5 py-2 ring-1 ring-white/10 transition hover:bg-white/15"
              >
                <div className="mb-0.5 flex items-center justify-between gap-3">
                  <span className="text-xs font-black text-sky-200">뉴스</span>
                  <span className="text-xs font-bold text-slate-400">
                    {news.time}
                  </span>
                </div>
                <p className="text-xs font-black leading-5 text-white">
                  {news.title}
                </p>
                <p className="mt-0.5 text-xs font-bold text-slate-400">
                  {news.source}
                </p>
              </div>
            ))}
          </div>
        </article>

        <article className="rounded-2xl bg-amber-50 p-4 shadow-[0_20px_60px_rgba(245,158,11,0.12)]">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <h2 className="mt-1 text-base font-black text-slate-950">
                오늘의 공시
              </h2>
            </div>
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-white text-amber-700 shadow-sm">
              <FiFileText className="h-4 w-4" />
            </span>
          </div>

          <div className="space-y-2">
            {mockDisclosures.slice(0, 3).map((item) => (
              <div
                key={`${item.company}-${item.title}`}
                className="group rounded-lg bg-white/85 px-2.5 py-2 transition hover:bg-white"
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-xs font-black text-amber-700">
                      {item.company}
                    </p>
                    <p className="mt-0.5 text-xs font-black leading-5 text-slate-950">
                      {item.title}
                    </p>
                    <p className="mt-0.5 text-xs font-bold text-slate-400">
                      오늘 {item.time}
                    </p>
                  </div>
                  <FiExternalLink className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-300 transition group-hover:text-amber-700" />
                </div>
              </div>
            ))}
          </div>
        </article>

        <article className="rounded-2xl bg-white p-4 shadow-[0_20px_60px_rgba(15,23,42,0.06)]">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <h2 className="mt-1 text-lg font-black text-slate-950">
                최근에 본 종목
              </h2>
            </div>
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-100 text-slate-500">
              <FiClock className="h-4 w-4" />
            </span>
          </div>

          <div className="space-y-2">
            {mockRecentStocks.map((stock) => (
              <div
                key={stock.code}
                className="flex items-center gap-3 rounded-lg border border-slate-100 p-2.5"
              >
                <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-slate-100 text-xs font-black text-slate-500">
                  {stock.name.slice(0, 1)}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-xs font-black text-slate-950">
                    {stock.name}
                  </p>
                  <p className="text-xs font-bold text-slate-400">
                    {stock.code}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-xs font-black text-slate-950">
                    {stock.price}
                  </p>
                  <p
                    className={`text-xs font-black ${stock.change.startsWith("-") ? "text-blue-500" : "text-rose-500"}`}
                  >
                    {stock.change}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </article>
      </section>
    </div>
  );
}
