"use client";

import Link from "next/link";
import {
  FiBookOpen,
  FiClock,
  FiExternalLink,
  FiFileText,
} from "react-icons/fi";
import {
  mockTodayNews,
  mockDisclosures,
  mockRecentStocks,
} from "@/lib/dashboard/mock";
import LiveChartSection from "@/components/dashboard/LiveChartSection";
import MarketOverviewSection from "@/components/dashboard/MarketOverviewSection";
import { useAccountAssets } from "@/lib/account/assetsStore";

function formatWon(value: number) {
  return `${value.toLocaleString("ko-KR")}원`;
}

function formatRate(value: number) {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

export default function DashboardPage() {
  const { assets, error: assetsError } = useAccountAssets();

  const totalAssetsText = assets ? formatWon(assets.totalKRW) : assetsError ? "조회 실패" : "불러오는 중";
  const accountText = assets ? `${assets.broker} ${assets.accountNo}` : "계좌 조회 중";
  const pnlText = assets ? `${formatWon(assets.investedPnlKRW)} (${formatRate(assets.investedPnlRate)})` : "-";
  const cashText = assets ? formatWon(assets.cashKRW) : "-";
  return (
    <div className="space-y-4">
      <section className="grid items-stretch gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
        <MarketOverviewSection />

        <Link
          href="/mypage?tab=assets"
          className="block rounded-2xl bg-blue-50 p-4 text-slate-800 shadow-[0_20px_60px_rgba(82,103,255,0.10)] transition hover:-translate-y-0.5 hover:bg-blue-100/60 hover:shadow-[0_28px_90px_rgba(82,103,255,0.16)]"
        >
          <p className="text-sm font-bold text-blue-500">내 자산</p>
          <p className="mt-2 text-2xl font-black text-slate-900">{totalAssetsText}</p>
          <p className="mt-2 text-sm font-bold text-blue-400">{accountText}</p>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <div className="rounded-xl bg-white/70 p-2.5 ring-1 ring-blue-100">
              <p className="text-xs text-slate-400">평가손익</p>
              <p className="mt-1 font-black text-slate-800">{pnlText}</p>
            </div>
            <div className="rounded-xl bg-white/70 p-2.5 ring-1 ring-blue-100">
              <p className="text-xs text-slate-400">현금</p>
              <p className="mt-1 font-black text-slate-800">{cashText}</p>
            </div>
          </div>
        </Link>
      </section>

      <LiveChartSection />

      <section className="grid gap-4 xl:grid-cols-3">
        <article className="rounded-2xl bg-violet-50 p-4 shadow-[0_20px_60px_rgba(139,92,246,0.10)]">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <h2 className="mt-1 text-base font-black text-slate-900">
                오늘의 뉴스
              </h2>
            </div>
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-violet-100 text-violet-500">
              <FiBookOpen className="h-4 w-4" />
            </span>
          </div>

          <div className="space-y-2">
            {mockTodayNews.slice(0, 3).map((news) => (
              <div
                key={news.title}
                className="rounded-lg bg-white/80 px-2.5 py-2 ring-1 ring-violet-100 transition hover:bg-white"
              >
                <div className="mb-0.5 flex items-center justify-between gap-3">
                  <span className="text-xs font-black text-violet-400">뉴스</span>
                  <span className="text-xs font-bold text-slate-400">
                    {news.time}
                  </span>
                </div>
                <p className="text-xs font-black leading-5 text-slate-800">
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
