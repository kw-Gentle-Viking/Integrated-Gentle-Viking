"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  FiBookOpen,
  FiClock,
  FiExternalLink,
  FiFileText,
} from "react-icons/fi";
import LiveChartSection from "@/components/dashboard/LiveChartSection";
import MarketOverviewSection from "@/components/dashboard/MarketOverviewSection";
import { useAccountAssets } from "@/lib/account/assetsStore";
import { resolveStockName } from "@/lib/stock-list/StockListContext";
import { fetchCurrentPrice } from "@/lib/api/prices";
import { getRecentStockCodes } from "@/lib/dashboard/recentStocks";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const MAX_RECENT_SHOWN = 3;

type NewsItem = { title: string; company: string; source: string; time: string; url: string };
type DisclosureItem = { company: string; title: string; time: string; url: string };
type RecentStock = { code: string; name: string; price: string; change: string; isUp: boolean | null };

function formatWon(value: number) {
  return `${value.toLocaleString("ko-KR")}원`;
}

function formatRate(value: number) {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

function useMarketFeed<T>(path: string) {
  const [items, setItems] = useState<T[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}${path}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(String(res.status));
        const data = (await res.json()) as { items: T[] };
        if (!cancelled) setItems(data.items);
      })
      .catch(() => {
        if (!cancelled) setItems([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [path]);

  return { items, loading };
}

function useRecentStocks() {
  const [stocks, setStocks] = useState<RecentStock[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const codes = getRecentStockCodes().slice(0, MAX_RECENT_SHOWN);
    if (codes.length === 0) {
      setLoading(false);
      return;
    }
    setStocks(codes.map((code) => ({ code, name: resolveStockName(code, code), price: "-", change: "-", isUp: null })));
    (async () => {
      for (const code of codes) {
        if (cancelled) return;
        try {
          const live = await fetchCurrentPrice(code);
          if (cancelled) return;
          setStocks((prev) =>
            prev.map((s) =>
              s.code === code
                ? { ...s, name: resolveStockName(code, live.name || code), price: live.priceFormatted, change: live.changeFormatted, isUp: live.isUp }
                : s,
            ),
          );
        } catch (error) {
          console.error(`recent stock price fetch failed for ${code}`, error);
        }
        await new Promise((resolve) => setTimeout(resolve, 300));
      }
      if (!cancelled) setLoading(false);
    })();
    return () => { cancelled = true; };
  }, []);

  return { stocks, loading };
}

export default function DashboardPage() {
  const { assets, error: assetsError } = useAccountAssets();
  const { items: news, loading: newsLoading } = useMarketFeed<NewsItem>("/market/news?limit=3");
  const { items: disclosures, loading: disclosuresLoading } = useMarketFeed<DisclosureItem>("/market/disclosures?limit=3");
  const { stocks: recentStocks, loading: recentLoading } = useRecentStocks();

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
            {newsLoading ? (
              <p className="px-2.5 py-2 text-xs font-bold text-slate-400">불러오는 중...</p>
            ) : news.length === 0 ? (
              <p className="px-2.5 py-2 text-xs font-bold text-slate-400">표시할 뉴스가 없습니다.</p>
            ) : (
              news.map((item) => (
                <a
                  key={item.url || item.title}
                  href={item.url || undefined}
                  target="_blank"
                  rel="noreferrer"
                  className="block rounded-lg bg-white/80 px-2.5 py-2 ring-1 ring-violet-100 transition hover:bg-white"
                >
                  <div className="mb-0.5 flex items-center justify-between gap-3">
                    <span className="text-xs font-black text-violet-400">{item.company}</span>
                    <span className="text-xs font-bold text-slate-400">
                      {item.time}
                    </span>
                  </div>
                  <p className="text-xs font-black leading-5 text-slate-800">
                    {item.title}
                  </p>
                  <p className="mt-0.5 text-xs font-bold text-slate-400">
                    {item.source}
                  </p>
                </a>
              ))
            )}
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
            {disclosuresLoading ? (
              <p className="px-2.5 py-2 text-xs font-bold text-slate-400">불러오는 중...</p>
            ) : disclosures.length === 0 ? (
              <p className="px-2.5 py-2 text-xs font-bold text-slate-400">표시할 공시가 없습니다.</p>
            ) : (
              disclosures.map((item) => (
                <a
                  key={`${item.company}-${item.title}-${item.time}`}
                  href={item.url || undefined}
                  target="_blank"
                  rel="noreferrer"
                  className="group block rounded-lg bg-white/85 px-2.5 py-2 transition hover:bg-white"
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
                        {item.time}
                      </p>
                    </div>
                    <FiExternalLink className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-300 transition group-hover:text-amber-700" />
                  </div>
                </a>
              ))
            )}
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
            {recentLoading && recentStocks.length === 0 ? (
              <p className="px-2.5 py-2 text-xs font-bold text-slate-400">불러오는 중...</p>
            ) : recentStocks.length === 0 ? (
              <p className="px-2.5 py-2 text-xs font-bold text-slate-400">최근에 조회한 종목이 없습니다.</p>
            ) : (
              recentStocks.map((stock) => (
                <Link
                  key={stock.code}
                  href={`/market/${stock.code}`}
                  className="flex items-center gap-3 rounded-lg border border-slate-100 p-2.5 transition hover:bg-slate-50"
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
                      className={`text-xs font-black ${stock.isUp === null ? "text-slate-400" : stock.isUp ? "text-rose-500" : "text-blue-500"}`}
                    >
                      {stock.change}
                    </p>
                  </div>
                </Link>
              ))
            )}
          </div>
        </article>
      </section>
    </div>
  );
}
