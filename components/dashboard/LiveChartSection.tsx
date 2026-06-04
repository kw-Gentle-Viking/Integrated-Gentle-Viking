"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { FiTrendingUp } from "react-icons/fi";
import type { CandleType } from "@/lib/chart/types";
import {
  fetchChartCandles,
  fetchCurrentPrice,
  fetchIntradayCandles,
  fetchVolumeRank,
} from "@/lib/api/prices";
import type { StockCurrentPrice, RankedStock } from "@/lib/api/prices";
import CandleChart from "@/components/chart/CandleChart";

const chartRanges = [
  { key: "1D",  label: "1일"  },
  { key: "1W",  label: "1주"  },
  { key: "3M",  label: "3달"  },
  { key: "1Y",  label: "1년"  },
  { key: "5Y",  label: "5년"  },
  { key: "ALL", label: "전체" },
];

const FALLBACK_CODE = "005930"; // 삼성전자 (순위 로드 전 기본값)

export default function LiveChartSection() {
  const router = useRouter();

  const [ranking, setRanking] = useState<RankedStock[]>([]);
  const [rankingLoading, setRankingLoading] = useState(true);
  const [selectedCode, setSelectedCode] = useState(FALLBACK_CODE);

  const [chartRange, setChartRange] = useState(chartRanges[0]);
  const [candles, setCandles] = useState<CandleType[]>([]);
  const [currentPrice, setCurrentPrice] = useState<StockCurrentPrice | null>(null);
  const [chartLoading, setChartLoading] = useState(true);
  const [chartError, setChartError] = useState<string | null>(null);
  const [chartRetryKey, setChartRetryKey] = useState(0);

  // 거래량 순위 조회
  useEffect(() => {
    fetchVolumeRank()
      .then((data) => {
        setRanking(data);
        if (data.length > 0) setSelectedCode(data[0].code);
      })
      .catch(() => {/* 실패 시 fallback code 유지 */})
      .finally(() => setRankingLoading(false));
  }, []);

  // 현재가 조회
  useEffect(() => {
    let cancelled = false;
    fetchCurrentPrice(selectedCode)
      .then((data) => { if (!cancelled) setCurrentPrice(data); })
      .catch(() => { if (!cancelled) setCurrentPrice(null); });
    return () => { cancelled = true; };
  }, [selectedCode]);

  const loadChart = useCallback(async () => {
    setChartLoading(true);
    setChartError(null);

    try {
      const data = chartRange.key === "1D"
        ? await fetchIntradayCandles(selectedCode)
        : await fetchChartCandles(selectedCode, chartRange.key);
      setCandles(data);
    } catch (err: unknown) {
      setChartError(err instanceof Error ? err.message : "데이터 조회 실패");
    } finally {
      setChartLoading(false);
    }
  }, [selectedCode, chartRange.key]);

  // 차트 데이터 조회
  useEffect(() => {
    void loadChart();
  }, [loadChart, chartRetryKey]);

  const selectedRanked = ranking.find((s) => s.code === selectedCode);
  const displayName   = currentPrice?.name   ?? selectedRanked?.name   ?? selectedCode;
  const displayPrice  = currentPrice?.priceFormatted  ?? selectedRanked?.price  ?? "-";
  const displayChange = currentPrice?.changeFormatted ?? selectedRanked?.change ?? "-";
  const displayIsUp   = currentPrice?.isUp   ?? selectedRanked?.isUp   ?? false;

  return (
    <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
      {/* 차트 카드 */}
      <article
        onClick={() => router.push("/market")}
        className="cursor-pointer rounded-2xl bg-white p-4 shadow-[0_20px_60px_rgba(15,23,42,0.06)] transition hover:shadow-[0_24px_70px_rgba(82,103,255,0.12)]"
      >
        <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="text-sm font-bold text-[#5267ff]">실시간 차트</p>
            <h2 className="mt-1 text-xl font-black text-slate-950">{displayName}</h2>
          </div>
          <div className="text-right">
            <p className="text-lg font-black text-slate-950">{displayPrice}</p>
            <p className={`text-sm font-black ${displayIsUp ? "text-rose-500" : "text-blue-500"}`}>
              {displayChange}
            </p>
          </div>
        </div>

        {chartLoading && candles.length === 0 ? (
          <div className="flex h-[316px] items-center justify-center rounded-2xl border border-slate-100 bg-slate-50">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-slate-200 border-t-[#5267ff]" />
          </div>
        ) : chartError && candles.length === 0 ? (
          <div className="flex h-[316px] flex-col items-center justify-center gap-2 rounded-2xl border border-slate-100 bg-slate-50">
            <p className="text-sm font-bold text-slate-400">{chartError}</p>
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); setChartRetryKey((key) => key + 1); }}
              className="rounded-lg bg-slate-100 px-3 py-1.5 text-xs font-bold text-slate-600 transition hover:bg-slate-200"
            >
              다시 시도
            </button>
          </div>
        ) : candles.length > 0 ? (
          <div className="relative overflow-hidden rounded-2xl border border-slate-100">
            <CandleChart candles={candles} height={316} />
            {(chartLoading || chartError) ? (
              <div className="pointer-events-none absolute inset-x-3 top-3 flex justify-end">
                <span className="rounded-full bg-white/90 px-3 py-1 text-xs font-black text-slate-500 shadow-sm">
                  {chartLoading ? "업데이트 중" : "마지막 정상 차트"}
                </span>
              </div>
            ) : null}
          </div>
        ) : (
          <div className="flex h-[316px] items-center justify-center rounded-2xl border border-slate-100 bg-slate-50 text-sm font-bold text-slate-400">
            데이터가 없습니다.
          </div>
        )}

        <div className="mt-3 grid grid-cols-6 rounded-2xl bg-slate-100 p-1">
          {chartRanges.map((range) => (
            <button
              key={range.key}
              type="button"
              onClick={(e) => { e.stopPropagation(); setChartRange(range); }}
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

      {/* 랭킹 카드 */}
      <article
        onClick={() => router.push("/market")}
        className="cursor-pointer rounded-2xl bg-white p-4 shadow-[0_20px_60px_rgba(15,23,42,0.06)] transition hover:shadow-[0_24px_70px_rgba(82,103,255,0.12)]"
      >
        <div className="mb-3 flex items-center justify-between">
          <h2 className="mt-1 text-lg font-black text-slate-950">실시간 종목 랭킹</h2>
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#eef2ff] text-[#5267ff]">
            <FiTrendingUp className="h-4 w-4" />
          </span>
        </div>

        {rankingLoading ? (
          <div className="flex h-40 items-center justify-center">
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-slate-200 border-t-[#5267ff]" />
          </div>
        ) : (
          <div className="space-y-2">
            {(ranking.length > 0 ? ranking : []).slice(0, 6).map((stock) => {
              const isSelected = selectedCode === stock.code;
              return (
                <button
                  key={stock.code}
                  type="button"
                  onClick={(e) => { e.stopPropagation(); setSelectedCode(stock.code); }}
                  className={`flex w-full items-center gap-2.5 rounded-lg border p-2.5 text-left transition ${
                    isSelected
                      ? "border-[#5267ff]/40 bg-[#eef2ff]"
                      : "border-slate-100 hover:bg-slate-50"
                  }`}
                >
                  <span
                    className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-xs font-black ${
                      stock.rank <= 3 ? "bg-[#5267ff] text-white" : "bg-slate-100 text-slate-500"
                    }`}
                  >
                    {stock.rank}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-black text-slate-950">{stock.name}</p>
                    <p className="text-xs font-bold text-slate-400">{stock.volume}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs font-black text-slate-950">{stock.price}</p>
                    <p className={`text-xs font-black ${stock.isUp ? "text-rose-500" : "text-blue-500"}`}>
                      {stock.change}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </article>
    </section>
  );
}
