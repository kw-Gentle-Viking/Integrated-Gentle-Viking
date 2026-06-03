"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { FiArrowLeft, FiHeart, FiShoppingBag } from "react-icons/fi";
import CandleChart from "@/components/chart/CandleChart";
import CartConfirmModal from "@/components/stock-list/CartConfirmModal";
import TradeModal from "@/components/market/TradeModal";
import { useStockList } from "@/lib/stock-list/StockListContext";
import { fetchChartCandles, fetchCurrentPrice, fetchIntradayCandles } from "@/lib/api/prices";
import type { StockCurrentPrice } from "@/lib/api/prices";
import type { CandleType } from "@/lib/chart/types";
import type { OrderSide } from "@/lib/api/trade";

const chartRanges = [
  { key: "1D", label: "1일" },
  { key: "1W", label: "1주" },
  { key: "3M", label: "3달" },
  { key: "1Y", label: "1년" },
  { key: "5Y", label: "5년" },
  { key: "ALL", label: "전체" },
];

export default function StockDetailPage() {
  const { code } = useParams<{ code: string }>();
  const router = useRouter();
  const { toggleFavorite, toggleCart, isFavorite, isInCart } = useStockList();

  const [currentPrice, setCurrentPrice] = useState<StockCurrentPrice | null>(null);
  const [candles, setCandles] = useState<CandleType[]>([]);
  const [chartRange, setChartRange] = useState(chartRanges[2]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pendingCart, setPendingCart] = useState(false);
  const [tradeModal, setTradeModal] = useState<OrderSide | null>(null);

  // 현재가
  useEffect(() => {
    if (!code) return;
    fetchCurrentPrice(code)
      .then(setCurrentPrice)
      .catch(() => setCurrentPrice(null));
  }, [code]);

  // 차트
  useEffect(() => {
    if (!code) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    const fetcher = chartRange.key === "1D"
      ? fetchIntradayCandles(code)
      : fetchChartCandles(code, chartRange.key);
    fetcher
      .then((data) => { if (!cancelled) { setCandles(data); setLoading(false); } })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "데이터 조회 실패");
          setCandles([]);
          setLoading(false);
        }
      });
    return () => { cancelled = true; };
  }, [code, chartRange.key]);

  const name = currentPrice?.name ?? code;
  const priceStr = currentPrice?.priceFormatted ?? "-";
  const changeStr = currentPrice?.changeFormatted ?? "-";
  const isUp = currentPrice?.isUp ?? false;

  const stockListItem = {
    code,
    name,
    price: priceStr,
    change: changeStr,
    logoText: name.slice(0, 1),
  };

  return (
    <div className="space-y-4">
      {/* 상단 네비 */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => router.back()}
          className="flex h-10 w-10 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-500 transition hover:text-slate-950"
        >
          <FiArrowLeft className="h-4 w-4" />
        </button>
        <div className="min-w-0 flex-1">
          <p className="text-xs font-bold text-[#5267ff]">{code}</p>
          <h1 className="truncate text-2xl font-black text-slate-950">{name}</h1>
        </div>
        <button
          type="button"
          onClick={() => toggleFavorite(code, stockListItem)}
          title="관심 종목"
          className={`flex h-10 w-10 items-center justify-center rounded-xl border transition ${
            isFavorite(code)
              ? "border-rose-100 bg-rose-50 text-rose-500"
              : "border-slate-200 text-slate-400 hover:border-rose-100 hover:bg-rose-50 hover:text-rose-500"
          }`}
        >
          <FiHeart className={`h-5 w-5 ${isFavorite(code) ? "fill-current" : ""}`} />
        </button>
        <button
          type="button"
          onClick={() => setPendingCart(true)}
          title="포트폴리오"
          className={`flex h-10 w-10 items-center justify-center rounded-xl border transition ${
            isInCart(code)
              ? "border-amber-200 bg-amber-50 text-amber-500"
              : "border-slate-200 text-slate-400 hover:border-amber-200 hover:bg-amber-50 hover:text-amber-500"
          }`}
        >
          <FiShoppingBag className="h-5 w-5" />
        </button>
      </div>

      {/* 가격 카드 */}
      <div className="rounded-2xl bg-white p-5 shadow-[0_20px_60px_rgba(15,23,42,0.06)]">
        <p className="text-sm font-bold text-slate-400">현재가</p>
        <div className="mt-1 flex items-end gap-3">
          <p className="text-4xl font-black text-slate-950">{priceStr}</p>
          <p className={`mb-1 text-lg font-black ${isUp ? "text-rose-500" : "text-blue-500"}`}>
            {changeStr}
          </p>
        </div>
      </div>

      {/* 차트 */}
      <div className="rounded-2xl bg-white p-4 shadow-[0_20px_60px_rgba(15,23,42,0.06)]">
        <div className="mb-3 flex items-center justify-between">
          <p className="text-sm font-bold text-[#5267ff]">가격 차트</p>
        </div>

        {loading ? (
          <div className="flex h-[400px] items-center justify-center rounded-2xl border border-slate-100 bg-slate-50">
            <div className="h-7 w-7 animate-spin rounded-full border-2 border-slate-200 border-t-[#5267ff]" />
          </div>
        ) : error ? (
          <div className="flex h-[400px] flex-col items-center justify-center gap-3 rounded-2xl border border-slate-100 bg-slate-50">
            <p className="text-sm font-bold text-slate-400">{error}</p>
            <button
              type="button"
              onClick={() => setChartRange({ ...chartRange })}
              className="rounded-lg bg-slate-100 px-4 py-2 text-xs font-bold text-slate-600 transition hover:bg-slate-200"
            >
              다시 시도
            </button>
          </div>
        ) : candles.length > 0 ? (
          <div className="overflow-hidden rounded-2xl border border-slate-100">
            <CandleChart candles={candles} height={400} />
          </div>
        ) : (
          <div className="flex h-[400px] items-center justify-center rounded-2xl border border-slate-100 bg-slate-50 text-sm font-bold text-slate-400">
            데이터가 없습니다.
          </div>
        )}

        {/* 범위 탭 */}
        <div className="mt-3 grid grid-cols-6 rounded-2xl bg-slate-100 p-1">
          {chartRanges.map((range) => (
            <button
              key={range.key}
              type="button"
              onClick={() => setChartRange(range)}
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
      </div>

      {/* 매수/매도 버튼 */}
      <div className="grid grid-cols-2 gap-3">
        <button
          type="button"
          onClick={() => setTradeModal("buy")}
          className="h-14 rounded-2xl bg-rose-500 text-base font-black text-white shadow-[0_8px_24px_rgba(239,68,68,0.3)] transition hover:bg-rose-600"
        >
          매수
        </button>
        <button
          type="button"
          onClick={() => setTradeModal("sell")}
          className="h-14 rounded-2xl bg-blue-500 text-base font-black text-white shadow-[0_8px_24px_rgba(59,130,246,0.3)] transition hover:bg-blue-600"
        >
          매도
        </button>
      </div>

      {/* 모달 */}
      {pendingCart && (
        <CartConfirmModal
          stockName={name}
          mode={isInCart(code) ? "remove" : "add"}
          onConfirm={() => { toggleCart(code, stockListItem); setPendingCart(false); }}
          onCancel={() => setPendingCart(false)}
        />
      )}
      {tradeModal && (
        <TradeModal
          stock={{ code, name, price: priceStr, isUp, change: changeStr }}
          initialSide={tradeModal}
          onClose={() => setTradeModal(null)}
        />
      )}
    </div>
  );
}
