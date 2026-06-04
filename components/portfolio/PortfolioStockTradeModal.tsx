"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { FiRefreshCw, FiX } from "react-icons/fi";
import CandleChart from "@/components/chart/CandleChart";
import TradeModal from "@/components/market/TradeModal";
import {
  fetchChartCandles,
  fetchCurrentPrice,
  fetchIntradayCandles,
  type StockCurrentPrice,
} from "@/lib/api/prices";
import type { OrderSide } from "@/lib/api/trade";
import type { CandleType } from "@/lib/chart/types";
import type { StockListItem } from "@/lib/stock-list/StockListContext";

type PortfolioStockTradeModalProps = {
  stock: StockListItem;
  onClose: () => void;
};

const chartRanges = [
  { key: "1D", label: "1일" },
  { key: "1W", label: "1주" },
  { key: "3M", label: "3달" },
  { key: "1Y", label: "1년" },
  { key: "5Y", label: "5년" },
  { key: "ALL", label: "전체" },
] as const;

function inferIsUp(change: string) {
  return !change.trim().startsWith("-");
}

export default function PortfolioStockTradeModal({
  stock,
  onClose,
}: PortfolioStockTradeModalProps) {
  const [currentPrice, setCurrentPrice] = useState<StockCurrentPrice | null>(null);
  const [candles, setCandles] = useState<CandleType[]>([]);
  const [chartRange, setChartRange] = useState<(typeof chartRanges)[number]>(chartRanges[2]);
  const [chartLoading, setChartLoading] = useState(true);
  const [chartError, setChartError] = useState<string | null>(null);
  const [retryKey, setRetryKey] = useState(0);
  const [tradeSide, setTradeSide] = useState<OrderSide | null>(null);

  useEffect(() => {
    let cancelled = false;

    fetchCurrentPrice(stock.code)
      .then((data) => {
        if (!cancelled) setCurrentPrice(data);
      })
      .catch(() => {
        if (!cancelled) setCurrentPrice(null);
      });

    return () => {
      cancelled = true;
    };
  }, [stock.code]);

  useEffect(() => {
    let cancelled = false;

    void Promise.resolve().then(() => {
      if (!cancelled) {
        setChartLoading(true);
        setChartError(null);
      }
    });

    const request =
      chartRange.key === "1D"
        ? fetchIntradayCandles(stock.code)
        : fetchChartCandles(stock.code, chartRange.key);

    request
      .then((data) => {
        if (!cancelled) setCandles(data);
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setCandles([]);
          setChartError(error instanceof Error ? error.message : "차트 데이터 조회 실패");
        }
      })
      .finally(() => {
        if (!cancelled) setChartLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [stock.code, chartRange.key, retryKey]);

  if (typeof document === "undefined") return null;

  const name = currentPrice?.name ?? stock.name;
  const price = currentPrice?.priceFormatted ?? stock.price;
  const change = currentPrice?.changeFormatted ?? stock.change;
  const isUp = currentPrice?.isUp ?? inferIsUp(change);

  const tradeStock = {
    code: stock.code,
    name,
    price,
    isUp,
    change,
  };

  return (
    <>
      {createPortal(
        <div className="fixed inset-0 z-40 flex min-h-dvh items-start justify-center overflow-y-auto bg-slate-950/35 p-4 py-6 backdrop-blur-sm sm:items-center sm:p-6">
          <button
            type="button"
            aria-label="종목 거래 창 닫기"
            className="absolute inset-0 cursor-default"
            onClick={onClose}
          />
          <aside className="relative w-full max-w-[900px] overflow-hidden rounded-2xl bg-white shadow-[0_28px_90px_rgba(15,23,42,0.24)]">
            <div className="max-h-[calc(100vh-32px)] overflow-y-auto sm:max-h-[calc(100vh-48px)]">
              <div className="p-5 sm:p-6 lg:p-7">
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <p className="text-sm font-bold text-[#5267ff]">Portfolio Trading</p>
                    <div className="mt-2 flex flex-wrap items-center gap-2">
                      <h2 className="break-keep text-3xl font-black text-slate-950">{name}</h2>
                      <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-black text-slate-500">
                        {stock.code}
                      </span>
                    </div>
                    <div className="mt-3 flex flex-wrap items-end gap-3">
                      <p className="text-3xl font-black text-slate-950">{price}</p>
                      <p className={`pb-1 text-base font-black ${isUp ? "text-rose-500" : "text-blue-500"}`}>
                        {change}
                      </p>
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

                <div className="mt-6 overflow-hidden rounded-2xl border border-slate-100">
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 bg-slate-50 px-4 py-3">
                    <p className="text-sm font-black text-slate-700">가격 차트</p>
                    <div className="flex flex-wrap gap-1">
                      {chartRanges.map((range) => (
                        <button
                          key={range.key}
                          type="button"
                          onClick={() => setChartRange(range)}
                          className={`h-8 rounded-lg px-3 text-xs font-black transition ${
                            chartRange.key === range.key
                              ? "bg-white text-slate-950 shadow-sm"
                              : "text-slate-400 hover:bg-white hover:text-slate-700"
                          }`}
                        >
                          {range.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {chartLoading ? (
                    <div className="flex h-[340px] items-center justify-center bg-white">
                      <div className="h-7 w-7 animate-spin rounded-full border-2 border-slate-200 border-t-[#5267ff]" />
                    </div>
                  ) : chartError ? (
                    <div className="flex h-[340px] flex-col items-center justify-center gap-3 bg-white px-4 text-center">
                      <p className="text-sm font-bold text-slate-400">{chartError}</p>
                      <button
                        type="button"
                        onClick={() => setRetryKey((key) => key + 1)}
                        className="inline-flex h-10 items-center gap-2 rounded-xl bg-slate-100 px-4 text-xs font-black text-slate-600 transition hover:bg-slate-200"
                      >
                        <FiRefreshCw className="h-4 w-4" />
                        다시 시도
                      </button>
                    </div>
                  ) : candles.length > 0 ? (
                    <div className="bg-white">
                      <CandleChart candles={candles} height={340} />
                    </div>
                  ) : (
                    <div className="flex h-[340px] items-center justify-center bg-white text-sm font-bold text-slate-400">
                      차트 데이터가 없습니다.
                    </div>
                  )}
                </div>

                <div className="mt-5 grid grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={() => setTradeSide("buy")}
                    className="h-14 rounded-2xl bg-rose-500 text-base font-black text-white shadow-[0_8px_24px_rgba(239,68,68,0.28)] transition hover:bg-rose-600"
                  >
                    매수
                  </button>
                  <button
                    type="button"
                    onClick={() => setTradeSide("sell")}
                    className="h-14 rounded-2xl bg-blue-500 text-base font-black text-white shadow-[0_8px_24px_rgba(59,130,246,0.28)] transition hover:bg-blue-600"
                  >
                    매도
                  </button>
                </div>
              </div>
            </div>
          </aside>
        </div>,
        document.body,
      )}

      {tradeSide ? (
        <TradeModal
          stock={tradeStock}
          initialSide={tradeSide}
          onClose={() => setTradeSide(null)}
        />
      ) : null}
    </>
  );
}
