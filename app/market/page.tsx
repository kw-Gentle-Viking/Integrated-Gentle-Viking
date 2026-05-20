"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { FiArrowLeft, FiChevronDown, FiHeart, FiShoppingBag, FiTrendingUp } from "react-icons/fi";
import { mockPopularStocks } from "@/lib/dashboard/mock";
import CandleChart from "@/components/chart/CandleChart";
import type { CandleType } from "@/lib/chart/types";
import { useStockList } from "@/lib/stock-list/StockListContext";
import CartConfirmModal from "@/components/stock-list/CartConfirmModal";

function seededRng(seed: number) {
  let s = (Math.abs(Math.floor(seed)) | 1) & 0x7fffffff;
  return () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

function generateMockCandles(name: string, count = 60): CandleType[] {
  const seed = name.split("").reduce((a, c) => a + c.charCodeAt(0), 0);
  const rng = seededRng(seed);
  const candles: CandleType[] = [];
  let price = 10000 + rng() * 290000;
  const now = Math.floor(Date.now() / 1000);
  const DAY = 86400;

  for (let i = count - 1; i >= 0; i--) {
    const vol = price * 0.025;
    const open = price;
    const close = open + (rng() - 0.48) * vol;
    const high = Math.max(open, close) + rng() * vol * 0.4;
    const low = Math.min(open, close) - rng() * vol * 0.4;
    candles.push({
      time: now - i * DAY,
      open: Math.round(open),
      high: Math.round(high),
      low: Math.round(Math.max(low, 1)),
      close: Math.round(close),
    });
    price = Math.max(close, 100);
  }
  return candles;
}

function StockRow({ stock }: { stock: (typeof mockPopularStocks)[number] }) {
  const [open, setOpen] = useState(false);
  const [pendingCart, setPendingCart] = useState(false);
  const candles = useMemo(() => generateMockCandles(stock.name), [stock.name]);
  const { toggleFavorite, toggleCart, isFavorite, isInCart } = useStockList();

  return (
    <div>
      {pendingCart && (
        <CartConfirmModal
          stockName={stock.name}
          mode={isInCart(stock.code) ? "remove" : "add"}
          onConfirm={() => { toggleCart(stock.code); setPendingCart(false); }}
          onCancel={() => setPendingCart(false)}
        />
      )}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="grid w-full grid-cols-[2rem_1fr_auto_auto_auto_auto_auto_1.5rem] items-center gap-x-2 py-3 text-left transition hover:bg-slate-50/60"
      >
        <span
          className={`flex h-7 w-7 items-center justify-center rounded-lg text-xs font-black ${
            stock.rank <= 3 ? "bg-[#5267ff] text-white" : "bg-slate-100 text-slate-500"
          }`}
        >
          {stock.rank}
        </span>
        <p className="truncate text-sm font-black text-slate-950">{stock.name}</p>
        <p className="text-sm font-black text-slate-950">{stock.price}</p>
        <p className={`w-20 text-right text-sm font-black ${stock.isUp ? "text-rose-500" : "text-blue-500"}`}>
          {stock.change}
        </p>
        <p className="hidden w-20 text-right text-xs font-bold text-slate-400 sm:block">
          {stock.volume}
        </p>
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); toggleFavorite(stock.code); }}
          title="관심 종목에 추가"
          className={`flex h-7 w-7 items-center justify-center rounded-lg border transition ${
            isFavorite(stock.code)
              ? "border-rose-100 bg-rose-50 text-rose-500"
              : "border-slate-200 text-slate-300 hover:border-rose-100 hover:bg-rose-50 hover:text-rose-500"
          }`}
        >
          <FiHeart className={`h-3.5 w-3.5 ${isFavorite(stock.code) ? "fill-current" : ""}`} />
        </button>
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); setPendingCart(true); }}
          title="포트폴리오에 추가"
          className={`flex h-7 w-7 items-center justify-center rounded-lg border transition ${
            isInCart(stock.code)
              ? "border-amber-200 bg-amber-50 text-amber-500"
              : "border-slate-200 text-slate-300 hover:border-amber-200 hover:bg-amber-50 hover:text-amber-500"
          }`}
        >
          <FiShoppingBag className="h-3.5 w-3.5" />
        </button>
        <FiChevronDown
          className={`h-4 w-4 text-slate-400 transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        />
      </button>

      {open && (
        <div className="pb-4 pt-1">
          <div className="mb-2 flex items-center justify-between px-1">
            <p className="text-xs font-bold text-slate-500">{stock.name} 가격 추이 (60일)</p>
            <p className={`text-xs font-black ${stock.isUp ? "text-rose-500" : "text-blue-500"}`}>
              {stock.change}
            </p>
          </div>
          <div className="overflow-hidden rounded-xl border border-slate-100">
            <CandleChart candles={candles} height={200} />
          </div>
        </div>
      )}
    </div>
  );
}

export default function MarketPage() {
  const router = useRouter();
  const [search, setSearch] = useState("");

  const filtered = mockPopularStocks.filter((s) =>
    s.name.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => router.back()}
          className="flex h-10 w-10 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-500 transition hover:text-slate-950"
        >
          <FiArrowLeft className="h-4 w-4" />
        </button>
        <div>
          <p className="text-sm font-bold text-[#5267ff]">Live Ranking</p>
          <h1 className="text-2xl font-black text-slate-950">실시간 종목 랭킹</h1>
        </div>
        <span className="ml-auto flex h-9 w-9 items-center justify-center rounded-xl bg-[#eef2ff] text-[#5267ff]">
          <FiTrendingUp className="h-4 w-4" />
        </span>
      </div>

      <div className="rounded-2xl bg-white p-4 shadow-[0_20px_60px_rgba(15,23,42,0.06)]">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="종목명 검색"
          className="mb-4 h-11 w-full rounded-xl border border-slate-100 bg-slate-50 px-4 text-sm text-slate-700 placeholder:text-slate-400 outline-none transition focus:border-[#5267ff] focus:shadow-[0_0_0_4px_rgba(82,103,255,0.12)]"
        />

        <div className="grid grid-cols-[2rem_1fr_auto_auto_auto_auto_auto_1.5rem] items-center gap-x-2 border-b border-slate-100 pb-2 text-xs font-bold text-slate-400">
          <span className="text-center">순위</span>
          <span>종목명</span>
          <span className="text-right">현재가</span>
          <span className="w-20 text-right">등락률</span>
          <span className="hidden w-20 text-right sm:block">거래대금</span>
          <span />
          <span />
          <span />
        </div>

        <div className="divide-y divide-slate-50">
          {filtered.map((stock) => (
            <StockRow key={stock.rank} stock={stock} />
          ))}
          {filtered.length === 0 && (
            <p className="py-10 text-center text-sm font-bold text-slate-400">
              검색 결과가 없습니다.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
