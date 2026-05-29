"use client";

import { FiPlus } from "react-icons/fi";
import PortfolioAside from "@/components/portfolio/PortfolioAside";
import PortfolioSummaryCards from "@/components/portfolio/PortfolioSummaryCards";
import PortfolioTable from "@/components/portfolio/PortfolioTable";
import { useStockList } from "@/lib/stock-list/StockListContext";

function parseWon(value: string) {
  const parsed = Number(value.replace(/[^0-9.-]/g, ""));
  return Number.isFinite(parsed) ? parsed : 0;
}

function formatWon(value: number) {
  return value.toLocaleString("ko-KR") + "원";
}

export default function PortfolioPage() {
  const { cartStocks } = useStockList();
  const totalValue = cartStocks.reduce(
    (sum, stock) => sum + parseWon(stock.price),
    0,
  );

  return (
    <div className="grid gap-6 xl:grid-cols-[1fr_320px]">
      <section className="rounded-2xl bg-white p-6 shadow-[0_24px_80px_rgba(15,23,42,0.06)] lg:p-8">
        <div className="flex flex-wrap items-start justify-between gap-5">
          <div>
            <p className="text-sm font-bold text-[#5267ff]">My Portfolio</p>
            <h1 className="mt-2 text-4xl font-black text-slate-950">
              내 주식보기
            </h1>
          </div>
          <button className="flex h-11 items-center gap-2 rounded-xl bg-[#5267ff] px-4 text-sm font-black text-white">
            <FiPlus /> 종목 추가
          </button>
        </div>

        <PortfolioSummaryCards
          totalValue={formatWon(totalValue)}
          stockCount={cartStocks.length}
        />
        <PortfolioTable stocks={cartStocks} />
      </section>

      <PortfolioAside stocks={cartStocks} />
    </div>
  );
}
