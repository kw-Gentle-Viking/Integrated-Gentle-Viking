"use client";

import { FiArrowUpRight, FiPlus } from "react-icons/fi";
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
  const totalValue = cartStocks.reduce((sum, stock) => sum + parseWon(stock.price), 0);

  return (
    <div className="grid gap-6 xl:grid-cols-[1fr_320px]">
      <section className="rounded-2xl bg-white p-6 shadow-[0_24px_80px_rgba(15,23,42,0.06)] lg:p-8">
        <div className="flex flex-wrap items-start justify-between gap-5">
          <div><p className="text-sm font-bold text-slate-400">My Portfolio</p><h1 className="mt-2 text-4xl font-black text-slate-950">내 주식보기</h1></div>
          <button className="flex h-11 items-center gap-2 rounded-xl bg-[#5267ff] px-4 text-sm font-black text-white"><FiPlus /> 종목 추가</button>
        </div>

        <div className="mt-8 grid gap-4 md:grid-cols-3">
          <div className="rounded-xl bg-[#f4f7ff] p-5"><p className="text-sm font-bold text-slate-500">평가 금액</p><p className="mt-2 text-3xl font-black">{formatWon(totalValue)}</p></div>
          <div className="rounded-xl bg-emerald-50 p-5"><p className="text-sm font-bold text-slate-500">담은 종목</p><p className="mt-2 text-3xl font-black text-emerald-500">{cartStocks.length}개</p></div>
          <div className="rounded-xl bg-rose-50 p-5"><p className="text-sm font-bold text-slate-500">오늘 변동</p><p className="mt-2 text-3xl font-black text-rose-500">AI 리포트 연동</p></div>
        </div>

        <div className="mt-8 overflow-hidden rounded-xl border border-slate-100">
          <div className="grid grid-cols-[1.4fr_0.8fr_0.9fr_0.9fr_0.7fr] bg-slate-50 px-5 py-3 text-xs font-bold text-slate-400">
            <span>종목</span><span>보유</span><span>현재가</span><span>평가금액</span><span className="text-right">수익률</span>
          </div>
          {cartStocks.length > 0 ? cartStocks.map((stock) => (
            <div key={stock.code} className="grid grid-cols-[1.4fr_0.8fr_0.9fr_0.9fr_0.7fr] items-center border-t border-slate-100 px-5 py-4 text-sm">
              <div className="flex items-center gap-3"><span className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-100 font-black">{stock.logoText ?? stock.name.slice(0, 1)}</span><div><p className="font-black text-slate-950">{stock.name}</p><p className="text-xs text-slate-400">{stock.code}</p></div></div>
              <span className="font-semibold text-slate-500">대기</span><span className="font-bold">{stock.price}</span><span className="font-bold">{stock.price}</span><span className={`text-right font-black ${stock.change.startsWith("-") ? "text-blue-500" : "text-rose-500"}`}>{stock.change}</span>
            </div>
          )) : (
            <div className="border-t border-slate-100 px-5 py-12 text-center">
              <p className="text-sm font-black text-slate-500">아직 포트폴리오에 담은 종목이 없습니다.</p>
              <p className="mt-2 text-sm text-slate-400">AI 리포트에서 가방 아이콘을 누르면 이곳에 바로 추가됩니다.</p>
            </div>
          )}
        </div>
      </section>

      <aside className="space-y-5">
        <section className="rounded-2xl bg-white p-6 shadow-[0_24px_80px_rgba(15,23,42,0.06)]">
          <h2 className="font-black">비중</h2>
          <div className="mt-5 space-y-4">
            {cartStocks.length > 0 ? cartStocks.slice(0, 3).map((stock) => {
              const width = Math.max(18, Math.round(100 / Math.min(cartStocks.length, 3))) + "%";
              return <div key={stock.code}><div className="mb-2 flex justify-between text-sm font-bold"><span>{stock.name}</span><span>{width}</span></div><div className="h-2 rounded-full bg-slate-100"><div className="h-2 rounded-full bg-[#5267ff]" style={{ width }} /></div></div>;
            }) : <p className="text-sm font-bold text-slate-400">담긴 종목이 없습니다.</p>}
          </div>
        </section>
        <section className="rounded-2xl bg-slate-950 p-6 text-white"><p className="text-xl font-black">AI 추천</p><p className="mt-3 text-sm leading-6 text-slate-300">AI 리포트에서 담은 종목을 기준으로 포트폴리오 구성을 확인합니다.</p><button className="mt-5 flex items-center gap-2 text-sm font-black text-emerald-300">자세히 보기 <FiArrowUpRight /></button></section>
      </aside>
    </div>
  );
}
