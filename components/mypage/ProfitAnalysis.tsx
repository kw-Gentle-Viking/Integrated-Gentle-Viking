"use client";

import { useAccountAssets } from "@/lib/account/assetsStore";

function fmt(value: number) {
  return value.toLocaleString("ko-KR");
}

export default function ProfitAnalysis() {
  const { assets, isLoading, error } = useAccountAssets();

  if (isLoading) {
    return (
      <div className="rounded-2xl border border-gray-100 bg-white p-6 text-sm text-slate-500 shadow-[0_2px_10px_rgba(0,0,0,0.02)]">
        한국투자증권 모의투자 자산을 불러오는 중입니다.
      </div>
    );
  }

  if (!assets) {
    return (
      <div className="rounded-2xl border border-rose-100 bg-rose-50 p-6 text-sm text-rose-600">
        {error ?? "자산 정보를 불러오지 못했습니다."}
      </div>
    );
  }

  const totalPnl = assets.investedPnlKRW;
  const isTotalPos = totalPnl >= 0;
  const holdings = assets.holdings;
  const maxAbsPnl = Math.max(1, ...holdings.map((h) => Math.abs(h.pnlKRW)));

  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-3">
        <section className="rounded-2xl bg-slate-950 p-5 text-white md:col-span-1">
          <p className="text-sm font-bold text-slate-300">보유 종목 평가손익</p>
          <p className={`mt-3 text-3xl font-black ${isTotalPos ? "text-rose-300" : "text-blue-300"}`}>
            {isTotalPos ? "+" : ""}
            {fmt(totalPnl)}원
          </p>
          <p className="mt-2 text-xs font-bold text-slate-400">
            {isTotalPos ? "+" : ""}
            {assets.investedPnlRate.toFixed(2)}% · {assets.broker}
          </p>
        </section>

        <section className="rounded-2xl border border-slate-100 bg-white p-5 shadow-[0_12px_32px_rgba(15,23,42,0.04)] md:col-span-2">
          <h2 className="text-lg font-black text-slate-950">수익 구성</h2>
          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            <div className="rounded-xl bg-slate-50 p-4">
              <p className="text-xs font-bold text-slate-400">투자 금액</p>
              <p className="mt-2 font-black text-slate-700">{fmt(assets.investedKRW)}원</p>
            </div>
            <div className="rounded-xl bg-slate-50 p-4">
              <p className="text-xs font-bold text-slate-400">평가 손익</p>
              <p className={`mt-2 font-black ${isTotalPos ? "text-rose-500" : "text-blue-500"}`}>
                {isTotalPos ? "+" : ""}
                {fmt(totalPnl)}원
              </p>
            </div>
            <div className="rounded-xl bg-slate-50 p-4">
              <p className="text-xs font-bold text-slate-400">보유 종목 수</p>
              <p className="mt-2 font-black text-slate-700">{holdings.length}개</p>
            </div>
          </div>
          <p className="mt-3 text-xs font-medium text-slate-400">
            실현 손익 · 배당 · 이자는 한국투자증권 모의투자 계좌에서 제공하지 않아 표시하지 않습니다.
          </p>
        </section>
      </div>

      <section className="rounded-2xl border border-slate-100 bg-white p-5 shadow-[0_12px_32px_rgba(15,23,42,0.04)]">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-black text-slate-950">종목별 평가손익</h2>
            <p className="mt-1 text-sm font-medium text-slate-400">현재 보유 중인 종목의 평가손익입니다.</p>
          </div>
        </div>

        {holdings.length === 0 ? (
          <p className="mt-6 text-sm font-bold text-slate-400">보유 중인 종목이 없습니다.</p>
        ) : (
          <div className="mt-6 space-y-4">
            {holdings.map((h) => {
              const width = `${Math.max(8, (Math.abs(h.pnlKRW) / maxAbsPnl) * 100)}%`;
              const isPos = h.pnlKRW >= 0;

              return (
                <div key={h.label} className="grid grid-cols-[88px_1fr_120px] items-center gap-3">
                  <span className="truncate text-sm font-black text-slate-500">{h.label}</span>
                  <div className="h-3 rounded-full bg-slate-100">
                    <div
                      className={`h-3 rounded-full ${isPos ? "bg-rose-400" : "bg-blue-400"}`}
                      style={{ width }}
                    />
                  </div>
                  <span className={`text-right text-sm font-black ${isPos ? "text-rose-500" : "text-blue-500"}`}>
                    {isPos ? "+" : ""}
                    {fmt(h.pnlKRW)}원
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
