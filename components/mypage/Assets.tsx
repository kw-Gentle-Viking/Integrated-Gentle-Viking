"use client";

import { invalidateAccountAssets, useAccountAssets } from "@/lib/account/assetsStore";

function fmt(v: number) {
  return v.toLocaleString("ko-KR");
}

function PnlText({ value, rate }: { value: number; rate: number }) {
  const isPos = value >= 0;
  const color = isPos ? "text-rose-500" : "text-blue-500";
  const sign = isPos ? "+" : "";
  return (
    <span className={`text-xs ${color}`}>
      {sign}
      {fmt(value)}원 ({sign}
      {rate.toFixed(2)}%)
    </span>
  );
}

export default function Assets() {
  const { assets, isLoading, isRefreshing, error } = useAccountAssets();
  const currentMonth = new Date().getMonth() + 1;

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

  const a = assets;
  const monthlyIsPos = a.monthly.totalKRW >= 0;

  return (
    <div className="space-y-6">
      {isRefreshing ? (
        <div className="rounded-2xl border border-blue-100 bg-blue-50 px-4 py-3 text-sm font-bold text-blue-600">
          최신 자산 정보를 확인 중입니다. 변경이 없으면 현재 화면을 유지합니다.
        </div>
      ) : null}
      {error ? (
        <div className="rounded-2xl border border-amber-100 bg-amber-50 px-4 py-3 text-sm font-bold text-amber-700">
          {error} 이전에 불러온 자산 정보를 유지합니다.
        </div>
      ) : null}
      <div>
        <p className="text-xs text-slate-400">
          {a.broker} 계좌{" "}
          <span className="text-blue-500">{a.accountNo}</span>
        </p>
        <div className="mt-1 flex items-center justify-between">
          <span className="text-2xl font-bold text-slate-900">
            {fmt(a.totalKRW)}원
          </span>
          <div className="flex gap-2">
            <button className="rounded-full border border-gray-200 bg-white px-4 py-1.5 text-sm font-semibold text-slate-700 shadow-[0_2px_10px_rgba(0,0,0,0.02)] transition hover:bg-gray-50">
              채우기
            </button>
            <button className="rounded-full border border-gray-200 bg-white px-4 py-1.5 text-sm font-semibold text-slate-700 shadow-[0_2px_10px_rgba(0,0,0,0.02)] transition hover:bg-gray-50">
              보내기
            </button>
            <button
              type="button"
              onClick={() => invalidateAccountAssets()}
              className="rounded-full border border-gray-200 bg-white px-4 py-1.5 text-sm font-semibold text-slate-700 shadow-[0_2px_10px_rgba(0,0,0,0.02)] transition hover:bg-gray-50"
            >
              새로고침
            </button>
          </div>
        </div>
      </div>

      <div className="overflow-hidden rounded-2xl border border-gray-100 bg-white shadow-[0_2px_10px_rgba(0,0,0,0.02)]">
        <div className="flex items-center justify-between border-b border-gray-50 px-5 py-4">
          <span className="text-sm text-gray-500">총 주문 가능 금액</span>
          <span className="text-sm font-bold text-slate-900">{fmt(a.orderableKRW)}원</span>
        </div>

        <div className="flex items-center justify-between border-b border-gray-50 px-5 py-4 pl-10">
          <span className="flex items-center gap-1.5 text-sm text-gray-400">
            <span className="flex h-4 w-4 items-center justify-center rounded-full bg-blue-500 text-[9px] font-bold text-white">
              ₩
            </span>
            원화
          </span>
          <span className="text-sm text-slate-700">{fmt(a.cashKRW)}원</span>
        </div>

        <div className="flex items-center justify-between border-b border-gray-50 px-5 py-4">
          <div>
            <p className="text-sm text-gray-500">총 투자 금액</p>
            <p className="text-xs text-gray-400">국내주식 평가금액</p>
          </div>
          <div className="text-right">
            <p className="text-sm font-bold text-slate-900">{fmt(a.investedKRW)}원</p>
            <PnlText value={a.investedPnlKRW} rate={a.investedPnlRate} />
          </div>
        </div>

        {a.holdings.length > 0 ? (
          a.holdings.map((h) => (
            <div key={h.label} className="flex items-center justify-between border-b border-gray-50 px-5 py-4 pl-10 last:border-none">
              <span className="flex items-center gap-1.5 text-sm text-gray-400">
                <span>{h.flag}</span>
                {h.label}
              </span>
              <div className="text-right">
                <p className="text-sm text-slate-700">{fmt(h.valueKRW)}원</p>
                <PnlText value={h.pnlKRW} rate={h.pnlRate} />
              </div>
            </div>
          ))
        ) : (
          <div className="px-5 py-4 pl-10 text-sm text-gray-400">
            보유 중인 국내주식이 없습니다.
          </div>
        )}
      </div>

      <div className="overflow-hidden rounded-2xl border border-gray-100 bg-white shadow-[0_2px_10px_rgba(0,0,0,0.02)]">
        <div className="flex items-center justify-between border-b border-gray-50 px-5 py-4">
          <span className="text-sm font-bold text-slate-900">{currentMonth}월 수익</span>
          <span className={`text-sm font-bold ${monthlyIsPos ? "text-red-500" : "text-blue-500"}`}>
            {monthlyIsPos ? "+" : ""}{fmt(a.monthly.totalKRW)}원
          </span>
        </div>

        {[
          { label: "판매수익", value: a.monthly.saleKRW },
          { label: "배당금", value: a.monthly.dividendKRW },
          { label: "이자", value: a.monthly.interestKRW },
        ].map(({ label, value }) => (
          <div key={label} className="flex items-center justify-between border-b border-gray-50 px-5 py-3.5 last:border-none">
            <span className="text-sm text-gray-500">{label}</span>
            <span className="text-sm text-slate-700">
              {value > 0 ? "+" : ""}{fmt(value)}원
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
