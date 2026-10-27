"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/signup/auth";
import { resolveStockName } from "@/lib/stock-list/StockListContext";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type TradeLog = {
  ticker: string;
  side: "BUY" | "SELL" | string;
  qty: number;
  price: number;
  amount: number;
  strategy_id: string;
  created_at: string;
  status?: "FILLED" | "FAILED" | string;
  order_no?: string | null;
};

const statusStyle: Record<string, string> = {
  FILLED: "bg-emerald-50 text-emerald-600",
  FAILED: "bg-rose-50 text-rose-500",
};

const statusLabel: Record<string, string> = {
  FILLED: "체결",
  FAILED: "실패",
};

function fmt(value: number) {
  return value.toLocaleString("ko-KR");
}

function formatDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export default function OrderHistory() {
  const [orders, setOrders] = useState<TradeLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    apiFetch(`${API_BASE}/trade/history`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`주문내역 조회 실패 (${res.status})`);
        const data = (await res.json()) as TradeLog[];
        if (!cancelled) {
          // 직접매매(수동 주문)만 보여준다. 자동매매 체결은 "AI자동거래내역" 탭 몫이다.
          setOrders(data.filter((log) => log.strategy_id === "manual"));
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setOrders([]);
          setError(err instanceof Error ? err.message : "주문내역을 불러오지 못했습니다.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="space-y-5">
      <div className="flex justify-end">
        <span className="rounded-full bg-[#eef2ff] px-4 py-2 text-xs font-black text-[#5267ff]">
          총 {orders.length}건
        </span>
      </div>

      <div className="overflow-x-auto rounded-2xl border border-slate-100 bg-white shadow-[0_12px_32px_rgba(15,23,42,0.04)]">
        <table className="w-full min-w-[760px] text-sm">
          <thead>
            <tr className="border-b border-slate-100 bg-slate-50/80 text-xs font-bold text-slate-400">
              <th className="px-5 py-3 text-left">주문일시</th>
              <th className="px-5 py-3 text-left">종목</th>
              <th className="px-5 py-3 text-center">구분</th>
              <th className="px-5 py-3 text-right">수량</th>
              <th className="px-5 py-3 text-right">주문가</th>
              <th className="px-5 py-3 text-right">주문금액</th>
              <th className="px-5 py-3 text-center">상태</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={7} className="py-16 text-center text-sm text-slate-400">불러오는 중...</td></tr>
            ) : error ? (
              <tr><td colSpan={7} className="py-16 text-center text-sm font-bold text-rose-500">{error}</td></tr>
            ) : orders.length === 0 ? (
              <tr><td colSpan={7} className="py-16 text-center text-sm text-slate-400">직접 주문한 내역이 없습니다.</td></tr>
            ) : (
              orders.map((order, index) => {
                const side = order.side === "BUY" ? "매수" : "매도";
                const status = order.status ?? "FILLED";
                const isMarket = !order.price;
                return (
                  <tr key={`${order.order_no ?? order.created_at}-${index}`} className="border-b border-slate-50 last:border-none hover:bg-slate-50/70">
                    <td className="whitespace-nowrap px-5 py-4 text-xs font-bold text-slate-400">
                      {formatDate(order.created_at)}
                    </td>
                    <td className="px-5 py-4">
                      <p className="font-black text-slate-950">{resolveStockName(order.ticker, order.ticker)}</p>
                      <p className="mt-0.5 text-xs font-bold text-slate-400">{order.ticker}</p>
                    </td>
                    <td className={`px-5 py-4 text-center font-black ${side === "매수" ? "text-rose-500" : "text-blue-500"}`}>
                      {side}
                    </td>
                    <td className="px-5 py-4 text-right font-semibold text-slate-700">{fmt(order.qty)}주</td>
                    <td className="px-5 py-4 text-right font-semibold text-slate-700">
                      {isMarket ? "시장가" : `${fmt(order.price)}원`}
                    </td>
                    <td className="px-5 py-4 text-right font-black text-slate-950">
                      {isMarket ? "-" : `${fmt(order.amount)}원`}
                    </td>
                    <td className="px-5 py-4 text-center">
                      <span className={`rounded-full px-3 py-1 text-xs font-black ${statusStyle[status] ?? "bg-slate-100 text-slate-500"}`}>
                        {statusLabel[status] ?? status}
                      </span>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
