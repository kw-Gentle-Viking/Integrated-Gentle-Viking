"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/signup/auth";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type AITradeAction = "매수" | "매도";

type TradeLog = {
  ticker: string;
  side: "BUY" | "SELL" | string;
  qty: number;
  price: number;
  amount: number;
  ai_signal: string;
  ai_confidence: number;
  strategy_id: string;
  created_at: string;
  status?: "FILLED" | "FAILED" | string;
};

const ACTION_STYLE: Record<AITradeAction, string> = {
  매수: "text-red-500",
  매도: "text-blue-500",
};

const FILTER_OPTIONS: (AITradeAction | "전체")[] = ["전체", "매수", "매도"];

function fmt(v: number) {
  return v.toLocaleString("ko-KR");
}

function toAction(side: string): AITradeAction {
  return side === "BUY" || side === "매수" ? "매수" : "매도";
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

function formatConfidence(value: number) {
  const pct = value <= 1 ? value * 100 : value;
  return `${pct.toFixed(1)}%`;
}

export default function AITradeHistory() {
  const [filter, setFilter] = useState<AITradeAction | "전체">("전체");
  const [trades, setTrades] = useState<TradeLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    apiFetch(`${API_BASE}/trade/history`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`거래내역 조회 실패 (${res.status})`);
        const data = (await res.json()) as TradeLog[];
        if (!cancelled) {
          setTrades(data);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setTrades([]);
          setError(err instanceof Error ? err.message : "거래내역을 불러오지 못했습니다.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  // 이 화면은 체결 건만 보여준다(마이페이지 문구): 실패한 주문(FAILED)은 제외한다.
  const filledTrades = trades.filter((trade) => trade.status !== "FAILED");
  const filtered = filter === "전체"
    ? filledTrades
    : filledTrades.filter((trade) => toAction(trade.side) === filter);

  return (
    <div className="space-y-5">
      <div className="border-b border-gray-100">
        <div className="flex items-end justify-start gap-5">
          {FILTER_OPTIONS.map((opt) => (
            <button
              key={opt}
              onClick={() => setFilter(opt)}
              className={`relative pb-3 text-sm font-bold transition-colors ${filter === opt ? "text-black" : "text-gray-400 hover:text-gray-600"}`}
            >
              {opt}
              {filter === opt && <span className="absolute bottom-0 left-0 h-0.5 w-full bg-black" />}
            </button>
          ))}
        </div>
      </div>

      <div className="overflow-x-auto rounded-2xl border border-gray-100 bg-white shadow-[0_2px_10px_rgba(0,0,0,0.02)]">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-100 bg-gray-50/50 text-xs text-gray-500">
              <th className="px-5 py-3 text-left font-medium">거래일시</th>
              <th className="px-5 py-3 text-left font-medium">종목</th>
              <th className="px-5 py-3 text-center font-medium">구분</th>
              <th className="px-5 py-3 text-right font-medium">수량</th>
              <th className="px-5 py-3 text-right font-medium">체결가</th>
              <th className="px-5 py-3 text-right font-medium">총 금액</th>
              <th className="px-5 py-3 text-left font-medium">AI 매매 사유</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={7} className="py-16 text-center text-sm text-gray-400">불러오는 중...</td></tr>
            ) : error ? (
              <tr><td colSpan={7} className="py-16 text-center text-sm font-bold text-rose-500">{error}</td></tr>
            ) : filtered.length === 0 ? (
              <tr><td colSpan={7} className="py-16 text-center text-sm text-gray-400">거래내역이 없습니다.</td></tr>
            ) : (
              filtered.map((trade, index) => {
                const action = toAction(trade.side);
                return (
                  <tr key={`${trade.created_at}-${trade.ticker}-${index}`} className="border-b border-gray-50 transition-colors hover:bg-gray-50 last:border-none">
                    <td className="whitespace-nowrap px-5 py-4 text-xs text-gray-400">{formatDate(trade.created_at)}</td>
                    <td className="px-5 py-4">
                      <p className="font-bold text-slate-900">{trade.ticker}</p>
                      <p className="text-xs text-gray-400">{trade.strategy_id}</p>
                    </td>
                    <td className="px-5 py-4 text-center"><span className={`font-bold ${ACTION_STYLE[action]}`}>{action}</span></td>
                    <td className="px-5 py-4 text-right text-slate-700">{fmt(trade.qty)}주</td>
                    <td className="px-5 py-4 text-right text-slate-700">{fmt(trade.price)}원</td>
                    <td className="px-5 py-4 text-right text-slate-700">{fmt(trade.amount)}원</td>
                    <td className="px-5 py-4">
                      <span className="flex items-start gap-1.5">
                        <span className="mt-0.5 shrink-0 rounded bg-violet-100 px-1 text-[10px] font-bold text-violet-500">AI</span>
                        <span className="text-xs text-gray-600">{trade.ai_signal} · 확신도 {formatConfidence(trade.ai_confidence)}</span>
                      </span>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      <p className="text-right text-xs text-gray-400">총 {filtered.length}건</p>
    </div>
  );
}
