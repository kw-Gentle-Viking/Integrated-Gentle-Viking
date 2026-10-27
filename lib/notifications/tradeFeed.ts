"use client";

// "알림" 화면/사이드바 뱃지가 쓰는 실제 알림 소스. 백엔드에 알림 전용 테이블/이벤트는
// 없어서(새로 만들기엔 과한 범위) 이미 실재하는 체결 내역(/trade/history, TradeLog)을
// "알림"으로 재구성한다 -- 직접 주문/AI 자동매매 체결 둘 다 실제로 일어난 일이다.
import { apiFetch } from "@/lib/signup/auth";
import { resolveStockName } from "@/lib/stock-list/StockListContext";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type TradeNotification = {
  id: string;
  title: string;
  desc: string;
  time: string;
  createdAt: string;
  isManual: boolean;
};

type TradeLog = {
  ticker: string;
  side: "BUY" | "SELL" | string;
  qty: number;
  price: number;
  amount: number;
  ai_signal?: string;
  ai_confidence?: number;
  strategy_id: string;
  created_at: string;
  status?: string;
};

function formatTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("ko-KR", {
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

export async function fetchTradeNotifications(limit = 10): Promise<TradeNotification[]> {
  const res = await apiFetch(`${API_BASE}/trade/history`);
  if (!res.ok) return [];
  const data = (await res.json()) as TradeLog[];

  const filled = data
    .filter((t) => t.status !== "FAILED")
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

  return filled.slice(0, limit).map((t) => {
    const isManual = t.strategy_id === "manual";
    const name = resolveStockName(t.ticker, t.ticker);
    const action = t.side === "BUY" ? "매수" : "매도";
    const qtyText = `${t.qty.toLocaleString("ko-KR")}주`;

    return {
      id: `${t.created_at}-${t.ticker}-${t.side}`,
      title: isManual ? "직접 주문 체결" : "AI 자동매매 체결",
      desc: isManual
        ? `${name} ${action} ${qtyText}`
        : `${name} ${action} ${qtyText} · AI 확신도 ${formatConfidence(t.ai_confidence ?? 0)}`,
      time: formatTime(t.created_at),
      createdAt: t.created_at,
      isManual,
    };
  });
}

export async function fetchRecentTradeCount(hours = 24): Promise<number> {
  const items = await fetchTradeNotifications(50);
  const cutoff = Date.now() - hours * 60 * 60 * 1000;
  return items.filter((item) => new Date(item.createdAt).getTime() >= cutoff).length;
}
