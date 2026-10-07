"use client";

import { useEffect, useRef } from "react";
import { FiTerminal } from "react-icons/fi";

// 자동매매는 대부분의 사이클에서 주문을 내지 않는다(확신도 기준·전략 조건을 통과하는 경우가 드묾).
// 그래서 "체결 기록"만 보면 화면이 비어 있어 프로그램이 멈춘 것처럼 보인다. 이 로그는 체결 여부와
// 무관하게 사이클마다 AI 판단과 그 처리 결과를 전부 보여줘서, "계속 평가는 하고 있다"는 걸 드러낸다.

export type ActivityDecision = {
  ticker: string;
  action: string; // HOLD | SKIP | ORDER_SUBMITTED
  reason: string; // 사유 코드 (ORDER_SUBMITTED일 때는 FILLED | FAILED)
  detail: string;
  ai_signal: string;
  ai_confidence: number;
  created_at: string;
};

export type ActivityFill = {
  ticker: string;
  side: string;
  qty: number;
  price: number;
  status?: string;
  order_no?: string | null;
  created_at: string;
};

const REASON_LABELS: Record<string, string> = {
  LOW_CONFIDENCE: "확신도 부족",
  STRATEGY_CONDITION_NOT_MET: "전략 조건 미충족",
  AI_STRATEGY_MISMATCH: "AI·전략 불일치",
  NO_MARKET_PRICE: "시세 없음",
  NO_POSITION_TO_SELL: "매도 보유수량 없음",
  NO_BUY_ALLOCATION: "매수 배분 없음",
  ZERO_ORDER_QUANTITY: "주문 수량 0",
  RISK_LIMIT_EXCEEDED: "리스크 한도 초과",
  UNEXPECTED_ERROR: "처리 중 오류",
};

function reasonLabel(reason: string) {
  return REASON_LABELS[reason] ?? reason;
}

function signalKor(signal: string) {
  if (signal === "BUY") return "매수";
  if (signal === "SELL") return "매도";
  if (signal === "HOLD") return "관망";
  return signal || "-";
}

function hhmmss(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "--:--:--";
  return date.toLocaleTimeString("ko-KR", { hour12: false });
}

// 같은 사이클에서 결정과 체결 로그가 거의 동시에(같은 트랜잭션) 기록되므로, 같은 종목·10초 이내면 같은
// 사이클로 본다. 둘을 합쳐야 체결가·주문번호까지 한 줄에 보여줄 수 있다(결정 기록에는 결정 시점 가격만 있음).
function matchFill(decision: ActivityDecision, fills: ActivityFill[]): ActivityFill | undefined {
  const t0 = new Date(decision.created_at).getTime();
  return fills.find(
    (f) => f.ticker === decision.ticker && Math.abs(new Date(f.created_at).getTime() - t0) <= 10_000,
  );
}

type Line = { key: string; ts: string; kind: "fill" | "fail" | "idle"; text: string };

function buildLines(decisions: ActivityDecision[], fills: ActivityFill[]): Line[] {
  const sorted = [...decisions].sort(
    (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
  );
  return sorted.map((d, i) => {
    const time = hhmmss(d.created_at);
    const ai = `AI ${signalKor(d.ai_signal)}(${(d.ai_confidence * 100).toFixed(0)}%)`;
    const head = `${time}  ${d.ticker.padEnd(6)}  ${ai}`;

    if (d.action === "ORDER_SUBMITTED") {
      const fill = matchFill(d, fills);
      if (d.reason === "FILLED" && fill) {
        const text = `${head}  →  체결 ${fill.qty}주 @ ${fill.price.toLocaleString("ko-KR")}원` +
          (fill.order_no ? `  (주문번호 ${fill.order_no})` : "");
        return { key: `${d.created_at}-${d.ticker}-${i}`, ts: d.created_at, kind: "fill", text };
      }
      const text = `${head}  →  주문 실패  —  ${d.detail}`;
      return { key: `${d.created_at}-${d.ticker}-${i}`, ts: d.created_at, kind: "fail", text };
    }

    const actionKor = d.action === "SKIP" ? "건너뜀" : "관망";
    const text = `${head}  →  ${actionKor} (${reasonLabel(d.reason)})`;
    return { key: `${d.created_at}-${d.ticker}-${i}`, ts: d.created_at, kind: "idle", text };
  });
}

export default function TradeActivityTerminal({
  decisions,
  fills,
}: {
  decisions: ActivityDecision[];
  fills: ActivityFill[];
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const lines = buildLines(decisions, fills);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines.length]);

  return (
    <div className="mt-6 overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
      <div className="flex items-center gap-2 border-b border-slate-800 bg-slate-900 px-5 py-3 text-sm font-black text-slate-200">
        <FiTerminal className="h-4 w-4 text-emerald-400" /> 자동매매 실행 로그
        <span className="ml-auto flex items-center gap-1.5 text-xs font-bold text-slate-500">
          <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" /> 실시간
        </span>
      </div>
      <div ref={scrollRef} className="max-h-80 overflow-y-auto px-5 py-4 font-mono text-xs leading-6">
        {lines.length === 0 ? (
          <p className="text-slate-500">
            아직 판단 기록이 없습니다. AI 추론이 들어올 때마다(5분 주기) 이 자리에 한 줄씩 쌓입니다.
          </p>
        ) : (
          lines.map((line) => (
            <p
              key={line.key}
              className={
                line.kind === "fill"
                  ? "font-bold text-emerald-400"
                  : line.kind === "fail"
                    ? "font-bold text-rose-400"
                    : "text-slate-500"
              }
            >
              {line.kind === "fill" ? "✔ " : line.kind === "fail" ? "✘ " : "· "}
              {line.text}
            </p>
          ))
        )}
      </div>
    </div>
  );
}
