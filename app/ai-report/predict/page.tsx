"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { FiAlertCircle, FiArrowLeft, FiCpu, FiRefreshCw, FiShoppingBag } from "react-icons/fi";
import { apiFetch } from "@/lib/signup/auth";
import CartConfirmModal from "@/components/stock-list/CartConfirmModal";
import { StockListItem, useStockList } from "@/lib/stock-list/StockListContext";

type Prediction = {
  buy: number;
  hold: number;
  sell: number;
};

type StockPrediction = {
  name: string;
  code: string;
  price: string;
  change: string;
};

type BackendPrediction = {
  ticker?: string;
  signal?: "BUY" | "HOLD" | "SELL" | string;
  confidence?: number;
  prob_buy?: number;
  prob_hold?: number;
  prob_sell?: number;
  trade_datetime?: string;
  model_version?: string;
  report?: string;
  gemini_report?: string;
  analysis?: string;
};

type PredictionResult = {
  prediction: Prediction;
  signal: string;
  confidence: number;
  tradeDatetime?: string;
  modelVersion?: string;
  report: string;
};

type ReportSection = {
  title: string;
  body: string[];
};

type ReportCard = {
  title: string;
  eyebrow: string;
  body: string[];
  tone: "summary" | "feature" | "attention" | "market" | "risk";
};

const predictionItems = [
  { key: "buy", label: "매수", color: "#f43f5e", desc: "상승 방향성과 거래량 확장 가능성" },
  { key: "hold", label: "관망", color: "#5267ff", desc: "추가 확인이 필요한 중립 구간" },
  { key: "sell", label: "매도", color: "#3b82f6", desc: "하방 위험 또는 과열 해소 가능성" },
] as const;

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const signalLabel: Record<string, string> = {
  BUY: "매수",
  HOLD: "관망",
  SELL: "매도",
};

function toPercent(value: unknown) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  const percent = numeric <= 1 ? numeric * 100 : numeric;
  return Math.max(0, Math.min(100, Math.round(percent)));
}

function normalizePrediction(data: BackendPrediction): PredictionResult {
  const prediction = {
    buy: toPercent(data.prob_buy),
    hold: toPercent(data.prob_hold),
    sell: toPercent(data.prob_sell),
  };
  const signal = signalLabel[String(data.signal ?? "").toUpperCase()] ?? "관망";
  const confidence = toPercent(data.confidence);

  return {
    prediction,
    signal,
    confidence,
    tradeDatetime: data.trade_datetime,
    modelVersion: data.model_version,
    report:
      data.report ||
      data.gemini_report ||
      data.analysis ||
      "백엔드 응답에 상세 분석 리포트가 포함되지 않았습니다.",
  };
}

function formatInferenceTime(value?: string) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return new Intl.DateTimeFormat("ko-KR", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function getLogoText(name: string, code: string) {
  if (name && name !== "종목명 없음") return name.slice(0, 1);
  return code.slice(0, 1);
}

function makeStockListItem(stock: StockPrediction): StockListItem {
  return {
    code: stock.code,
    name: stock.name,
    price: stock.price,
    change: stock.change,
    logoText: getLogoText(stock.name, stock.code),
    memo: "AI 매매 확률 보기에서 포트폴리오에 추가한 종목",
  };
}

function cleanMarkdown(line: string) {
  return line
    .replace(/^[-*]\s+/, "")
    .replace(/^\d+\.\s+/, "")
    .replace(/\*\*/g, "")
    .trim();
}

function parseReportSections(report: string): ReportSection[] {
  const lines = report
    .split(/\n+/)
    .map((line) => line.trim())
    .filter((line) => line && line !== "---");

  const sections: ReportSection[] = [];
  let current: ReportSection = { title: "AI 요약", body: [] };

  for (const line of lines) {
    const heading = line.match(/^#{1,4}\s+(.+)$/);
    if (heading) {
      if (current.body.length > 0) sections.push(current);
      current = { title: cleanMarkdown(heading[1]), body: [] };
      continue;
    }

    const cleaned = cleanMarkdown(line);
    if (cleaned) current.body.push(cleaned);
  }

  if (current.body.length > 0) sections.push(current);
  return sections.length > 0 ? sections : [{ title: "AI 분석 리포트", body: [report] }];
}

function findReportBody(sections: ReportSection[], keywords: string[], limit = 7) {
  const matches = sections.filter((section) => {
    const haystack = `${section.title} ${section.body.join(" ")}`.toLowerCase();
    return keywords.some((keyword) => haystack.includes(keyword.toLowerCase()));
  });

  const body = matches.flatMap((section) => section.body).filter(Boolean);
  if (body.length > 0) return body.slice(0, limit);
  return sections.flatMap((section) => section.body).filter(Boolean).slice(0, limit);
}

function buildReportCards(report: string): ReportCard[] {
  const sections = parseReportSections(report);

  return [
    {
      title: "예측 요약",
      eyebrow: "Prediction Summary",
      body: findReportBody(sections, ["예측 요약", "모델 예측", "매수 확률", "관망 확률", "매도 확률"], 5),
      tone: "summary",
    },
    {
      title: "주요 피쳐 분석",
      eyebrow: "Core Features",
      body: findReportBody(sections, ["핵심 피처", "지표 분석", "macd", "볼린저", "이격도", "순매수"], 8),
      tone: "feature",
    },
    {
      title: "시간 가중치 분석",
      eyebrow: "Attention Weight",
      body: findReportBody(sections, ["시간적", "어텐션", "분 전", "최근 1시간"], 6),
      tone: "attention",
    },
    {
      title: "종목 및 시장 환경",
      eyebrow: "Market Context",
      body: findReportBody(sections, ["섹터", "시장", "거시", "이벤트", "장 진행률", "금통위", "FOMC"], 7),
      tone: "market",
    },
    {
      title: "투자 유의 사항",
      eyebrow: "Risk Notice",
      body: findReportBody(sections, ["투자 유의", "원금 손실", "투자 결정", "불확실", "주의"], 7),
      tone: "risk",
    },
  ];
}

function cardToneClass(tone: ReportCard["tone"]) {
  if (tone === "summary") return "border-[#5267ff]/20 bg-[#f4f7ff]";
  return "border-slate-100 bg-white";
}

function cardEyebrowClass(tone: ReportCard["tone"]) {
  if (tone === "summary") return "text-[#5267ff]";
  return "text-slate-400";
}

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function ProbabilityRing({ label, value, color, desc }: { label: string; value: number; color: string; desc: string }) {
  const [animatedValue, setAnimatedValue] = useState(0);

  useEffect(() => {
    const duration = 900;
    const startedAt = performance.now();
    let frame = 0;

    const tick = (now: number) => {
      const progress = Math.min((now - startedAt) / duration, 1);
      setAnimatedValue(Math.round(value * progress));
      if (progress < 1) frame = window.requestAnimationFrame(tick);
    };

    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, [value]);

  return (
    <div className="rounded-2xl border border-slate-100 bg-white p-6 text-center shadow-[0_16px_44px_rgba(15,23,42,0.05)]">
      <div
        className="mx-auto flex h-36 w-36 items-center justify-center rounded-full transition-[background] duration-1000 ease-out"
        style={{ background: `conic-gradient(${color} ${animatedValue * 3.6}deg, #e5e7eb 0deg)` }}
      >
        <div className="flex h-24 w-24 flex-col items-center justify-center rounded-full bg-white shadow-inner">
          <span className="text-3xl font-black text-slate-950">{value}%</span>
          <span className="text-xs font-black text-slate-400">{label}</span>
        </div>
      </div>
      <p className="mt-5 text-lg font-black text-slate-950">{label}</p>
      <p className="mt-2 text-sm leading-6 text-slate-500">{desc}</p>
    </div>
  );
}

function PredictContent() {
  const searchParams = useSearchParams();
  const code = searchParams.get("code") ?? "005930";
  const stock: StockPrediction = {
    name: searchParams.get("name") || "종목명 없음",
    code,
    price: searchParams.get("price") || "-",
    change: searchParams.get("change") || "-",
  };
  const stockListItem = makeStockListItem(stock);
  const { toggleCart, isInCart } = useStockList();
  const [pendingCart, setPendingCart] = useState<StockListItem | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [requestMessage, setRequestMessage] = useState<string | null>(null);

  const fetchPrediction = useCallback(async () => {
    const res = await apiFetch(`${API_BASE}/ai/predictions/${encodeURIComponent(code)}`);
    if (res.status === 404) return false;
    if (!res.ok) throw new Error(`추론 결과 요청 실패 (${res.status})`);

    const data = (await res.json()) as BackendPrediction;
    setResult(normalizePrediction(data));
    setErrorMessage(null);
    setRequestMessage(null);
    return true;
  }, [code]);

  const requestPrediction = useCallback(async () => {
    setIsLoading(true);
    setResult(null);
    setErrorMessage(null);
    setRequestMessage(`${stock.name} 실제 AI 추론을 시작하는 중입니다.`);

    try {
      const res = await apiFetch(`${API_BASE}/ai/predictions/request`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tickers: [code] }),
      });
      const payload = await res.json().catch(() => ({})) as { job_id?: string; detail?: string; message?: string };
      if (!res.ok) throw new Error(payload.detail || payload.message || `추론 요청 실패 (${res.status})`);

      setRequestMessage("AI 서버가 분석 중입니다. 결과 콜백이 도착하면 자동으로 화면이 전환됩니다.");

      for (let attempt = 0; attempt < 24; attempt += 1) {
        await delay(2500);
        const hasResult = await fetchPrediction();
        if (hasResult) return;
      }

      setErrorMessage("추론 요청은 전달됐지만 아직 AI 서버 콜백 결과가 도착하지 않았습니다. 새로고침을 눌러 다시 확인해주세요.");
    } catch (error) {
      setResult(null);
      setErrorMessage(error instanceof Error ? error.message : "추론 요청 중 오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  }, [code, fetchPrediction, stock.name]);

  useEffect(() => {
    void requestPrediction();
  }, [requestPrediction]);

  const reportCards = useMemo(() => (result ? buildReportCards(result.report) : []), [result]);

  const dominant = result
    ? predictionItems.reduce((best, item) =>
        result.prediction[item.key] > result.prediction[best.key] ? item : best,
      )
    : null;

  return (
    <>
      <section className="rounded-2xl bg-white p-6 shadow-[0_24px_80px_rgba(15,23,42,0.06)] lg:p-8">
        <div className="flex flex-wrap items-start justify-between gap-5 border-b border-slate-100 pb-6">
          <div>
            <Link href="/ai-report" className="inline-flex items-center gap-2 text-sm font-black text-slate-400 transition hover:text-slate-950">
              <FiArrowLeft className="h-4 w-4" />
              투자 리포트로 돌아가기
            </Link>
            <p className="mt-5 text-sm font-bold text-[#5267ff]">AI Prediction</p>
            <h1 className="mt-2 text-4xl font-black text-slate-950">{stock.name} AI 예측</h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-500">
              Temporal Fusion Transformer가 백엔드와 AI 서버를 거쳐 실제 추론한 매수, 관망, 매도 확률을 표시합니다.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setPendingCart(stockListItem)}
              title={isInCart(stock.code) ? "포트폴리오에서 제외" : "포트폴리오에 추가"}
              className={`flex h-11 items-center gap-2 rounded-full border px-4 text-sm font-black transition ${
                isInCart(stock.code)
                  ? "border-amber-200 bg-amber-50 text-amber-600 hover:bg-amber-100"
                  : "border-slate-200 bg-white text-slate-500 hover:border-amber-200 hover:bg-amber-50 hover:text-amber-600"
              }`}
            >
              <FiShoppingBag className="h-4 w-4" />
              포트폴리오 담기
            </button>
            <button
              type="button"
              onClick={() => void requestPrediction()}
              disabled={isLoading}
              className="flex h-11 items-center gap-2 rounded-full border border-slate-200 bg-slate-100 px-5 text-sm font-black text-slate-600 transition hover:border-slate-300 hover:bg-slate-200 hover:text-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
            >
              <FiRefreshCw className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
              새로고침
            </button>
          </div>
        </div>

        {isLoading ? (
          <div className="mt-6 flex min-h-[480px] items-center justify-center rounded-2xl bg-[#f4f7ff] p-8">
            <div className="max-w-md text-center">
              <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-white shadow-[0_18px_50px_rgba(82,103,255,0.18)]">
                <FiCpu className="h-9 w-9 animate-pulse text-[#5267ff]" />
              </div>
              <h2 className="mt-6 text-2xl font-black text-slate-950">실제 AI 추론 중</h2>
              <p className="mt-3 text-sm font-bold leading-6 text-slate-500">
                {requestMessage ?? `${stock.name}의 추론 요청을 AI 서버에 전달하고 있습니다.`}
              </p>
              <div className="mt-6 h-2 overflow-hidden rounded-full bg-white">
                <div className="h-full w-1/2 animate-pulse rounded-full bg-[#5267ff]" />
              </div>
            </div>
          </div>
        ) : !result ? (
          <div className="mt-6 rounded-2xl border border-amber-100 bg-amber-50 px-5 py-8 text-center">
            <FiAlertCircle className="mx-auto h-8 w-8 text-amber-500" />
            <p className="mt-4 text-base font-black text-amber-900">추론 결과를 아직 받지 못했습니다.</p>
            <p className="mx-auto mt-2 max-w-2xl text-sm font-bold leading-6 text-amber-700">
              {errorMessage ?? "AI 서버가 백엔드로 추론 결과를 전송하면 이 화면에 확률과 리포트가 표시됩니다."}
            </p>
          </div>
        ) : (
          <div className="mt-6 space-y-4">
            <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
              <aside className="rounded-2xl bg-slate-950 p-6 text-white">
                <FiCpu className="h-8 w-8 text-emerald-300" />
                <p className="mt-5 text-sm font-bold text-slate-300">{stock.code}</p>
                <h2 className="mt-1 text-3xl font-black">{stock.name}</h2>
                <p className="mt-4 text-2xl font-black">{stock.price}</p>
                <p className={`mt-1 text-sm font-black ${stock.change.startsWith("-") ? "text-blue-300" : "text-rose-300"}`}>
                  {stock.change}
                </p>
              </aside>

              <div className="grid gap-4 md:grid-cols-3">
                {predictionItems.map((item) => (
                  <ProbabilityRing
                    key={item.key}
                    label={item.label}
                    value={result.prediction[item.key]}
                    color={item.color}
                    desc={item.desc}
                  />
                ))}
              </div>
            </div>

            <div className="grid gap-3 rounded-2xl border border-slate-100 bg-white p-4 text-xs font-bold text-slate-500 shadow-[0_12px_32px_rgba(15,23,42,0.04)] md:grid-cols-3">
              <div className="flex items-center justify-between gap-3 rounded-xl bg-slate-50 px-4 py-3">
                <span>모델 판단</span>
                <span className="font-black text-slate-950">{result.signal ?? dominant?.label} 우위</span>
              </div>
              <div className="flex items-center justify-between gap-3 rounded-xl bg-slate-50 px-4 py-3">
                <span>추론 시각</span>
                <span className="font-black text-slate-950">{formatInferenceTime(result.tradeDatetime)}</span>
              </div>
              <div className="flex items-center justify-between gap-3 rounded-xl bg-slate-50 px-4 py-3">
                <span>모델 버전</span>
                <span className="font-black text-slate-950">{result.modelVersion ?? "-"}</span>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
                <div>
                  <p className="text-xs font-black uppercase text-[#5267ff]">Inference Insight</p>
                  <h2 className="mt-1 text-2xl font-black text-slate-950">추론 결과 분석 리포트</h2>
                </div>
                <span className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-black text-emerald-700">실제 응답</span>
              </div>

              <div className="text-base font-normal text-slate-950">
                모델 최종 판단: {result.signal} | 확신도: {result.confidence}%
              </div>

              {reportCards.slice(0, 1).map((card) => (
                <article key={card.title} className={`rounded-2xl border p-6 shadow-[0_12px_32px_rgba(15,23,42,0.04)] ${cardToneClass(card.tone)}`}>
                  <p className={`text-xs font-black uppercase ${cardEyebrowClass(card.tone)}`}>{card.eyebrow}</p>
                  <h3 className="mt-3 text-2xl font-black text-slate-950">{card.title}</h3>
                  <div className="mt-5 space-y-2 text-sm font-bold leading-7 text-slate-600">
                    {card.body.map((paragraph, paragraphIndex) => (
                      <p key={`${card.title}-${paragraphIndex}`}>{paragraph}</p>
                    ))}
                  </div>
                </article>
              ))}

              <div className="grid gap-4 lg:grid-cols-2">
                {reportCards.slice(1).map((card, index) => (
                  <article key={card.title} className={`rounded-2xl border p-5 shadow-[0_12px_32px_rgba(15,23,42,0.04)] ${cardToneClass(card.tone)}`}>
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className={`text-xs font-black uppercase ${cardEyebrowClass(card.tone)}`}>{card.eyebrow}</p>
                        <h3 className="mt-2 text-lg font-black text-slate-950">{card.title}</h3>
                      </div>
                      <span className="rounded-full bg-slate-50 px-3 py-1 text-xs font-black text-slate-400">
                        {String(index + 3).padStart(2, "0")}
                      </span>
                    </div>
                    <div className="mt-4 space-y-2 text-sm font-bold leading-7 text-slate-600">
                      {card.body.map((paragraph, paragraphIndex) => (
                        <p key={`${card.title}-${paragraphIndex}`}>{paragraph}</p>
                      ))}
                    </div>
                  </article>
                ))}
              </div>
            </div>
          </div>
        )}
      </section>

      {pendingCart ? (
        <CartConfirmModal
          stockName={pendingCart.name}
          mode={isInCart(pendingCart.code) ? "remove" : "add"}
          onConfirm={() => {
            toggleCart(pendingCart.code, pendingCart);
            setPendingCart(null);
          }}
          onCancel={() => setPendingCart(null)}
        />
      ) : null}
    </>
  );
}

export default function PredictPage() {
  return (
    <Suspense fallback={null}>
      <PredictContent />
    </Suspense>
  );
}
