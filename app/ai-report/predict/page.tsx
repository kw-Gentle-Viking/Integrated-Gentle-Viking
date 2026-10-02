"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { FiAlertCircle, FiArrowLeft, FiCpu, FiRefreshCw, FiShoppingBag } from "react-icons/fi";
import { apiFetch } from "@/lib/signup/auth";
import { startPredictionJob, useEnsurePredictionJob, usePredictionJob } from "@/lib/ai/predictionJobStore";
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

type AgreementAnalysis = {
  status: string;
  recommendation_signal: string;
  tft_signal: string;
  alignment_label: string;
  alignment_level: "aligned" | "partial" | "diverged" | string;
  summary: string;
  interpretation: string;
  action_note: string;
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

function normalizeSignalCode(value: string) {
  const upper = value.toUpperCase();
  if (upper.includes("BUY") || value.includes("매수") || value.includes("비중확대")) return "BUY";
  if (upper.includes("SELL") || value.includes("매도") || value.includes("제외")) return "SELL";
  if (upper.includes("HOLD") || value.includes("관망") || value.includes("보류")) return "HOLD";
  return "HOLD";
}

function agreementTone(level?: string) {
  if (level === "aligned") return "border-emerald-100 bg-emerald-50 text-emerald-700";
  if (level === "partial") return "border-amber-100 bg-amber-50 text-amber-700";
  return "border-rose-100 bg-rose-50 text-rose-700";
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
  const recommendationSignal = searchParams.get("recommendationSignal") || "";
  const recommendationSummary = searchParams.get("recommendationSummary") || "";
  const recommendationReasons = searchParams.get("recommendationReasons") || "";
  const stockListItem = makeStockListItem(stock);
  const { toggleCart, isInCart } = useStockList();
  const { result, isLoading, errorMessage, requestMessage } = usePredictionJob(code);
  const [pendingCart, setPendingCart] = useState<StockListItem | null>(null);
  const [agreement, setAgreement] = useState<AgreementAnalysis | null>(null);
  const [isAgreementLoading, setIsAgreementLoading] = useState(false);

  useEnsurePredictionJob(code, stock.name);

  const requestAgreement = useCallback(async (predictionResult: PredictionResult) => {
    if (!recommendationSignal) {
      setAgreement({
        status: "missing_recommendation_context",
        recommendation_signal: "-",
        tft_signal: predictionResult.signal,
        alignment_label: "추천 맥락 없음",
        alignment_level: "partial",
        summary: "추천 리포트에서 진입한 경우 합치성 분석이 표시됩니다.",
        interpretation: "현재 페이지 URL에 추천 리포트의 판단과 근거가 포함되어 있지 않아 TFT 예측과 비교할 수 없습니다.",
        action_note: "AI 추천 카드의 'AI 매매 확률 보기' 버튼으로 진입하면 추천 모델과 TFT 모델의 의견 차이를 함께 확인할 수 있습니다.",
      });
      return;
    }

    setIsAgreementLoading(true);
    try {
      const res = await apiFetch(`${API_BASE}/ai/agreement`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ticker: stock.code,
          name: stock.name,
          recommendation_signal: recommendationSignal,
          recommendation_summary: recommendationSummary,
          recommendation_reasons: recommendationReasons,
          tft_signal: normalizeSignalCode(predictionResult.signal),
          confidence: predictionResult.confidence,
          prob_buy: predictionResult.prediction.buy,
          prob_hold: predictionResult.prediction.hold,
          prob_sell: predictionResult.prediction.sell,
        }),
      });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(payload.detail || `합치성 분석 실패 (${res.status})`);
      setAgreement(payload as AgreementAnalysis);
    } catch (error) {
      const tftSignal = predictionResult.signal;
      setAgreement({
        status: "fallback_frontend",
        recommendation_signal: recommendationSignal,
        tft_signal: tftSignal,
        alignment_label: "분석 대기",
        alignment_level: "partial",
        summary: "추천 리포트와 TFT 예측의 관점 차이를 확인하는 중입니다.",
        interpretation: error instanceof Error ? error.message : "합치성 분석을 불러오지 못했습니다.",
        action_note: "잠시 후 새로고침하면 Gemini 해석이 다시 요청됩니다.",
      });
    } finally {
      setIsAgreementLoading(false);
    }
  }, [recommendationReasons, recommendationSignal, recommendationSummary, stock.code, stock.name]);

  useEffect(() => {
    if (!result) {
      setAgreement(null);
      return;
    }

    void requestAgreement(result);
  }, [requestAgreement, result]);

  const requestPrediction = useCallback(() => {
    setAgreement(null);
    void startPredictionJob(code, stock.name, { force: true });
  }, [code, stock.name]);

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
            <h1 className="mt-2 text-4xl font-black text-slate-950">{stock.name} AI 매매 확률 예측</h1>
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
                {/* modelVersion은 AI 서버가 실제로 이 종목을 push했을 때만 내려온다 -- 종목이 AI
                   유니버스(코스피 200종목) 밖이면 백엔드가 confidence 0 HOLD(또는 설정에 따라 무작위
                   신호)로 채워 보내는데, 이 배지가 조건 없이 항상 "실제 응답"이라 둘을 구분할 수
                   없었다(2026-10-02 통합 감사). */}
                {result.modelVersion ? (
                  <span className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-black text-emerald-700">실제 응답</span>
                ) : (
                  <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-black text-amber-700">
                    AI 미응답 (해당 종목 추론 결과 없음)
                  </span>
                )}
              </div>

              <div className="text-base font-normal text-slate-950">
                모델 최종 판단: {result.signal} | 확신도: {result.confidence}%
              </div>

              <div className="flex flex-wrap items-center gap-2 text-base font-normal text-slate-950">
                {isAgreementLoading && !agreement ? (
                  <span className="text-slate-500">추천 결과: 분석 중 | 모델 추론 결과: {result.signal}</span>
                ) : agreement ? (
                  <span>
                    추천 결과: {agreement.recommendation_signal} | 모델 추론 결과: {agreement.tft_signal}
                  </span>
                ) : (
                  <span>추천 결과: - | 모델 추론 결과: {result.signal}</span>
                )}
                <span className={`rounded-full border px-3 py-1 text-xs font-black ${agreementTone(agreement?.alignment_level)}`}>
                  {isAgreementLoading ? "분석 중" : agreement?.alignment_label ?? "분석 대기"}
                </span>
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
