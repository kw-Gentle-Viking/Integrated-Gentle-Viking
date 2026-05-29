"use client";

import { Suspense, useEffect, useState } from "react";
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

const previewPrediction: PredictionResult = {
  prediction: { buy: 58, hold: 31, sell: 11 },
  signal: "매수",
  confidence: 58,
  tradeDatetime: new Date().toISOString(),
  modelVersion: "tft-preview",
  report:
    "매수 확률이 가장 높게 산출된 예시입니다. 최근 가격 흐름과 거래량 확장 가능성이 상승 방향성을 지지하지만, 관망 확률도 31%로 남아 있어 신규 진입은 분할 접근이 적합합니다. 실제 Gemini 리포트가 수신되면 이 영역에 종목별 추론 근거, 리스크 요인, 대응 전략이 긴 문단 형태로 표시됩니다.",
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
      "Gemini 리포트가 아직 응답에 포함되지 않았습니다. 추론 확률은 정상 수신됐지만, 상세 분석 문구는 백엔드 리포트 필드가 연결되면 이 영역에 표시됩니다.",
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

function ProbabilityRing({ label, value, color, desc, isPreview = false }: { label: string; value: number; color: string; desc: string; isPreview?: boolean }) {
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
        style={{
          background: `conic-gradient(${color} ${animatedValue * 3.6}deg, #e5e7eb 0deg)`,
        }}
      >
        <div className="flex h-24 w-24 flex-col items-center justify-center rounded-full bg-white shadow-inner">
          <span className="text-3xl font-black text-slate-950">{value}%</span>
          <span className="text-xs font-black text-slate-400">{label}</span>
        </div>
      </div>
      <div className="mt-5 flex items-center justify-center gap-2">
        <p className="text-lg font-black text-slate-950">{label}</p>
        {isPreview ? (
          <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-black text-slate-400">예시</span>
        ) : null}
      </div>
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

  const displayResult = result ?? previewPrediction;
  const isPreview = Boolean(errorMessage && !result);
  const dominant = predictionItems.reduce((best, item) =>
    displayResult.prediction[item.key] > displayResult.prediction[best.key] ? item : best,
  );

  const fetchPrediction = async () => {
    setIsLoading(true);
    setErrorMessage(null);

    try {
      const res = await apiFetch(`${API_BASE}/ai/predictions/${encodeURIComponent(code)}`);
      if (res.status === 404) {
        setResult(null);
        setErrorMessage("아직 백엔드에 수신된 최신 추론 결과가 없습니다.");
        return;
      }
      if (!res.ok) {
        throw new Error("prediction request failed");
      }

      const data = (await res.json()) as BackendPrediction;
      setResult(normalizePrediction(data));
    } catch {
      setResult(null);
      setErrorMessage("추론 결과를 불러오지 못했습니다. 백엔드 연결 상태를 확인해주세요.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void fetchPrediction();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [code]);

  const runInference = () => {
    void fetchPrediction();
  };

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
            Temporal Fusion Transformer 추론 결과로 N일 후 매수, 관망, 매도 확률을 표시합니다.
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
            onClick={runInference}
            disabled={isLoading}
            className="flex h-11 items-center gap-2 rounded-full border border-slate-200 bg-slate-100 px-5 text-sm font-black text-slate-600 transition hover:border-slate-300 hover:bg-slate-200 hover:text-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <FiRefreshCw className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
            다시 조회
          </button>
          
        </div>
      </div>

      {isLoading ? (
        <div className="mt-6 flex min-h-[420px] items-center justify-center rounded-2xl bg-[#f4f7ff] p-8">
          <div className="text-center">
            <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-white shadow-[0_18px_50px_rgba(82,103,255,0.18)]">
              <FiCpu className="h-9 w-9 animate-pulse text-[#5267ff]" />
            </div>
            <h2 className="mt-6 text-2xl font-black text-slate-950">AI 추론 결과 조회 중</h2>
            <p className="mt-3 text-sm font-bold leading-6 text-slate-500">
              {stock.name}의 최신 추론 확률을 백엔드에서 불러오고 있습니다.
            </p>
            <div className="mx-auto mt-6 h-2 w-72 overflow-hidden rounded-full bg-white">
              <div className="h-full w-1/2 animate-[loadingSlide_1.1s_ease-in-out_infinite] rounded-full bg-[#5267ff]" />
            </div>
          </div>
        </div>
      ) : (
        <div className="mt-6 space-y-4">
          {isPreview ? (
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-amber-100 bg-amber-50 px-5 py-4">
              <div className="flex items-start gap-3">
                <FiAlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-amber-500" />
                <div>
                  <p className="text-sm font-black text-amber-900">추론 결과 수신 대기 중</p>
                  <p className="mt-1 text-xs font-bold leading-5 text-amber-700">
                    {errorMessage} 아래 카드는 실제 응답이 들어오면 같은 위치에 값이 교체되는 예시 화면입니다.
                  </p>
                </div>
              </div>
              
            </div>
          ) : null}

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
                value={displayResult.prediction[item.key]}
                color={item.color}
                desc={item.desc}
                isPreview={isPreview}
              />
            ))}
          </div>
        </div>

          <article className="rounded-2xl border border-slate-100 bg-slate-50 p-5 lg:p-6">
            <div className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-200/70 pb-4">
              <div>
                <p className="text-xs font-black uppercase text-[#5267ff]">Gemini Report</p>
                <h2 className="mt-1 text-2xl font-black text-slate-950">추론 결과 분석 리포트</h2>
              </div>
              {/* <span className={`rounded-full px-3 py-1 text-xs font-black ${isPreview ? "bg-amber-100 text-amber-700" : "bg-emerald-100 text-emerald-700"}`}>
                {isPreview ? "예시 미리보기" : "실제 응답"}
              </span> */}
            </div>
            <div className="mt-5 space-y-3 text-sm font-bold leading-7 text-slate-600">
              {(isPreview
                ? displayResult.report
                : displayResult.report || `백엔드 /ai/predictions/${stock.code} 응답의 Gemini 리포트가 이 영역에 표시됩니다.`
              )
                .split(/\n+/)
                .map((paragraph, index) => (
                  <p key={`${paragraph}-${index}`}>{paragraph}</p>
                ))}
            </div>
          </article>

          <div className="grid gap-3 rounded-2xl border border-slate-100 bg-white p-4 text-xs font-bold text-slate-500 shadow-[0_12px_32px_rgba(15,23,42,0.04)] md:grid-cols-3">
            <div className="flex items-center justify-between gap-3 rounded-xl bg-slate-50 px-4 py-3">
              <span>모델 판단</span>
              <span className="font-black text-slate-950">{displayResult.signal ?? dominant.label} 우위</span>
            </div>
            <div className="flex items-center justify-between gap-3 rounded-xl bg-slate-50 px-4 py-3">
              <span>추론 시각</span>
              <span className="font-black text-slate-950">{isPreview ? "수신 대기" : formatInferenceTime(displayResult.tradeDatetime)}</span>
            </div>
            <div className="flex items-center justify-between gap-3 rounded-xl bg-slate-50 px-4 py-3">
              <span>모델 버전</span>
              <span className="font-black text-slate-950">{displayResult.modelVersion ?? "-"}</span>
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
