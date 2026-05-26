"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { FiArrowLeft, FiCpu, FiRefreshCw } from "react-icons/fi";

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
  prediction: Prediction;
};

const defaultPrediction: Prediction = { buy: 45, hold: 40, sell: 15 };

const predictions: Record<string, StockPrediction> = {
  "005930": { name: "삼성전자", code: "005930", price: "81,200원", change: "+0.61%", prediction: { buy: 58, hold: 31, sell: 11 } },
  "000660": { name: "SK하이닉스", code: "000660", price: "163,000원", change: "-1.25%", prediction: { buy: 42, hold: 45, sell: 13 } },
  "035420": { name: "NAVER", code: "035420", price: "186,700원", change: "+2.10%", prediction: { buy: 49, hold: 39, sell: 12 } },
  "373220": { name: "LG에너지솔루션", code: "373220", price: "348,500원", change: "+2.08%", prediction: { buy: 37, hold: 46, sell: 17 } },
  "005380": { name: "현대차", code: "005380", price: "212,000원", change: "+0.84%", prediction: { buy: 33, hold: 52, sell: 15 } },
};

const predictionItems = [
  { key: "buy", label: "매수", color: "#f43f5e", desc: "상승 방향성과 거래량 확장 가능성" },
  { key: "hold", label: "관망", color: "#5267ff", desc: "추가 확인이 필요한 중립 구간" },
  { key: "sell", label: "매도", color: "#3b82f6", desc: "하방 위험 또는 과열 해소 가능성" },
] as const;

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
        style={{
          background: `conic-gradient(${color} ${animatedValue * 3.6}deg, #e5e7eb 0deg)`,
        }}
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
  const baseStock = predictions[code];
  const stock: StockPrediction = {
    name: searchParams.get("name") || baseStock?.name || "종목명 없음",
    code,
    price: searchParams.get("price") || baseStock?.price || "-",
    change: searchParams.get("change") || baseStock?.change || "-",
    prediction: baseStock?.prediction || defaultPrediction,
  };
  const [loadingCode, setLoadingCode] = useState(code);
  const [isRerunning, setIsRerunning] = useState(true);
  const dominant = predictionItems.reduce((best, item) =>
    stock.prediction[item.key] > stock.prediction[best.key] ? item : best,
  );
  const isLoading = isRerunning || loadingCode !== code;

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setLoadingCode(code);
      setIsRerunning(false);
    }, 1200);
    return () => window.clearTimeout(timer);
  }, [code]);

  const runInference = () => {
    setIsRerunning(true);
    window.setTimeout(() => setIsRerunning(false), 1200);
  };

  return (
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
            학습시킨 모델(Temporal Fusion Transformer) 가중치를 연결해 N일 후 매수, 관망, 매도 확률을 계산했습니다.
          </p>
        </div>
        <button
          type="button"
          onClick={runInference}
          className="flex h-11 items-center gap-2 rounded-full bg-slate-950 px-5 text-sm font-black text-white transition hover:bg-slate-800"
        >
          <FiRefreshCw className="h-4 w-4" />
          다시 예측
        </button>
      </div>

      {isLoading ? (
        <div className="mt-6 flex min-h-[420px] items-center justify-center rounded-2xl bg-[#f4f7ff] p-8">
          <div className="text-center">
            <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-white shadow-[0_18px_50px_rgba(82,103,255,0.18)]">
              <FiCpu className="h-9 w-9 animate-pulse text-[#5267ff]" />
            </div>
            <h2 className="mt-6 text-2xl font-black text-slate-950">AI 모델 추론 중</h2>
            <p className="mt-3 text-sm font-bold leading-6 text-slate-500">
              {stock.name}의 최근 가격 흐름과 리포트 근거를 모델 입력값으로 변환하고 있습니다.
            </p>
            <div className="mx-auto mt-6 h-2 w-72 overflow-hidden rounded-full bg-white">
              <div className="h-full w-1/2 animate-[loadingSlide_1.1s_ease-in-out_infinite] rounded-full bg-[#5267ff]" />
            </div>
          </div>
        </div>
      ) : (
        <div className="mt-6 grid gap-4 lg:grid-cols-[320px_1fr]">
          <aside className="rounded-2xl bg-slate-950 p-6 text-white">
            <FiCpu className="h-8 w-8 text-emerald-300" />
            <p className="mt-5 text-sm font-bold text-slate-300">{stock.code}</p>
            <h2 className="mt-1 text-3xl font-black">{stock.name}</h2>
            <p className="mt-4 text-2xl font-black">{stock.price}</p>
            <p className={`mt-1 text-sm font-black ${stock.change.startsWith("-") ? "text-blue-300" : "text-rose-300"}`}>
              {stock.change}
            </p>
            <div className="mt-6 rounded-2xl bg-white/10 p-4 ring-1 ring-white/10">
              <p className="text-xs font-bold text-slate-300">모델 판단</p>
              <p className="mt-2 text-xl font-black text-white">{dominant.label} 우위</p>
            </div>
          </aside>

          <div className="grid gap-4 md:grid-cols-3">
            {predictionItems.map((item) => (
              <ProbabilityRing
                key={item.key}
                label={item.label}
                value={stock.prediction[item.key]}
                color={item.color}
                desc={item.desc}
              />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

export default function PredictPage() {
  return (
    <Suspense fallback={null}>
      <PredictContent />
    </Suspense>
  );
}
