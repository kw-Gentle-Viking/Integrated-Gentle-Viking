"use client";

import { useEffect, useSyncExternalStore } from "react";
import { apiFetch } from "@/lib/signup/auth";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Prediction = {
  buy: number;
  hold: number;
  sell: number;
};

export type BackendPrediction = {
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

export type PredictionResult = {
  prediction: Prediction;
  signal: string;
  confidence: number;
  tradeDatetime?: string;
  modelVersion?: string;
  report: string;
};

type PredictionJob = {
  result: PredictionResult | null;
  isLoading: boolean;
  errorMessage: string | null;
  requestMessage: string | null;
  requestedAt: number | null;
};

const signalLabel: Record<string, string> = {
  BUY: "매수",
  HOLD: "관망",
  SELL: "매도",
};

const emptyJob: PredictionJob = {
  result: null,
  isLoading: false,
  errorMessage: null,
  requestMessage: null,
  requestedAt: null,
};

const jobs = new Map<string, PredictionJob>();
const running = new Map<string, Promise<void>>();
const listeners = new Set<() => void>();

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function toPercent(value: unknown) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  const percent = numeric <= 1 ? numeric * 100 : numeric;
  return Math.max(0, Math.min(100, Math.round(percent)));
}

export function normalizePrediction(data: BackendPrediction): PredictionResult {
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

function emit() {
  listeners.forEach((listener) => listener());
}

function getJob(code: string) {
  return jobs.get(code) ?? emptyJob;
}

function setJob(code: string, next: Partial<PredictionJob>) {
  jobs.set(code, { ...getJob(code), ...next });
  emit();
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

async function fetchPrediction(code: string) {
  const res = await apiFetch(`${API_BASE}/ai/predictions/${encodeURIComponent(code)}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`추론 결과 요청 실패 (${res.status})`);
  return normalizePrediction((await res.json()) as BackendPrediction);
}

export async function startPredictionJob(code: string, stockName: string, options: { force?: boolean } = {}) {
  if (running.has(code)) return running.get(code);
  const existing = getJob(code);
  if (existing.result && !options.force) return Promise.resolve();

  const task = (async () => {
    setJob(code, {
      isLoading: true,
      result: options.force ? null : existing.result,
      errorMessage: null,
      requestMessage: `${stockName} 실제 AI 추론을 시작하는 중입니다.`,
      requestedAt: Date.now(),
    });

    try {
      if (!options.force) {
        const cached = await fetchPrediction(code);
        if (cached) {
          setJob(code, {
            result: cached,
            isLoading: false,
            errorMessage: null,
            requestMessage: null,
          });
          return;
        }
      }

      const res = await apiFetch(`${API_BASE}/ai/predictions/request`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tickers: [code] }),
      });
      const payload = await res.json().catch(() => ({})) as { job_id?: string; detail?: string; message?: string };
      if (!res.ok) throw new Error(payload.detail || payload.message || `추론 요청 실패 (${res.status})`);

      setJob(code, {
        requestMessage: "AI 서버가 분석 중입니다. 다른 페이지로 이동해도 결과 대기를 계속합니다.",
      });

      for (let attempt = 0; attempt < 24; attempt += 1) {
        await delay(2500);
        const result = await fetchPrediction(code);
        if (result) {
          setJob(code, {
            result,
            isLoading: false,
            errorMessage: null,
            requestMessage: null,
          });
          return;
        }
      }

      setJob(code, {
        isLoading: false,
        errorMessage: "추론 요청은 전달됐지만 아직 AI 서버 콜백 결과가 도착하지 않았습니다. 새로고침을 눌러 다시 확인해주세요.",
      });
    } catch (error) {
      setJob(code, {
        result: options.force ? null : getJob(code).result,
        isLoading: false,
        errorMessage: error instanceof Error ? error.message : "추론 요청 중 오류가 발생했습니다.",
      });
    } finally {
      running.delete(code);
    }
  })();

  running.set(code, task);
  return task;
}

export function usePredictionJob(code: string) {
  return useSyncExternalStore(
    subscribe,
    () => getJob(code),
    () => getJob(code),
  );
}

export function useEnsurePredictionJob(code: string, stockName: string) {
  useEffect(() => {
    void startPredictionJob(code, stockName);
  }, [code, stockName]);
}
