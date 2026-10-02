"use client";

import { useCallback, useEffect, useState } from "react";
import { FiActivity, FiCpu, FiPlay, FiRefreshCw, FiSquare, FiZap } from "react-icons/fi";
import PortfolioAside from "@/components/portfolio/PortfolioAside";
import PortfolioStockTradeModal from "@/components/portfolio/PortfolioStockTradeModal";
import PortfolioSummaryCards from "@/components/portfolio/PortfolioSummaryCards";
import PortfolioTable from "@/components/portfolio/PortfolioTable";
import { StockListItem, useStockList } from "@/lib/stock-list/StockListContext";
import { apiFetch } from "@/lib/signup/auth";
import { isOnceJobDone } from "@/lib/ai/predictionJobStore";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type TradeStatus = { status: "RUNNING" | "STOPPED" | string };

type TradeLog = {
  ticker: string;
  side: string;
  qty: number;
  price: number;
  amount: number;
  ai_signal: string;
  ai_confidence: number;
  strategy_id: string;
  status?: string;
  created_at: string;
};

type AutoTradeDecision = {
  ticker: string;
  action: string;
  reason: string;
  detail: string;
  ai_signal: string;
  ai_confidence: number;
  strategy_id: string;
  price: number;
  created_at: string;
};

type BackendPrediction = {
  ticker?: string;
  signal?: string;
  confidence?: number;
  prob_buy?: number;
  prob_hold?: number;
  prob_sell?: number;
  trade_datetime?: string;
  model_version?: string;
  stale?: boolean;
};

type PortfolioPrediction = {
  ticker: string;
  name: string;
  signal: string;
  confidence: number;
  probBuy: number;
  probHold: number;
  probSell: number;
  tradeDatetime?: string;
  modelVersion?: string;
  stale?: boolean;
  hasResult: boolean;
  error?: string;
};

type TradeActionPayload = {
  message?: string;
  detail?: string;
  job_id?: string;
  tickers?: string[];
  excluded_manual_tickers?: string[];
  excluded_unsupported_tickers?: string[];
};

type MarketStatus = {
  label: string;
  isOpen: boolean;
};

function getKoreanMarketStatus(date = new Date()): MarketStatus {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "Asia/Seoul",
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).formatToParts(date);
  const getPart = (type: Intl.DateTimeFormatPartTypes) =>
    parts.find((part) => part.type === type)?.value ?? "";
  const weekday = getPart("weekday");
  const hour = Number(getPart("hour"));
  const minute = Number(getPart("minute"));
  const totalMinutes = hour * 60 + minute;
  const isWeekend = weekday === "Sat" || weekday === "Sun";

  if (isWeekend) return { label: "주말 휴장", isOpen: false };
  if (totalMinutes < 9 * 60) return { label: "장 개장 전", isOpen: false };
  if (totalMinutes <= 15 * 60 + 30) return { label: "장중", isOpen: true };
  return { label: "장 마감", isOpen: false };
}

function parseWon(value: string) {
  const parsed = Number(value.replace(/[^0-9.-]/g, ""));
  return Number.isFinite(parsed) ? parsed : 0;
}

function formatWon(value: number) {
  return value.toLocaleString("ko-KR") + "원";
}

function formatDate(value?: string) {
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

function toPercent(value: unknown) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return numeric <= 1 ? numeric * 100 : numeric;
}

function formatConfidence(value: number) {
  return `${toPercent(value).toFixed(1)}%`;
}

function signalLabel(signal?: string) {
  const normalized = (signal ?? "").toUpperCase();
  if (normalized === "BUY") return "매수";
  if (normalized === "SELL") return "매도";
  if (normalized === "HOLD") return "관망";
  return signal || "대기";
}

function signalClass(signal: string) {
  if (signal === "매수") return "bg-rose-50 text-rose-600 border-rose-100";
  if (signal === "매도") return "bg-blue-50 text-blue-600 border-blue-100";
  if (signal === "관망") return "bg-amber-50 text-amber-600 border-amber-100";
  return "bg-slate-50 text-slate-500 border-slate-100";
}

function decisionActionClass(action: string) {
  if (action === "ORDER_SUBMITTED") return "bg-emerald-50 text-emerald-700 border-emerald-100";
  if (action === "SKIP") return "bg-rose-50 text-rose-600 border-rose-100";
  return "bg-amber-50 text-amber-700 border-amber-100";
}

function decisionActionLabel(action: string) {
  if (action === "ORDER_SUBMITTED") return "주문 시도";
  if (action === "SKIP") return "스킵";
  return "보류";
}

function decisionReasonLabel(reason: string) {
  const labels: Record<string, string> = {
    LOW_CONFIDENCE: "확신도 부족",
    STRATEGY_CONDITION_NOT_MET: "전략 조건 미충족",
    AI_STRATEGY_MISMATCH: "AI·전략 불일치",
    NO_MARKET_PRICE: "시세 없음",
    NO_POSITION_TO_SELL: "매도 보유수량 없음",
    NO_BUY_ALLOCATION: "매수 배분 없음",
    ZERO_ORDER_QUANTITY: "주문 수량 0",
    FILLED: "주문 체결",
    FAILED: "주문 실패",
  };
  return labels[reason] ?? reason;
}

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export default function PortfolioPage() {
  const { cartStocks } = useStockList();
  const [tradeStatus, setTradeStatus] = useState<TradeStatus | null>(null);
  const [tradeLogs, setTradeLogs] = useState<TradeLog[]>([]);
  const [tradeDecisions, setTradeDecisions] = useState<AutoTradeDecision[]>([]);
  const [portfolioPredictions, setPortfolioPredictions] = useState<PortfolioPrediction[]>([]);
  const [isLoadingTrade, setIsLoadingTrade] = useState(false);
  const [isLoadingPredictions, setIsLoadingPredictions] = useState(false);
  const [tradeMessage, setTradeMessage] = useState<string | null>(null);
  const [tradeError, setTradeError] = useState<string | null>(null);
  const [marketStatus, setMarketStatus] = useState<MarketStatus>(() => getKoreanMarketStatus());
  const [selectedStock, setSelectedStock] = useState<StockListItem | null>(null);
  const [autoTradeEnabled, setAutoTradeEnabled] = useState(false);

  const totalValue = cartStocks.reduce(
    (sum, stock) => sum + parseWon(stock.price),
    0,
  );

  const loadPortfolioPredictions = useCallback(async () => {
    if (cartStocks.length === 0) {
      setPortfolioPredictions([]);
      return false;
    }

    setIsLoadingPredictions(true);
    try {
      const results = await Promise.all(
        cartStocks.map(async (stock) => {
          try {
            const res = await apiFetch(`${API_BASE}/ai/predictions/${encodeURIComponent(stock.code)}`);
            if (res.status === 404) {
              return {
                ticker: stock.code,
                name: stock.name,
                signal: "대기",
                confidence: 0,
                probBuy: 0,
                probHold: 0,
                probSell: 0,
                hasResult: false,
                error: "AI 결과 대기중",
              } satisfies PortfolioPrediction;
            }
            if (!res.ok) {
              const payload = await res.json().catch(() => ({})) as { detail?: string; message?: string };
              throw new Error(payload.detail || payload.message || `AI 결과 조회 실패 (${res.status})`);
            }

            const data = await res.json() as BackendPrediction;
            return {
              ticker: data.ticker || stock.code,
              name: stock.name,
              signal: signalLabel(data.signal),
              confidence: toPercent(data.confidence),
              probBuy: toPercent(data.prob_buy),
              probHold: toPercent(data.prob_hold),
              probSell: toPercent(data.prob_sell),
              tradeDatetime: data.trade_datetime,
              modelVersion: data.model_version,
              stale: data.stale,
              hasResult: true,
            } satisfies PortfolioPrediction;
          } catch (error) {
            return {
              ticker: stock.code,
              name: stock.name,
              signal: "대기",
              confidence: 0,
              probBuy: 0,
              probHold: 0,
              probSell: 0,
              hasResult: false,
              error: error instanceof Error ? error.message : "AI 결과 조회 실패",
            } satisfies PortfolioPrediction;
          }
        }),
      );

      setPortfolioPredictions(results);
      return results.some((result) => result.hasResult);
    } finally {
      setIsLoadingPredictions(false);
    }
  }, [cartStocks]);

  const loadTradeState = useCallback(async () => {
    try {
      const [statusRes, historyRes, decisionsRes] = await Promise.all([
        apiFetch(`${API_BASE}/trade/status`),
        apiFetch(`${API_BASE}/trade/history`),
        apiFetch(`${API_BASE}/trade/decisions`),
      ]);

      if (statusRes.ok) {
        const status = (await statusRes.json()) as TradeStatus;
        setTradeStatus(status);
        if (status.status === "RUNNING") {
          setAutoTradeEnabled(true);
        }
      }

      if (historyRes.ok) {
        setTradeLogs((await historyRes.json()) as TradeLog[]);
      }

      if (decisionsRes.ok) {
        setTradeDecisions((await decisionsRes.json()) as AutoTradeDecision[]);
      }

      // 셋 중 하나라도 비정상 응답(401/500 등)이면 화면은 이전 상태를 그대로 보여주면서 아무 알림도
      // 없이 넘어갔다 -- 사용자는 "자동매매 상태를 못 불러왔다"는 사실 자체를 알 수 없었다(2026-10-02 통합 감사).
      if (!statusRes.ok || !historyRes.ok || !decisionsRes.ok) {
        console.error("자동매매 상태 조회 실패", statusRes.status, historyRes.status, decisionsRes.status);
        setTradeError("자동매매 상태를 최신으로 불러오지 못했습니다. 화면이 실제와 다를 수 있습니다.");
      }
    } catch (error) {
      console.error(error);
      setTradeError("자동매매 상태를 불러오지 못했습니다.");
    }
  }, []);

  useEffect(() => {
    void loadTradeState();
  }, [loadTradeState]);

  useEffect(() => {
    if (!autoTradeEnabled && tradeStatus?.status !== "RUNNING") return;

    const timer = window.setInterval(() => {
      void loadTradeState();
    }, 15_000);

    return () => window.clearInterval(timer);
  }, [autoTradeEnabled, tradeStatus?.status, loadTradeState]);

  useEffect(() => {
    void loadPortfolioPredictions();
  }, [loadPortfolioPredictions]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setMarketStatus(getKoreanMarketStatus());
    }, 60_000);
    return () => window.clearInterval(timer);
  }, []);

  const refreshAll = async () => {
    await Promise.all([loadTradeState(), loadPortfolioPredictions()]);
  };

  const runTradeAction = async (action: "start" | "stop" | "once") => {
    const currentMarketStatus = getKoreanMarketStatus();
    setMarketStatus(currentMarketStatus);

    if ((action === "start" || action === "once") && !autoTradeEnabled) {
      setTradeMessage(null);
      setTradeError("자동매매가 OFF 상태입니다. 자동매매를 켠 뒤 다시 시도해주세요.");
      return;
    }

    if ((action === "start" || action === "once") && !currentMarketStatus.isOpen) {
      setTradeMessage(null);
      setTradeError(`현재 ${currentMarketStatus.label} 상태라 자동매매를 시작할 수 없습니다. 국내 정규장(09:00-15:30)에 다시 시도해주세요.`);
      return;
    }

    setIsLoadingTrade(true);
    setTradeError(null);
    setTradeMessage(null);

    try {
      const res = await apiFetch(`${API_BASE}/trade/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          // total_capital 은 보내지 않는다: 바구니 종목의 '1주 가격 합'은 투자 원금이 아니고, 새로고침 후에는
          // 가격이 비어 값이 사라져 서버가 계좌 전체 예수금을 쓰게 되어 결과가 달라졌다. 서버가 항상 KIS 예수금을 조회한다.
          basket: cartStocks.map((stock) => ({
            ticker: stock.code,
            ticker_name: stock.name,
          })),
          // 전략을 고르는 UI가 아직 없어서 종목별로 보낼 실제 전략 ID가 없다. "rsi_reversal" 같은
          // 가짜 placeholder를 보내면 백엔드 STRATEGY_CONFIG에 없는 값이라 매번 조용히 conservative로
          // 대체돼 사용자가 그 사실을 알 길이 없었다(2026-10-02 통합 감사). ticker_strategies를 비워
          // 보내면 백엔드가 자기 기본 전략(ultra_safe)을 그대로 쓴다.
          ticker_strategies: [],
        }),
      });

      const payload = await res.json().catch(() => ({})) as TradeActionPayload;
      if (!res.ok) {
        throw new Error(payload.detail || payload.message || `자동매매 요청 실패 (${res.status})`);
      }

      if (action === "once") {
        setTradeMessage("AI 서버에 1회 분석 요청을 보냈습니다. 결과 콜백을 기다리는 중입니다.");
        await loadTradeState();

        let hasPrediction = false;
        for (let attempt = 0; attempt < 20; attempt += 1) {
          await delay(3000);
          // job_id로 ONCE 콜백 도착(GET /ai/once/{job_id} 200)을 먼저 확인한다 -- 그냥
          // /ai/predictions/{ticker}를 바로 조회하면 콜백 도착 전 직전 5분 주기 push나 과거 ONCE
          // 결과를 "새로 받은 결과"로 오인할 수 있다(lib/ai/predictionJobStore.ts와 같은 문제,
          // 2026-10-02 통합 감사).
          if (payload.job_id && !(await isOnceJobDone(payload.job_id))) continue;
          hasPrediction = await loadPortfolioPredictions();
          if (hasPrediction) break;
        }

        setTradeMessage(
          hasPrediction
            ? "AI 분석 결과를 수신했습니다. 아래 최신 AI 판단을 확인하세요."
            : "AI 분석 요청은 전달됐지만 아직 결과가 도착하지 않았습니다. 잠시 후 새로고침을 눌러 확인하세요.",
        );
      } else {
        setTradeMessage(payload.message || "자동매매 요청이 처리됐습니다.");
        await refreshAll();
      }
    } catch (error) {
      setTradeError(error instanceof Error ? error.message : "자동매매 요청 중 오류가 발생했습니다.");
    } finally {
      setIsLoadingTrade(false);
    }
  };

  const isRunning = tradeStatus?.status === "RUNNING";
  const canRequestTrade = autoTradeEnabled && marketStatus.isOpen && cartStocks.length > 0;

  const handleAutoTradeToggle = () => {
    const nextEnabled = !autoTradeEnabled;
    setAutoTradeEnabled(nextEnabled);
    setTradeMessage(null);
    setTradeError(null);

    if (!nextEnabled && isRunning) {
      void runTradeAction("stop");
    }
  };

  return (
    <div className="grid gap-6 xl:grid-cols-[1fr_320px]">
      <section className="space-y-6">
        <div className="rounded-2xl bg-white p-6 shadow-[0_24px_80px_rgba(15,23,42,0.06)] lg:p-8">
          <div className="flex flex-wrap items-start justify-between gap-5">
            <div>
              <p className="text-sm font-bold text-[#5267ff]">My Portfolio</p>
              <h1 className="mt-2 text-4xl font-black text-slate-950">
                내 주식보기
              </h1>
            </div>
            <span className="rounded-full bg-slate-100 px-4 py-2 text-xs font-black text-slate-500">
              백엔드 바구니 연동
            </span>
          </div>

          <PortfolioSummaryCards
            totalValue={formatWon(totalValue)}
            stockCount={cartStocks.length}
          />
          <PortfolioTable stocks={cartStocks} onSelectStock={setSelectedStock} />
        </div>

        <div className="rounded-2xl bg-white p-6 shadow-[0_24px_80px_rgba(15,23,42,0.06)] lg:p-8">
          <div className="border-b border-slate-100 pb-5">
            <div className="min-w-0">
              <p className="text-sm font-bold text-[#5267ff]">AI Auto Trading</p>
              <div className="mt-2 flex flex-wrap items-center justify-between gap-3">
                <div className="flex flex-wrap items-center gap-2">
                  <h2 className="text-2xl font-black text-slate-950">실시간 AI 자동 거래</h2>
                  <div className="flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1.5 text-xs font-black text-slate-500">
                    <span className={`h-2.5 w-2.5 rounded-full ${isRunning ? "bg-emerald-400" : "bg-slate-300"}`} />
                    {isRunning ? "실행 중" : "중지됨"}
                  </div>
                </div>
                <div className="ml-auto flex shrink-0 items-center gap-2">
                  <span className="text-xs font-black text-slate-500">자동매매</span>
                  <button
                    type="button"
                    onClick={handleAutoTradeToggle}
                    aria-label="자동매매 켜기 끄기"
                    aria-pressed={autoTradeEnabled}
                    className={`relative h-7 w-12 rounded-full p-1 transition ${autoTradeEnabled ? "bg-[#5267ff]" : "bg-slate-300"}`}
                  >
                    <span className={`block h-5 w-5 rounded-full bg-white shadow-sm transition ${autoTradeEnabled ? "translate-x-5" : ""}`} />
                  </button>
                  <span className={`w-6 text-left text-xs font-black ${autoTradeEnabled ? "text-[#5267ff]" : "text-slate-400"}`}>
                    {autoTradeEnabled ? "ON" : "OFF"}
                  </span>
                </div>
              </div>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                포트폴리오 종목을 기준으로 백엔드가 AI 서버에 START/ONCE 커맨드를 전달하고, 수신된 추론 결과와 전략 조건으로 주문을 판단합니다.
              </p>
            </div>
          </div>

          {autoTradeEnabled ? (
            <>
              <div className="mt-5 flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => void runTradeAction("start")}
                  disabled={isLoadingTrade || !canRequestTrade}
                  className="inline-flex h-11 items-center gap-2 rounded-xl bg-[#5267ff] px-4 text-sm font-black text-white transition hover:bg-[#4054e8] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <FiPlay className="h-4 w-4" /> 자동매매 시작
                </button>
                <button
                  type="button"
                  onClick={() => void runTradeAction("once")}
                  disabled={isLoadingTrade || !canRequestTrade}
                  className="inline-flex h-11 items-center gap-2 rounded-xl border border-slate-200 px-4 text-sm font-black text-slate-600 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <FiZap className="h-4 w-4" /> {isLoadingTrade ? "AI 분석 대기" : "1회 분석/실행"}
                </button>
                <button
                  type="button"
                  onClick={() => void runTradeAction("stop")}
                  disabled={isLoadingTrade || !isRunning}
                  className="inline-flex h-11 items-center gap-2 rounded-xl border border-rose-100 bg-rose-50 px-4 text-sm font-black text-rose-600 transition hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <FiSquare className="h-4 w-4" /> 중지
                </button>
                <button
                  type="button"
                  onClick={() => void refreshAll()}
                  disabled={isLoadingTrade || isLoadingPredictions}
                  className="inline-flex h-11 items-center gap-2 rounded-xl border border-slate-200 px-4 text-sm font-black text-slate-500 transition hover:bg-slate-50 disabled:opacity-50"
                >
                  <FiRefreshCw className={`h-4 w-4 ${isLoadingPredictions ? "animate-spin" : ""}`} /> 새로고침
                </button>
              </div>

              {cartStocks.length === 0 ? (
                <div className="mt-5 rounded-2xl border border-amber-100 bg-amber-50 px-4 py-3 text-sm font-bold text-amber-700">
                  자동매매를 시작하려면 AI 리포트나 종목 랭킹에서 포트폴리오에 종목을 먼저 담아주세요.
                </div>
              ) : null}
              {!marketStatus.isOpen ? (
                <div className="mt-5 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-bold text-slate-600">
                  현재 {marketStatus.label} 상태입니다. 자동매매 시작과 1회 분석/실행은 국내 정규장(09:00-15:30)에만 가능합니다.
                </div>
              ) : null}
              {tradeMessage ? (
                <div className="mt-5 rounded-2xl border border-emerald-100 bg-emerald-50 px-4 py-3 text-sm font-bold text-emerald-700">
                  {tradeMessage}
                </div>
              ) : null}
              {tradeError ? (
                <div className="mt-5 rounded-2xl border border-rose-100 bg-rose-50 px-4 py-3 text-sm font-bold text-rose-600">
                  {tradeError}
                </div>
              ) : null}

              <div className="mt-6 overflow-hidden rounded-xl border border-slate-100">
                <div className="flex items-center justify-between gap-3 bg-slate-50 px-5 py-3 text-sm font-black text-slate-700">
                  <span className="inline-flex items-center gap-2">
                    <FiCpu className="h-4 w-4 text-[#5267ff]" /> 최신 AI 판단
                  </span>
                  {isLoadingPredictions ? (
                    <span className="text-xs text-slate-400">조회 중</span>
                  ) : null}
                </div>
                {portfolioPredictions.length > 0 ? (
                  <div className="divide-y divide-slate-100">
                    {portfolioPredictions.map((prediction) => (
                      <div key={prediction.ticker} className="grid gap-4 px-5 py-4 text-sm lg:grid-cols-[1.2fr_0.8fr_1.3fr] lg:items-center">
                            <div>
                              <p className="font-black text-slate-950">{prediction.name}</p>
                              <p className="mt-0.5 text-xs font-bold text-slate-400">
                                {prediction.ticker} · {prediction.hasResult ? formatDate(prediction.tradeDatetime) : prediction.error || "결과 대기중"}
                              </p>
                            </div>
                            <div className="flex flex-wrap items-center gap-2">
                              <span className={`rounded-full border px-3 py-1 text-xs font-black ${signalClass(prediction.signal)}`}>
                                {prediction.signal}
                              </span>
                              <span className="text-xs font-black text-slate-500">
                                확신도 {prediction.hasResult ? `${prediction.confidence.toFixed(1)}%` : "-"}
                              </span>
                              {prediction.stale ? (
                                <span className="rounded-full border border-amber-300 bg-amber-50 px-3 py-1 text-xs font-black text-amber-600">
                                  오래된 신호
                                </span>
                              ) : null}
                            </div>
                            <div className="grid grid-cols-3 gap-2 text-xs font-black">
                              <div className="rounded-lg bg-rose-50 px-3 py-2 text-rose-600">매수 {prediction.probBuy.toFixed(1)}%</div>
                              <div className="rounded-lg bg-amber-50 px-3 py-2 text-amber-600">관망 {prediction.probHold.toFixed(1)}%</div>
                              <div className="rounded-lg bg-blue-50 px-3 py-2 text-blue-600">매도 {prediction.probSell.toFixed(1)}%</div>
                            </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="px-5 py-12 text-center text-sm font-bold text-slate-400">
                    포트폴리오 종목의 AI 판단 결과가 아직 없습니다.
                  </div>
                )}
              </div>

              <div className="mt-6 overflow-hidden rounded-xl border border-slate-100">
                <div className="flex items-center gap-2 bg-slate-50 px-5 py-3 text-sm font-black text-slate-700">
                  <FiCpu className="h-4 w-4 text-[#5267ff]" /> 최근 자동매매 판단 기록
                </div>
                {tradeDecisions.length > 0 ? (
                  <div className="divide-y divide-slate-100">
                    {tradeDecisions.slice(0, 10).map((decision, index) => (
                      <div key={`${decision.created_at}-${decision.ticker}-${index}`} className="grid gap-3 px-5 py-4 text-sm md:grid-cols-[1fr_auto_auto] md:items-center">
                        <div>
                          <div className="flex flex-wrap items-center gap-2">
                            <p className="font-black text-slate-950">{decision.ticker}</p>
                            <span className={`rounded-full border px-2.5 py-1 text-xs font-black ${decisionActionClass(decision.action)}`}>
                              {decisionActionLabel(decision.action)}
                            </span>
                          </div>
                          <p className="mt-1 text-xs font-bold text-slate-400">{formatDate(decision.created_at)} · {decision.strategy_id || "-"}</p>
                          <p className="mt-1 text-xs font-bold text-slate-600">{decisionReasonLabel(decision.reason)}</p>
                          {decision.detail ? <p className="mt-1 text-xs leading-5 text-slate-500">{decision.detail}</p> : null}
                        </div>
                        <p className="font-bold text-slate-600">{formatWon(decision.price)}</p>
                        <p className="text-right text-xs font-bold text-slate-400">AI {decision.ai_signal || "-"} {formatConfidence(decision.ai_confidence)}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="px-5 py-10 text-center text-sm font-bold text-slate-400">
                    아직 자동매매 판단 기록이 없습니다. 다음 AI 실시간 판단 이후 HOLD/SKIP 사유가 여기에 표시됩니다.
                  </div>
                )}
              </div>

              <div className="mt-6 overflow-hidden rounded-xl border border-slate-100">
                <div className="flex items-center gap-2 bg-slate-50 px-5 py-3 text-sm font-black text-slate-700">
                  <FiActivity className="h-4 w-4 text-[#5267ff]" /> 최근 자동매매 주문 기록
                </div>
                {tradeLogs.length > 0 ? (
                  <div className="divide-y divide-slate-100">
                    {tradeLogs.slice(0, 8).map((log, index) => (
                      <div key={`${log.created_at}-${log.ticker}-${index}`} className="grid gap-3 px-5 py-4 text-sm md:grid-cols-[1fr_auto_auto_auto] md:items-center">
                            <div>
                              <p className="font-black text-slate-950">
                                {log.ticker}
                                {log.status === "FAILED" ? (
                                  <span className="ml-2 rounded-full border border-rose-300 bg-rose-50 px-2 py-0.5 text-xs font-black text-rose-600">
                                    실패
                                  </span>
                                ) : null}
                              </p>
                              <p className="mt-0.5 text-xs font-bold text-slate-400">{formatDate(log.created_at)} · {log.strategy_id}</p>
                            </div>
                            <p className={`font-black ${log.status === "FAILED" ? "text-slate-400 line-through" : log.side === "BUY" ? "text-rose-500" : "text-blue-500"}`}>{log.side}</p>
                            <p className="font-bold text-slate-600">{log.qty.toLocaleString("ko-KR")}주 · {formatWon(log.price)}</p>
                            <p className="text-right text-xs font-bold text-slate-400">AI {log.ai_signal} {formatConfidence(log.ai_confidence)}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="px-5 py-12 text-center text-sm font-bold text-slate-400">
                    아직 백엔드에 저장된 주문 기록이 없습니다. AI 판단은 위 최신 AI 판단 카드에 표시되고, 전략 조건과 주문 조건이 맞아 실제 주문을 시도할 때 이 영역에 기록됩니다.
                  </div>
                )}
              </div>
            </>
          ) : null}
        </div>
      </section>

      <PortfolioAside stocks={cartStocks} />

      {selectedStock ? (
        <PortfolioStockTradeModal
          stock={selectedStock}
          onClose={() => setSelectedStock(null)}
        />
      ) : null}
    </div>
  );
}
