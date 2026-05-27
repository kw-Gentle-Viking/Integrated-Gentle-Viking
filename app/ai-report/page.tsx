"use client";

import { useCallback, useEffect, useState } from "react";
import { FiRefreshCw, FiZap } from "react-icons/fi";
import { IoSparkles } from "react-icons/io5";
import ReportCard from "@/components/ai-report/ReportCard";
import ReportDetailModal from "@/components/ai-report/ReportDetailModal";
import type {
  GeminiReason,
  ReportSource,
  ReportStock,
} from "@/lib/ai-report/types";
import { AUTH_EVENT_NAME, apiFetch, getCurrentUser } from "@/lib/signup/auth";
import { StockListItem, useStockList } from "@/lib/stock-list/StockListContext";
import CartConfirmModal from "@/components/stock-list/CartConfirmModal";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const REPORT_CACHE_PREFIX = "gv-ai-report";
const REPORT_CACHE_VERSION = 4;
const DAILY_REPORT_HOUR = 8;
const DAILY_REPORT_MINUTE = 30;


type BackendReason = {
  title?: string;
  headline?: string;
  details?: string;
  sources?: ReportSource[];
  tags?: string[];
};

type BackendRecommendation = {
  rank?: number;
  ticker?: string;
  name?: string;
  recommendationScore?: number;
  currentPrice?: number;
  changeRate?: number;
  summary?: string;
  strategy?: string;
  reasons?: {
    news?: BackendReason;
    disclosure?: BackendReason;
    materialFlow?: BackendReason;
  };
};

type BackendReport = {
  recommendations?: BackendRecommendation[];
};


type CachedReport = {
  version?: number;
  slotKey: string;
  generatedAt: string;
  refreshedAt: string;
  stocks: ReportStock[];
};


const fallbackReportStocks: ReportStock[] = [
  {
    rank: 1,
    name: "삼성전자",
    code: "005930",
    logoText: "삼",
    price: "81,200원",
    change: "+0.61%",
    signal: "메모리 반등",
    score: 86,
    summary: "메모리 반등 흐름이 확인되며, 뉴스와 수급 근거가 동시에 점수를 끌어올렸습니다.",
    reasons: {
      news: {
        summary: "HBM 공급 확대 기대와 외국인 순매수 유입이 함께 확인됐습니다.",
        details:
          "메모리 가격 반등 기대, AI 서버 수요, 대형 반도체주 수급 회복 기사가 동시에 잡혔습니다. 추후 Gemini 리포트 JSON의 `reasons.news.summary/details/sources`를 그대로 연결할 수 있습니다.",
        sources: ["Market Brief", "반도체 섹터 뉴스"],
      },
      disclosure: {
        summary:
          "기업설명회 일정이 예정되어 실적 가이던스 확인 가능성이 있습니다.",
        details:
          "IR 일정은 단기 가격 재료라기보다 컨센서스 조정의 트리거가 될 수 있어 확인 우선순위를 높였습니다.",
        sources: ["DART", "IR 일정"],
      },
      flow: {
        summary:
          "대형 반도체 업종 내 거래대금이 증가하며 수급 집중도가 높아졌습니다.",
        details:
          "외국인 순매수와 거래대금 증가가 함께 나타나 단기 주목도를 높였습니다. 다만 급등 후 추격 매수 위험은 별도 체크가 필요합니다.",
        sources: ["KRX", "수급 데이터"],
      },
    },
  },
  {
    rank: 2,
    name: "SK하이닉스",
    code: "000660",
    logoText: "S",
    price: "163,000원",
    change: "-1.25%",
    signal: "HBM 모멘텀",
    score: 81,
    summary: "AI 서버 투자 확대와 HBM 모멘텀이 유지되지만 단기 차익실현도 함께 확인됩니다.",
    reasons: {
      news: {
        summary:
          "AI 서버 투자 확대 뉴스가 반도체 장비·메모리 섹터 기대를 유지했습니다.",
        details:
          "HBM 중심의 중장기 성장 스토리는 유지되지만 단기 가격에는 기대가 일부 반영되어 있습니다.",
        sources: ["AI 인프라 뉴스"],
      },
      disclosure: {
        summary:
          "주요 공시는 제한적이나 업종 실적 컨센서스 상향이 이어지고 있습니다.",
        details:
          "개별 공시보다 섹터 컨센서스 변화가 더 중요한 구간으로 판단했습니다.",
        sources: ["컨센서스 데이터"],
      },
      flow: {
        summary:
          "기관 매수는 유지됐지만 단기 차익실현 물량이 일부 출회됐습니다.",
        details:
          "수급의 질은 양호하나 상승 피로도가 있어 관망 확률을 함께 높게 둡니다.",
        sources: ["기관/외국인 수급"],
      },
    },
  },
  {
    rank: 3,
    name: "NAVER",
    code: "035420",
    logoText: "N",
    price: "186,700원",
    change: "+2.10%",
    signal: "AI 서비스 재료",
    score: 77,
    summary: "AI 서비스와 커머스 개선 기대가 플랫폼주 반등 재료로 작용했습니다.",
    reasons: {
      news: {
        summary:
          "AI 검색과 커머스 전환율 개선 기대가 플랫폼주 반등 재료로 작용했습니다.",
        details: "AI 서비스 관련 언급이 늘고 있어 성장 재료로 분류했습니다.",
        sources: ["플랫폼 업종 뉴스"],
      },
      disclosure: {
        summary:
          "주요 경영사항 공시 이후 신규 서비스 투자 방향 확인이 필요합니다.",
        details:
          "투자 확대가 비용 부담인지 성장 동력인지 다음 공시와 실적에서 확인해야 합니다.",
        sources: ["DART"],
      },
      flow: {
        summary: "최근 낙폭 이후 저가 매수세가 유입되며 거래량이 증가했습니다.",
        details:
          "저가 매수세는 확인되지만 추세 전환 여부는 아직 추가 확인이 필요합니다.",
        sources: ["거래량 데이터"],
      },
    },
  },
  {
    rank: 4,
    name: "LG에너지솔루션",
    code: "373220",
    logoText: "L",
    price: "348,500원",
    change: "+2.08%",
    signal: "배터리 수요 회복",
    score: 73,
    summary: "전기차 수요 회복 기대와 원재료 가격 안정이 투자심리를 개선했습니다.",
    reasons: {
      news: {
        summary:
          "전기차 수요 회복 기대와 원재료 가격 안정이 투자심리를 개선했습니다.",
        details:
          "전기차 수요와 소재 가격 변수가 동시에 완화되는지가 핵심입니다.",
        sources: ["2차전지 뉴스"],
      },
      disclosure: {
        summary:
          "단일판매 공급계약 정정 공시로 중장기 매출 가시성이 재점검됐습니다.",
        details: "계약 변경의 방향과 규모를 확인해 매출 가시성에 반영합니다.",
        sources: ["DART 계약 공시"],
      },
      flow: {
        summary:
          "2차전지 업종 거래대금이 늘었지만 변동성은 여전히 높은 편입니다.",
        details:
          "섹터 순환매가 들어왔지만 가격 변동폭이 커 리스크 점검이 필요합니다.",
        sources: ["KRX 섹터 거래대금"],
      },
    },
  },
  {
    rank: 5,
    name: "현대차",
    code: "005380",
    logoText: "현",
    price: "212,000원",
    change: "+0.84%",
    signal: "환율 수혜",
    score: 69,
    summary: "환율과 수출주 민감도가 다시 부각되며 안정적인 하방 방어 수급이 확인됩니다.",
    reasons: {
      news: {
        summary:
          "원달러 환율 보합권 흐름 속 수출주 민감도가 다시 부각됐습니다.",
        details:
          "환율과 판매 믹스가 이익 전망에 우호적인지 확인하는 구간입니다.",
        sources: ["FX Desk", "자동차 업종 뉴스"],
      },
      disclosure: {
        summary: "특이 공시는 없지만 주주환원 기대가 밸류에이션을 지지합니다.",
        details: "신규 공시보다는 배당/자사주 기대와 실적 안정성이 핵심입니다.",
        sources: ["기업 공시 캘린더"],
      },
      flow: {
        summary: "외국인 매수 강도는 완만하나 하방 방어 수급이 확인됩니다.",
        details: "공격적 매수보다 안정적 보유 관점의 수급이 우세합니다.",
        sources: ["외국인 수급"],
      },
    },
  },
];


function formatPrice(value?: number) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${value.toLocaleString("ko-KR")}원`;
}

function formatChange(value?: number) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function makeLogoText(name: string) {
  return name.trim().slice(0, 1) || "?";
}

function mapReason(reason?: BackendReason): GeminiReason {
  const sources = reason?.sources?.filter(Boolean) ?? [];
  return {
    summary: reason?.headline || "확인된 데이터가 아직 없습니다.",
    details: reason?.details || "추가 데이터가 들어오면 상세 근거를 표시합니다.",
    sources: sources.length > 0 ? sources : reason?.tags ?? [reason?.title || "데이터"],
  };
}

function mapReportStock(item: BackendRecommendation, index: number): ReportStock {
  const name = item.name || "종목명 없음";
  const signal = item.strategy || item.reasons?.news?.tags?.[0] || "AI 추천";
  return {
    rank: item.rank ?? index + 1,
    name,
    code: item.ticker || "",
    logoText: makeLogoText(name),
    price: formatPrice(item.currentPrice),
    change: formatChange(item.changeRate),
    signal,
    score: item.recommendationScore ?? 0,
    summary: item.summary || "뉴스, 공시, 재료와 수급 근거를 종합해 산출한 추천입니다.",
    reasons: {
      news: mapReason(item.reasons?.news),
      disclosure: mapReason(item.reasons?.disclosure),
      flow: mapReason(item.reasons?.materialFlow),
    },
  };
}

function parseRecommendationReport(report: unknown): ReportStock[] {
  const parsed = typeof report === "string" ? JSON.parse(report) as BackendReport : report as BackendReport;
  return (parsed.recommendations ?? []).map(mapReportStock);
}

function getReportSlot(now = new Date()) {
  const scheduledAt = new Date(now);
  scheduledAt.setHours(DAILY_REPORT_HOUR, DAILY_REPORT_MINUTE, 0, 0);

  if (now < scheduledAt) {
    scheduledAt.setDate(scheduledAt.getDate() - 1);
  }

  const year = scheduledAt.getFullYear();
  const month = String(scheduledAt.getMonth() + 1).padStart(2, "0");
  const day = String(scheduledAt.getDate()).padStart(2, "0");

  return {
    key: `${year}-${month}-${day}-0830`,
    scheduledAt,
  };
}

function formatReportTime(date: Date) {
  return date.toLocaleTimeString("ko-KR", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

function getReportCacheKey(userId: string) {
  return `${REPORT_CACHE_PREFIX}:${userId}`;
}

function readCachedReport(userId: string, slotKey: string): CachedReport | null {
  if (typeof window === "undefined") return null;

  try {
    const rawCache = window.localStorage.getItem(getReportCacheKey(userId));
    if (!rawCache) return null;

    const cache = JSON.parse(rawCache) as CachedReport;
    if (cache.version !== REPORT_CACHE_VERSION) return null;
    if (cache.slotKey !== slotKey || !Array.isArray(cache.stocks)) return null;
    if (cache.stocks.length === 0) return null;

    return cache;
  } catch {
    return null;
  }
}

function writeCachedReport(userId: string, cache: CachedReport) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(getReportCacheKey(userId), JSON.stringify(cache));
}



export default function AIReportPage() {
  const { toggleFavorite, toggleCart, isFavorite, isInCart } = useStockList();
  const [selectedStock, setSelectedStock] = useState<ReportStock | null>(null);
  const [pendingCart, setPendingCart] = useState<StockListItem | null>(null);
  const [refreshedAt, setRefreshedAt] = useState("09:30");
  const [userId, setUserId] = useState<string | null>(null);
  const [nickname, setNickname] = useState("회원");
  const [isGenerating, setIsGenerating] = useState(false);
  const [stocks, setStocks] = useState<ReportStock[]>(fallbackReportStocks);
  const [reportError, setReportError] = useState<string | null>(null);

  useEffect(() => {
    const syncUser = () => {
      const user = getCurrentUser();
      setUserId(user?.id ?? user?.email ?? "guest");
      setNickname(user?.nickname ?? "회원");
    };

    syncUser();
    window.addEventListener(AUTH_EVENT_NAME, syncUser);
    window.addEventListener("storage", syncUser);

    return () => {
      window.removeEventListener(AUTH_EVENT_NAME, syncUser);
      window.removeEventListener("storage", syncUser);
    };
  }, []);

  const loadReport = useCallback(async ({ force = false } = {}) => {
    if (!userId) return;

    const slot = getReportSlot();
    const cachedReport = readCachedReport(userId, slot.key);

    if (!force && cachedReport) {
      setSelectedStock(null);
      setReportError(null);
      setStocks(cachedReport.stocks);
      setRefreshedAt(cachedReport.refreshedAt);
      return;
    }

    setIsGenerating(true);
    setSelectedStock(null);
    setReportError(null);

    try {
      const res = await apiFetch(`${API_BASE}/recommendation`, {
        headers: { Accept: "application/json" },
      });

      if (!res.ok) {
        throw new Error(`추천 리포트 요청 실패 (${res.status})`);
      }

      const payload = (await res.json()) as { report?: unknown; timestamp?: string };
      const nextStocks = parseRecommendationReport(payload.report);

      if (nextStocks.length === 0) {
        throw new Error("추천 종목이 비어 있습니다.");
      }

      setStocks(nextStocks);
      const reportTime = payload.timestamp ? new Date(payload.timestamp) : new Date();
      const displayTime = force ? reportTime : slot.scheduledAt;
      const refreshedTime = formatReportTime(displayTime);
      setRefreshedAt(refreshedTime);
      writeCachedReport(userId, {
        version: REPORT_CACHE_VERSION,
        slotKey: slot.key,
        generatedAt: reportTime.toISOString(),
        refreshedAt: refreshedTime,
        stocks: nextStocks,
      });
    } catch (error) {
      console.error(error);
      setReportError(
        error instanceof Error
          ? error.message
          : "추천 리포트를 불러오지 못했습니다.",
      );
      setStocks(fallbackReportStocks);
    } finally {
      setIsGenerating(false);
    }
  }, [userId]);

  useEffect(() => {
    void loadReport();
  }, [loadReport]);

  const refreshReport = () => {
    void loadReport({ force: true });
  };

  const displayedStocks = stocks.map((stock, index) => ({ ...stock, rank: index + 1 }));

  const makeStockListItem = (stock: ReportStock): StockListItem => ({
    code: stock.code,
    name: stock.name,
    price: stock.price,
    change: stock.change,
    logoText: stock.logoText,
    memo: stock.summary,
  });

  return (
    <section className="rounded-2xl bg-white p-6 shadow-[0_24px_80px_rgba(15,23,42,0.06)] lg:p-8">
      <div className="flex flex-wrap items-start justify-between gap-5 border-b border-slate-100 pb-6">
        <div>
          <p className="text-sm font-bold text-[#5267ff]">
            AI Portfolio Report
          </p>
          <h1 className="mt-2 text-4xl font-black text-slate-950">
            {nickname} 님을 위한 오늘 주목해볼 종목
          </h1>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-500">
            뉴스, 공시, 재료와 수급 변화, 그리고 사용자 투자 성향을 함께 읽어
            오늘 확인할 Top 5 종목을 정리했습니다.
          </p>
        </div>
        <button
          type="button"
          onClick={refreshReport}
          disabled={isGenerating}
          className="flex h-11 items-center gap-2 rounded-full border border-slate-200 bg-slate-100 px-5 text-sm font-black text-slate-600 transition hover:border-slate-300 hover:bg-slate-200 hover:text-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <FiRefreshCw
            className={`h-4 w-4 ${isGenerating ? "animate-spin" : ""}`}
          />
          {isGenerating ? "생성 중" : "새로 고침"}
        </button>
      </div>

      <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 rounded-full bg-[#eef2ff] px-4 py-2 text-xs font-black text-[#5267ff]">
          <FiZap className="h-4 w-4" />
          모델 기준 시각 {refreshedAt}
        </div>
        <p className="text-xs font-bold text-slate-400">
          하트는 관심 종목, 가방은 포트폴리오에 저장됩니다.
        </p>
      </div>

      {reportError ? (
        <div className="mt-5 rounded-2xl border border-amber-100 bg-amber-50 px-4 py-3 text-sm font-bold text-amber-700">
          {reportError} 기존 예시 리포트를 표시합니다.
        </div>
      ) : null}

      {isGenerating ? (
        <div className="mt-6 flex min-h-[520px] items-center justify-center rounded-2xl bg-[#f4f7ff] p-8">
          <div className="text-center">
            <div className="mx-auto flex h-24 w-24 items-center justify-center rounded-full bg-white shadow-[0_20px_60px_rgba(82,103,255,0.2)]">
              <IoSparkles className="h-12 w-12 animate-pulse text-[#5267ff]" />
            </div>
            <h2 className="mt-7 text-3xl font-black text-slate-950">
              AI가 리포트를 새롭게 생성 중입니다...
            </h2>
            <p className="mt-3 text-sm font-bold leading-6 text-slate-500">
              Gemini 리포트 결과를 불러와 뉴스, 공시, 재료와 수급 근거를 다시
              정리하고 있습니다.
            </p>
            <div className="mx-auto mt-7 h-2 w-80 overflow-hidden rounded-full bg-white">
              <div className="h-full w-1/2 animate-[loadingSlide_1.1s_ease-in-out_infinite] rounded-full bg-[#5267ff]" />
            </div>
          </div>
        </div>
      ) : (
        <div className="mt-6 space-y-4">
          {displayedStocks.map((stock) => (
            <ReportCard
              key={stock.code}
              stock={stock}
              isFavorite={isFavorite(stock.code)}
              isInCart={isInCart(stock.code)}
              onSelect={setSelectedStock}
              onToggleFavorite={(target) => {
                toggleFavorite(target.code, makeStockListItem(target));
              }}
              onRequestCart={setPendingCart}
              makeStockListItem={makeStockListItem}
            />
          ))}
        </div>
      )}

      {pendingCart && (
        <CartConfirmModal
          stockName={pendingCart.name}
          mode={isInCart(pendingCart.code) ? "remove" : "add"}
          onConfirm={() => { toggleCart(pendingCart.code, pendingCart); setPendingCart(null); }}
          onCancel={() => setPendingCart(null)}
        />
      )}

      {selectedStock ? (
        <ReportDetailModal
          stock={selectedStock}
          onClose={() => setSelectedStock(null)}
        />
      ) : null}
    </section>
  );
}
