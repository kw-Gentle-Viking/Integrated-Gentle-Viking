"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  FiActivity,
  FiFileText,
  FiHeart,
  FiRefreshCw,
  FiTrendingUp,
  FiX,
  FiZap,
} from "react-icons/fi";
import { IoSparkles } from "react-icons/io5";
import { AUTH_EVENT_NAME, getCurrentUser } from "@/lib/signup/auth";

type ReasonKey = "news" | "disclosure" | "flow";

type GeminiReason = {
  summary: string;
  details: string;
  sources: string[];
};

type ReportStock = {
  rank: number;
  name: string;
  code: string;
  logoText: string;
  price: string;
  change: string;
  signal: string;
  score: number;
  reasons: Record<ReasonKey, GeminiReason>;
};

const reportStocks: ReportStock[] = [
  {
    rank: 1,
    name: "삼성전자",
    code: "005930",
    logoText: "삼",
    price: "81,200원",
    change: "+0.61%",
    signal: "메모리 반등",
    score: 86,
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

const reasonCards = [
  { key: "news", label: "뉴스", icon: FiFileText },
  { key: "disclosure", label: "공시", icon: FiFileText },
  { key: "flow", label: "재료/수급", icon: FiTrendingUp },
] as const;

function DetailChart({ stock }: { stock: ReportStock }) {
  const points = stock.change.startsWith("-")
    ? "M0 54 C55 42 84 58 128 49 C176 38 216 70 268 66 C326 62 364 84 420 76"
    : "M0 82 C48 76 84 88 126 68 C176 44 214 58 266 38 C316 18 362 28 420 12";
  const fill = `${points} L 420 116 L 0 116 Z`;
  const color = stock.change.startsWith("-") ? "#60a5fa" : "#fb7185";

  return (
    <div className="relative h-40 overflow-hidden rounded-2xl bg-white p-4 ring-1 ring-slate-100">
      <div className="pointer-events-none absolute inset-x-4 top-1/3 h-px bg-slate-100" />
      <div className="pointer-events-none absolute inset-x-4 bottom-1/3 h-px bg-slate-100" />
      <svg
        viewBox="0 0 420 120"
        className="absolute inset-0 h-full w-full"
        aria-hidden="true"
      >
        <defs>
          <linearGradient
            id={`detailFill-${stock.code}`}
            x1="0"
            x2="0"
            y1="0"
            y2="1"
          >
            <stop offset="0%" stopColor={color} stopOpacity="0.22" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={fill} fill={`url(#detailFill-${stock.code})`} />
        <path
          d={points}
          fill="none"
          stroke={color}
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="4"
        />
      </svg>
    </div>
  );
}

export default function AIReportPage() {
  const [favoriteCodes, setFavoriteCodes] = useState<string[]>([]);
  const [selectedStock, setSelectedStock] = useState<ReportStock | null>(null);
  const [refreshedAt, setRefreshedAt] = useState("09:30");
  const [nickname, setNickname] = useState("회원");
  const [isGenerating, setIsGenerating] = useState(false);
  const [reportVersion, setReportVersion] = useState(0);

  useEffect(() => {
    const syncUser = () => {
      const user = getCurrentUser();
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

  const toggleFavorite = (code: string) => {
    setFavoriteCodes((current) =>
      current.includes(code)
        ? current.filter((item) => item !== code)
        : [...current, code],
    );
  };

  const refreshReport = () => {
    setIsGenerating(true);
    setSelectedStock(null);

    window.setTimeout(() => {
      const now = new Date();
      setRefreshedAt(
        now.toLocaleTimeString("ko-KR", {
          hour: "2-digit",
          minute: "2-digit",
          hour12: false,
        }),
      );
      setReportVersion((current) => current + 1);
      setIsGenerating(false);
    }, 1300);
  };

  const rotatedStocks = reportVersion % reportStocks.length;
  const displayedStocks = [
    ...reportStocks.slice(rotatedStocks),
    ...reportStocks.slice(0, rotatedStocks),
  ].map((stock, index) => ({ ...stock, rank: index + 1 }));

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
          className="flex h-11 items-center gap-2 rounded-full bg-slate-950 px-5 text-sm font-black text-white transition hover:bg-slate-800"
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
          하트를 누르면 관심 종목 후보에 저장됩니다.
        </p>
      </div>

      {isGenerating ? (
        <div className="mt-6 flex min-h-[520px] items-center justify-center rounded-2xl bg-[#f4f7ff] p-8">
          <div className="text-center">
            <div className="mx-auto flex h-24 w-24 items-center justify-center rounded-full bg-white shadow-[0_20px_60px_rgba(82,103,255,0.2)]">
              <IoSparkles className="h-12 w-12 animate-pulse text-[#5267ff]" />
            </div>
            <h2 className="mt-7 text-3xl font-black text-slate-950">
              AI가 리포트를 생성 중입니다...
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
          {displayedStocks.map((stock) => {
            const isFavorite = favoriteCodes.includes(stock.code);

            return (
              <button
                key={stock.code}
                type="button"
                onClick={() => setSelectedStock(stock)}
                className="block w-full rounded-2xl border border-slate-100 bg-white p-5 text-left shadow-[0_12px_32px_rgba(15,23,42,0.04)] transition hover:border-[#5267ff]/30 hover:shadow-[0_18px_44px_rgba(15,23,42,0.08)]"
              >
                <div>
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div className="flex items-start gap-4">
                      <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-slate-950 text-sm font-black text-white shadow-[0_10px_24px_rgba(15,23,42,0.16)]">
                        {stock.logoText}
                      </span>
                      <div>
                        <div className="flex flex-wrap items-center gap-2">
                          <h2 className="text-2xl font-black text-slate-950">
                            {stock.name}
                          </h2>
                          <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-black text-slate-500">
                            {stock.code}
                          </span>
                          <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-black text-emerald-600">
                            {stock.signal}
                          </span>
                        </div>
                        <div className="mt-2 flex items-center gap-3">
                          <p className="text-lg font-black text-slate-950">
                            {stock.price}
                          </p>
                          <p
                            className={`text-sm font-black ${stock.change.startsWith("-") ? "text-blue-500" : "text-rose-500"}`}
                          >
                            {stock.change}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          toggleFavorite(stock.code);
                        }}
                        title="관심 종목에 추가"
                        className={`flex h-10 w-10 items-center justify-center rounded-xl border transition ${
                          isFavorite
                            ? "border-rose-100 bg-rose-50 text-rose-500"
                            : "border-slate-200 text-slate-400 hover:border-rose-100 hover:bg-rose-50 hover:text-rose-500"
                        }`}
                      >
                        <FiHeart
                          className={`h-5 w-5 ${isFavorite ? "fill-current" : ""}`}
                        />
                      </button>
                      <Link
                        href={`/ai-report/predict?code=${stock.code}`}
                        onClick={(event) => event.stopPropagation()}
                        className="flex h-10 items-center gap-2 rounded-xl bg-[#5267ff] px-4 text-sm font-black text-white transition hover:bg-[#4054df]"
                      >
                        <FiActivity className="h-4 w-4" />
                        AI 예측하기
                      </Link>
                    </div>
                  </div>

                  <div className="mt-5 grid gap-3 md:grid-cols-[1.1fr_1fr_1fr_120px]">
                    {reasonCards.map((item) => {
                      const Icon = item.icon;
                      const reason = stock.reasons[item.key];

                      return (
                        <div
                          key={item.key}
                          className="rounded-xl bg-slate-50 p-3"
                        >
                          <p className="flex items-center gap-2 text-xs font-black text-slate-500">
                            <Icon className="h-4 w-4" />
                            {item.label}
                          </p>
                          <p className="mt-1.5 line-clamp-2 text-xs font-bold leading-5 text-slate-600">
                            {reason.summary}
                          </p>
                        </div>
                      );
                    })}
                    <div className="flex items-center justify-end">
                      <span className="rounded-full bg-[#eef2ff] px-4 py-2 text-xs font-black text-[#5267ff]">
                        상세 보기
                      </span>
                    </div>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      )}

      {selectedStock ? (
        <div className="fixed inset-0 z-50 flex justify-end bg-slate-950/35 backdrop-blur-sm">
          <button
            type="button"
            aria-label="상세 패널 닫기"
            className="absolute inset-0 cursor-default"
            onClick={() => setSelectedStock(null)}
          />
          <aside className="relative h-full w-full max-w-[520px] translate-x-0 overflow-y-auto bg-white p-6 shadow-[0_24px_80px_rgba(15,23,42,0.25)] animate-[slideInRight_0.28s_ease-out]">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm font-bold text-[#5267ff]">Stock Detail</p>
                <h2 className="mt-2 text-3xl font-black text-slate-950">
                  {selectedStock.name}
                </h2>
                <div className="mt-2 flex items-center gap-2">
                  <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-black text-slate-500">
                    {selectedStock.code}
                  </span>
                  <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-black text-emerald-600">
                    {selectedStock.signal}
                  </span>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setSelectedStock(null)}
                title="닫기"
                className="flex h-10 w-10 items-center justify-center rounded-xl text-slate-400 transition hover:bg-slate-100 hover:text-slate-950"
              >
                <FiX className="h-5 w-5" />
              </button>
            </div>

            <div className="mt-6">
              <DetailChart stock={selectedStock} />
            </div>

            <div className="mt-5 grid grid-cols-3 gap-3">
              <div className="rounded-2xl bg-slate-950 p-4 text-white">
                <p className="text-xs font-bold text-slate-300">추천 점수</p>
                <p className="mt-2 text-3xl font-black">
                  {selectedStock.score}
                </p>
              </div>
              <div className="rounded-2xl bg-slate-50 p-4">
                <p className="text-xs font-bold text-slate-400">현재가</p>
                <p className="mt-2 text-lg font-black text-slate-950">
                  {selectedStock.price}
                </p>
              </div>
              <div className="rounded-2xl bg-slate-50 p-4">
                <p className="text-xs font-bold text-slate-400">등락률</p>
                <p
                  className={`mt-2 text-lg font-black ${selectedStock.change.startsWith("-") ? "text-blue-500" : "text-rose-500"}`}
                >
                  {selectedStock.change}
                </p>
              </div>
            </div>

            <section className="mt-5 rounded-2xl bg-[#f4f7ff] p-5">
              <p className="text-xs font-black text-[#5267ff]">핵심 요약</p>
              <p className="mt-3 text-lg font-black leading-8 text-slate-900">
                {selectedStock.signal} 흐름이 확인되며, 뉴스와 수급 근거가
                동시에 점수를 끌어올렸습니다.
              </p>
            </section>

            <section className="mt-5 space-y-3">
              <h3 className="text-lg font-black text-slate-950">핵심 데이터</h3>
              {reasonCards.map((item) => {
                const Icon = item.icon;
                const reason = selectedStock.reasons[item.key];

                return (
                  <article
                    key={item.key}
                    className="rounded-2xl border border-slate-100 p-4"
                  >
                    <p className="flex items-center gap-2 text-sm font-black text-slate-700">
                      <Icon className="h-4 w-4 text-[#5267ff]" />
                      {item.label}
                    </p>
                    <p className="mt-2 text-sm font-bold leading-6 text-slate-700">
                      {reason.summary}
                    </p>
                    <p className="mt-3 text-sm leading-6 text-slate-500">
                      {reason.details}
                    </p>
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {reason.sources.map((source) => (
                        <span
                          key={source}
                          className="rounded-full bg-slate-100 px-2 py-1 text-[11px] font-bold text-slate-500"
                        >
                          {source}
                        </span>
                      ))}
                    </div>
                  </article>
                );
              })}
            </section>

            <div className="sticky bottom-0 mt-6 bg-white pb-1 pt-4">
              <Link
                href={`/ai-report/predict?code=${selectedStock.code}`}
                className="flex h-12 w-full items-center justify-center gap-2 rounded-xl bg-[#5267ff] text-sm font-black text-white transition hover:bg-[#4054df]"
              >
                <FiActivity className="h-4 w-4" />
                AI 예측하기
              </Link>
            </div>
          </aside>
        </div>
      ) : null}
    </section>
  );
}
