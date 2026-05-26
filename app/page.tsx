"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useState } from "react";
import {
  IoAnalyticsOutline,
  IoArrowForward,
  IoCheckmarkCircle,
  IoDocumentTextOutline,
  IoFlashOutline,
  IoNewspaperOutline,
  IoPulseOutline,
  IoSearchOutline,
  IoShieldCheckmarkOutline,
  IoSparklesOutline,
  IoStatsChartOutline,
  IoTrendingUpOutline,
} from "react-icons/io5";
import {
  AUTH_EVENT_NAME,
  AuthUser,
  getCurrentUser,
  logoutUser,
} from "@/lib/signup/auth";

const floatingSignals = [
  {
    label: "NEWS",
    icon: IoNewspaperOutline,
    className: "signal-news left-[7%] top-[24%] rotate-[-13deg]",
  },
  {
    label: "DART",
    icon: IoDocumentTextOutline,
    className: "signal-dart right-[10%] top-[18%] rotate-[12deg]",
  },
  {
    label: "KRX",
    icon: IoStatsChartOutline,
    className: "signal-krx left-[18%] top-[55%] rotate-[10deg]",
  },
  {
    label: "KIS",
    icon: IoPulseOutline,
    className: "signal-kis right-[18%] top-[52%] rotate-[-9deg]",
  },
  {
    label: "AI",
    icon: IoSparklesOutline,
    className: "signal-ai left-[6%] bottom-[16%] rotate-[16deg]",
  },
  {
    label: "RISK",
    icon: IoShieldCheckmarkOutline,
    className: "signal-risk right-[6%] bottom-[18%] rotate-[-14deg]",
  },
];


const workflowCards = [
  {
    eyebrow: "01 Data",
    title: "흩어진 시장 단서 수집",
    description:
      "뉴스, 공시, 가격, 사용자 성향을 한 화면에서 함께 읽을 수 있게 정리합니다.",
    icon: IoSearchOutline,
  },
  {
    eyebrow: "02 Insight",
    title: "오늘 살펴볼 종목 제안",
    description:
      "투자 결정을 강요하지 않고 종목별 흐름과 관찰 포인트를 먼저 보여줍니다.",
    icon: IoAnalyticsOutline,
  },
  {
    eyebrow: "03 Action",
    title: "리포트와 실행 흐름 연결",
    description:
      "근거 리포트 확인 후 사용자가 원할 때만 자동매매 설정으로 이어집니다.",
    icon: IoFlashOutline,
  },
];

const partnerLogos = ["NAVER", "DART", "Gemini", "KIS API", "KRX"];

export default function LandingPage() {
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);

  useEffect(() => {
    const syncAuth = () => setCurrentUser(getCurrentUser());

    syncAuth();
    window.addEventListener(AUTH_EVENT_NAME, syncAuth);
    window.addEventListener("storage", syncAuth);

    return () => {
      window.removeEventListener(AUTH_EVENT_NAME, syncAuth);
      window.removeEventListener("storage", syncAuth);
    };
  }, []);

  const handleLogout = () => {
    logoutUser();
    setCurrentUser(null);
  };

  return (
    <div className="landing-shell min-h-screen text-slate-950">
      <section className="relative min-h-screen overflow-visible">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_54%,rgba(125,211,252,0.36),transparent_28%),radial-gradient(circle_at_61%_56%,rgba(196,181,253,0.32),transparent_25%),linear-gradient(180deg,#f8fbff_0%,#f4f7fb_58%,#eaf1fb_100%)]" />
        <div className="pointer-events-none absolute inset-x-0 top-[43%] h-px bg-slate-200/70" />

        <nav className="relative z-20 flex h-[76px] w-full items-center border-b border-slate-200/70 bg-white/78 shadow-[0_18px_60px_rgba(15,23,42,0.06)] backdrop-blur-xl">
          <div className="mx-auto flex w-full max-w-[1180px] items-center justify-between px-5">
            <Link href="/" className="flex items-center gap-3">
              <Image
                src="/genvi.png"
                alt="Genvi"
                width={118}
                height={58}
                priority
                className="h-12 w-auto object-contain"
              />
            </Link>

            <div className="hidden items-center gap-9 text-xs font-bold text-slate-500 md:flex">
              <a
                href="#service-flow"
                className="transition hover:text-slate-950"
              >
                서비스 흐름
              </a>
              <a href="#daily-view" className="transition hover:text-slate-950">
                데일리 브리핑
              </a>
              <a
                href="#report-view"
                className="transition hover:text-slate-950"
              >
                분석 노트
              </a>
            </div>

            <div className="flex items-center gap-3">
              {currentUser ? (
                <button
                  type="button"
                  onClick={handleLogout}
                  className="hidden h-11 items-center justify-center rounded-full border border-slate-200 bg-white/60 px-5 text-sm font-extrabold text-slate-700 transition hover:border-slate-300 hover:bg-white sm:inline-flex"
                >
                  로그아웃
                </button>
              ) : (
                <Link
                  href="/login"
                  className="hidden h-11 items-center justify-center rounded-full border border-slate-200 bg-white/60 px-5 text-sm font-extrabold text-slate-700 transition hover:border-slate-300 hover:bg-white sm:inline-flex"
                >
                  로그인
                </Link>
              )}
              <Link
                href={currentUser ? "/dashboard" : "/signup"}
                className="inline-flex h-11 items-center justify-center rounded-full bg-gradient-to-r from-indigo-600 via-violet-600 to-sky-500 px-6 text-sm font-extrabold text-white shadow-[0_14px_34px_rgba(79,70,229,0.28)] transition hover:scale-[1.02]"
              >
                {currentUser ? "대시보드" : "회원가입"}
              </Link>
            </div>
          </div>
        </nav>

        {floatingSignals.map((signal) => {
          const Icon = signal.icon;

          return (
            <div
              key={signal.label}
              className={`landing-signal absolute z-10 hidden h-20 w-20 md:block ${signal.className}`}
            >
              <div className="landing-signal-card flex h-full w-full items-center justify-center rounded-2xl border border-white/80 bg-white/75 text-slate-500 shadow-[0_22px_60px_rgba(30,41,59,0.12)] backdrop-blur-xl">
                <div className="text-center">
                  <Icon className="mx-auto h-7 w-7 text-indigo-500" />
                  <p className="mt-1 text-[10px] font-black tracking-[0.14em]">
                    {signal.label}
                  </p>
                </div>
              </div>
            </div>
          );
        })}

        <div className="relative z-10 mx-auto flex min-h-[calc(100vh-76px)] max-w-[1180px] flex-col items-center justify-start pb-[300px] pt-16 text-center sm:pb-[330px] sm:pt-20 lg:pb-[360px] lg:pt-24">
          <div className="landing-enter landing-enter-1 mb-5 inline-flex items-center gap-2 rounded-full border border-white/90 bg-white/70 px-4 py-2 text-xs font-black text-slate-600 shadow-sm backdrop-blur">
            <IoSparklesOutline className="h-4 w-4 text-indigo-500" />
            PERSONALIZED STOCK BRIEFING
          </div>

          <h1 className="landing-enter landing-enter-2 max-w-[820px] text-[42px] font-black leading-[1.06] tracking-normal text-slate-950 sm:text-[60px] lg:text-[72px]">
            Welcome to
            <br />
            Genvi !
          </h1>

          <p className="landing-enter landing-enter-3 mt-6 max-w-[570px] text-sm leading-7 text-slate-600 sm:text-base">
            흩어진 시장 정보를 AI가 한데 모아,
            <br />
            오늘 주목할 종목의 흐름과 관찰 포인트를
            <br />내 투자 성향에 맞춘 Daily Brief로
          </p>

          <div className="landing-enter landing-enter-4 mt-8 flex flex-wrap justify-center gap-3">
            <Link
              href={currentUser ? "/dashboard" : "/signup"}
              className="inline-flex h-12 items-center justify-center gap-2 rounded-full bg-slate-950 px-6 text-sm font-extrabold text-white shadow-[0_16px_40px_rgba(15,23,42,0.18)] transition hover:bg-slate-800"
            >
              {currentUser ? "오늘의 화면 보기" : "서비스 시작하기"}
              <IoArrowForward className="h-4 w-4" />
            </Link>
          </div>
        </div>

        <div className="landing-preview landing-enter landing-enter-5 absolute inset-x-5 top-[560px] z-20 mx-auto max-w-[1080px] sm:top-[585px] lg:top-[610px]">
          <div className="overflow-hidden rounded-[24px] border border-white/80 bg-[#f8fafc] shadow-[0_35px_100px_rgba(30,41,59,0.18)]">
            <div className="dashboard-intake pointer-events-none absolute left-1/2 top-[-18px] h-10 w-28 -translate-x-1/2 rounded-full bg-sky-300/25 blur-xl" />
            <div className="flex h-14 items-center justify-between border-b border-slate-200/70 bg-white/90 px-5 backdrop-blur">
              <div className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-full bg-rose-300" />
                <span className="h-3 w-3 rounded-full bg-amber-300" />
                <span className="h-3 w-3 rounded-full bg-emerald-300" />
              </div>
              <div className="hidden h-8 w-[240px] items-center gap-2 rounded-full bg-slate-50 px-3 text-xs font-bold text-slate-400 sm:flex">
                <IoSearchOutline className="h-4 w-4" />
                종목, 뉴스, 공시 검색
              </div>
            </div>

            <div className="grid min-h-[500px] bg-[#f8fafc] lg:grid-cols-[190px_1fr]">
              <aside className="hidden border-r border-slate-200/70 bg-[#f6f8fc] p-5 lg:block">
                <Image
                  src="/genvi.png"
                  alt="Genvi"
                  width={96}
                  height={48}
                  className="h-10 w-auto object-contain"
                />
                <div className="mt-7 space-y-2 text-xs font-bold">
                  {["홈", "AI 리포트", "내 주식보기", "관심 종목", "마이페이지"].map((item, index) => (
                    <div
                      key={item}
                      className={`rounded-lg px-3 py-2.5 ${
                        index === 0
                          ? "bg-white/55 text-slate-700 shadow-[0_14px_34px_rgba(15,23,42,0.13)] ring-1 ring-white/70 backdrop-blur-md"
                          : "text-slate-400"
                      }`}
                    >
                      {item}
                    </div>
                  ))}
                </div>
              </aside>

              <div className="p-4 sm:p-6">
                <div className="grid gap-4 lg:grid-cols-[1fr_240px]">
                  <section className="rounded-2xl bg-white p-4 shadow-[0_20px_60px_rgba(15,23,42,0.06)]">
                    <div className="mb-4 flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 shadow-[0_0_0_4px_rgba(52,211,153,0.14)]" />
                        <p className="text-sm font-black text-slate-950">국내 정규장</p>
                      </div>
                      <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-black text-emerald-600">
                        장중
                      </span>
                    </div>
                    <div className="grid gap-3 sm:grid-cols-3">
                      {[
                        { name: "KOSPI", value: "2,742.31", change: "+0.82%", up: true },
                        { name: "KOSDAQ", value: "873.94", change: "-0.18%", up: false },
                        { name: "USD/KRW", value: "1,346.20", change: "+0.21%", up: true },
                      ].map((index) => (
                        <div key={index.name} className="rounded-xl border border-slate-100 bg-slate-50/70 p-3">
                          <p className="text-xs font-bold text-slate-500">{index.name}</p>
                          <p className="mt-1 text-xl font-black text-slate-950">{index.value}</p>
                          <p className={`mt-1 text-xs font-black ${index.up ? "text-rose-500" : "text-blue-500"}`}>
                            {index.change}
                          </p>
                        </div>
                      ))}
                    </div>
                  </section>

                  <section className="rounded-2xl bg-blue-50 p-4 text-slate-800 shadow-[0_20px_60px_rgba(82,103,255,0.10)]">
                    <p className="text-sm font-bold text-blue-500">내 자산</p>
                    <p className="mt-2 text-2xl font-black text-slate-900">139만 7,380원</p>
                    <p className="mt-2 text-sm font-bold text-blue-400">+2.37% 오늘</p>
                    <div className="mt-4 grid grid-cols-2 gap-3">
                      <div className="rounded-xl bg-white/70 p-2.5 ring-1 ring-blue-100">
                        <p className="text-xs text-slate-400">수익</p>
                        <p className="mt-1 text-sm font-black text-slate-800">+29,770원</p>
                      </div>
                      <div className="rounded-xl bg-white/70 p-2.5 ring-1 ring-blue-100">
                        <p className="text-xs text-slate-400">현금</p>
                        <p className="mt-1 text-sm font-black text-slate-800">86,539원</p>
                      </div>
                    </div>
                  </section>
                </div>

                <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_260px]">
                  <section className="rounded-2xl bg-white p-4 shadow-[0_20px_60px_rgba(15,23,42,0.06)]">
                    <div className="mb-3 flex items-end justify-between gap-3">
                      <div>
                        <p className="text-sm font-bold text-[#5267ff]">실시간 차트</p>
                        <h3 className="mt-1 text-xl font-black text-slate-950">삼성전자</h3>
                      </div>
                      <div className="text-right">
                        <p className="text-lg font-black text-slate-950">81,200원</p>
                        <p className="text-sm font-black text-rose-500">+0.61%</p>
                      </div>
                    </div>
                    <div className="relative h-[170px] overflow-hidden rounded-2xl border border-slate-100 bg-white">
                      <div className="absolute inset-x-4 top-1/3 h-px bg-slate-100" />
                      <div className="absolute inset-x-4 bottom-1/3 h-px bg-slate-100" />
                      <svg viewBox="0 0 520 180" className="absolute inset-0 h-full w-full" aria-hidden="true">
                        <defs>
                          <linearGradient id="landingChartFill" x1="0" x2="0" y1="0" y2="1">
                            <stop offset="0%" stopColor="#fb7185" stopOpacity="0.2" />
                            <stop offset="100%" stopColor="#fb7185" stopOpacity="0" />
                          </linearGradient>
                        </defs>
                        <path d="M0 132 C42 126 68 138 104 112 C148 82 182 98 224 70 C268 40 310 54 352 30 C404 0 452 18 520 8 L520 180 L0 180 Z" fill="url(#landingChartFill)" />
                        <path d="M0 132 C42 126 68 138 104 112 C148 82 182 98 224 70 C268 40 310 54 352 30 C404 0 452 18 520 8" fill="none" stroke="#fb7185" strokeLinecap="round" strokeLinejoin="round" strokeWidth="5" />
                      </svg>
                    </div>

                  </section>

                  <section className="rounded-2xl bg-white p-4 shadow-[0_20px_60px_rgba(15,23,42,0.06)]">
                    <div className="mb-3 flex items-center justify-between">
                      <h3 className="text-lg font-black text-slate-950">실시간 종목 랭킹</h3>
                      <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#eef2ff] text-[#5267ff]">
                        <IoTrendingUpOutline className="h-4 w-4" />
                      </span>
                    </div>
                    <div className="space-y-2">
                      {[
                        { rank: 1, name: "삼성전자", price: "81,200", change: "+0.61%", up: true },
                        { rank: 2, name: "SK하이닉스", price: "163,000", change: "-1.25%", up: false },
                        { rank: 3, name: "NAVER", price: "186,700", change: "+2.10%", up: true },
                      ].map((stock) => (
                        <div key={stock.rank} className="flex items-center gap-2.5 rounded-lg border border-slate-100 p-2.5">
                          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-slate-100 text-xs font-black text-slate-500">
                            {stock.rank}
                          </span>
                          <div className="min-w-0 flex-1">
                            <p className="truncate text-sm font-black text-slate-950">{stock.name}</p>
                            <p className="text-xs font-bold text-slate-400">{stock.price}원</p>
                          </div>
                          <p className={`text-xs font-black ${stock.up ? "text-rose-500" : "text-blue-500"}`}>
                            {stock.change}
                          </p>
                        </div>
                      ))}
                    </div>
                  </section>
                </div>

                <div className="mt-4 grid gap-4 lg:grid-cols-3">
                  <article className="rounded-2xl bg-violet-50 p-4 shadow-[0_20px_60px_rgba(139,92,246,0.10)]">
                    <div className="mb-3 flex items-center justify-between">
                      <h4 className="text-base font-black text-slate-900">오늘의 뉴스</h4>
                      <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-violet-100 text-violet-500">
                        <IoNewspaperOutline className="h-4 w-4" />
                      </span>
                    </div>
                    <div className="space-y-2">
                      {[
                        { title: "HBM 공급 확대 기대감", meta: "09:12 Market" },
                        { title: "외국인 대형주 순매수", meta: "08:48 Brief" },
                      ].map((news) => (
                        <div key={news.title} className="rounded-lg bg-white/85 px-2.5 py-2 ring-1 ring-violet-100">
                          <p className="text-xs font-black leading-5 text-slate-800">{news.title}</p>
                          <p className="mt-0.5 text-[11px] font-bold text-slate-400">{news.meta}</p>
                        </div>
                      ))}
                    </div>
                  </article>

                  <article className="rounded-2xl bg-amber-50 p-4 shadow-[0_20px_60px_rgba(245,158,11,0.12)]">
                    <div className="mb-3 flex items-center justify-between">
                      <h4 className="text-base font-black text-slate-950">오늘의 공시</h4>
                      <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-white text-amber-700 shadow-sm">
                        <IoDocumentTextOutline className="h-4 w-4" />
                      </span>
                    </div>
                    <div className="space-y-2">
                      {[
                        { company: "삼성전자", title: "기업설명회 일정" },
                        { company: "현대차", title: "주주환원 관련 안내" },
                      ].map((item) => (
                        <div key={`${item.company}-${item.title}`} className="rounded-lg bg-white/85 px-2.5 py-2">
                          <p className="text-[11px] font-black text-amber-700">{item.company}</p>
                          <p className="mt-0.5 text-xs font-black leading-5 text-slate-950">{item.title}</p>
                        </div>
                      ))}
                    </div>
                  </article>

                  <article className="rounded-2xl bg-white p-4 shadow-[0_20px_60px_rgba(15,23,42,0.06)] ring-1 ring-slate-100">
                    <div className="mb-3 flex items-center justify-between">
                      <h4 className="text-base font-black text-slate-950">최근 본 종목</h4>
                      <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-100 text-slate-500">
                        <IoPulseOutline className="h-4 w-4" />
                      </span>
                    </div>
                    <div className="space-y-2">
                      {[
                        { name: "NAVER", code: "035420", change: "+2.10%", up: true },
                        { name: "LG에너지솔루션", code: "373220", change: "-0.42%", up: false },
                      ].map((stock) => (
                        <div key={stock.code} className="flex items-center gap-2 rounded-lg border border-slate-100 p-2.5">
                          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-slate-100 text-xs font-black text-slate-500">
                            {stock.name.slice(0, 1)}
                          </span>
                          <div className="min-w-0 flex-1">
                            <p className="truncate text-xs font-black text-slate-950">{stock.name}</p>
                            <p className="text-[11px] font-bold text-slate-400">{stock.code}</p>
                          </div>
                          <p className={`text-[11px] font-black ${stock.up ? "text-rose-500" : "text-blue-500"}`}>
                            {stock.change}
                          </p>
                        </div>
                      ))}
                    </div>
                  </article>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <main className="relative z-10 px-5 pb-24 pt-[650px] sm:pt-[560px] lg:pt-[490px]">
        <section id="service-flow" className="mx-auto max-w-[1120px] py-20">
          <div className="max-w-[760px]">
            <p className="text-xs font-black uppercase tracking-[0.18em] text-indigo-500">
              How Genvi Works
            </p>
            <h2 className="mt-4 text-3xl font-black tracking-normal text-slate-950 sm:text-5xl">
              투자 결정을 대신하지 않고,
              <br className="hidden sm:block" />
              필요한 장면을 또렷하게 정리합니다.
            </h2>
          </div>

          <div className="mt-10 grid gap-4 md:grid-cols-3">
            {workflowCards.map((card) => {
              const Icon = card.icon;

              return (
                <article
                  key={card.title}
                  className="rounded-3xl border border-slate-100 bg-slate-50 p-6"
                >
                  <div className="mb-8 flex h-12 w-12 items-center justify-center rounded-2xl bg-white text-indigo-500 shadow-sm">
                    <Icon className="h-6 w-6" />
                  </div>
                  <p className="text-xs font-black uppercase tracking-[0.14em] text-slate-400">
                    {card.eyebrow}
                  </p>
                  <h3 className="mt-3 text-xl font-black text-slate-950">
                    {card.title}
                  </h3>
                  <p className="mt-4 text-sm leading-6 text-slate-500">
                    {card.description}
                  </p>
                </article>
              );
            })}
          </div>
        </section>

        <section
          id="daily-view"
          className="mx-auto grid max-w-[1120px] gap-8 py-20 lg:grid-cols-[0.9fr_1.1fr]"
        >
          <div className="flex flex-col justify-center">
            <p className="text-xs font-black uppercase tracking-[0.18em] text-emerald-500">
              Daily UX
            </p>
            <h2 className="mt-4 text-3xl font-black tracking-normal text-slate-950 sm:text-5xl">
              매일 확인할 화면은 <br />
              짧고 분명하게.
            </h2>
            <p className="mt-5 max-w-[560px] text-sm leading-7 text-slate-500">
              종목을 단순히 나열하기보다, 오늘 먼저 살펴볼 후보와 관련 근거를{" "}
              <span className="whitespace-nowrap">함께 정리합니다.</span>
              사용자는 흐름과 근거를 확인한 뒤 다음 행동을 직접 선택합니다.
            </p>
          </div>

          <div className="rounded-[28px] border border-slate-100 bg-slate-950 p-5 text-white shadow-[0_24px_80px_rgba(15,23,42,0.18)]">
            <div className="flex items-center justify-between">
              <p className="font-black">Daily Picks</p>
              <span className="rounded-full bg-white/10 px-3 py-1 text-xs font-bold text-white/70">
                보수형 프로필
              </span>
            </div>
            <div className="mt-6 space-y-3">
              {[
                { title: "시장 소식 변화", detail: "뉴스 흐름 요약" },
                { title: "기업 공시 확인", detail: "주요 공시 포인트" },
                { title: "가격 흐름 점검", detail: "시세 변화 요약" },
              ].map((item, index) => (
                <div
                  key={item.title}
                  className="grid grid-cols-[44px_1fr_auto] items-center gap-4 rounded-2xl bg-white/8 p-4"
                >
                  <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white font-black text-slate-950">
                    {index + 1}
                  </div>
                  <div>
                    <p className="font-black">{item.title}</p>
                    <p className="mt-1 text-xs font-bold text-white/45">
                      {item.detail}
                    </p>
                  </div>
                  <IoArrowForward className="h-5 w-5 text-white/45" />
                </div>
              ))}
            </div>
          </div>
        </section>

        <section
          id="report-view"
          className="mx-auto grid max-w-[1120px] gap-8 rounded-[32px] bg-[#f5f8fc] p-6 py-10 sm:p-10 lg:grid-cols-[1.05fr_0.95fr]"
        >
          <div className="rounded-[24px] bg-white p-6 shadow-sm">
            <div className="mb-6 flex items-center justify-between">
              <p className="font-black text-slate-950">AI 리포트 미리보기</p>
              <span className="rounded-full bg-indigo-50 px-3 py-1 text-xs font-black text-indigo-600">
                근거 중심
              </span>
            </div>
            <div className="space-y-4">
              {[
                "최근 뉴스에서 수요 회복 관련 키워드가 증가했습니다.",
                "공시 이벤트는 단기 변동성보다 중장기 계획에 가깝습니다.",
                "사용자 성향 기준으로 과도한 비중 확대는 주의가 필요합니다.",
              ].map((line) => (
                <div
                  key={line}
                  className="flex gap-3 rounded-2xl bg-slate-50 p-4"
                >
                  <IoCheckmarkCircle className="mt-0.5 h-5 w-5 shrink-0 text-emerald-500" />
                  <p className="text-sm font-semibold leading-6 text-slate-600">
                    {line}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <div className="flex flex-col justify-center">
            <p className="text-xs font-black uppercase tracking-[0.18em] text-indigo-500">
              Report to Automation
            </p>
            <h2 className="mt-4 text-3xl font-black tracking-normal text-slate-950 sm:text-4xl">
              자동매매는 선택 사항으로, <br /> 리포트 확인 뒤에만 이어집니다.
            </h2>
            <p className="mt-5 text-sm leading-7 text-slate-500">
              서비스의 중심은 매수 유도가 아니라 정보 정리와 근거 확인입니다.
              실행 기능은 사용자가 원할 때만 별도 설정으로 연결됩니다.
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              {partnerLogos.map((logo) => (
                <span
                  key={logo}
                  className="rounded-full border border-slate-200 bg-white px-4 py-2 text-xs font-black text-slate-500"
                >
                  {logo}
                </span>
              ))}
            </div>
          </div>
        </section>

        <footer className="mx-auto max-w-[1120px] px-4 pb-4 pt-16 text-center">
          <div className="flex flex-wrap items-center justify-center gap-3 text-xs font-semibold text-slate-500 sm:text-sm">
            <Link href="/privacy" className="transition hover:text-slate-900">
              개인정보 처리방침
            </Link>
            <span className="text-slate-300">|</span>
            <Link href="/terms" className="transition hover:text-slate-900">
              유의사항
            </Link>
            <span className="text-slate-300">|</span>
            <span>Gentle Viking</span>
          </div>
          <p className="mt-6 text-xs font-medium leading-6 text-slate-400 sm:text-sm">
            제공하는 투자 정보는 고객의 투자 판단을 위한 단순 참고용일 뿐, 투자
            제안 및 권유, 종목 추천을 위해 작성된 것이 아닙니다.
          </p>
        </footer>
      </main>
    </div>
  );
}
