"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  FiBarChart2,
  FiBell,
  FiChevronLeft,
  FiCreditCard,
  FiFileText,
  FiGrid,
  FiLogOut,
  FiHeart,
  FiPieChart,
  FiSettings,
  FiShield,
  FiUser,
} from "react-icons/fi";
import { TbReportAnalytics } from "react-icons/tb";
import { useSidebar } from "@/components/sidebar/SidebarContext";
import { logoutUser } from "@/lib/signup/auth";

const NAV_ITEMS = [
  { name: "홈", href: "/dashboard", icon: FiGrid },
  { name: "AI 리포트", href: "/ai-report", icon: TbReportAnalytics },
  { name: "내 주식보기", href: "/portfolio", icon: FiBarChart2 },
  { name: "관심 종목", href: "/watchlist", icon: FiHeart },
  { name: "마이페이지", href: "/mypage?tab=assets", icon: FiUser },
];

const MYPAGE_ITEMS = [
  { key: "assets", label: "자산", icon: FiCreditCard },
  { key: "orders", label: "주문내역", icon: FiShield },
  { key: "AItrades", label: "AI자동거래내역", icon: FiFileText },
  { key: "analysis", label: "수익분석", icon: FiPieChart },
  { key: "profile", label: "내 정보 변경", icon: FiUser },
];

const SECONDARY_ITEMS = [
  { name: "알림", href: "/notifications", icon: FiBell, badge: "3" },
  { name: "설정", href: "/settings", icon: FiSettings },
];

export default function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const searchParams = useSearchParams();
  const { isOpen, toggleSidebar } = useSidebar();
  const isStandalonePage =
    pathname === "/" || pathname === "/login" || pathname === "/signup";
  if (isStandalonePage) return null;

  const activeMypageTab = searchParams.get("tab") ?? "assets";
  const isMypage = pathname === "/mypage";

  const handleLogout = async () => {
    await logoutUser();
    router.push("/");
  };

  const linkBaseClass = isOpen
    ? "flex items-center justify-between rounded-lg px-3 py-2.5 text-sm font-semibold transition-all"
    : "mx-auto flex h-11 w-11 items-center justify-center rounded-lg text-sm font-semibold transition-all";

  const renderLink = (
    item: (typeof NAV_ITEMS)[number] | (typeof SECONDARY_ITEMS)[number],
  ) => {
    const Icon = item.icon;
    const itemPath = item.href.split("?")[0];
    const isActive =
      pathname === itemPath || (itemPath === "/dashboard" && pathname === "/");

    return (
      <Link
        key={`${item.name}-${item.href}`}
        href={item.href}
        title={isOpen ? undefined : item.name}
        className={`${linkBaseClass} ${
          isActive
            ? "bg-[#5267ff] text-white shadow-[0_10px_24px_rgba(82,103,255,0.28)]"
            : "text-slate-500 hover:bg-white hover:text-slate-950"
        }`}
      >
        <span
          className={`flex items-center ${isOpen ? "gap-3" : "justify-center"}`}
        >
          <Icon className="h-4 w-4 shrink-0" />
          {isOpen ? item.name : null}
        </span>
        {isOpen && "badge" in item && item.badge ? (
          <span
            className={`flex h-5 min-w-5 items-center justify-center rounded-full px-1 text-[10px] ${isActive ? "bg-white/20 text-white" : "bg-orange-100 text-orange-600"}`}
          >
            {item.badge}
          </span>
        ) : null}
      </Link>
    );
  };

  return (
    <aside
      className={`fixed inset-y-0 left-0 z-40 hidden flex-col border-r border-slate-200/70 bg-[#f6f8fc] py-6 transition-all duration-300 lg:flex ${
        isOpen ? "w-[340px] px-8" : "w-[88px] items-center px-4"
      }`}
    >
      <div className="pointer-events-none absolute left-0 top-0 h-[360px] w-[300px] rounded-full bg-[radial-gradient(circle_at_35%_35%,rgba(82,103,255,0.26),rgba(125,211,252,0.18)_38%,rgba(244,114,182,0.12)_62%,transparent_72%)] blur-3xl" />
      <div
        className={`relative z-10 mb-8 flex w-full items-center ${isOpen ? "justify-between" : "justify-center"}`}
      >
        {isOpen ? (
          <Link href="/dashboard" className="flex items-center">
            <Image
              src="/genvi.png"
              alt="Gentle Viking"
              width={118}
              height={58}
              priority
              className="h-12 w-auto object-contain"
            />
          </Link>
        ) : (
          <button
            type="button"
            onClick={toggleSidebar}
            title="사이드바 열기"
            className="group relative flex h-12 w-12 items-center justify-center rounded-xl transition hover:bg-white hover:shadow-[0_10px_24px_rgba(15,23,42,0.08)]"
          >
            <Image
              src="/only_logo.png"
              alt="Gentle Viking"
              width={44}
              height={44}
              priority
              className="h-11 w-11 rounded-xl object-contain"
            />
            <span className="pointer-events-none absolute left-[calc(100%+10px)] top-1/2 z-50 -translate-y-1/2 whitespace-nowrap rounded-lg bg-slate-950 px-3 py-2 text-xs font-bold text-white opacity-0 shadow-lg transition group-hover:opacity-100">
              사이드바 열기
            </span>
          </button>
        )}
        {isOpen ? (
          <button
            type="button"
            onClick={toggleSidebar}
            title="사이드바 닫기"
            className="group relative flex h-12 w-12 items-center justify-center rounded-xl text-slate-400 transition hover:bg-white hover:text-slate-950 hover:shadow-[0_10px_24px_rgba(15,23,42,0.08)]"
          >
            <FiChevronLeft className="h-5 w-5" />
            <span className="pointer-events-none absolute left-[calc(100%+10px)] top-1/2 z-50 -translate-y-1/2 whitespace-nowrap rounded-lg bg-slate-950 px-3 py-2 text-xs font-bold text-white opacity-0 shadow-lg transition group-hover:opacity-100">
              사이드바 닫기
            </span>
          </button>
        ) : null}
      </div>

      <nav
        className={`relative z-10 w-full space-y-1.5 ${isOpen ? "" : "flex flex-col items-center"}`}
      >
        {NAV_ITEMS.map((item) => (
          <div key={item.href} className="w-full">
            {renderLink(item)}
            {isOpen && item.name === "마이페이지" && isMypage ? (
              <div className="mt-2 space-y-1 border-l border-slate-200 pl-4">
                {MYPAGE_ITEMS.map((subItem) => {
                  const Icon = subItem.icon;
                  const isActive = activeMypageTab === subItem.key;

                  return (
                    <Link
                      key={subItem.key}
                      href={`/mypage?tab=${subItem.key}`}
                      className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${
                        isActive
                          ? "bg-[#eef2ff] font-black text-[#5267ff]"
                          : "font-bold text-slate-500 hover:bg-white hover:text-slate-950"
                      }`}
                    >
                      <Icon className="h-4 w-4 shrink-0" />
                      {subItem.label}
                    </Link>
                  );
                })}
              </div>
            ) : null}
          </div>
        ))}
      </nav>

      <div
        className={`relative z-10 mt-auto w-full space-y-1.5 ${isOpen ? "" : "flex flex-col items-center"}`}
      >
        {SECONDARY_ITEMS.map(renderLink)}
        <button
          onClick={handleLogout}
          title={isOpen ? undefined : "로그아웃"}
          className={`${linkBaseClass} w-full ${isOpen ? "" : "mx-auto"} text-slate-500 hover:bg-white hover:text-slate-950`}
        >
          <span
            className={`flex items-center ${isOpen ? "gap-3" : "justify-center"}`}
          >
            <FiLogOut className="h-4 w-4" />
            {isOpen ? "로그아웃" : null}
          </span>
        </button>
      </div>
    </aside>
  );
}
