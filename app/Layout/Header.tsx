"use client";

import type { CSSProperties } from "react";
import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { FiBell, FiSearch } from "react-icons/fi";
import { useSidebar } from "@/components/sidebar/SidebarContext";
import { AUTH_EVENT_NAME, AuthUser, getCurrentUser } from "@/lib/signup/auth";

export default function Header() {
  const pathname = usePathname();
  const { isOpen } = useSidebar();
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);
  const isStandalonePage =
    pathname === "/" || pathname === "/login" || pathname === "/signup";

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

  if (isStandalonePage) return null;

  const shellStyle = {
    "--sidebar-width": isOpen ? "340px" : "88px",
  } as CSSProperties;
  const profileName = currentUser?.nickname ?? "Gentle Viking";

  return (
    <header
      className="sticky top-0 z-30 border-b border-slate-200/70 bg-white backdrop-blur"
      style={shellStyle}
    >
      <div className="ml-0 flex h-[72px] items-center justify-between px-5 transition-[margin] duration-300 lg:ml-[var(--sidebar-width)] lg:px-8">
        <div className="relative w-full max-w-md">
          <FiSearch className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="종목, 리포트, 뉴스 검색"
            className="h-11 w-full rounded-xl border border-slate-100 bg-white pl-11 pr-4 text-sm text-slate-700 placeholder:text-slate-400 outline-none transition focus:border-[#5267ff] focus:shadow-[0_0_0_4px_rgba(82,103,255,0.12)]"
          />
        </div>

        <div className="ml-4 flex items-center gap-3">
          <Link
            href="/notifications"
            title="알림"
            className="relative flex h-11 w-11 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-500 transition hover:text-slate-950"
          >
            <FiBell className="h-5 w-5" />
            <span className="absolute right-2.5 top-2.5 h-2 w-2 rounded-full bg-orange-400" />
          </Link>
          <Link
            href="/mypage?tab=profile"
            className="hidden items-center gap-3 rounded-xl border border-slate-200 bg-white px-3 py-2 transition hover:border-[#5267ff]/40 sm:flex"
          >
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-slate-950 text-xs font-black text-white">
              {profileName.slice(0, 1)}
            </div>
            <div>
              <p className="text-xs font-bold text-slate-950">
                {profileName}님
              </p>
            </div>
          </Link>
        </div>
      </div>
    </header>
  );
}
