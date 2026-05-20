"use client";

import { useEffect, useState } from "react";
import { AUTH_EVENT_NAME, getCurrentUser } from "@/lib/signup/auth";

export default function ProfileSettings() {
  const [nickname, setNickname] = useState("Gentle Viking");
  const [email, setEmail] = useState("test@gentleviking.com");
  const [riskLevel, setRiskLevel] = useState("균형형");
  const [autoTrade, setAutoTrade] = useState(true);

  useEffect(() => {
    const syncUser = () => {
      const user = getCurrentUser();
      if (!user) return;
      setNickname(user.nickname);
      setEmail(user.email);
    };

    syncUser();
    window.addEventListener(AUTH_EVENT_NAME, syncUser);
    window.addEventListener("storage", syncUser);

    return () => {
      window.removeEventListener(AUTH_EVENT_NAME, syncUser);
      window.removeEventListener("storage", syncUser);
    };
  }, []);

  return (
    <form className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2">
        <label className="block">
          <span className="text-sm font-black text-slate-700">닉네임</span>
          <input
            value={nickname}
            onChange={(event) => setNickname(event.target.value)}
            className="mt-2 h-12 w-full rounded-xl border border-slate-200 bg-white px-4 text-sm font-bold text-slate-900 outline-none transition focus:border-[#5267ff] focus:shadow-[0_0_0_4px_rgba(82,103,255,0.12)]"
          />
        </label>
        <label className="block">
          <span className="text-sm font-black text-slate-700">이메일</span>
          <input
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            className="mt-2 h-12 w-full rounded-xl border border-slate-200 bg-white px-4 text-sm font-bold text-slate-900 outline-none transition focus:border-[#5267ff] focus:shadow-[0_0_0_4px_rgba(82,103,255,0.12)]"
          />
        </label>
      </div>

      <section className="rounded-2xl border border-slate-100 bg-white p-5 shadow-[0_12px_32px_rgba(15,23,42,0.04)]">
        <h2 className="text-lg font-black text-slate-950">투자 성향</h2>
        <div className="mt-4 grid gap-2 sm:grid-cols-3">
          {["안정형", "균형형", "공격형"].map((level) => (
            <button
              key={level}
              type="button"
              onClick={() => setRiskLevel(level)}
              className={`h-11 rounded-full border text-sm font-black transition ${
                riskLevel === level
                  ? "border-[#5267ff] bg-[#eef2ff] text-[#5267ff]"
                  : "border-slate-200 bg-white text-slate-500 hover:border-slate-300"
              }`}
            >
              {level}
            </button>
          ))}
        </div>
      </section>

      <section className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-slate-100 bg-slate-50 p-5">
        <div>
          <h2 className="text-lg font-black text-slate-950">
            AI 자동거래 알림
          </h2>
          <p className="mt-1 text-sm font-medium text-slate-400">
            체결과 리스크 변동을 알림으로 받습니다.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setAutoTrade((current) => !current)}
          className={`flex h-8 w-14 items-center rounded-full p-1 transition ${autoTrade ? "bg-[#5267ff]" : "bg-slate-300"}`}
          aria-pressed={autoTrade}
        >
          <span
            className={`h-6 w-6 rounded-full bg-white shadow-sm transition ${autoTrade ? "translate-x-6" : ""}`}
          />
        </button>
      </section>

      <div className="flex justify-end">
        <button
          type="button"
          className="h-12 rounded-full bg-slate-950 px-6 text-sm font-black text-white transition hover:bg-slate-800"
        >
          변경사항 저장
        </button>
      </div>
    </form>
  );
}
