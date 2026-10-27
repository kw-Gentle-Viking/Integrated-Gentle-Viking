"use client";

import { FiBell, FiLock, FiSettings } from "react-icons/fi";

const settings = [
  { title: "알림 설정", desc: "가격 알림, AI 리포트 알림, 거래 알림 수신 여부를 관리합니다.", icon: FiBell },
  { title: "보안 설정", desc: "로그인 보안과 계정 접근 옵션을 관리합니다.", icon: FiLock },
];

export default function SettingsPage() {
  return (
    <section className="rounded-2xl bg-white p-6 shadow-[0_24px_80px_rgba(15,23,42,0.06)] lg:p-8">
      <div className="mb-8 flex items-start gap-4 border-b border-slate-100 pb-6">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-[#eef2ff] text-[#5267ff]">
          <FiSettings className="h-5 w-5" />
        </div>
        <div>
          <p className="text-sm font-bold text-[#5267ff]">Settings</p>
          <h1 className="mt-2 text-4xl font-black text-slate-950">설정</h1>
          <p className="mt-3 text-sm leading-6 text-slate-500">내 정보가 아닌 앱 이용 환경과 보안 관련 사항을 관리합니다.</p>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {settings.map((item) => {
          const Icon = item.icon;
          return (
            <article key={item.title} className="rounded-xl border border-slate-100 p-5">
              <div className="flex items-start justify-between gap-3">
                <Icon className="h-6 w-6 text-[#5267ff]" />
                <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] font-black text-slate-400">준비 중</span>
              </div>
              <p className="mt-4 text-lg font-black text-slate-950">{item.title}</p>
              <p className="mt-2 text-sm leading-6 text-slate-500">{item.desc}</p>
            </article>
          );
        })}
      </div>
    </section>
  );
}
