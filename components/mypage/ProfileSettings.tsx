"use client";

import { useEffect, useState } from "react";
import { AUTH_EVENT_NAME, apiFetch, getCurrentUser, updateCurrentUserFields } from "@/lib/signup/auth";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function ProfileSettings() {
  const [nickname, setNickname] = useState("Gentle Viking");
  const [email, setEmail] = useState("test@gentleviking.com");
  const [riskLabel, setRiskLabel] = useState("-");
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    const syncUser = () => {
      const user = getCurrentUser();
      if (!user) return;
      setNickname(user.nickname);
      setEmail(user.email);
      setRiskLabel(user.riskLabel || "-");
    };

    syncUser();
    window.addEventListener(AUTH_EVENT_NAME, syncUser);
    window.addEventListener("storage", syncUser);

    return () => {
      window.removeEventListener(AUTH_EVENT_NAME, syncUser);
      window.removeEventListener("storage", syncUser);
    };
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setSaveMessage(null);
    setSaveError(null);
    try {
      const res = await apiFetch(`${API_BASE}/users/me`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nickname }),
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({})) as { detail?: string };
        throw new Error(payload.detail || `저장 실패 (${res.status})`);
      }
      updateCurrentUserFields({ nickname });
      setSaveMessage("닉네임을 저장했습니다.");
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : "저장 중 오류가 발생했습니다.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <form className="space-y-6" onSubmit={(event) => event.preventDefault()}>
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
            disabled
            readOnly
            title="이메일은 이 화면에서 변경할 수 없습니다."
            className="mt-2 h-12 w-full cursor-not-allowed rounded-xl border border-slate-200 bg-slate-50 px-4 text-sm font-bold text-slate-400 outline-none"
          />
        </label>
      </div>

      <section className="rounded-2xl border border-slate-100 bg-white p-5 shadow-[0_12px_32px_rgba(15,23,42,0.04)]">
        <h2 className="text-lg font-black text-slate-950">투자 성향</h2>
        <p className="mt-2 text-sm font-bold text-[#5267ff]">{riskLabel}</p>
        <p className="mt-1 text-xs font-medium text-slate-400">
          회원가입 설문 결과로 정해집니다. 바꾸려면 다시 설문해야 하며, 이 화면에서는 변경할 수 없습니다.
        </p>
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
        <span className="rounded-full bg-slate-200 px-3 py-1 text-xs font-black text-slate-500">
          준비 중
        </span>
      </section>

      <div className="flex items-center justify-end gap-3">
        {saveMessage ? <span className="text-sm font-bold text-emerald-600">{saveMessage}</span> : null}
        {saveError ? <span className="text-sm font-bold text-rose-600">{saveError}</span> : null}
        <button
          type="button"
          onClick={handleSave}
          disabled={saving}
          className="h-12 rounded-full bg-slate-950 px-6 text-sm font-black text-white transition hover:bg-slate-800 disabled:opacity-50"
        >
          {saving ? "저장 중..." : "변경사항 저장"}
        </button>
      </div>
    </form>
  );
}
