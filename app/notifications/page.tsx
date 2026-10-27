"use client";

import { useEffect, useState } from "react";
import { FiBell, FiCheckCircle } from "react-icons/fi";
import { fetchTradeNotifications, type TradeNotification } from "@/lib/notifications/tradeFeed";

export default function NotificationsPage() {
  const [notifications, setNotifications] = useState<TradeNotification[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchTradeNotifications(20)
      .then((items) => {
        if (!cancelled) setNotifications(items);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "알림을 불러오지 못했습니다.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  return (
    <section className="rounded-2xl bg-white p-6 shadow-[0_24px_80px_rgba(15,23,42,0.06)] lg:p-8">
      <div className="mb-8 flex items-start gap-4 border-b border-slate-100 pb-6">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-[#eef2ff] text-[#5267ff]">
          <FiBell className="h-5 w-5" />
        </div>
        <div>
          <p className="text-sm font-bold text-[#5267ff]">Notifications</p>
          <h1 className="mt-2 text-4xl font-black text-slate-950">알림</h1>
          <p className="mt-3 text-sm leading-6 text-slate-500">직접 주문과 AI 자동매매 체결 내역을 알림으로 보여줍니다.</p>
        </div>
      </div>

      <div className="space-y-3">
        {loading ? (
          <p className="py-10 text-center text-sm font-bold text-slate-400">불러오는 중...</p>
        ) : error ? (
          <p className="py-10 text-center text-sm font-bold text-rose-500">{error}</p>
        ) : notifications.length === 0 ? (
          <p className="py-10 text-center text-sm font-bold text-slate-400">아직 체결된 거래가 없습니다.</p>
        ) : (
          notifications.map((item) => (
            <article key={item.id} className="flex items-center justify-between gap-5 rounded-xl border border-slate-100 p-5 transition hover:bg-slate-50">
              <div className="flex items-start gap-3">
                <FiCheckCircle className="mt-0.5 h-5 w-5 text-[#5267ff]" />
                <div>
                  <p className="font-black text-slate-950">{item.title}</p>
                  <p className="mt-1 text-sm text-slate-500">{item.desc}</p>
                </div>
              </div>
              <span className="text-xs font-bold text-slate-400">{item.time}</span>
            </article>
          ))
        )}
      </div>
    </section>
  );
}
