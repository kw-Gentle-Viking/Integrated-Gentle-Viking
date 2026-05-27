"use client";

import Link from "next/link";
import {
  FiActivity,
  FiFileText,
  FiHeart,
  FiShoppingBag,
  FiTrendingUp,
} from "react-icons/fi";
import type { StockListItem } from "@/lib/stock-list/StockListContext";
import type { ReasonKey, ReportStock } from "@/lib/ai-report/types";
import { getPredictHref, getSignalBadgeClass } from "./reportHelpers";

const reasonCards: { key: ReasonKey; label: string; icon: typeof FiFileText }[] =
  [
    { key: "news", label: "뉴스", icon: FiFileText },
    { key: "disclosure", label: "공시", icon: FiFileText },
    { key: "flow", label: "재료/수급", icon: FiTrendingUp },
  ];

type ReportCardProps = {
  stock: ReportStock;
  isFavorite: boolean;
  isInCart: boolean;
  onSelect: (stock: ReportStock) => void;
  onToggleFavorite: (stock: ReportStock) => void;
  onRequestCart: (stock: StockListItem) => void;
  makeStockListItem: (stock: ReportStock) => StockListItem;
};

export default function ReportCard({
  stock,
  isFavorite,
  isInCart,
  onSelect,
  onToggleFavorite,
  onRequestCart,
  makeStockListItem,
}: ReportCardProps) {
  return (
    <div
      onClick={() => onSelect(stock)}
      className="block w-full cursor-pointer rounded-2xl border border-slate-100 bg-white p-5 text-left shadow-[0_12px_32px_rgba(15,23,42,0.04)] transition hover:border-[#5267ff]/30 hover:shadow-[0_18px_44px_rgba(15,23,42,0.08)]"
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
                <span
                  className={`rounded-full px-2.5 py-1 text-xs font-black ${getSignalBadgeClass(
                    stock.signal,
                  )}`}
                >
                  {stock.signal}
                </span>
              </div>
              <div className="mt-2 flex items-center gap-3">
                <p className="text-lg font-black text-slate-950">
                  {stock.price}
                </p>
                <p
                  className={`text-sm font-black ${
                    stock.change.startsWith("-")
                      ? "text-blue-500"
                      : "text-rose-500"
                  }`}
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
                onToggleFavorite(stock);
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
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                onRequestCart(makeStockListItem(stock));
              }}
              title="포트폴리오에 추가"
              className={`flex h-10 w-10 items-center justify-center rounded-xl border transition ${
                isInCart
                  ? "border-amber-200 bg-amber-50 text-amber-500"
                  : "border-slate-200 text-slate-400 hover:border-amber-200 hover:bg-amber-50 hover:text-amber-500"
              }`}
            >
              <FiShoppingBag className="h-5 w-5" />
            </button>
            <Link
              href={getPredictHref(stock)}
              onClick={(event) => event.stopPropagation()}
              className="flex h-10 items-center gap-2 rounded-xl border border-indigo-100 bg-gradient-to-r from-indigo-50 via-fuchsia-50 to-rose-50 px-4 text-sm font-black shadow-sm transition hover:border-indigo-200 hover:from-indigo-100 hover:via-fuchsia-100 hover:to-rose-100"
            >
              <FiActivity className="h-4 w-4 text-[#5267ff]" />
              <span className="bg-gradient-to-r from-[#5267ff] via-fuchsia-500 to-rose-500 bg-clip-text text-transparent">
                AI 매매 확률 보기
              </span>
            </Link>
          </div>
        </div>

        <div className="mt-5 grid gap-3 md:grid-cols-[1.1fr_1fr_1fr_120px]">
          {reasonCards.map((item) => {
            const Icon = item.icon;
            const reason = stock.reasons[item.key];

            return (
              <div key={item.key} className="rounded-xl bg-slate-50 p-3">
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
    </div>
  );
}
