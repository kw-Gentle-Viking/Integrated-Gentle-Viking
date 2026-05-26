"use client";

import { useState } from "react";
import { createPortal } from "react-dom";
import { FiHeart, FiTrash2, FiX } from "react-icons/fi";
import { StockListItem, useStockList } from "@/lib/stock-list/StockListContext";

export default function WatchlistPage() {
  const { favoriteStocks, toggleFavorite } = useStockList();
  const [deleteTarget, setDeleteTarget] = useState<StockListItem | null>(null);
  const modalRoot = typeof document === "undefined" ? null : document.body;

  const handleDelete = () => {
    if (!deleteTarget) return;
    toggleFavorite(deleteTarget.code, deleteTarget);
    setDeleteTarget(null);
  };

  return (
    <>
      <div>
        <section className="rounded-2xl bg-white p-6 shadow-[0_24px_80px_rgba(15,23,42,0.06)] lg:p-8">
          <div className="flex flex-wrap items-start justify-between gap-5 border-b border-slate-100 pb-6">
            <div>
              <p className="text-sm font-bold text-[#5267ff]">Watchlist</p>
              <h1 className="mt-2 text-4xl font-black text-slate-950">관심 종목</h1>
              <p className="mt-3 text-sm leading-6 text-slate-500">
                AI 추천 리포트나 실시간 종목 랭킹에서 하트를 누른 종목을 모아 확인합니다.
              </p>
            </div>
          </div>

          <div className="mt-6 overflow-hidden rounded-xl border border-slate-100">
            <div className="grid grid-cols-[1.2fr_0.8fr_0.8fr_1.4fr_72px] bg-slate-50 px-5 py-3 text-xs font-bold text-slate-400">
              <span>종목</span>
              <span>현재가</span>
              <span>등락률</span>
              <span>메모</span>
              <span className="text-right">관리</span>
            </div>
            {favoriteStocks.length > 0 ? favoriteStocks.map((stock) => (
              <div key={stock.code} className="grid grid-cols-[1.2fr_0.8fr_0.8fr_1.4fr_72px] items-center border-t border-slate-100 px-5 py-4 text-sm">
                <div className="flex items-center gap-3">
                  <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-rose-50 text-rose-500">
                    <FiHeart className="h-4 w-4 fill-current" />
                  </span>
                  <div>
                    <p className="font-black text-slate-950">{stock.name}</p>
                    <p className="text-xs text-slate-400">{stock.code}</p>
                  </div>
                </div>
                <span className="font-bold text-slate-950">{stock.price}</span>
                <span className={`font-black ${stock.change.startsWith("-") ? "text-blue-500" : "text-rose-500"}`}>{stock.change}</span>
                <span className="line-clamp-1 text-slate-500">{stock.memo ?? "AI 추천 리포트에서 저장한 관심 종목"}</span>
                <button
                  type="button"
                  onClick={() => setDeleteTarget(stock)}
                  title={`${stock.name} 삭제`}
                  className="ml-auto flex h-9 w-9 items-center justify-center rounded-lg text-slate-400 transition hover:bg-rose-50 hover:text-rose-500"
                >
                  <FiTrash2 className="h-4 w-4" />
                </button>
              </div>
            )) : (
              <div className="border-t border-slate-100 px-5 py-12 text-center">
                <p className="text-sm font-black text-slate-500">아직 관심 종목이 없습니다.</p>
                <p className="mt-2 text-sm text-slate-400">AI 리포트에서 하트를 누르면 이곳에 바로 추가됩니다.</p>
              </div>
            )}
          </div>
        </section>
      </div>

      {modalRoot && deleteTarget
        ? createPortal(
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 px-5 backdrop-blur-sm">
          <div className="w-full max-w-sm rounded-2xl bg-white p-6 shadow-[0_24px_80px_rgba(15,23,42,0.22)]">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-lg font-black text-slate-950">관심 종목 삭제</p>
                <p className="mt-2 text-sm leading-6 text-slate-500">
                  {deleteTarget.name}을 관심 종목에서 삭제할까요?
                </p>
              </div>
              <button
                type="button"
                onClick={() => setDeleteTarget(null)}
                title="닫기"
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-slate-400 transition hover:bg-slate-100 hover:text-slate-950"
              >
                <FiX className="h-4 w-4" />
              </button>
            </div>
            <div className="mt-6 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setDeleteTarget(null)}
                className="h-11 rounded-xl bg-slate-100 px-4 text-sm font-black text-slate-600 transition hover:bg-slate-200"
              >
                취소
              </button>
              <button
                type="button"
                onClick={handleDelete}
                className="h-11 rounded-xl bg-rose-500 px-4 text-sm font-black text-white transition hover:bg-rose-600"
              >
                삭제하기
              </button>
            </div>
          </div>
        </div>,
        modalRoot,
        )
        : null}
    </>
  );
}
