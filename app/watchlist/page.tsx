"use client";

import { useState } from "react";
import WatchlistDeleteModal from "@/components/watchlist/WatchlistDeleteModal";
import WatchlistTable from "@/components/watchlist/WatchlistTable";
import { StockListItem, useStockList } from "@/lib/stock-list/StockListContext";

export default function WatchlistPage() {
  const { favoriteStocks, toggleFavorite } = useStockList();
  const [deleteTarget, setDeleteTarget] = useState<StockListItem | null>(null);

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
              <h1 className="mt-2 text-4xl font-black text-slate-950">
                관심 종목
              </h1>
              <p className="mt-3 text-sm leading-6 text-slate-500">
                AI 추천 리포트나 실시간 종목 랭킹에서 하트를 누른 종목을 모아 확인합니다.
              </p>
            </div>
          </div>

          <WatchlistTable
            stocks={favoriteStocks}
            onRequestDelete={setDeleteTarget}
          />
        </section>
      </div>

      {deleteTarget ? (
        <WatchlistDeleteModal
          stock={deleteTarget}
          onConfirm={handleDelete}
          onCancel={() => setDeleteTarget(null)}
        />
      ) : null}
    </>
  );
}
