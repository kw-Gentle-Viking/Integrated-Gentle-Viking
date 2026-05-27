"use client";

import { createPortal } from "react-dom";
import { FiX } from "react-icons/fi";
import type { StockListItem } from "@/lib/stock-list/StockListContext";

type WatchlistDeleteModalProps = {
  stock: StockListItem;
  onConfirm: () => void;
  onCancel: () => void;
};

export default function WatchlistDeleteModal({
  stock,
  onConfirm,
  onCancel,
}: WatchlistDeleteModalProps) {
  if (typeof document === "undefined") return null;

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 px-5 backdrop-blur-sm">
      <div className="w-full max-w-sm rounded-2xl bg-white p-6 shadow-[0_24px_80px_rgba(15,23,42,0.22)]">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-lg font-black text-slate-950">관심 종목 삭제</p>
            <p className="mt-2 text-sm leading-6 text-slate-500">
              {stock.name}을 관심 종목에서 삭제할까요?
            </p>
          </div>
          <button
            type="button"
            onClick={onCancel}
            title="닫기"
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-slate-400 transition hover:bg-slate-100 hover:text-slate-950"
          >
            <FiX className="h-4 w-4" />
          </button>
        </div>
        <div className="mt-6 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="h-11 rounded-xl bg-slate-100 px-4 text-sm font-black text-slate-600 transition hover:bg-slate-200"
          >
            취소
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className="h-11 rounded-xl bg-rose-500 px-4 text-sm font-black text-white transition hover:bg-rose-600"
          >
            삭제하기
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
