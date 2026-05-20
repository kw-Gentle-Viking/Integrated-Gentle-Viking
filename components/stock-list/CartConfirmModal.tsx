"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { FiShoppingBag, FiX } from "react-icons/fi";

interface Props {
  stockName: string;
  mode: "add" | "remove";
  onConfirm: () => void;
  onCancel: () => void;
}

export default function CartConfirmModal({ stockName, mode, onConfirm, onCancel }: Props) {
  const isAdd = mode === "add";
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  if (!mounted) return null;

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/40 backdrop-blur-sm">
      <div className="relative w-full max-w-sm rounded-2xl bg-white p-6 shadow-[0_32px_80px_rgba(15,23,42,0.18)]">
        <button
          type="button"
          onClick={onCancel}
          className="absolute right-4 top-4 flex h-8 w-8 items-center justify-center rounded-lg text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
        >
          <FiX className="h-4 w-4" />
        </button>

        <div className={`flex h-12 w-12 items-center justify-center rounded-2xl ${isAdd ? "bg-amber-50" : "bg-slate-100"}`}>
          <FiShoppingBag className={`h-6 w-6 ${isAdd ? "text-amber-500" : "text-slate-500"}`} />
        </div>

        <h2 className="mt-4 text-xl font-black text-slate-950">
          {isAdd ? "포트폴리오에 추가" : "포트폴리오에서 제외"}
        </h2>
        <p className="mt-1 text-sm font-bold text-slate-500">
          <span className="text-slate-800">{stockName}</span>
        </p>

        <div className={`mt-4 rounded-xl px-4 py-3 ${isAdd ? "bg-amber-50" : "bg-slate-50"}`}>
          <p className={`text-xs font-bold leading-5 ${isAdd ? "text-amber-700" : "text-slate-500"}`}>
            {isAdd
              ? <>추가된 종목은 <span className="font-black">익영업일 주식장 시작(오전 9시)</span>에 포트폴리오에 반영됩니다.</>
              : <>제외된 종목은 <span className="font-black">익영업일 주식장 시작(오전 9시)</span>부터 포트폴리오에서 제거됩니다.</>
            }
          </p>
        </div>

        <div className="mt-5 grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="h-11 rounded-xl border border-slate-200 text-sm font-bold text-slate-600 transition hover:bg-slate-50"
          >
            취소
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className={`h-11 rounded-xl text-sm font-black text-white transition ${isAdd ? "bg-amber-400 hover:bg-amber-500" : "bg-slate-700 hover:bg-slate-800"}`}
          >
            확인
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
