"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { FiX, FiAlertTriangle, FiCheckCircle } from "react-icons/fi";
import { placeOrder } from "@/lib/api/trade";
import type { OrderSide, PriceType } from "@/lib/api/trade";

interface Props {
  stock: {
    code: string;
    name: string;
    price: string;
    isUp: boolean;
    change: string;
  };
  initialSide: OrderSide;
  onClose: () => void;
}

function parsePrice(str: string): number {
  return parseInt(str.replace(/[^0-9]/g, ""), 10) || 0;
}

export default function TradeModal({ stock, initialSide, onClose }: Props) {
  const [mounted, setMounted] = useState(false);
  const [side, setSide] = useState<OrderSide>(initialSide);
  const [priceType, setPriceType] = useState<PriceType>("limit");
  const [qty, setQty] = useState(1);
  const [price, setPrice] = useState(parsePrice(stock.price));
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [resultMsg, setResultMsg] = useState("");

  useEffect(() => setMounted(true), []);

  const total = priceType === "market" ? null : qty * price;
  const isBuy = side === "buy";

  const handleQty = (delta: number) =>
    setQty((v) => Math.max(1, v + delta));

  const handleSubmit = async () => {
    setStatus("loading");
    try {
      const result = await placeOrder({
        code: stock.code,
        side,
        qty,
        price: priceType === "market" ? 0 : price,
        priceType,
      });
      setResultMsg(result.message || "주문이 완료됐습니다.");
      setStatus("success");
    } catch (err) {
      setResultMsg(err instanceof Error ? err.message : "주문 처리 중 오류가 발생했습니다.");
      setStatus("error");
    }
  };

  if (!mounted) return null;

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/40 backdrop-blur-sm">
      <div className="relative w-full max-w-sm rounded-2xl bg-white shadow-[0_32px_80px_rgba(15,23,42,0.18)]">

        {/* 헤더 */}
        <div className={`flex items-center justify-between rounded-t-2xl px-6 py-4 ${isBuy ? "bg-rose-50" : "bg-blue-50"}`}>
          <div>
            <p className={`text-xs font-bold ${isBuy ? "text-rose-400" : "text-blue-400"}`}>
              모의투자 · {stock.code}
            </p>
            <h2 className="mt-0.5 text-xl font-black text-slate-950">{stock.name}</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-400 transition hover:bg-white/60 hover:text-slate-700"
          >
            <FiX className="h-4 w-4" />
          </button>
        </div>

        <div className="p-6">
          {status === "success" || status === "error" ? (
            /* 결과 화면 */
            <div className="py-4 text-center">
              {status === "success" ? (
                <FiCheckCircle className="mx-auto h-12 w-12 text-emerald-500" />
              ) : (
                <FiAlertTriangle className="mx-auto h-12 w-12 text-rose-500" />
              )}
              <p className="mt-4 text-base font-black text-slate-950">
                {status === "success" ? "주문 완료" : "주문 실패"}
              </p>
              <p className="mt-1 text-sm font-bold text-slate-500">{resultMsg}</p>
              <button
                type="button"
                onClick={onClose}
                className="mt-6 h-11 w-full rounded-xl bg-slate-950 text-sm font-black text-white transition hover:bg-slate-800"
              >
                확인
              </button>
            </div>
          ) : (
            <>
              {/* 매수/매도 탭 */}
              <div className="grid grid-cols-2 rounded-xl bg-slate-100 p-1">
                {(["buy", "sell"] as OrderSide[]).map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setSide(s)}
                    className={`h-9 rounded-lg text-sm font-black transition ${
                      side === s
                        ? s === "buy"
                          ? "bg-rose-500 text-white shadow-sm"
                          : "bg-blue-500 text-white shadow-sm"
                        : "text-slate-400 hover:text-slate-700"
                    }`}
                  >
                    {s === "buy" ? "매수" : "매도"}
                  </button>
                ))}
              </div>

              {/* 현재가 */}
              <div className="mt-4 flex items-center justify-between rounded-xl bg-slate-50 px-4 py-3">
                <span className="text-xs font-bold text-slate-400">현재가</span>
                <div className="text-right">
                  <span className="text-base font-black text-slate-950">{stock.price}</span>
                  <span className={`ml-2 text-xs font-bold ${stock.isUp ? "text-rose-500" : "text-blue-500"}`}>
                    {stock.change}
                  </span>
                </div>
              </div>

              {/* 주문유형 */}
              <div className="mt-4">
                <p className="mb-1.5 text-xs font-bold text-slate-400">주문유형</p>
                <div className="grid grid-cols-2 gap-2">
                  {(["limit", "market"] as PriceType[]).map((t) => (
                    <button
                      key={t}
                      type="button"
                      onClick={() => setPriceType(t)}
                      className={`h-9 rounded-xl border text-sm font-bold transition ${
                        priceType === t
                          ? "border-[#5267ff] bg-[#eef2ff] text-[#5267ff]"
                          : "border-slate-200 text-slate-400 hover:border-slate-300"
                      }`}
                    >
                      {t === "limit" ? "지정가" : "시장가"}
                    </button>
                  ))}
                </div>
              </div>

              {/* 수량 */}
              <div className="mt-4">
                <p className="mb-1.5 text-xs font-bold text-slate-400">수량</p>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => handleQty(-1)}
                    className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-slate-200 text-lg font-bold text-slate-500 transition hover:bg-slate-50"
                  >
                    −
                  </button>
                  <input
                    type="number"
                    min={1}
                    value={qty}
                    onChange={(e) => setQty(Math.max(1, parseInt(e.target.value) || 1))}
                    className="h-10 w-full rounded-xl border border-slate-200 text-center text-sm font-black text-slate-950 outline-none transition focus:border-[#5267ff] focus:shadow-[0_0_0_3px_rgba(82,103,255,0.12)]"
                  />
                  <button
                    type="button"
                    onClick={() => handleQty(1)}
                    className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-slate-200 text-lg font-bold text-slate-500 transition hover:bg-slate-50"
                  >
                    +
                  </button>
                </div>
              </div>

              {/* 가격 (지정가만) */}
              {priceType === "limit" && (
                <div className="mt-4">
                  <p className="mb-1.5 text-xs font-bold text-slate-400">주문가격</p>
                  <div className="flex items-center gap-2 rounded-xl border border-slate-200 px-4 transition focus-within:border-[#5267ff] focus-within:shadow-[0_0_0_3px_rgba(82,103,255,0.12)]">
                    <input
                      type="number"
                      min={1}
                      value={price}
                      onChange={(e) => setPrice(Math.max(1, parseInt(e.target.value) || 1))}
                      className="h-10 w-full text-sm font-black text-slate-950 outline-none"
                    />
                    <span className="text-xs font-bold text-slate-400">원</span>
                  </div>
                </div>
              )}

              {/* 주문금액 */}
              <div className="mt-4 flex items-center justify-between rounded-xl bg-slate-50 px-4 py-3">
                <span className="text-xs font-bold text-slate-400">주문금액</span>
                <span className="text-base font-black text-slate-950">
                  {total != null ? `${total.toLocaleString("ko-KR")}원` : "시장가"}
                </span>
              </div>

              {/* 안내 */}
              <div className="mt-3 flex items-start gap-2 rounded-xl bg-amber-50 px-3 py-2.5">
                <FiAlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" />
                <p className="text-xs font-bold leading-5 text-amber-700">
                  모의투자 계좌에 적용됩니다. 실제 자산에 영향을 주지 않습니다.
                </p>
              </div>

              {/* 버튼 */}
              <div className="mt-5 grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={onClose}
                  className="h-11 rounded-xl border border-slate-200 text-sm font-bold text-slate-600 transition hover:bg-slate-50"
                >
                  취소
                </button>
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={status === "loading"}
                  className={`h-11 rounded-xl text-sm font-black text-white transition disabled:opacity-60 ${
                    isBuy ? "bg-rose-500 hover:bg-rose-600" : "bg-blue-500 hover:bg-blue-600"
                  }`}
                >
                  {status === "loading" ? "처리 중..." : isBuy ? "매수 주문" : "매도 주문"}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
