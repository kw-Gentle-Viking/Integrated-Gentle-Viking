import { FiHeart, FiTrash2 } from "react-icons/fi";
import type { StockListItem } from "@/lib/stock-list/StockListContext";

type WatchlistTableProps = {
  stocks: StockListItem[];
  onRequestDelete: (stock: StockListItem) => void;
};

export default function WatchlistTable({
  stocks,
  onRequestDelete,
}: WatchlistTableProps) {
  return (
    <div className="mt-6 overflow-hidden rounded-xl border border-slate-100">
      <div className="grid grid-cols-[1.2fr_0.8fr_0.8fr_1.4fr_72px] bg-slate-50 px-5 py-3 text-xs font-bold text-slate-400">
        <span>종목</span>
        <span>현재가</span>
        <span>등락률</span>
        <span>메모</span>
        <span className="text-right">관리</span>
      </div>
      {stocks.length > 0 ? (
        stocks.map((stock) => (
          <div
            key={stock.code}
            className="grid grid-cols-[1.2fr_0.8fr_0.8fr_1.4fr_72px] items-center border-t border-slate-100 px-5 py-4 text-sm"
          >
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
            <span
              className={`font-black ${
                stock.change.startsWith("-") ? "text-blue-500" : "text-rose-500"
              }`}
            >
              {stock.change}
            </span>
            <span className="line-clamp-1 text-slate-500">
              {stock.memo ?? "AI 추천 리포트에서 저장한 관심 종목"}
            </span>
            <button
              type="button"
              onClick={() => onRequestDelete(stock)}
              title={`${stock.name} 삭제`}
              className="ml-auto flex h-9 w-9 items-center justify-center rounded-lg text-slate-400 transition hover:bg-rose-50 hover:text-rose-500"
            >
              <FiTrash2 className="h-4 w-4" />
            </button>
          </div>
        ))
      ) : (
        <div className="border-t border-slate-100 px-5 py-12 text-center">
          <p className="text-sm font-black text-slate-500">
            아직 관심 종목이 없습니다.
          </p>
          <p className="mt-2 text-sm text-slate-400">
            AI 리포트에서 하트를 누르면 이곳에 바로 추가됩니다.
          </p>
        </div>
      )}
    </div>
  );
}
