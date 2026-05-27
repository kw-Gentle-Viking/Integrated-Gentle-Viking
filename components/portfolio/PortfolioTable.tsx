import type { StockListItem } from "@/lib/stock-list/StockListContext";

type PortfolioTableProps = {
  stocks: StockListItem[];
};

export default function PortfolioTable({ stocks }: PortfolioTableProps) {
  return (
    <div className="mt-8 overflow-hidden rounded-xl border border-slate-100">
      <div className="grid grid-cols-[1.4fr_0.8fr_0.9fr_0.9fr_0.7fr] bg-slate-50 px-5 py-3 text-xs font-bold text-slate-400">
        <span>종목</span>
        <span>보유</span>
        <span>현재가</span>
        <span>평가금액</span>
        <span className="text-right">수익률</span>
      </div>
      {stocks.length > 0 ? (
        stocks.map((stock) => (
          <div
            key={stock.code}
            className="grid grid-cols-[1.4fr_0.8fr_0.9fr_0.9fr_0.7fr] items-center border-t border-slate-100 px-5 py-4 text-sm"
          >
            <div className="flex items-center gap-3">
              <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-100 font-black">
                {stock.logoText ?? stock.name.slice(0, 1)}
              </span>
              <div>
                <p className="font-black text-slate-950">{stock.name}</p>
                <p className="text-xs text-slate-400">{stock.code}</p>
              </div>
            </div>
            <span className="font-semibold text-slate-500">대기</span>
            <span className="font-bold">{stock.price}</span>
            <span className="font-bold">{stock.price}</span>
            <span
              className={`text-right font-black ${
                stock.change.startsWith("-") ? "text-blue-500" : "text-rose-500"
              }`}
            >
              {stock.change}
            </span>
          </div>
        ))
      ) : (
        <div className="border-t border-slate-100 px-5 py-12 text-center">
          <p className="text-sm font-black text-slate-500">
            아직 포트폴리오에 담은 종목이 없습니다.
          </p>
          <p className="mt-2 text-sm text-slate-400">
            AI 리포트에서 가방 아이콘을 누르면 이곳에 바로 추가됩니다.
          </p>
        </div>
      )}
    </div>
  );
}
