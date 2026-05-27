import { FiArrowUpRight } from "react-icons/fi";
import type { StockListItem } from "@/lib/stock-list/StockListContext";

type PortfolioAsideProps = {
  stocks: StockListItem[];
};

export default function PortfolioAside({ stocks }: PortfolioAsideProps) {
  return (
    <aside className="space-y-5">
      <section className="rounded-2xl bg-white p-6 shadow-[0_24px_80px_rgba(15,23,42,0.06)]">
        <h2 className="font-black">비중</h2>
        <div className="mt-5 space-y-4">
          {stocks.length > 0 ? (
            stocks.slice(0, 3).map((stock) => {
              const width =
                Math.max(18, Math.round(100 / Math.min(stocks.length, 3))) +
                "%";

              return (
                <div key={stock.code}>
                  <div className="mb-2 flex justify-between text-sm font-bold">
                    <span>{stock.name}</span>
                    <span>{width}</span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-100">
                    <div
                      className="h-2 rounded-full bg-[#5267ff]"
                      style={{ width }}
                    />
                  </div>
                </div>
              );
            })
          ) : (
            <p className="text-sm font-bold text-slate-400">
              담긴 종목이 없습니다.
            </p>
          )}
        </div>
      </section>
      <section className="rounded-2xl bg-slate-950 p-6 text-white">
        <p className="text-xl font-black">AI 추천</p>
        <p className="mt-3 text-sm leading-6 text-slate-300">
          AI 리포트에서 담은 종목을 기준으로 포트폴리오 구성을 확인합니다.
        </p>
        <button className="mt-5 flex items-center gap-2 text-sm font-black text-emerald-300">
          자세히 보기 <FiArrowUpRight />
        </button>
      </section>
    </aside>
  );
}
