import { FiArrowUpRight } from "react-icons/fi";
import type { StockListItem } from "@/lib/stock-list/StockListContext";

type PortfolioAsideProps = {
  stocks: StockListItem[];
};

function parseWon(value: string) {
  const parsed = Number(value.replace(/[^0-9.-]/g, ""));
  return Number.isFinite(parsed) ? parsed : 0;
}

export default function PortfolioAside({ stocks }: PortfolioAsideProps) {
  // 아직 실제 보유 수량이 없는 종목들이라 "보유 평가액 기준 비중"은 계산할 수 없다. 대신
  // 바구니 종목의 현재가 비율로 보여준다(동일 비중 1/3/1/3/1/3으로 고정돼 있던 걸 실제 시세
  // 기반 계산으로 교체 -- 전부 같은 수량을 산다고 가정했을 때의 비중).
  const prices = stocks.map((stock) => parseWon(stock.price));
  const totalPrice = prices.reduce((sum, price) => sum + price, 0);

  return (
    <aside className="space-y-5">
      <section className="rounded-2xl bg-white p-6 shadow-[0_24px_80px_rgba(15,23,42,0.06)]">
        <h2 className="font-black">비중</h2>
        <p className="mt-1 text-xs font-medium text-slate-400">
          종목당 1주씩 산다고 가정했을 때 현재가 기준 비중
        </p>
        <div className="mt-5 space-y-4">
          {stocks.length > 0 && totalPrice > 0 ? (
            stocks.map((stock, index) => {
              const ratio = prices[index] / totalPrice;
              const width = `${Math.round(ratio * 100)}%`;

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
              {stocks.length > 0 ? "현재가를 불러오는 중입니다." : "담긴 종목이 없습니다."}
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
