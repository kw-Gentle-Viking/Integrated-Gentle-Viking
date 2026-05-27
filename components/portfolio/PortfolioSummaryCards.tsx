type PortfolioSummaryCardsProps = {
  totalValue: string;
  stockCount: number;
};

export default function PortfolioSummaryCards({
  totalValue,
  stockCount,
}: PortfolioSummaryCardsProps) {
  return (
    <div className="mt-8 grid gap-4 md:grid-cols-3">
      <div className="rounded-xl bg-[#f4f7ff] p-5">
        <p className="text-sm font-bold text-slate-500">평가 금액</p>
        <p className="mt-2 text-3xl font-black">{totalValue}</p>
      </div>
      <div className="rounded-xl bg-emerald-50 p-5">
        <p className="text-sm font-bold text-slate-500">담은 종목</p>
        <p className="mt-2 text-3xl font-black text-emerald-500">
          {stockCount}개
        </p>
      </div>
      <div className="rounded-xl bg-rose-50 p-5">
        <p className="text-sm font-bold text-slate-500">오늘 변동</p>
        <p className="mt-2 text-3xl font-black text-rose-500">
          AI 리포트 연동
        </p>
      </div>
    </div>
  );
}
