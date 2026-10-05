"""한 달 검증 리포트: (1) AI 신호 적중률(다음 거래일 수익률 기준), (2) 체결 기록의 실현 손익(FIFO).

    python -m scripts.return_report --start 2026-10-06 --end 2026-11-05

입력: 백엔드 DB(trading: ai_prediction_history, trade_logs), 운영 DB(stock_db: price_daily 종가).
장 마감 후 확정 신호(하루 마지막 예측)만 적중률 계산에 쓴다(학습 조건과 맞추기 위함).
"""
import argparse
import os

import pandas as pd


def forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """prices: ticker, trade_date, close → 같은 종목의 다음 거래일 수익률(next_ret)."""
    df = prices.sort_values(["ticker", "trade_date"]).copy()
    df["next_close"] = df.groupby("ticker")["close"].shift(-1)
    df["next_ret"] = df["next_close"] / df["close"] - 1
    return df[["ticker", "trade_date", "next_ret"]].dropna()


def signal_hits(signals: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    """signals: ticker, trade_datetime, signal(BUY/HOLD/SELL). 하루 마지막 신호만 본다.
    BUY는 다음 날 상승이면 적중, SELL은 하락이면 적중. HOLD는 평가 대상에서 제외."""
    s = signals.copy()
    s["trade_date"] = pd.to_datetime(s["trade_datetime"]).dt.date
    last = (s.sort_values("trade_datetime").groupby(["ticker", "trade_date"], as_index=False).tail(1))
    f = fwd.copy()
    f["trade_date"] = pd.to_datetime(f["trade_date"]).dt.date
    m = last.merge(f, on=["ticker", "trade_date"], how="inner")
    m = m[m["signal"].isin(["BUY", "SELL"])].copy()
    m["hit"] = ((m["signal"] == "BUY") & (m["next_ret"] > 0)) | ((m["signal"] == "SELL") & (m["next_ret"] < 0))
    return m


def fifo_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """trades: ticker, side(BUY/SELL), qty, price, created_at → 매도로 실현된 손익 행."""
    out = []
    for ticker, g in trades.sort_values("created_at").groupby("ticker"):
        lots: list[list[float]] = []  # [qty, price]
        for _, t in g.iterrows():
            if t["side"] == "BUY":
                lots.append([float(t["qty"]), float(t["price"])])
                continue
            remain = float(t["qty"])
            while remain > 0 and lots:
                take = min(remain, lots[0][0])
                out.append({"ticker": ticker, "qty": take, "pnl": take * (float(t["price"]) - lots[0][1]),
                            "sold_at": t["created_at"]})
                lots[0][0] -= take
                remain -= take
                if lots[0][0] <= 0:
                    lots.pop(0)
    return pd.DataFrame(out, columns=["ticker", "qty", "pnl", "sold_at"])


def main() -> int:
    import psycopg2
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    trading = psycopg2.connect(os.environ["DATABASE_URL"])
    prod = psycopg2.connect(os.environ["PROD_STOCK_DB_DSN"])
    sig = pd.read_sql("SELECT ticker, trade_datetime, signal FROM ai_prediction_history "
                      "WHERE trade_datetime >= %s AND trade_datetime < %s", trading, params=(args.start, args.end))
    tr = pd.read_sql("SELECT ticker, side, qty, price, created_at FROM trade_logs "
                     "WHERE status = 'FILLED' AND created_at >= %s AND created_at < %s", trading,
                     params=(args.start, args.end))
    px = pd.read_sql("SELECT ticker, trade_date, close_price AS close FROM price_daily "
                     "WHERE trade_date >= %s::date - 5 AND trade_date <= %s::date + 5", prod,
                     params=(args.start, args.end))
    fwd = forward_returns(px)
    hits = signal_hits(sig, fwd) if len(sig) else pd.DataFrame()
    pnl = fifo_pnl(tr) if len(tr) else pd.DataFrame()
    print(f"신호 적중률: {hits['hit'].mean():.1%} ({int(hits['hit'].sum())}/{len(hits)})" if len(hits) else "신호 적중률: 데이터 없음")
    print(f"실현 손익 합계: {pnl['pnl'].sum():,.0f}원 ({len(pnl)}건)" if len(pnl) else "실현 손익: 체결 없음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
