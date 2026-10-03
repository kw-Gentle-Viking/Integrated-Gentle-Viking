"""stock_db_v2(AI 200종목)의 공통 소스 테이블을 운영 stock_db에서 증분 복사한다.

운영 수집 크론(collector_batch / collector_kis / collector_yf_fred / collector_dart)은 stock_db(350종목)에만
저장하므로, AI가 읽는 stock_db_v2는 이 스크립트로 채운다. 테이블별로 stock_db_v2의 현재 최대 날짜 이후
행만 가져와 ON CONFLICT DO NOTHING으로 넣는다(멱등).

    python -m data_collection.sync_v2_from_prod
    python -m data_collection.sync_v2_from_prod --start 2026-09-09 --end 2026-10-02   # 명시 범위
"""
import argparse
import os
from datetime import date

import psycopg2
import psycopg2.extras

PROD_DSN = os.getenv("PROD_STOCK_DB_DSN", "postgresql://localhost/stock_db")

# (table, columns, date_col, ticker_filtered, select_exprs)
TABLES = [
    ("price_daily",
     ["ticker", "trade_date", "open_price", "high_price", "low_price", "close_price", "volume", "turnover",
      "shares_outstanding"], "trade_date", True, None),
    ("daily_valuation", ["ticker", "trade_date", "per", "pbr", "market_cap"], "trade_date", True, None),
    ("investor_flow_daily",
     ["ticker", "trade_date", "individual_net_amt", "foreign_net_amt", "inst_net_amt", "market_cap"],
     "trade_date", True, None),
    ("stock_events", ["ticker", "event_date", "event_type", "description"], "event_date", True, None),
    ("market_global",
     ["trade_date", "snp500_close", "nasdaq_close", "phlx_semi_close", "vix", "wti_crude_oil", "gold_price",
      "usd_krw", "us_10y_yield", "fed_rate", "kr_base_rate"], "trade_date", False, None),
    ("market_index_daily", ["index_code", "trade_date", "close_price"], "trade_date", False, None),
    ("sector_daily_ohlcv", ["sector_code", "trade_date", "open", "high", "low", "close", "volume"],
     "trade_date", False, None),
    ("market_events", ["event_date", "event_type", "is_bok", "is_fomc", "is_witching_kr", "is_witching_us"],
     "event_date", False,
     ["event_date", "event_type", "is_bok::boolean", "is_fomc::boolean", "is_witching_kr::boolean",
      "is_witching_us::boolean"]),
    ("calendar", ["base_date", "day_of_week", "is_market_open", "is_holiday", "is_short_selling_banned"],
     "base_date", False,
     ["base_date", "day_of_week", "is_market_open::boolean", "is_holiday::boolean",
      "is_short_selling_banned::boolean"]),
]


def _ai_tickers(v2):
    with v2.cursor() as c:
        c.execute("SELECT ticker FROM ticker_universe")
        return [r[0] for r in c.fetchall()]


def _v2_max(v2, table, date_col):
    with v2.cursor() as c:
        c.execute(f"SELECT MAX({date_col}) FROM {table}")
        return c.fetchone()[0]


def sync(v2_dsn, start=None, end=None):
    v2 = psycopg2.connect(v2_dsn)
    prod = psycopg2.connect(PROD_DSN)
    end = end or date.today().isoformat()
    tickers = _ai_tickers(v2)
    print(f"AI tickers: {len(tickers)}, end={end}")
    try:
        for table, cols, date_col, ticker_filtered, select_exprs in TABLES:
            lo = start or (_v2_max(v2, table, date_col) or date(2019, 1, 1)).isoformat()
            with prod.cursor() as pc:
                sel = ", ".join(select_exprs or cols)
                if ticker_filtered:
                    pc.execute(f"SELECT {sel} FROM {table} WHERE {date_col} > %s AND {date_col} <= %s "
                               f"AND ticker = ANY(%s)", (lo, end, tickers))
                else:
                    pc.execute(f"SELECT {sel} FROM {table} WHERE {date_col} > %s AND {date_col} <= %s",
                               (lo, end))
                rows = pc.fetchall()
            if rows:
                with v2.cursor() as vc:
                    psycopg2.extras.execute_values(
                        vc, f"INSERT INTO {table} ({', '.join(cols)}) VALUES %s ON CONFLICT DO NOTHING",
                        rows, page_size=1000)
                v2.commit()
            print(f"{table}: {lo} < date <= {end} -> {len(rows)} rows")
    finally:
        v2.close()
        prod.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=None, help="이 날짜 초과분만 복사 (기본: v2 테이블별 최대 날짜)")
    p.add_argument("--end", default=None)
    args = p.parse_args()
    v2_dsn = os.environ["STOCK_DB_V2_DSN"]
    sync(v2_dsn, args.start, args.end)


if __name__ == "__main__":
    main()
