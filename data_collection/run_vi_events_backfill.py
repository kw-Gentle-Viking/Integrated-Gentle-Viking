"""Runner script to backfill vi_events (VI 발동/해제 이력) for the 200-ticker universe,
using KIS TR FHPST01390000 (변동성완화장치(VI) 현황).

See data_collection/backfill_vi_status.py module docstring for the full research trail behind
why this TR (not the FHKST02900200 tried in Task 8, and not KRX's APIs) is used.

Strategy:
1. For each KRX trading day (calendar.is_market_open = true) in [2019-01-02, latest price_daily
   date], query market-wide VI status twice: FID_MRKT_CLS_CODE='K' (KOSPI/거래소) and 'Q' (KOSDAQ),
   with FID_INPUT_ISCD='' (no ticker filter). Filter each response's rows down to the 200-ticker
   universe and upsert immediately (idempotent, ON CONFLICT DO UPDATE).
2. This market-wide query is observed to cap at 30 rows per call (verified live 2026-09-09).
   Any (market, date) whose raw (unfiltered) response hits exactly 30 rows is recorded as
   "capped" — on such a day, events beyond the top 30 for that market may have been silently
   dropped from the sweep, so it is NOT reliable evidence of completeness for that day.
3. For every capped day, re-verify with per-ticker queries (FID_INPUT_ISCD=<ticker>) for each
   universe ticker in that market — per-ticker queries are NOT capped (verified live: a ticker
   with 4 real events on 2026-05-27 returned exactly its own 4, not clipped to 30). This closes
   the gap for exactly the volatile days we most care about (e.g. 2026-05-27). Bounded by
   MAX_SUPPLEMENTARY_CALLS to avoid unbounded runtime if far more days turn out capped than
   expected; anything beyond the cap is logged as an explicit, uncorrected limitation.

Environment variables required:
- STOCK_DB_V2_DSN: PostgreSQL connection string
- KIS_APP_KEY, KIS_APP_SECRET: Korea Investment & Securities API credentials

IMPORTANT: Run OUTSIDE the production KIS collection window (08:55-15:30, 15:50-16:10 KST
weekdays) — data_collection.kis_client.assert_outside_production_window enforces this per call.
"""

import os
import sys
import time
import psycopg2
from data_collection.kis_client import KisClient
from data_collection.backfill_vi_status import parse_vi_status_response, raw_row_count
from data_collection.backfill_leverage import upsert_vi_events

KIS_API_INTERVAL = 0.056
TR_ID = "FHPST01390000"
API_PATH = "/uapi/domestic-stock/v1/quotations/inquire-vi-status"
MAX_SUPPLEMENTARY_CALLS = 4000

MARKETS = ["K", "Q"]  # K=거래소(KOSPI), Q=코스닥(KOSDAQ) — per FID_MRKT_CLS_CODE docstring


def call_vi_status(client, mrkt_cls_code, iscd, date_str):
    params = {
        "FID_DIV_CLS_CODE": "0",
        "FID_COND_SCR_DIV_CODE": "20139",
        "FID_MRKT_CLS_CODE": mrkt_cls_code,
        "FID_INPUT_ISCD": iscd,
        "FID_RANK_SORT_CLS_CODE": "0",
        "FID_INPUT_DATE_1": date_str,
        "FID_TRGT_CLS_CODE": "",
        "FID_TRGT_EXLS_CLS_CODE": "",
    }
    raw = client.request(path=API_PATH, tr_id=TR_ID, params=params)
    time.sleep(KIS_API_INTERVAL)
    return raw


def main():
    dsn = os.environ.get("STOCK_DB_V2_DSN")
    kis_app_key = os.environ.get("KIS_APP_KEY")
    kis_app_secret = os.environ.get("KIS_APP_SECRET")
    if not dsn or not kis_app_key or not kis_app_secret:
        raise RuntimeError("Missing STOCK_DB_V2_DSN / KIS_APP_KEY / KIS_APP_SECRET")

    client = KisClient(app_key=kis_app_key, app_secret=kis_app_secret)

    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("SELECT ticker, is_kospi FROM ticker_universe")
        universe_rows = cur.fetchall()
        cur.execute("SELECT MIN(trade_date), MAX(trade_date) FROM price_daily")
        min_date, max_date = cur.fetchone()
        cur.execute(
            "SELECT base_date FROM calendar WHERE is_market_open = true "
            "AND base_date BETWEEN %s AND %s ORDER BY base_date",
            (min_date, max_date),
        )
        trading_days = [row[0] for row in cur.fetchall()]
    conn.close()

    universe_tickers = {t for t, _ in universe_rows}
    kospi_tickers = {t for t, is_kospi in universe_rows if is_kospi}
    kosdaq_tickers = {t for t, is_kospi in universe_rows if not is_kospi}
    market_ticker_sets = {"K": kospi_tickers, "Q": kosdaq_tickers}

    print(f"Universe: {len(universe_tickers)} tickers ({len(kospi_tickers)} KOSPI, "
          f"{len(kosdaq_tickers)} KOSDAQ)")
    print(f"Trading days: {len(trading_days)} ({trading_days[0]} .. {trading_days[-1]})")

    capped_days = []  # list of (market, date_str)
    total_calls = 0
    total_rows_upserted = 0
    day_errors = 0

    for i, day in enumerate(trading_days, 1):
        date_str = day.strftime("%Y%m%d")
        day_rows = []
        for mrkt in MARKETS:
            try:
                raw = call_vi_status(client, mrkt, "", date_str)
                total_calls += 1
            except Exception as e:
                print(f"  [warn] {date_str} mrkt={mrkt}: {e}")
                day_errors += 1
                continue
            n = raw_row_count(raw)
            if n == 30:
                capped_days.append((mrkt, date_str))
            day_rows.extend(parse_vi_status_response(raw, market_ticker_sets[mrkt]))

        if day_rows:
            upsert_vi_events(dsn, day_rows)
            total_rows_upserted += len(day_rows)

        if i % 200 == 0 or i == len(trading_days):
            print(f"[{i}/{len(trading_days)}] {date_str} — "
                  f"{total_calls} calls, {total_rows_upserted} rows upserted so far, "
                  f"{len(capped_days)} capped (market,date) pairs so far")

    print(f"\nMain sweep done: {total_calls} calls, {total_rows_upserted} rows, "
          f"{day_errors} day-level errors, {len(capped_days)} capped (market, date) pairs.")

    # Supplementary per-ticker verification for capped days
    supplementary_calls = 0
    supplementary_rows = 0
    skipped_capped_days = []
    if capped_days:
        print(f"\nRe-verifying {len(capped_days)} capped (market, date) pairs via per-ticker "
              f"queries (bounded to {MAX_SUPPLEMENTARY_CALLS} extra calls)...")
        for mrkt, date_str in capped_days:
            if supplementary_calls >= MAX_SUPPLEMENTARY_CALLS:
                skipped_capped_days.append((mrkt, date_str))
                continue
            for ticker in market_ticker_sets[mrkt]:
                if supplementary_calls >= MAX_SUPPLEMENTARY_CALLS:
                    skipped_capped_days.append((mrkt, date_str))
                    break
                try:
                    raw = call_vi_status(client, mrkt, ticker, date_str)
                    supplementary_calls += 1
                except Exception as e:
                    print(f"  [warn] supplementary {date_str} {ticker}: {e}")
                    continue
                rows = parse_vi_status_response(raw, {ticker})
                if rows:
                    upsert_vi_events(dsn, rows)
                    supplementary_rows += len(rows)

        print(f"Supplementary sweep done: {supplementary_calls} calls, "
              f"{supplementary_rows} additional rows upserted.")
        if skipped_capped_days:
            print(f"  ⚠ {len(skipped_capped_days)} capped (market,date) pairs were NOT "
                  f"re-verified (MAX_SUPPLEMENTARY_CALLS reached): {skipped_capped_days}")

    # Verification summary
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM vi_events")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT ticker) FROM vi_events")
        distinct_tickers = cur.fetchone()[0]
        cur.execute("SELECT MIN(triggered_at), MAX(triggered_at) FROM vi_events")
        min_t, max_t = cur.fetchone()
        cur.execute(
            "SELECT DATE(triggered_at) AS d, COUNT(*) FROM vi_events "
            "GROUP BY d ORDER BY COUNT(*) DESC LIMIT 15"
        )
        top_days = cur.fetchall()
        cur.execute("SELECT vi_type, COUNT(*) FROM vi_events GROUP BY vi_type")
        by_type = cur.fetchall()
    conn.close()

    print("\n" + "=" * 60)
    print(f"vi_events final: {total} rows, {distinct_tickers} distinct tickers, "
          f"range {min_t} .. {max_t}")
    print(f"By vi_type: {by_type}")
    print("Top 15 busiest days (event count):")
    for d, c in top_days:
        print(f"  {d}: {c}")
    print(f"\nCapped (market,date) pairs found: {len(capped_days)}")
    print(f"Capped pairs NOT re-verified: {len(skipped_capped_days)}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
