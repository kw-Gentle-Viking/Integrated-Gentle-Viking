"""Fully per-ticker (uncapped) VI verification for the 200-ticker universe over a given
trading-day window.

Why this exists: `run_vi_events_backfill.py`'s main sweep queries VI status market-wide
(one call per (market, date)), which is efficient but caps at ~30 rows per call — hit on
~87% of trading days in the live 2019-01-02~2026-09-08 run (2026-09-09 backlog item, see
`docs/data_units.md`'s "vi_events 데이터 소스 재조사 + 백필" section and
`.superpowers/sdd/2026-09-08-ai-model-redesign-plan/backlog-vi-events-report.md`). The
main sweep's own chronological fallback (see `run_vi_events_backfill.py`) re-verifies
capped days one ticker at a time, but its fixed call budget gets exhausted working through
capped days in date order — so it never reaches recent, high-interest windows if the
early years alone have more capped days than the budget covers.

This script closes that gap for one window at a time: for every ticker in
`ticker_universe`, one call per ticker per trading day in [start_date, end_date] (never
capped, since it's one ticker's own VI events per call, not a market-wide list). Use it
for any window where you need trustworthy VI coverage regardless of how capped the main
sweep's market-wide response was that day — e.g. the original run was created to verify
the 2026-05-27 single-stock leveraged-ETF launch window.

Usage:
    set -a && source .env && set +a
    PYTHONPATH=. /home/user/miniconda3/envs/kis_collector/bin/python \
        data_collection/run_vi_events_targeted_window.py 2026-05-13 2026-06-10

Do NOT run during production collection windows (08:55-15:30, 15:50-16:10 KST) —
KisClient.request() enforces this via assert_outside_production_window().
"""
import os
import sys
import time

import psycopg2

from data_collection.kis_client import KisClient
from data_collection.backfill_vi_status import parse_vi_status_response
from data_collection.backfill_leverage import upsert_vi_events

KIS_API_INTERVAL = 0.056
TR_ID = "FHPST01390000"
API_PATH = "/uapi/domestic-stock/v1/quotations/inquire-vi-status"


def load_universe_and_days(dsn: str, start_date: str, end_date: str):
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT ticker, is_kospi FROM ticker_universe")
            universe = cur.fetchall()
            cur.execute(
                "SELECT base_date FROM calendar WHERE is_market_open=true "
                "AND base_date BETWEEN %s AND %s ORDER BY base_date",
                (start_date, end_date),
            )
            days = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()
    return universe, days


def run_targeted_window(dsn: str, client: KisClient, start_date: str, end_date: str) -> tuple[int, int]:
    """Per-ticker, per-day VI status calls (uncapped) for every ticker across the window.

    Returns (total_calls, total_rows_upserted).
    """
    universe, days = load_universe_and_days(dsn, start_date, end_date)
    print(f"Targeted verification: {len(universe)} tickers x {len(days)} days = "
          f"{len(universe) * len(days)} calls", flush=True)

    total_calls = 0
    total_rows = 0
    for ticker, is_kospi in universe:
        mrkt = "K" if is_kospi else "Q"
        for day in days:
            date_str = day.strftime("%Y%m%d")
            params = {
                "FID_DIV_CLS_CODE": "0", "FID_COND_SCR_DIV_CODE": "20139",
                "FID_MRKT_CLS_CODE": mrkt, "FID_INPUT_ISCD": ticker,
                "FID_RANK_SORT_CLS_CODE": "0", "FID_INPUT_DATE_1": date_str,
                "FID_TRGT_CLS_CODE": "", "FID_TRGT_EXLS_CLS_CODE": "",
            }
            try:
                raw = client.request(path=API_PATH, tr_id=TR_ID, params=params)
                total_calls += 1
            except Exception as e:
                print(f"  [warn] {ticker} {date_str}: {e}", flush=True)
                continue
            time.sleep(KIS_API_INTERVAL)
            rows = parse_vi_status_response(raw, {ticker})
            if rows:
                upsert_vi_events(dsn, rows)
                total_rows += len(rows)
        if total_calls % 1000 < len(days):
            print(f"  progress: {total_calls} calls, {total_rows} rows so far", flush=True)

    print(f"DONE: {total_calls} calls, {total_rows} rows upserted (targeted window "
          f"{start_date}~{end_date})", flush=True)
    return total_calls, total_rows


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: run_vi_events_targeted_window.py <start_date YYYY-MM-DD> <end_date YYYY-MM-DD>")
        sys.exit(1)
    start_date, end_date = sys.argv[1], sys.argv[2]

    dsn = os.environ["STOCK_DB_V2_DSN"]
    client = KisClient(app_key=os.environ["KIS_APP_KEY"], app_secret=os.environ["KIS_APP_SECRET"])
    run_targeted_window(dsn, client, start_date, end_date)


if __name__ == "__main__":
    main()
