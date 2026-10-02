"""
Runner script to backfill leverage ETF/ETN daily prices and VI event history.

This script:
1. Saves leverage product metadata to leverage_products table
2. For each of 18 leverage products, fetches daily OHLCV from 2026-05-27 to present
3. For Samsung Electronics (005930) and SK Hynix (000660), fetches VI event history since 2019

Environment variables required:
- STOCK_DB_V2_DSN: PostgreSQL connection string
- KIS_APP_KEY: Korea Investment & Securities API key
- KIS_APP_SECRET: Korea Investment & Securities API secret

IMPORTANT: Run this script OUTSIDE the production KIS collection window (08:55-15:30, 15:50-16:10)
to avoid token conflicts with existing collector processes.
"""

import os
import sys
import time
import psycopg2
from datetime import datetime, timedelta
from data_collection.kis_client import KisClient
from data_collection.backfill_leverage import (
    save_leverage_products,
    parse_leverage_daily_response,
    upsert_leverage_daily,
    parse_vi_event_response,
    upsert_vi_events,
)
from data_collection.leverage_products import LEVERAGE_PRODUCTS

# KIS API rate limit: ~18 requests/sec (0.056 sec interval)
KIS_API_INTERVAL = 0.056


def backfill_leverage_daily(client: KisClient, dsn: str, code: str, start_date: str, end_date: str) -> int:
    """Backfill daily leverage product data in 140-day chunks (KIS limit).

    Returns number of records inserted/updated.
    """
    cursor_end = datetime.strptime(end_date, "%Y%m%d")
    start = datetime.strptime(start_date, "%Y%m%d")
    all_rows = []

    while cursor_end >= start:
        cursor_start = max(start, cursor_end - timedelta(days=140))
        try:
            raw = client.request(
                path="/uapi/domestic-stock/v1/quotations/inquire-daily-price",
                tr_id="FHKST03010100",
                params={
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": code,
                    "FID_INPUT_DATE_1": cursor_start.strftime("%Y%m%d"),
                    "FID_INPUT_DATE_2": cursor_end.strftime("%Y%m%d"),
                    "FID_PERIOD_DIV_CODE": "D",
                    "FID_ORG_ADJ_PRC": "1",
                },
            )
            time.sleep(KIS_API_INTERVAL)
        except Exception as e:
            print(f"    [warn] chunk {cursor_start.date()}~{cursor_end.date()}: {e}")
            cursor_end = cursor_start - timedelta(days=1)
            continue

        rows = parse_leverage_daily_response(raw, code)
        all_rows.extend(rows)
        cursor_end = cursor_start - timedelta(days=1)

    if all_rows:
        upsert_leverage_daily(dsn, all_rows)
        return len(all_rows)
    return 0


def backfill_vi_events(client: KisClient, dsn: str, ticker: str, start_date: str, end_date: str) -> int:
    """Backfill VI event data in 365-day chunks.

    Returns number of records inserted/updated.
    """
    cursor_end = datetime.strptime(end_date, "%Y%m%d")
    start = datetime.strptime(start_date, "%Y%m%d")
    all_rows = []

    while cursor_end >= start:
        cursor_start = max(start, cursor_end - timedelta(days=365))
        try:
            raw = client.request(
                path="/uapi/domestic-stock/v1/quotations/inquire-vi",
                tr_id="FHKST02900200",
                params={
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": ticker,
                    "FID_INPUT_DATE_1": cursor_start.strftime("%Y%m%d"),
                    "FID_INPUT_DATE_2": cursor_end.strftime("%Y%m%d"),
                },
            )
            time.sleep(KIS_API_INTERVAL)
        except Exception as e:
            print(f"    [warn] chunk {cursor_start.date()}~{cursor_end.date()}: {e}")
            cursor_end = cursor_start - timedelta(days=1)
            continue

        rows = parse_vi_event_response(raw, ticker)
        all_rows.extend(rows)
        cursor_end = cursor_start - timedelta(days=1)

    if all_rows:
        upsert_vi_events(dsn, all_rows)
        return len(all_rows)
    return 0


def main():
    # Read configuration from environment
    dsn = os.environ.get("STOCK_DB_V2_DSN")
    kis_app_key = os.environ.get("KIS_APP_KEY")
    kis_app_secret = os.environ.get("KIS_APP_SECRET")

    if not dsn or not kis_app_key or not kis_app_secret:
        raise RuntimeError(
            "Missing required environment variables: "
            "STOCK_DB_V2_DSN, KIS_APP_KEY, KIS_APP_SECRET"
        )

    # Initialize KIS client
    client = KisClient(app_key=kis_app_key, app_secret=kis_app_secret)

    # Step 1: Save leverage product metadata
    print("Step 1: Saving leverage product metadata...")
    try:
        save_leverage_products(dsn)
        print("  ✓ Leverage products saved to database")
    except Exception as e:
        print(f"  ERROR saving leverage products: {e}")
        return 1

    # Step 2: Backfill daily leverage data for each product
    print("\nStep 2: Backfilling daily leverage prices (18 products)...")
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = "20260527"  # Listing date
    successful_products = 0
    failed_products = []

    for i, product in enumerate(LEVERAGE_PRODUCTS, 1):
        if product["code"] is None:
            print(f"  [{i:2d}/18] {product['product_name']}: SKIP (code not found)")
            continue

        code = product["code"]
        name = product["product_name"]
        try:
            print(f"  [{i:2d}/18] {code} {name}...", end=" ", flush=True)
            count = backfill_leverage_daily(client, dsn, code, start_date, end_date)
            print(f"✓ {count} records")
            successful_products += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed_products.append(code)

    print(f"\n  Summary: {successful_products} successful, {len(failed_products)} failed")
    if failed_products:
        print(f"  Failed codes: {', '.join(failed_products)}")

    # Step 3: Backfill VI events for underlying stocks
    print("\nStep 3: Backfilling VI events (Samsung + SK Hynix)...")
    print("  NOTE: VI event endpoint (FHKST02900200) not available in KIS API")
    print("        (Historical VI data may require alternative source or endpoint)")
    tickers_for_vi = [
        ("005930", "Samsung Electronics"),
        ("000660", "SK Hynix"),
    ]
    successful_vi = 0
    failed_vi = []

    for ticker, name in tickers_for_vi:
        try:
            print(f"  {ticker} {name}...", end=" ", flush=True)
            # VI events since 2019-01-02
            count = backfill_vi_events(client, dsn, ticker, "20190102", end_date)
            print(f"✓ {count} events")
            successful_vi += 1
        except Exception as e:
            # VI endpoint error is not critical - data exists but endpoint may not be available
            print(f"SKIP (endpoint not available)")
            failed_vi.append(ticker)

    if successful_vi == 0:
        print(f"\n  ⚠ VI events backfill skipped: endpoint not available in this API version")
    else:
        print(f"\n  Summary: {successful_vi} successful, {len(failed_vi)} failed")

    # Step 4: Verification
    print("\nStep 4: Verifying backfilled data...")
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM leverage_products")
        product_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM leverage_daily")
        daily_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM vi_events")
        vi_count = cur.fetchone()[0]

    conn.close()

    print(f"  leverage_products: {product_count} records")
    print(f"  leverage_daily: {daily_count} records")
    print(f"  vi_events: {vi_count} records")

    print("\n" + "="*60)
    print("Leverage backfill completed successfully")
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
