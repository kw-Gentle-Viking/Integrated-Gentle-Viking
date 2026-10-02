"""
Runner script to backfill daily OHLCV price history for the top-200 ticker universe
from 2019-01-02 to present.

This script:
1. Queries ticker_universe table for the 200-ticker list
2. For each ticker, calls backfill_ticker() to fetch daily price history in 140-day chunks
3. Upserts data into price_daily table

Environment variables required:
- STOCK_DB_V2_DSN: PostgreSQL connection string
- KIS_APP_KEY: Korea Investment & Securities API key
- KIS_APP_SECRET: Korea Investment & Securities API secret

IMPORTANT: Run this script OUTSIDE the production KIS collection window (08:55-15:30, 15:50-16:10)
to avoid token conflicts with existing collector processes. Use nohup for long-running backfills:
  nohup /path/to/python run_daily_price_backfill.py >> backfill_daily_price.log 2>&1 &
"""

import os
import sys
import time
import psycopg2
from data_collection.kis_client import KisClient
from data_collection.backfill_daily_price import backfill_ticker

# KIS API rate limit: ~18 requests/sec (0.056 sec interval)
KIS_API_INTERVAL = 0.056


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

    # Query ticker universe
    print("Fetching ticker universe from database...")
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("SELECT ticker FROM ticker_universe ORDER BY rank")
        tickers = [row[0] for row in cur.fetchall()]
    conn.close()

    print(f"Loaded {len(tickers)} tickers from universe")

    # Backfill daily prices for each ticker
    start_date = "20190102"
    successful = 0
    failed = 0
    failed_tickers = []

    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"[{i:3d}/{len(tickers)}] Backfilling {ticker}...")
            backfill_ticker(client, dsn, ticker, start_date=start_date)
            successful += 1
            # Rate limiting: pause between requests to avoid overwhelming KIS API
            time.sleep(KIS_API_INTERVAL)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
            failed_tickers.append(ticker)
            # Continue with next ticker even if one fails
            time.sleep(KIS_API_INTERVAL)

    # Summary
    print("\n" + "="*60)
    print(f"Backfill completed: {successful} successful, {failed} failed")
    if failed_tickers:
        print(f"Failed tickers: {', '.join(failed_tickers)}")
    print("="*60)

    # Verification
    print("\nVerifying backfilled data...")
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ticker, COUNT(*) FROM price_daily GROUP BY ticker "
            "ORDER BY COUNT(*) DESC LIMIT 20"
        )
        print("\nTop 20 tickers by record count:")
        for ticker, count in cur.fetchall():
            print(f"  {ticker}: {count:6d} records")
        cur.execute("SELECT COUNT(DISTINCT ticker) FROM price_daily")
        total_tickers_with_data = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM price_daily")
        total_records = cur.fetchone()[0]
    conn.close()

    print(f"\nTotal: {total_tickers_with_data} tickers with data, {total_records} price records")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
