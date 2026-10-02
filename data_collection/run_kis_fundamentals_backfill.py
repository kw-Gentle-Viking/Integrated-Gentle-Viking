"""
Runner script to backfill valuation, investor flow, and market index data for the top-200 ticker universe
from 2019-01-02 to present.

This script:
1. Queries ticker_universe table for the 200-ticker list
2. For each ticker, backfills daily valuation (PER, PBR, market cap)
3. For each ticker, backfills daily investor flow (individual/foreign/institutional net buy/sell)
4. Backfills market index data (KOSPI 0001 and KOSDAQ 1001)

Environment variables required:
- STOCK_DB_V2_DSN: PostgreSQL connection string
- KIS_APP_KEY: Korea Investment & Securities API key
- KIS_APP_SECRET: Korea Investment & Securities API secret

IMPORTANT: Run this script OUTSIDE the production KIS collection window (08:55-15:30, 15:50-16:10)
to avoid token conflicts with existing collector processes. Use nohup for long-running backfills:
  nohup /path/to/python run_kis_fundamentals_backfill.py >> backfill_fundamentals.log 2>&1 &
"""

import os
import sys
import time
import psycopg2
from data_collection.kis_client import KisClient
from data_collection.backfill_kis_fundamentals import (
    backfill_valuation,
    backfill_investor_flow,
    backfill_market_indices,
)

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

    # Start date for backfill
    start_date = "20190102"

    # Backfill market indices (once for entire date range)
    print("\nBackfilling market indices (KOSPI 0001 and KOSDAQ 1001)...")
    try:
        backfill_market_indices(client, dsn, start_date=start_date)
        print("  Market indices completed successfully")
    except Exception as e:
        print(f"  ERROR backfilling market indices: {e}")

    # Backfill valuation and investor flow for each ticker
    print("\nBackfilling valuation and investor flow data for each ticker...")
    successful = 0
    failed = 0
    failed_tickers = []

    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"[{i:3d}/{len(tickers)}] Backfilling {ticker}...")

            # Backfill valuation (PER, PBR, market cap)
            backfill_valuation(client, dsn, ticker, start_date=start_date)
            time.sleep(KIS_API_INTERVAL)

            # Backfill investor flow (individual/foreign/institutional net amounts)
            backfill_investor_flow(client, dsn, ticker, start_date=start_date)
            time.sleep(KIS_API_INTERVAL)

            successful += 1
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
        # Check valuation data
        cur.execute(
            "SELECT ticker, COUNT(*) FROM daily_valuation GROUP BY ticker "
            "ORDER BY COUNT(*) DESC LIMIT 10"
        )
        print("\nTop 10 tickers by valuation record count:")
        for ticker, count in cur.fetchall():
            print(f"  {ticker}: {count:6d} records")

        # Check investor flow data
        cur.execute(
            "SELECT ticker, COUNT(*) FROM investor_flow_daily GROUP BY ticker "
            "ORDER BY COUNT(*) DESC LIMIT 10"
        )
        print("\nTop 10 tickers by investor flow record count:")
        for ticker, count in cur.fetchall():
            print(f"  {ticker}: {count:6d} records")

        # Check market index data
        cur.execute(
            "SELECT index_code, COUNT(*) FROM market_index_daily GROUP BY index_code"
        )
        print("\nMarket index record counts:")
        for index_code, count in cur.fetchall():
            index_name = "KOSPI" if index_code == "0001" else "KOSDAQ" if index_code == "1001" else index_code
            print(f"  {index_name} ({index_code}): {count:6d} records")

        # Overall counts
        cur.execute("SELECT COUNT(DISTINCT ticker) FROM daily_valuation")
        val_tickers = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM daily_valuation")
        val_records = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT ticker) FROM investor_flow_daily")
        flow_tickers = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM investor_flow_daily")
        flow_records = cur.fetchone()[0]

    conn.close()

    print(f"\nValuation: {val_tickers} tickers with {val_records} records")
    print(f"Investor Flow: {flow_tickers} tickers with {flow_records} records")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
