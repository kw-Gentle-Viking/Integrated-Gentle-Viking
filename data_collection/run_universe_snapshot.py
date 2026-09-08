"""
Runner script to compute the 2019-01-02 top-200 market cap universe snapshot.

This script:
1. Loads candidate tickers from KOSPI200 and KOSDAQ150 CSVs
2. Fetches closing prices and shares outstanding as of 2019-01-02 via KIS API
3. Merges candidates with price data
4. Ranks by market cap and selects top 200
5. Saves to ticker_universe table

Environment variables required:
- STOCK_DB_V2_DSN: PostgreSQL connection string
- KIS_APP_KEY: Korea Investment & Securities API key
- KIS_APP_SECRET: Korea Investment & Securities API secret

IMPORTANT: Run this script OUTSIDE the production KIS collection window (08:55-15:30, 15:50-16:10)
to avoid token conflicts with existing collector processes.
"""

import os
import sys
from pathlib import Path
from data_collection.universe import (
    load_candidate_tickers,
    fetch_snapshot_prices,
    compute_top_n_snapshot,
    save_universe,
)
from data_collection.kis_client import KisClient


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

    snapshot_date = "20190102"
    kospi_csv = "/home/user/KOSPI200_종목리스트.csv"
    kosdaq_csv = "/home/user/KOSDAQ150_종목리스트.csv"

    # Step 1: Load candidate tickers
    print(f"Loading candidate tickers from {kospi_csv} and {kosdaq_csv}...")
    candidates = load_candidate_tickers(kospi_csv, kosdaq_csv)
    print(f"Loaded {len(candidates)} candidates")

    # Step 2: Fetch snapshot prices as of 2019-01-02
    print(f"Fetching snapshot prices as of {snapshot_date}...")
    client = KisClient(app_key=kis_app_key, app_secret=kis_app_secret)
    ticker_list = [c["ticker"] for c in candidates]
    prices = fetch_snapshot_prices(client, ticker_list, snapshot_date)
    print(f"Fetched prices for {len(prices)} tickers")

    # Step 3: Merge candidates with price data
    prices_by_ticker = {p["ticker"]: p for p in prices}
    merged = []
    for candidate in candidates:
        ticker = candidate["ticker"]
        if ticker in prices_by_ticker:
            price_data = prices_by_ticker[ticker]
            merged.append({
                "ticker": ticker,
                "is_kospi": candidate["is_kospi"],
                "close_price": price_data["close_price"],
                "shares_outstanding": price_data["shares_outstanding"],
            })

    print(f"Merged data for {len(merged)} tickers with price information")

    # Step 4: Rank and select top 200
    print("Computing top-200 universe...")
    universe = compute_top_n_snapshot(merged, n=200)
    print(f"Universe contains {len(universe)} stocks")

    # Step 5: Save to database
    print(f"Saving universe to database (snapshot_date={snapshot_date})...")
    save_universe(dsn, universe, snapshot_date)
    print("Universe snapshot saved successfully")

    # Verification
    print("\nTop 10 by market cap:")
    for stock in universe[:10]:
        print(
            f"  Rank {stock['rank']:3d}: {stock['ticker']} "
            f"(market_cap={stock['market_cap']:,.0f})"
        )


if __name__ == "__main__":
    main()
