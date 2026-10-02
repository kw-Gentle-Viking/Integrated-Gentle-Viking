#!/usr/bin/env python
"""
Runner script for macro data backfill from yfinance and FRED.
Fetches global macro indicators and upserts into market_global table.
"""
import os
from datetime import datetime, timedelta
import pandas as pd
from data_collection.backfill_yf_fred import fetch_yfinance, fetch_fred, upsert_market_global


def run_backfill(start_date: str = "2019-01-01", dsn: str = None):
    """
    Backfill market_global table with yfinance and FRED data.

    Args:
        start_date: Target start date (YYYY-MM-DD format). Actual fetch starts ~30 days earlier.
        dsn: Database DSN. If None, reads from STOCK_DB_V2_DSN environment variable.
    """
    if dsn is None:
        dsn = os.environ.get("STOCK_DB_V2_DSN")
        if not dsn:
            raise ValueError("STOCK_DB_V2_DSN environment variable must be set")

    # Parse dates
    target_start = datetime.strptime(start_date, "%Y-%m-%d")
    # Fetch from ~30 days before target start to ensure ffill has data
    actual_fetch_start = target_start - timedelta(days=30)
    today = datetime.now()

    fetch_start_str = actual_fetch_start.strftime("%Y-%m-%d")
    fetch_end_str = today.strftime("%Y-%m-%d")
    target_start_str = start_date

    print(f"Fetching yfinance data from {fetch_start_str} to {fetch_end_str}...")
    yf_data = fetch_yfinance(fetch_start_str, fetch_end_str)

    print(f"Fetching FRED data from {fetch_start_str} to {fetch_end_str}...")
    fred_data = fetch_fred(fetch_start_str, fetch_end_str)

    # Merge data by date
    print("Merging data by date...")
    all_dates = sorted(set(yf_data.keys()) | set(fred_data.keys()))
    merged_data = {}
    for date in all_dates:
        row = {}
        if date in yf_data:
            row.update(yf_data[date])
        if date in fred_data:
            row.update(fred_data[date])
        merged_data[date] = row

    # Convert to DataFrame for forward fill and weekend removal
    df = pd.DataFrame.from_dict(merged_data, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Forward fill missing values
    print("Forward filling missing values...")
    df = df.fillna(method="ffill")

    # Remove weekends (weekday() returns 5=Sat, 6=Sun)
    print("Removing weekends...")
    df = df[df.index.weekday < 5]

    # Filter to target date range and later
    df = df[df.index >= pd.to_datetime(target_start_str)]

    # Upsert each row to database
    print(f"Upserting {len(df)} rows to market_global table...")
    for trade_date, row in df.iterrows():
        row_dict = {col: val for col, val in row.items() if pd.notna(val)}
        trade_date_str = trade_date.strftime("%Y-%m-%d")
        try:
            upsert_market_global(dsn, trade_date_str, row_dict)
            print(f"  Upserted {trade_date_str}: {len(row_dict)} fields")
        except Exception as e:
            print(f"  ERROR upserting {trade_date_str}: {e}")

    print("Backfill complete!")


if __name__ == "__main__":
    run_backfill()
