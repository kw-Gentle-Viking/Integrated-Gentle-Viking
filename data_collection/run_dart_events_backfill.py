#!/usr/bin/env python3
"""
Backfill stock_events from DART disclosure reports ONLY.

This is a dedicated, narrow runner extracted from run_dart_calendar_sector_backfill.py's
main(), which also backfills trading calendar / sector OHLCV (KIS API) / market_events —
all three already have live data in stock_db_v2 (2026-09-08 run), so re-running main()
in full would just be redundant KIS API calls for no new data. This script does ONLY the
DART part, which had never been run live (DART_API_KEY was blank at Task 7 time, so
stock_events was left empty — see docs/data_units.md and the 2026-09-09 backfill report).

Usage:
    set -a && source .env && set +a
    /home/user/miniconda3/envs/kis_collector/bin/python data_collection/run_dart_events_backfill.py

Does not touch the KIS API, but still calls assert_outside_production_window() for
consistency with the rest of this project's backfill scripts.
"""
import os
import sys

from data_collection.kis_client import assert_outside_production_window
from data_collection.run_dart_calendar_sector_backfill import (
    backfill_dart_events, load_tickers_from_db,
)


def main():
    assert_outside_production_window()

    dsn = os.environ.get("STOCK_DB_V2_DSN")
    dart_key = os.environ.get("DART_API_KEY")

    if not dsn:
        print("ERROR: STOCK_DB_V2_DSN not set")
        sys.exit(1)
    if not dart_key:
        print("ERROR: DART_API_KEY not set")
        sys.exit(1)

    print("Starting DART disclosure events backfill (stock_events only)...")
    tickers = load_tickers_from_db(dsn)
    print(f"  Loaded {len(tickers)} tickers from universe")
    backfill_dart_events(dsn, tickers)
    print("\n✓ DART events backfill complete")


if __name__ == "__main__":
    main()
