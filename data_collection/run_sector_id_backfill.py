#!/usr/bin/env python
"""
Backfill ticker_universe.sector_id from KIS current-quote industry names.

No task in the original 18-task plan ever populated ticker_universe.sector_id
(Task 3's save_universe() only wrote ticker/snapshot_date/market_cap/rank/is_kospi).
This is Task 10 review-fix: back-fill sector_id for all 200 tickers using the KIS
current-quote endpoint's bstp_kor_isnm (업종명) field, matched against the 20 known
"업종지수" sector names already used for sector_daily_ohlcv (sector_id 0-19), falling
back to sector_id=20 ("unclassified") for names outside that 20-name taxonomy.

Usage:
    STOCK_DB_V2_DSN="postgresql://..." KIS_APP_KEY="..." KIS_APP_SECRET="..." \
    python -m data_collection.run_sector_id_backfill

Do NOT run during production collection windows (08:55-15:30, 15:50-16:10).
"""
import os
import sys
from data_collection.kis_client import KisClient
from data_collection.backfill_sector_id import run_backfill, UNCLASSIFIED_SECTOR_ID


def main() -> None:
    dsn = os.environ.get("STOCK_DB_V2_DSN")
    app_key = os.environ.get("KIS_APP_KEY")
    app_secret = os.environ.get("KIS_APP_SECRET")
    if not dsn or not app_key or not app_secret:
        print("ERROR: STOCK_DB_V2_DSN, KIS_APP_KEY, KIS_APP_SECRET must all be set")
        sys.exit(1)

    client = KisClient(app_key, app_secret)
    sector_id_by_ticker = run_backfill(dsn, client)

    matched = sum(1 for sid in sector_id_by_ticker.values() if sid != UNCLASSIFIED_SECTOR_ID)
    unclassified = sum(1 for sid in sector_id_by_ticker.values() if sid == UNCLASSIFIED_SECTOR_ID)
    print(f"\nDone. {matched} tickers matched a real sector (0-19), "
          f"{unclassified} fell back to unclassified (20).")


if __name__ == "__main__":
    main()
