#!/usr/bin/env python3
"""
Backfill DART disclosure events, trading calendar, and sector index OHLCV.

Usage:
    STOCK_DB_V2_DSN="postgresql://..." KIS_APP_KEY="..." KIS_APP_SECRET="..." \
    DART_API_KEY="..." python run_dart_calendar_sector_backfill.py

This script:
1. Backfills stock_events from DART disclosure reports (via dart_fss API)
2. Builds and upserts trading calendar with short-selling ban periods
3. Backfills sector daily OHLCV from KIS API (20 sector codes)
4. Upserts market_events with BOK, FOMC, and witching day schedules

Do NOT run during production collection windows (08:55-15:30, 15:50-16:10).
"""
import os
import sys
import time
from datetime import datetime, timedelta
import psycopg2
import dart_fss as dart
from data_collection.kis_client import KisClient, assert_outside_production_window
from data_collection.backfill_dart_calendar_sector import (
    parse_dart_reports, parse_sector_daily_response, upsert_stock_events,
    upsert_sector_daily_ohlcv, upsert_calendar, build_calendar_rows,
)


# 원본 캡스톤 프로젝트의 SECTOR_ID_TO_CODE 매핑과 동일(sector_id 0~19 → 20개 섹터).
# 2026-09-08 라이브로 0005~0026 전체 검증: 0022/0023만 무효(빈 응답), 나머지 20개 실존 확인.
SECTOR_CODES = [
    "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0012", "0013", "0014",
    "0015", "0016", "0017", "0018", "0019", "0020", "0021", "0024", "0025", "0026",
]

# KIS API rate limit: ~18 calls per second
KIS_API_INTERVAL = 0.056


def load_tickers_from_db(dsn: str) -> list[str]:
    """Load the 200-ticker universe from ticker_universe table."""
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("SELECT ticker FROM ticker_universe ORDER BY ticker")
        tickers = [row[0] for row in cur.fetchall()]
    conn.close()
    return tickers


def backfill_dart_events(dsn: str, tickers: list[str],
                         start_date: str = "20190102", end_date: str | None = None) -> None:
    """Backfill stock_events from DART disclosure reports."""
    dart_key = os.environ.get("DART_API_KEY")
    if not dart_key:
        print("WARNING: DART_API_KEY not set, skipping DART backfill")
        return

    dart.set_api_key(dart_key)

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] Fetching DART reports for {ticker}...", end=" ", flush=True)
        try:
            # Get DART corp_code from ticker (assuming simple mapping)
            # In real implementation, this would need a ticker-to-corp_code lookup table
            corp_code = None
            try:
                # Try to use dart_fss's built-in ticker lookup
                from dart_fss.api.filings import get_corp_code
                corp_list = get_corp_code()
                for code, info in corp_list.items():
                    if info.get("stock_code") == ticker:
                        corp_code = code
                        break
            except Exception as lookup_err:
                print(f"(skipped: {lookup_err.__class__.__name__})")
                continue

            if not corp_code:
                print("(no corp_code mapping)")
                continue

            # Search for reports from start_date to end_date
            end = end_date or datetime.now().strftime("%Y%m%d")
            reports = dart.search(
                corp_code=corp_code,
                bgn_de=start_date,
                end_de=end,
                sort="date",
                page_count=100,
            ).get("list", [])

            if reports:
                rows = parse_dart_reports(reports, ticker=ticker)
                if rows:
                    upsert_stock_events(dsn, rows)
                    print(f"({len(rows)} events)")
                else:
                    print("(no matching events)")
            else:
                print("(no reports)")
        except Exception as e:
            print(f"(ERROR: {e})")


def backfill_calendar(dsn: str, start_date: str = "2019-01-01",
                      end_date: str | None = None) -> None:
    """Build and upsert trading calendar with Korean holidays and short-selling bans."""
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")

    # Short-selling ban periods (공매도금지기간) — hardcoded from historical policy
    short_selling_ban_periods = [
        ("2020-03-16", "2021-05-02"),  # COVID-19 crisis
        ("2023-11-06", "2099-12-31"),  # Extended ban from Nov 2023
    ]

    rows = build_calendar_rows(start_date, end_date, short_selling_ban_periods=short_selling_ban_periods)
    print(f"  Building calendar {start_date} to {end_date}: {len(rows)} rows")
    upsert_calendar(dsn, rows)


def backfill_sector_daily_ohlcv(client: KisClient, dsn: str,
                                start_date: str = "20190102", end_date: str | None = None) -> None:
    """Backfill sector_daily_ohlcv from KIS API (60-day chunks per KIS quirk)."""
    end_date = end_date or datetime.now().strftime("%Y%m%d")
    cursor_end = datetime.strptime(end_date, "%Y%m%d")
    start = datetime.strptime(start_date, "%Y%m%d")

    for sector_code in SECTOR_CODES:
        print(f"  Sector {sector_code}...", end=" ", flush=True)
        cursor_end_idx = cursor_end
        call_count = 0

        while cursor_end_idx >= start:
            # CRITICAL: inquire-daily-indexchartprice silently caps at 50 rows per call
            # (live-verified 2026-09-08). Use 60-day chunks (not 140) to stay safely under cap.
            cursor_start = max(start, cursor_end_idx - timedelta(days=60))
            try:
                raw = client.request(
                    path="/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice",
                    tr_id="FHKUP03500100",
                    params={
                        "FID_COND_MRKT_DIV_CODE": "U",  # 업종(sector)
                        "FID_INPUT_ISCD": sector_code,
                        "FID_INPUT_DATE_1": cursor_start.strftime("%Y%m%d"),
                        "FID_INPUT_DATE_2": cursor_end_idx.strftime("%Y%m%d"),
                        "FID_PERIOD_DIV_CODE": "D",
                    },
                )
            except Exception as exc:
                # 한 청크의 일시적 API 오류(500 등)로 섹터 전체 백필이 중단되면 안 된다 —
                # 스킵하고 다음 청크로 진행(2026-09-08 라이브 실행 중 섹터 0180에서 발생 확인).
                print(f"[skip {cursor_start.strftime('%Y%m%d')}-{cursor_end_idx.strftime('%Y%m%d')}: {exc}]", end=" ", flush=True)
                time.sleep(KIS_API_INTERVAL)
                cursor_end_idx = cursor_start - timedelta(days=1)
                continue
            time.sleep(KIS_API_INTERVAL)
            call_count += 1

            rows = raw.get("output2", [])
            if rows:
                parsed = parse_sector_daily_response(raw, sector_code=sector_code)
                upsert_sector_daily_ohlcv(dsn, parsed)

            cursor_end_idx = cursor_start - timedelta(days=1)

        print(f"({call_count} calls, backfilled)")


def backfill_market_events(dsn: str) -> None:
    """Upsert market_events with BOK, FOMC, and witching day schedules."""
    # Hardcoded schedule (in production, this would come from a calendar source)
    events = []

    # BOK monetary policy meeting dates (quarterly)
    bok_dates = [
        "2023-01-12", "2023-03-16", "2023-05-18", "2023-07-13",
        "2023-08-17", "2023-10-12", "2023-11-16",
        "2024-01-18", "2024-02-15", "2024-03-21", "2024-04-18",
        "2024-05-16", "2024-06-13", "2024-07-18", "2024-08-15",
        "2024-09-19", "2024-10-17", "2024-11-21", "2024-12-19",
        "2025-01-16", "2025-02-20", "2025-03-20", "2025-04-17",
        "2025-05-15", "2025-06-19", "2025-07-17", "2025-08-21",
        "2025-09-18", "2025-10-16", "2025-11-20", "2025-12-18",
    ]

    # FOMC meeting dates (8 per year, approximately)
    fomc_dates = [
        "2023-01-31", "2023-03-21", "2023-05-02", "2023-06-13",
        "2023-07-25", "2023-09-19", "2023-10-31", "2023-12-12",
        "2024-01-30", "2024-03-19", "2024-05-01", "2024-06-18",
        "2024-07-31", "2024-09-17", "2024-11-06", "2024-12-17",
        "2025-01-28", "2025-03-18", "2025-05-06", "2025-06-17",
        "2025-07-29", "2025-09-16", "2025-11-04", "2025-12-16",
    ]

    # Witching day (3rd Friday of each quarter month in US, roughly)
    # Korean market witching days (3rd Friday of contracts month)
    witching_kr_dates = [
        "2023-03-17", "2023-06-16", "2023-09-15", "2023-12-15",
        "2024-03-15", "2024-06-21", "2024-09-20", "2024-12-20",
        "2025-03-21", "2025-06-20", "2025-09-19", "2025-12-19",
    ]

    # US witching days (3rd Friday of month)
    witching_us_dates = [
        "2023-01-20", "2023-02-17", "2023-03-17", "2023-04-21",
        "2023-05-19", "2023-06-16", "2023-07-21", "2023-08-18",
        "2023-09-15", "2023-10-20", "2023-11-17", "2023-12-15",
        "2024-01-19", "2024-02-16", "2024-03-15", "2024-04-19",
        "2024-05-17", "2024-06-21", "2024-07-19", "2024-08-16",
        "2024-09-20", "2024-10-18", "2024-11-15", "2024-12-20",
        "2025-01-17", "2025-02-21", "2025-03-21", "2025-04-18",
        "2025-05-16", "2025-06-20", "2025-07-18", "2025-08-15",
        "2025-09-19", "2025-10-17", "2025-11-21", "2025-12-19",
    ]

    # Create event rows
    for d in bok_dates:
        events.append({"event_date": d, "event_type": "bok",
                       "is_bok": True, "is_fomc": False, "is_witching_kr": False, "is_witching_us": False})
    for d in fomc_dates:
        events.append({"event_date": d, "event_type": "fomc",
                       "is_bok": False, "is_fomc": True, "is_witching_kr": False, "is_witching_us": False})
    for d in witching_kr_dates:
        events.append({"event_date": d, "event_type": "witching_kr",
                       "is_bok": False, "is_fomc": False, "is_witching_kr": True, "is_witching_us": False})
    for d in witching_us_dates:
        events.append({"event_date": d, "event_type": "witching_us",
                       "is_bok": False, "is_fomc": False, "is_witching_kr": False, "is_witching_us": True})

    # Upsert to DB
    if events:
        conn = psycopg2.connect(os.environ.get("STOCK_DB_V2_DSN", ""))
        with conn.cursor() as cur:
            for r in events:
                cur.execute(
                    """INSERT INTO market_events (event_date, event_type, is_bok, is_fomc, is_witching_kr, is_witching_us)
                       VALUES (%(event_date)s, %(event_type)s, %(is_bok)s, %(is_fomc)s, %(is_witching_kr)s, %(is_witching_us)s)
                       ON CONFLICT (event_date, event_type) DO UPDATE SET
                         is_bok = EXCLUDED.is_bok, is_fomc = EXCLUDED.is_fomc,
                         is_witching_kr = EXCLUDED.is_witching_kr, is_witching_us = EXCLUDED.is_witching_us""",
                    r,
                )
        conn.commit()
        conn.close()
        print(f"  Upserted {len(events)} market events")


def main():
    # Validate environment
    assert_outside_production_window()

    dsn = os.environ.get("STOCK_DB_V2_DSN")
    kis_key = os.environ.get("KIS_APP_KEY")
    kis_secret = os.environ.get("KIS_APP_SECRET")
    dart_key = os.environ.get("DART_API_KEY")

    if not dsn:
        print("ERROR: STOCK_DB_V2_DSN not set")
        sys.exit(1)
    if not kis_key or not kis_secret:
        print("ERROR: KIS_APP_KEY and KIS_APP_SECRET not set")
        sys.exit(1)

    if not dart_key:
        print("WARNING: DART_API_KEY not set — DART backfill will be skipped")

    print("Starting DART/calendar/sector backfill...")
    print(f"  DSN: {dsn[:50]}...")
    print(f"  Sector codes: {len(SECTOR_CODES)}")

    # 1. Backfill DART events
    print("\nBackfilling DART disclosure events...")
    tickers = load_tickers_from_db(dsn)
    print(f"  Loaded {len(tickers)} tickers from universe")
    backfill_dart_events(dsn, tickers)

    # 2. Build calendar
    print("\nBackfilling trading calendar...")
    backfill_calendar(dsn)

    # 3. Backfill sector OHLCV
    print("\nBackfilling sector daily OHLCV...")
    client = KisClient(kis_key, kis_secret)
    backfill_sector_daily_ohlcv(client, dsn)

    # 4. Backfill market events
    print("\nBackfilling market events...")
    backfill_market_events(dsn)

    print("\n✓ Backfill complete")


if __name__ == "__main__":
    main()
