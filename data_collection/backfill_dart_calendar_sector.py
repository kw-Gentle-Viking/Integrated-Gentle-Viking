import time
from datetime import datetime, timedelta
import psycopg2
from data_collection.kis_client import KisClient

# KIS API rate limit: ~18 calls per second
KIS_API_INTERVAL = 0.056

EVENT_KEYWORD_MAP = {
    "유상증자": "유상증자", "무상증자": "무상증자", "배당": "배당",
    "주식분할": "액면분할", "실적": "실적발표", "잠정": "실적발표",
    "매출액또는손익": "실적발표", "합병": "합병",
}


def classify_dart_event(report_name: str) -> str | None:
    for keyword, event_type in EVENT_KEYWORD_MAP.items():
        if keyword in report_name:
            return event_type
    return None


def build_calendar_rows(start: str, end: str,
                         short_selling_ban_periods: list[tuple[str, str]]) -> list[dict]:
    rows = []
    d = datetime.strptime(start, "%Y-%m-%d")
    end_d = datetime.strptime(end, "%Y-%m-%d")
    bans = [(datetime.strptime(s, "%Y-%m-%d"), datetime.strptime(e, "%Y-%m-%d"))
            for s, e in short_selling_ban_periods]
    while d <= end_d:
        dow = d.weekday()
        is_open = dow < 5
        banned = any(s <= d <= e for s, e in bans)
        rows.append({
            "base_date": d.strftime("%Y-%m-%d"), "day_of_week": dow,
            "is_market_open": is_open, "is_holiday": not is_open,
            "is_short_selling_banned": banned,
        })
        d += timedelta(days=1)
    return rows


def parse_dart_reports(raw_reports: list[dict], ticker: str) -> list[dict]:
    rows = []
    for r in raw_reports:
        event_type = classify_dart_event(r["report_nm"])
        if event_type is None:
            continue
        d = r["rcept_dt"]
        rows.append({"ticker": ticker, "event_date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
                      "event_type": event_type, "description": r["report_nm"]})
    return rows


def upsert_stock_events(dsn: str, rows: list[dict]) -> None:
    if not rows:
        return
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """INSERT INTO stock_events (ticker, event_date, event_type, description)
                   VALUES (%(ticker)s, %(event_date)s, %(event_type)s, %(description)s)
                   ON CONFLICT (ticker, event_date, event_type) DO NOTHING""",
                r,
            )
    conn.commit()
    conn.close()


def parse_sector_daily_response(raw: dict, sector_code: str) -> list[dict]:
    rows = []
    for r in raw.get("output2", []):
        d = r["stck_bsop_date"]
        rows.append({
            "sector_code": sector_code, "trade_date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
            "open": float(r["bstp_nmix_oprc"]), "high": float(r["bstp_nmix_hgpr"]),
            "low": float(r["bstp_nmix_lwpr"]), "close": float(r["bstp_nmix_prpr"]),
            "volume": int(r["acml_vol"]),
        })
    return rows


def upsert_sector_daily_ohlcv(dsn: str, rows: list[dict]) -> None:
    if not rows:
        return
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """INSERT INTO sector_daily_ohlcv (sector_code, trade_date, open, high, low, close, volume)
                   VALUES (%(sector_code)s, %(trade_date)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s)
                   ON CONFLICT (sector_code, trade_date) DO UPDATE SET
                     open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                     close = EXCLUDED.close, volume = EXCLUDED.volume""",
                r,
            )
    conn.commit()
    conn.close()


def upsert_calendar(dsn: str, rows: list[dict]) -> None:
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """INSERT INTO calendar (base_date, day_of_week, is_market_open, is_holiday, is_short_selling_banned)
                   VALUES (%(base_date)s, %(day_of_week)s, %(is_market_open)s, %(is_holiday)s, %(is_short_selling_banned)s)
                   ON CONFLICT (base_date) DO UPDATE SET
                     day_of_week = EXCLUDED.day_of_week, is_market_open = EXCLUDED.is_market_open,
                     is_holiday = EXCLUDED.is_holiday, is_short_selling_banned = EXCLUDED.is_short_selling_banned""",
                r,
            )
    conn.commit()
    conn.close()
