from datetime import datetime, timedelta
import time
import psycopg2
from data_collection.kis_client import KisClient

# KIS API rate limit: ~18 calls per second
KIS_API_INTERVAL = 0.056


def parse_daily_price_response(raw: dict, ticker: str) -> list[dict]:
    rows = []
    for r in raw.get("output2", []):
        volume = int(r.get("acml_vol") or 0)
        if volume == 0:
            continue
        d = r["stck_bsop_date"]
        rows.append({
            "ticker": ticker,
            "trade_date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
            "open_price": float(r["stck_oprc"]), "high_price": float(r["stck_hgpr"]),
            "low_price": float(r["stck_lwpr"]), "close_price": float(r["stck_clpr"]),
            "volume": volume, "turnover": float(r["acml_tr_pbmn"]),
            "shares_outstanding": int(r["lstn_stcn"]),
        })
    return rows


def upsert_price_daily(dsn: str, rows: list[dict]) -> None:
    if not rows:
        return
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """INSERT INTO price_daily
                   (ticker, trade_date, open_price, high_price, low_price, close_price,
                    volume, turnover, shares_outstanding)
                   VALUES (%(ticker)s, %(trade_date)s, %(open_price)s, %(high_price)s,
                           %(low_price)s, %(close_price)s, %(volume)s, %(turnover)s,
                           %(shares_outstanding)s)
                   ON CONFLICT (ticker, trade_date) DO UPDATE SET
                     open_price = EXCLUDED.open_price, high_price = EXCLUDED.high_price,
                     low_price = EXCLUDED.low_price, close_price = EXCLUDED.close_price,
                     volume = EXCLUDED.volume, turnover = EXCLUDED.turnover,
                     shares_outstanding = EXCLUDED.shares_outstanding""",
                r,
            )
    conn.commit()
    conn.close()


def backfill_ticker(client: KisClient, dsn: str, ticker: str,
                     start_date: str = "20190102", end_date: str | None = None) -> None:
    """KIS는 1회 조회당 최대 약 100영업일만 반환하므로 청크 단위로 역순 페이징한다."""
    end_date = end_date or datetime.now().strftime("%Y%m%d")
    cursor_end = datetime.strptime(end_date, "%Y%m%d")
    start = datetime.strptime(start_date, "%Y%m%d")
    while cursor_end >= start:
        cursor_start = max(start, cursor_end - timedelta(days=140))
        raw = client.request(
            path="/uapi/domestic-stock/v1/quotations/inquire-daily-price",
            tr_id="FHKST03010100",
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker,
                    "FID_INPUT_DATE_1": cursor_start.strftime("%Y%m%d"),
                    "FID_INPUT_DATE_2": cursor_end.strftime("%Y%m%d"),
                    "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "1"},
        )
        time.sleep(KIS_API_INTERVAL)  # Respect KIS API rate limit: ~18 calls/sec
        upsert_price_daily(dsn, parse_daily_price_response(raw, ticker))
        cursor_end = cursor_start - timedelta(days=1)
