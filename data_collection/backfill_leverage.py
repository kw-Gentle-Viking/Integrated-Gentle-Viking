import psycopg2
from data_collection.kis_client import KisClient
from data_collection.leverage_products import LEVERAGE_PRODUCTS


def save_leverage_products(dsn: str) -> None:
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for p in LEVERAGE_PRODUCTS:
            if p["code"] is None:
                continue  # ETN 코드 미확인 — Step 6 이후 채워지면 재실행
            cur.execute(
                """INSERT INTO leverage_products
                   (code, product_name, underlying_ticker, multiple, product_type, issuer, listed_date)
                   VALUES (%(code)s, %(product_name)s, %(underlying_ticker)s, %(multiple)s,
                           %(product_type)s, %(issuer)s, %(listed_date)s)
                   ON CONFLICT (code) DO NOTHING""",
                p,
            )
    conn.commit()
    conn.close()


def upsert_leverage_daily(dsn: str, rows: list[dict]) -> None:
    if not rows:
        return
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """INSERT INTO leverage_daily (code, trade_date, close_price, volume, turnover, nav, aum)
                   VALUES (%(code)s, %(trade_date)s, %(close_price)s, %(volume)s,
                           %(turnover)s, %(nav)s, %(aum)s)
                   ON CONFLICT (code, trade_date) DO UPDATE SET
                     close_price = EXCLUDED.close_price, volume = EXCLUDED.volume,
                     turnover = EXCLUDED.turnover, nav = EXCLUDED.nav, aum = EXCLUDED.aum""",
                r,
            )
    conn.commit()
    conn.close()


def parse_vi_event_response(raw: dict, ticker: str) -> list[dict]:
    results = []
    for r in raw.get("output", []):
        t = r["vi_bsop_time"]
        triggered = f"{t[:4]}-{t[4:6]}-{t[6:8]} {t[8:10]}:{t[10:12]}:{t[12:]}"
        released = None
        if r.get("vi_rls_time"):
            rt = r["vi_rls_time"]
            released = f"{rt[:4]}-{rt[4:6]}-{rt[6:8]} {rt[8:10]}:{rt[10:12]}:{rt[12:]}"
        results.append({"ticker": ticker, "triggered_at": triggered,
                          "released_at": released, "vi_type": r["vi_type_cd"]})
    return results


def upsert_vi_events(dsn: str, rows: list[dict]) -> None:
    if not rows:
        return
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """INSERT INTO vi_events (ticker, triggered_at, released_at, vi_type)
                   VALUES (%(ticker)s, %(triggered_at)s, %(released_at)s, %(vi_type)s)
                   ON CONFLICT (ticker, triggered_at) DO UPDATE SET
                     released_at = EXCLUDED.released_at, vi_type = EXCLUDED.vi_type""",
                r,
            )
    conn.commit()
    conn.close()


def parse_leverage_daily_response(raw: dict, code: str) -> dict | None:
    """Parse ETF/ETN daily price response from KIS.

    Fields are in raw won units:
    - close_price: from stck_clpr (원)
    - volume: from acml_vol (shares)
    - turnover: from acml_tr_pbmn (원)
    - nav: NULL (only available in current price inquiry, not historical daily data)
    - aum: NULL (only available in current price inquiry, not historical daily data)
    """
    rows = raw.get("output2", [])
    results = []
    for r in rows:
        trade_date = r.get("stck_bsop_date")  # YYYYMMDD format
        if not trade_date:
            continue
        # Convert YYYYMMDD to YYYY-MM-DD
        date_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
        results.append({
            "code": code,
            "trade_date": date_str,
            "close_price": float(r.get("stck_clpr") or 0) or None,
            "volume": int(r.get("acml_vol") or 0) or None,
            "turnover": float(r.get("acml_tr_pbmn") or 0) or None,
            "nav": None,  # Not available in daily history endpoint
            "aum": None,  # Not available in daily history endpoint
        })
    return results
