import time
from datetime import datetime, timedelta
import psycopg2
from data_collection.kis_client import KisClient

# KIS API rate limit: ~18 calls per second
KIS_API_INTERVAL = 0.056


def parse_valuation_response(raw: dict, ticker: str, trade_date: str) -> dict | None:
    output = raw.get("output")
    if not output:
        return None
    return {
        "ticker": ticker, "trade_date": trade_date,
        "per": float(output["per"]), "pbr": float(output["pbr"]),
        "market_cap": float(output["hts_avls"]) * 1_000_000,
    }


def parse_investor_flow_response(raw: dict, ticker: str, trade_date: str) -> dict | None:
    rows = raw.get("output")
    if not rows:
        return None
    r = rows[0]
    return {
        "ticker": ticker, "trade_date": trade_date,
        "individual_net_amt": float(r["prsn_ntby_tr_pbmn"]) * 1_000_000,
        "foreign_net_amt": float(r["frgn_ntby_tr_pbmn"]) * 1_000_000,
        "inst_net_amt": float(r["orgn_ntby_tr_pbmn"]) * 1_000_000,
    }


def upsert_daily_valuation(dsn: str, row: dict) -> None:
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO daily_valuation (ticker, trade_date, per, pbr, market_cap)
               VALUES (%(ticker)s, %(trade_date)s, %(per)s, %(pbr)s, %(market_cap)s)
               ON CONFLICT (ticker, trade_date) DO UPDATE SET
                 per = EXCLUDED.per, pbr = EXCLUDED.pbr, market_cap = EXCLUDED.market_cap""",
            row,
        )
    conn.commit()
    conn.close()


def upsert_investor_flow_daily(dsn: str, row: dict) -> None:
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO investor_flow_daily
               (ticker, trade_date, individual_net_amt, foreign_net_amt, inst_net_amt)
               VALUES (%(ticker)s, %(trade_date)s, %(individual_net_amt)s,
                       %(foreign_net_amt)s, %(inst_net_amt)s)
               ON CONFLICT (ticker, trade_date) DO UPDATE SET
                 individual_net_amt = EXCLUDED.individual_net_amt,
                 foreign_net_amt = EXCLUDED.foreign_net_amt, inst_net_amt = EXCLUDED.inst_net_amt""",
            row,
        )
    conn.commit()
    conn.close()


def upsert_market_index_daily(dsn: str, index_code: str, trade_date: str, close_price: float) -> None:
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO market_index_daily (index_code, trade_date, close_price)
               VALUES (%s, %s, %s)
               ON CONFLICT (index_code, trade_date) DO UPDATE SET close_price = EXCLUDED.close_price""",
            (index_code, trade_date, close_price),
        )
    conn.commit()
    conn.close()


def backfill_valuation(client: KisClient, dsn: str, ticker: str,
                       start_date: str = "20190102", end_date: str | None = None) -> None:
    """Backfill daily valuation data (PER, PBR, market cap) for a single ticker."""
    end_date = end_date or datetime.now().strftime("%Y%m%d")
    cursor_end = datetime.strptime(end_date, "%Y%m%d")
    start = datetime.strptime(start_date, "%Y%m%d")

    while cursor_end >= start:
        cursor_start = max(start, cursor_end - timedelta(days=140))
        raw = client.request(
            path="/uapi/domestic-stock/v1/quotations/inquire-daily-price",
            tr_id="FHKST01010100",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": ticker,
                "FID_INPUT_DATE_1": cursor_start.strftime("%Y%m%d"),
                "FID_INPUT_DATE_2": cursor_end.strftime("%Y%m%d"),
                "FID_PERIOD_DIV_CODE": "D",
            },
        )
        time.sleep(KIS_API_INTERVAL)

        # Parse valuation response (single daily record)
        rows = raw.get("output2", [])
        for r in rows:
            d = r["stck_bsop_date"]
            trade_date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
            valuation_data = parse_valuation_response(
                {"output": r}, ticker=ticker, trade_date=trade_date
            )
            if valuation_data:
                upsert_daily_valuation(dsn, valuation_data)

        cursor_end = cursor_start - timedelta(days=1)


def backfill_investor_flow(client: KisClient, dsn: str, ticker: str,
                           start_date: str = "20190102", end_date: str | None = None) -> None:
    """Backfill daily investor flow data (net buy/sell by category) for a single ticker."""
    end_date = end_date or datetime.now().strftime("%Y%m%d")
    cursor_end = datetime.strptime(end_date, "%Y%m%d")
    start = datetime.strptime(start_date, "%Y%m%d")

    while cursor_end >= start:
        cursor_start = max(start, cursor_end - timedelta(days=140))
        raw = client.request(
            path="/uapi/domestic-stock/v1/quotations/inquire-investor-flow",
            tr_id="FHKST01010900",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": ticker,
                "FID_INPUT_DATE_1": cursor_start.strftime("%Y%m%d"),
                "FID_INPUT_DATE_2": cursor_end.strftime("%Y%m%d"),
                "FID_PERIOD_DIV_CODE": "D",
            },
        )
        time.sleep(KIS_API_INTERVAL)

        # Parse investor flow responses
        rows = raw.get("output", [])
        for r in rows:
            d = r["stck_bsop_date"]
            trade_date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
            flow_data = parse_investor_flow_response(
                {"output": [r]}, ticker=ticker, trade_date=trade_date
            )
            if flow_data:
                upsert_investor_flow_daily(dsn, flow_data)

        cursor_end = cursor_start - timedelta(days=1)


def backfill_market_indices(client: KisClient, dsn: str,
                            start_date: str = "20190102", end_date: str | None = None) -> None:
    """Backfill market index data (KOSPI 0001 and KOSDAQ 1001) for the entire period."""
    end_date = end_date or datetime.now().strftime("%Y%m%d")
    cursor_end = datetime.strptime(end_date, "%Y%m%d")
    start = datetime.strptime(start_date, "%Y%m%d")

    for index_code in ["0001", "1001"]:  # KOSPI, KOSDAQ
        cursor_end_idx = cursor_end
        while cursor_end_idx >= start:
            cursor_start = max(start, cursor_end_idx - timedelta(days=140))
            raw = client.request(
                path="/uapi/domestic-stock/v1/quotations/inquire-daily-index",
                tr_id="FHKST01010000",
                params={
                    "FID_INPUT_ISCD": index_code,
                    "FID_INPUT_DATE_1": cursor_start.strftime("%Y%m%d"),
                    "FID_INPUT_DATE_2": cursor_end_idx.strftime("%Y%m%d"),
                    "FID_PERIOD_DIV_CODE": "D",
                },
            )
            time.sleep(KIS_API_INTERVAL)

            # Parse index data
            rows = raw.get("output2", [])
            for r in rows:
                d = r["stck_bsop_date"]
                trade_date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
                close_price = float(r["stck_clpr"])
                upsert_market_index_daily(dsn, index_code, trade_date, close_price)

            cursor_end_idx = cursor_start - timedelta(days=1)
