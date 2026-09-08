import csv
from pathlib import Path
from typing import Optional
import psycopg2
from data_collection.kis_client import KisClient


def load_candidate_tickers(kospi_csv: str, kosdaq_csv: str) -> list[dict]:
    candidates = []
    for path, is_kospi in [(kospi_csv, True), (kosdaq_csv, False)]:
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                candidates.append({"ticker": row["ticker"].zfill(6), "is_kospi": is_kospi})
    return candidates


def fetch_snapshot_prices(client: KisClient, tickers: list[str], snapshot_date: str) -> list[dict]:
    results = []
    for ticker in tickers:
        data = client.request(
            path="/uapi/domestic-stock/v1/quotations/inquire-daily-price",
            tr_id="FHKST03010100",
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker,
                    "FID_INPUT_DATE_1": snapshot_date, "FID_INPUT_DATE_2": snapshot_date,
                    "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "1"},
        )
        rows = data.get("output2", [])
        if not rows:
            continue
        row = rows[0]
        results.append({
            "ticker": ticker,
            "close_price": float(row.get("stck_clpr") or 0) or None,
            "shares_outstanding": int(row.get("lstn_stcn") or 0) or None,
        })
    return results


def compute_top_n_snapshot(candidates: list[dict], n: int = 200) -> list[dict]:
    valid = [
        c for c in candidates
        if c.get("close_price") is not None and c.get("shares_outstanding") is not None
    ]
    for c in valid:
        c["market_cap"] = c["close_price"] * c["shares_outstanding"]
    ranked = sorted(valid, key=lambda c: c["market_cap"], reverse=True)[:n]
    for i, c in enumerate(ranked):
        c["rank"] = i + 1
    return ranked


def save_universe(dsn: str, universe: list[dict], snapshot_date: str) -> None:
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for c in universe:
            cur.execute(
                """INSERT INTO ticker_universe (ticker, snapshot_date, market_cap, rank, is_kospi)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (ticker) DO UPDATE SET
                     snapshot_date = EXCLUDED.snapshot_date, market_cap = EXCLUDED.market_cap,
                     rank = EXCLUDED.rank, is_kospi = EXCLUDED.is_kospi""",
                (c["ticker"], snapshot_date, c["market_cap"], c["rank"], c.get("is_kospi", True)),
            )
    conn.commit()
    conn.close()
