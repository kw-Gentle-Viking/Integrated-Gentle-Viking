#!/usr/bin/env python
"""
Backfill 11 derived macro return/change/spread columns into the live feature_pool
table (Task 10.5 — closes a scope gap found during Task 13's config review:
feature_pool only had raw macro-indicator levels, not their day-over-day
derivatives, even though those levels are non-stationary and were already
excluded from clip/scaling in Task 11).

Exact formulas (see task-10.5-brief.md — do not invent alternates):
  kospi_ret          = compute_return(index_0001)
  kosdaq_ret          = compute_return(index_1001)
  snp500_ret          = compute_return(snp500_close)
  nasdaq_ret          = compute_return(nasdaq_close)
  phlx_semi_ret       = compute_return(phlx_semi_close)
  wti_ret             = compute_return(wti_crude_oil)
  gold_ret            = compute_return(gold_price)
  vix_chg             = compute_change(vix)
  usd_krw_chg         = compute_change(usd_krw)
  us_10y_yield_chg    = compute_change(us_10y_yield)
  rate_spread_us_kr   = compute_rate_spread(us_10y_yield, kr_base_rate)  # same-day level diff, not a day-over-day delta

Calculation grain: these are date-level values broadcast identically across all
200 tickers in feature_pool (already joined this way — see Task 10). Computing
pct_change/diff must happen on a DISTINCT-ON(trade_date), date-ordered series —
never per-ticker (same date repeats 200x per ticker, which would zero out every
diff). Results are then joined back onto every ticker row for that trade_date.

Usage:
  set -a && source .env && set +a
  python features/add_macro_features.py
"""

import os
import sys

import pandas as pd
import psycopg2
import psycopg2.extras

from features.macro_features import compute_return, compute_change, compute_rate_spread

# 12 raw level columns per the brief (fed_rate is loaded for completeness but has
# no derived column in the formula table — only kr_base_rate feeds rate_spread_us_kr).
RAW_LEVEL_COLS = [
    "index_0001", "index_1001",
    "snp500_close", "nasdaq_close", "phlx_semi_close",
    "vix", "wti_crude_oil", "gold_price", "usd_krw",
    "us_10y_yield", "fed_rate", "kr_base_rate",
]

# New column name -> derivation. "chg"/"ret" entries use a single source column;
# rate_spread_us_kr uses two.
RETURN_COLS = {
    "kospi_ret": "index_0001",
    "kosdaq_ret": "index_1001",
    "snp500_ret": "snp500_close",
    "nasdaq_ret": "nasdaq_close",
    "phlx_semi_ret": "phlx_semi_close",
    "wti_ret": "wti_crude_oil",
    "gold_ret": "gold_price",
}
CHANGE_COLS = {
    "vix_chg": "vix",
    "usd_krw_chg": "usd_krw",
    "us_10y_yield_chg": "us_10y_yield",
}
NEW_COLUMNS = list(RETURN_COLS.keys()) + list(CHANGE_COLS.keys()) + ["rate_spread_us_kr"]

assert len(NEW_COLUMNS) == 11, f"expected 11 new columns, got {len(NEW_COLUMNS)}"


def load_date_level_series(dsn: str) -> pd.DataFrame:
    """DISTINCT ON (trade_date), sorted ascending — one row per trading day."""
    conn = psycopg2.connect(dsn)
    try:
        cols_sql = ", ".join(RAW_LEVEL_COLS)
        query = (
            f"SELECT DISTINCT ON (trade_date) trade_date, {cols_sql} "
            f"FROM feature_pool ORDER BY trade_date ASC, ticker ASC"
        )
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    return df


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("trade_date").reset_index(drop=True)
    for col in RAW_LEVEL_COLS:
        df[col] = df[col].astype("float64")

    out = pd.DataFrame({"trade_date": df["trade_date"]})
    for new_col, src_col in RETURN_COLS.items():
        out[new_col] = compute_return(df[src_col])
    for new_col, src_col in CHANGE_COLS.items():
        out[new_col] = compute_change(df[src_col])
    out["rate_spread_us_kr"] = compute_rate_spread(df["us_10y_yield"], df["kr_base_rate"])
    return out


def add_columns(conn) -> None:
    with conn.cursor() as cur:
        for col in NEW_COLUMNS:
            cur.execute(f"ALTER TABLE feature_pool ADD COLUMN IF NOT EXISTS {col} NUMERIC")
    conn.commit()


def join_back(conn, derived: pd.DataFrame) -> None:
    """Write derived date-level values into a temp table, then UPDATE...FROM join
    them onto every ticker row for that trade_date in one statement."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TEMP TABLE macro_features_staging (
                trade_date DATE PRIMARY KEY,
                {", ".join(f"{c} NUMERIC" for c in NEW_COLUMNS)}
            ) ON COMMIT DROP
            """
        )

        rows = [
            (row["trade_date"],)
            + tuple(None if pd.isna(row[c]) else float(row[c]) for c in NEW_COLUMNS)
            for _, row in derived.iterrows()
        ]

        placeholders = ", ".join(["%s"] * (1 + len(NEW_COLUMNS)))
        psycopg2.extras.execute_values(
            cur,
            f"INSERT INTO macro_features_staging (trade_date, {', '.join(NEW_COLUMNS)}) VALUES %s",
            rows,
            template=f"({placeholders})",
        )

        set_clause = ", ".join(f"{c} = s.{c}" for c in NEW_COLUMNS)
        cur.execute(
            f"""
            UPDATE feature_pool AS f
            SET {set_clause}
            FROM macro_features_staging AS s
            WHERE f.trade_date = s.trade_date
            """
        )
        updated = cur.rowcount
    conn.commit()
    print(f"  UPDATEd {updated} feature_pool rows")


def main() -> None:
    dsn = os.environ.get("STOCK_DB_V2_DSN")
    if not dsn:
        print("ERROR: STOCK_DB_V2_DSN environment variable must be set")
        sys.exit(1)

    print("Loading date-level macro series (DISTINCT ON trade_date) from feature_pool...")
    date_df = load_date_level_series(dsn)
    print(f"  Loaded {len(date_df)} distinct trading days")

    print(f"Computing {len(NEW_COLUMNS)} derived columns...")
    derived = compute_derived(date_df)
    first_day_nulls = derived.iloc[0].isna().sum()
    print(f"  First trading day ({derived.iloc[0]['trade_date']}): {first_day_nulls} NULL columns")

    conn = psycopg2.connect(dsn)
    try:
        print("Adding 11 columns to feature_pool (ADD COLUMN IF NOT EXISTS)...")
        add_columns(conn)

        print("Joining derived values back onto all ticker rows by trade_date...")
        join_back(conn, derived)
    finally:
        conn.close()

    print("Done.")


if __name__ == "__main__":
    main()
