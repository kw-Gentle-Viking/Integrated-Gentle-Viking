#!/usr/bin/env python
"""S2: derive volatility-normalised label params on the TRAIN split only, persist them to
training/threshold_vn.json, and populate `label_vn` (SMALLINT) on `labels` and `feature_pool`
for the full period. Additive: never touches `label` / `next_day_return` / threshold.json.
Idempotent: ALTER ... IF NOT EXISTS + UPDATE-from-VALUES; re-running yields identical results.

    set -a && source .env && set +a && PYTHONPATH=. python training/run_derive_volnorm_labels.py
"""
import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import psycopg2
import psycopg2.extras

from training.label import assign_label_volnorm, derive_volnorm_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TRAIN_START = "2019-01-02"
TRAIN_END = "2023-12-31"
TARGET_HOLD_RATIO = 0.5
FLOOR_QUANTILE = 0.01
PARAMS_PATH = "training/threshold_vn.json"
TABLES = ("labels", "feature_pool")


def compute_volnorm_labels(df: pd.DataFrame, train_start: str, train_end: str,
                           target_hold_ratio: float = TARGET_HOLD_RATIO,
                           floor_quantile: float = FLOOR_QUANTILE) -> tuple[dict, list]:
    """df needs trade_date, next_day_return, volatility_20d. Params come from rows inside
    [train_start, train_end] only; labels are assigned to every row (df order)."""
    d = pd.to_datetime(df["trade_date"])
    in_train = (d >= pd.Timestamp(train_start)) & (d <= pd.Timestamp(train_end))
    tr = df[in_train]
    params = derive_volnorm_params(list(tr["next_day_return"].astype(float)),
                                   list(tr["volatility_20d"].astype(float)),
                                   target_hold_ratio=target_hold_ratio,
                                   floor_quantile=floor_quantile)
    labels = assign_label_volnorm(list(df["next_day_return"].astype(float)),
                                  list(df["volatility_20d"].astype(float)),
                                  params["k"], params["floor"])
    return params, labels


def load_inputs(dsn: str) -> pd.DataFrame:
    conn = psycopg2.connect(dsn)
    try:
        return pd.read_sql("""SELECT ticker, trade_date, next_day_return, volatility_20d
                              FROM feature_pool ORDER BY ticker, trade_date""", conn)
    finally:
        conn.close()


def persist(dsn: str, df: pd.DataFrame, labels: list) -> None:
    values = [(t, d, None if l is None else int(l))
              for t, d, l in zip(df["ticker"], df["trade_date"], labels)]
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            for table in TABLES:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS label_vn SMALLINT")
        conn.commit()
        with conn.cursor() as cur:
            for table in TABLES:
                psycopg2.extras.execute_values(
                    cur,
                    f"""UPDATE {table} AS t SET label_vn = v.label_vn
                        FROM (VALUES %s) AS v(ticker, trade_date, label_vn)
                        WHERE t.ticker = v.ticker AND t.trade_date = v.trade_date::date""",
                    values, template="(%s, %s, %s::smallint)", page_size=5000)
                logger.info("updated %s (%d candidate rows)", table, len(values))
        conn.commit()
    finally:
        conn.close()


def main():
    dsn = os.environ.get("STOCK_DB_V2_DSN")
    if not dsn:
        raise ValueError("STOCK_DB_V2_DSN environment variable not set")
    df = load_inputs(dsn)
    params, labels = compute_volnorm_labels(df, TRAIN_START, TRAIN_END)
    logger.info("k=%.6f floor=%.6f (rows=%d)", params["k"], params["floor"], len(df))
    with open(PARAMS_PATH, "w") as f:
        json.dump({**params, "target_hold_ratio": TARGET_HOLD_RATIO, "floor_quantile": FLOOR_QUANTILE,
                   "computed_on": f"train, {TRAIN_START}~{TRAIN_END}",
                   "timestamp": datetime.now().isoformat()}, f, indent=2)
    persist(dsn, df, labels)


if __name__ == "__main__":
    main()
