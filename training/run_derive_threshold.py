#!/usr/bin/env python
"""
Step 9: Derive threshold from train split, compute labels for full period.

1. Query train split (2019-01-02~2023-12-31) price_daily for all 200 tickers
2. Compute next-day returns for all data
3. Derive threshold targeting 50% hold ratio
4. Save threshold to training/threshold.json
5. Apply threshold to full period (2019-01-02~2026-09-08) and persist to labels table
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
import psycopg2.extras
from training.label import (
    compute_next_day_return,
    derive_threshold,
    build_label_rows,
    upsert_labels,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load DSN from environment
STOCK_DB_V2_DSN = os.environ.get('STOCK_DB_V2_DSN')
if not STOCK_DB_V2_DSN:
    raise ValueError("STOCK_DB_V2_DSN environment variable not set")

# Date ranges
TRAIN_START = "2019-01-02"
TRAIN_END = "2023-12-31"
FULL_START = "2019-01-02"
FULL_END = "2026-09-08"

TARGET_HOLD_RATIO = 0.5


def get_all_tickers(dsn: str) -> list[str]:
    """Get list of all unique tickers in price_daily."""
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT ticker FROM price_daily ORDER BY ticker")
        tickers = [row[0] for row in cur.fetchall()]
    conn.close()
    return tickers


def get_ticker_data(dsn: str, ticker: str, start_date: str, end_date: str) -> tuple[list[str], list[float]]:
    """Get sorted (trade_date, close_price) pairs for a ticker in a date range.

    Returns (dates, prices) where prices may contain None for missing data.
    """
    conn = psycopg2.connect(dsn)
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            """SELECT trade_date, close_price
               FROM price_daily
               WHERE ticker = %s AND trade_date >= %s AND trade_date <= %s
               ORDER BY trade_date""",
            (ticker, start_date, end_date),
        )
        rows = cur.fetchall()
    conn.close()

    if not rows:
        return [], []

    dates = [str(row['trade_date']) for row in rows]
    prices = [float(row['close_price']) if row['close_price'] is not None else None for row in rows]
    return dates, prices


def derive_threshold_from_train_split() -> float:
    """
    Compute threshold from train split (2019-01-02~2023-12-31) for all tickers.
    Returns threshold value targeting target_hold_ratio.
    """
    logger.info(f"Deriving threshold from train split ({TRAIN_START} to {TRAIN_END})...")

    all_tickers = get_all_tickers(STOCK_DB_V2_DSN)
    logger.info(f"Found {len(all_tickers)} tickers")

    all_returns = []
    skipped_tickers = []

    for ticker in all_tickers:
        try:
            dates, prices = get_ticker_data(STOCK_DB_V2_DSN, ticker, TRAIN_START, TRAIN_END)

            if not prices:
                logger.warning(f"  {ticker}: No data in train split")
                skipped_tickers.append((ticker, "no data"))
                continue

            returns = compute_next_day_return(prices)
            # Filter out None values
            valid_returns = [r for r in returns if r is not None]

            if not valid_returns:
                logger.warning(f"  {ticker}: No valid returns computed")
                skipped_tickers.append((ticker, "no valid returns"))
                continue

            all_returns.extend(valid_returns)
            logger.info(f"  {ticker}: {len(valid_returns)} returns")

        except Exception as e:
            logger.error(f"  {ticker}: Error - {e}")
            skipped_tickers.append((ticker, str(e)))

    if skipped_tickers:
        logger.warning(f"Skipped {len(skipped_tickers)} tickers: {skipped_tickers[:5]}")

    if not all_returns:
        raise ValueError("No valid returns found in train split")

    logger.info(f"Total returns collected: {len(all_returns)}")
    threshold = derive_threshold(all_returns, target_hold_ratio=TARGET_HOLD_RATIO)
    logger.info(f"Derived threshold: {threshold:.6f}")

    return threshold


def persist_labels_for_full_period(threshold: float) -> None:
    """
    Apply threshold to full period (2019-01-02~2026-09-08) and persist to labels table.
    """
    logger.info(f"Persisting labels for full period ({FULL_START} to {FULL_END})...")

    all_tickers = get_all_tickers(STOCK_DB_V2_DSN)

    total_rows = 0
    skipped_tickers = []

    for ticker in all_tickers:
        try:
            dates, prices = get_ticker_data(STOCK_DB_V2_DSN, ticker, FULL_START, FULL_END)

            if not prices:
                logger.warning(f"  {ticker}: No data in full period")
                skipped_tickers.append((ticker, "no data"))
                continue

            # Build label rows
            rows = build_label_rows(ticker, dates, prices, threshold)

            # Persist to labels table
            upsert_labels(STOCK_DB_V2_DSN, rows)

            total_rows += len(rows)
            logger.info(f"  {ticker}: {len(rows)} rows persisted")

        except Exception as e:
            logger.error(f"  {ticker}: Error - {e}")
            skipped_tickers.append((ticker, str(e)))

    if skipped_tickers:
        logger.warning(f"Skipped {len(skipped_tickers)} tickers: {skipped_tickers[:5]}")

    logger.info(f"Total rows persisted: {total_rows}")


def verify_label_distribution() -> dict:
    """Verify label distribution in labels table."""
    conn = psycopg2.connect(STOCK_DB_V2_DSN)
    with conn.cursor() as cur:
        # Count labels
        cur.execute("SELECT label, COUNT(*) FROM labels WHERE label IS NOT NULL GROUP BY label ORDER BY label")
        label_counts = {row[0]: row[1] for row in cur.fetchall()}

        # Total non-null labels
        cur.execute("SELECT COUNT(*) FROM labels WHERE label IS NOT NULL")
        total = cur.fetchone()[0]
    conn.close()

    if total == 0:
        return {"error": "No labels found"}

    distribution = {
        "buy": (label_counts.get(0, 0) / total) * 100,
        "hold": (label_counts.get(1, 0) / total) * 100,
        "sell": (label_counts.get(2, 0) / total) * 100,
        "total": total,
    }

    logger.info(f"Label distribution: Buy={distribution['buy']:.1f}%, Hold={distribution['hold']:.1f}%, Sell={distribution['sell']:.1f}%")

    return distribution


def main():
    logger.info("=" * 80)
    logger.info("Task 9: Derive threshold and persist labels")
    logger.info("=" * 80)

    # Step 1: Derive threshold from train split
    threshold = derive_threshold_from_train_split()

    # Step 2: Save threshold to file
    threshold_data = {
        "threshold": threshold,
        "computed_on": f"1단계 train, {TRAIN_START}~{TRAIN_END}",
        "target_hold_ratio": TARGET_HOLD_RATIO,
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs("training", exist_ok=True)
    with open("training/threshold.json", "w") as f:
        json.dump(threshold_data, f, indent=2)
    logger.info(f"Saved threshold to training/threshold.json")

    # Step 3: Persist labels for full period
    persist_labels_for_full_period(threshold)

    # Step 4: Verify distribution
    distribution = verify_label_distribution()

    logger.info("=" * 80)
    logger.info("Task 9 completed successfully")
    logger.info("=" * 80)

    return threshold, distribution


if __name__ == "__main__":
    main()
