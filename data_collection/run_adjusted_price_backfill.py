"""Re-backfill price_daily with KIS ADJUSTED prices (FID_ORG_ADJ_PRC="0") for the whole universe.

The 2026-09-08 backfill stored raw ("1") prices, so every split/bonus/rights event produced a
price jump. This overwrites every (ticker, trade_date) row via upsert (all price/volume columns
are updated on conflict).

- Universe: all tickers in ticker_universe; range 2019-01-02 .. latest trade_date in price_daily.
- Resumable: per-ticker progress JSON (safe to interrupt/re-run; done tickers are skipped).
- Error isolation: a failing ticker is logged and skipped; failures are retried at the end.
- Conservative rate: KIS keys are shared with live production collectors.

    set -a && source .env && set +a && PYTHONPATH=. python -m data_collection.run_adjusted_price_backfill

Run only outside production windows (KisClient enforces this).
"""
import json
import logging
import os
import sys
import time

import psycopg2

from data_collection.backfill_daily_price import backfill_ticker
from data_collection.kis_client import KisClient

# Slower than the original 0.056s (~18/s): keys are shared with production collectors.
KIS_API_INTERVAL = float(os.environ.get("KIS_ADJ_BACKFILL_INTERVAL", "0.25"))
START_DATE = "20190102"
PROGRESS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "adjusted_backfill_progress.json")

logger = logging.getLogger("adjusted_backfill")


def load_progress(path: str) -> dict:
    try:
        with open(path) as f:
            d = json.load(f)
        return {"done": list(d.get("done", [])), "failed": dict(d.get("failed", {}))}
    except (FileNotFoundError, json.JSONDecodeError):
        return {"done": [], "failed": {}}


def save_progress(path: str, progress: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=1)
    os.replace(tmp, path)


def pending_tickers(tickers: list[str], progress: dict) -> list[str]:
    done = set(progress["done"])
    return [t for t in tickers if t not in done]


def run_backfill(tickers, backfill_fn, progress_path, retry_rounds: int = 2,
                 sleep=time.sleep, retry_wait: float = 30.0) -> dict:
    """Run backfill_fn(ticker) for every not-yet-done ticker; isolate errors; retry failures."""
    progress = load_progress(progress_path)
    total = len(tickers)
    t0 = time.time()

    def attempt(todo):
        for i, tk in enumerate(todo, 1):
            try:
                backfill_fn(tk)
                progress["done"].append(tk)
                progress["failed"].pop(tk, None)
                logger.info("[%d/%d] %s ok (done=%d/%d, %.0fs)", i, len(todo), tk,
                            len(progress["done"]), total, time.time() - t0)
            except Exception as e:  # noqa: BLE001 -- isolate per ticker
                progress["failed"][tk] = repr(e)
                logger.error("[%d/%d] %s FAILED: %r", i, len(todo), tk, e)
            save_progress(progress_path, progress)

    attempt(pending_tickers(tickers, progress))
    for r in range(retry_rounds):
        retry = pending_tickers(tickers, progress)
        if not retry:
            break
        logger.warning("retry round %d: %d tickers: %s", r + 1, len(retry), retry)
        sleep(retry_wait)
        attempt(retry)
    return progress


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dsn = os.environ["STOCK_DB_V2_DSN"]
    client = KisClient(app_key=os.environ["KIS_APP_KEY"], app_secret=os.environ["KIS_APP_SECRET"])
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("SELECT ticker FROM ticker_universe ORDER BY rank")
        tickers = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT max(trade_date) FROM price_daily")
        end_date = cur.fetchone()[0].strftime("%Y%m%d")
    conn.close()
    logger.info("universe=%d tickers, range %s..%s, interval=%.3fs", len(tickers), START_DATE,
                end_date, KIS_API_INTERVAL)

    import data_collection.backfill_daily_price as bdp
    bdp.KIS_API_INTERVAL = KIS_API_INTERVAL  # backfill_ticker sleeps this after every call

    def fn(tk):
        backfill_ticker(client, dsn, tk, start_date=START_DATE, end_date=end_date, adjusted=True)

    progress = run_backfill(tickers, fn, PROGRESS_PATH)
    failed = {t: e for t, e in progress["failed"].items() if t not in progress["done"]}
    logger.info("finished: done=%d/%d failed=%d %s", len(progress["done"]), len(tickers),
                len(failed), failed)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
