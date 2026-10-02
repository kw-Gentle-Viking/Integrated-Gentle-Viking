"""
serving/universe_batch.py
==========================
Batch inference over the whole ticker_universe (200 tickers) on the latest COMPLETED trading day
in stock_db_v2.feature_pool (the "as-of" day), then a cross-sectional rank percentile.

For every ticker:
    59 real feature_pool rows before as-of + the as-of row itself (a completed daily bar, wrapped as
    one 5-min bar and run through serving.feature_builder.build_encoder_df exactly as live serving
    would, at 15:30 of the as-of day) -> serving.inference.run_inference
    -> p_buy / p_hold / p_sell, score = p_buy - p_sell.
Then, inside that one as-of date only, rank_pct in [0, 1] (higher = more buy-leaning, ties share
their average rank; NaN/inf scores and universes with fewer than 2 valid scores get None).

Output JSON (default training/artifacts/universe_scores_<asof>.json): scores (best first),
excluded (data insufficient: no row on as-of / short history / trading-day gap), failed (per-ticker
exceptions / non-finite outputs), plus run metadata (model path+version, git commit, timing).
One ticker failing never stops the batch.

Model: serving.model.get_model() (env TFT_MODEL_PATH swaps the weights; CHAMPION_CONFIG_PATH the
architecture). DB access is SELECT-only against stock_db_v2 (STOCK_DB_V2_DSN); no KIS/prod DB.

    python -m serving.universe_batch [--asof YYYY-MM-DD] [--out path]

Recommended schedule (NOT installed, see docs/universe_batch.md): once per trading day AFTER the
daily feature_pool build has landed (evening, e.g. 19:00 KST, weekdays), running on CPU.
"""

import argparse
import json
import logging
import math
import os
import subprocess
import time
from collections import defaultdict
from datetime import date, datetime, time as dtime
from typing import Callable, Mapping

from serving.feature_builder import FUTURE_DB_COLS, build_encoder_df

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_HISTORY = 59
N_ROWS = N_HISTORY + 1  # 59 history rows + the as-of row
MIN_DATE_COVERAGE = 0.5  # a date counts as a market trading day / usable as-of if >=50% of the universe has a row
# The as-of run is a historical-style computation; long holiday blocks between rows are legitimate.
BATCH_MAX_GAP_DAYS = 15


# ============================================================ pure functions
def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def compute_rank_pct(scores: Mapping[str, float | None]) -> dict[str, float | None]:
    """Cross-sectional rank percentile in [0, 1]; higher score -> higher percentile.

    rank_pct = average_rank_index / (n_valid - 1), rank index 0-based over ascending scores, so the
    lowest score is 0.0 and the highest 1.0; equal scores share the average of their ranks. Scores
    that are None/NaN/inf are excluded (None, and they don't count towards n). With fewer than 2
    valid scores a percentile is meaningless -> None for everyone.
    """
    valid = sorted(((s, t) for t, s in scores.items() if _finite(s)))
    out: dict[str, float | None] = {t: None for t in scores}
    n = len(valid)
    if n < 2:
        return out
    i = 0
    while i < n:
        j = i
        while j + 1 < n and valid[j + 1][0] == valid[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            out[valid[k][1]] = avg_rank / (n - 1)
        i = j + 1
    return out


def pick_asof(coverage: Mapping[date, int], n_universe: int, min_coverage: float = MIN_DATE_COVERAGE) -> date:
    """Latest date on which at least `min_coverage` of the universe has a feature_pool row (guards
    against a partially-loaded newest day being treated as the completed as-of day)."""
    ok = [d for d, n in coverage.items() if n >= min_coverage * n_universe]
    if not ok:
        raise ValueError(f"no feature_pool date has >= {min_coverage:.0%} of the {n_universe}-ticker universe")
    return max(ok)


def default_out_path(asof: date) -> str:
    return os.path.join(REPO_ROOT, "training", "artifacts", f"universe_scores_{asof.isoformat()}.json")


def _as_date(d) -> date:
    return d.date() if isinstance(d, datetime) else d


def score_universe(
    universe: list[str],
    rows_by_ticker: Mapping[str, list[dict]],
    market_dates: set[date],
    asof: date,
    predict_fn: Callable[[str, "pd.DataFrame"], dict],  # noqa: F821
    cols: dict,
) -> dict:
    """Score every ticker (no DB access). rows_by_ticker[t] = t's feature_pool rows, oldest first,
    up to and including as-of (extra older rows are ignored; only the last 60 are used).
    predict_fn(ticker, encoder_df) -> {"prob_buy","prob_hold","prob_sell",...}.
    Returns {"scores": [...best first...], "excluded": [...], "failed": [...]}."""
    expected_dates = set(sorted(d for d in market_dates if d <= asof)[-N_ROWS:])
    scored: list[dict] = []
    excluded: list[dict] = []
    failed: list[dict] = []

    for ticker in universe:
        rows = list(rows_by_ticker.get(ticker) or [])
        last = _as_date(rows[-1]["trade_date"]) if rows else None
        if last != asof:
            excluded.append({"ticker": ticker, "reason": "no_row_on_asof",
                             "detail": f"latest feature_pool row: {last}, as-of {asof}"})
            continue
        if len(rows) < N_ROWS:
            excluded.append({"ticker": ticker, "reason": "insufficient_history",
                             "detail": f"{len(rows)}/{N_ROWS} feature_pool rows"})
            continue
        window = rows[-N_ROWS:]
        missing = sorted(expected_dates - {_as_date(r["trade_date"]) for r in window})
        if missing:
            excluded.append({"ticker": ticker, "reason": "history_gap",
                             "detail": f"missing {len(missing)} market trading day(s) in the 60-row window, "
                                       f"e.g. {', '.join(str(d) for d in missing[:3])}"})
            continue
        try:
            history, day_row = window[:-1], window[-1]
            bar = {"datetime": datetime.combine(asof, dtime(15, 25)),
                   "open": float(day_row["open_price"]), "high": float(day_row["high_price"]),
                   "low": float(day_row["low_price"]), "close": float(day_row["close_price"]),
                   "volume": float(day_row["volume"])}
            df = build_encoder_df(ticker, history, [bar], datetime.combine(asof, dtime(15, 30)),
                                  cols["historical"], cols["future"], cols["static"],
                                  market_dates=None, max_gap_days=BATCH_MAX_GAP_DAYS)
            pred = predict_fn(ticker, df)
            p_buy, p_hold, p_sell = (float(pred["prob_buy"]), float(pred["prob_hold"]), float(pred["prob_sell"]))
        except Exception as e:  # per-ticker isolation
            logger.error("%s 배치 추론 실패: %s", ticker, e)
            failed.append({"ticker": ticker, "reason": "error", "detail": f"{type(e).__name__}: {e}"})
            continue
        if not all(math.isfinite(p) for p in (p_buy, p_hold, p_sell)):
            failed.append({"ticker": ticker, "reason": "non_finite_output",
                           "detail": f"p_buy={p_buy} p_hold={p_hold} p_sell={p_sell}"})
            continue
        scored.append({"ticker": ticker, "p_buy": p_buy, "p_hold": p_hold, "p_sell": p_sell,
                       "score": p_buy - p_sell})

    ranks = compute_rank_pct({r["ticker"]: r["score"] for r in scored})
    for r in scored:
        r["rank_pct"] = ranks[r["ticker"]]
    scored.sort(key=lambda r: (-r["score"], r["ticker"]))
    return {"scores": scored, "excluded": excluded, "failed": failed}


# ============================================================ DB access (SELECT-only)
def load_universe(dsn: str) -> list[str]:
    import psycopg2

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT ticker FROM ticker_universe ORDER BY rank")
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def resolve_asof(dsn: str, universe: list[str], asof: date | None = None) -> date:
    if asof is not None:
        return asof
    import psycopg2

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT trade_date, COUNT(DISTINCT ticker) FROM feature_pool "
                        "WHERE ticker = ANY(%s) GROUP BY trade_date ORDER BY trade_date DESC LIMIT 10",
                        (universe,))
            coverage = {r[0]: r[1] for r in cur.fetchall()}
    finally:
        conn.close()
    return pick_asof(coverage, len(universe))


def load_universe_rows(dsn: str, universe: list[str], asof: date, cols: dict,
                       n_rows: int = N_ROWS, lookback_days: int = 150):
    """One SELECT for the whole universe: rows in (asof - lookback_days, asof]. Returns
    (rows_by_ticker oldest-first trimmed to the last n_rows, market_dates) where market_dates are the
    dates on which >= MIN_DATE_COVERAGE of the universe has a row."""
    import psycopg2
    from datetime import timedelta

    want = ["ticker", "trade_date", "close_price", "open_price", "high_price", "low_price", "volume"] \
        + cols["historical"] + FUTURE_DB_COLS + cols["static"]
    seen: set[str] = set()
    want = [c for c in want if not (c in seen or seen.add(c))]
    query = (f"SELECT {', '.join(want)} FROM feature_pool "
             "WHERE ticker = ANY(%s) AND trade_date <= %s AND trade_date > %s "
             "ORDER BY ticker, trade_date")
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(query, (universe, asof, asof - timedelta(days=lookback_days)))
            colnames = [d[0] for d in cur.description]
            fetched = cur.fetchall()
    finally:
        conn.close()
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    date_count: dict[date, int] = defaultdict(int)
    for r in fetched:
        row = dict(zip(colnames, r))
        by_ticker[row["ticker"]].append(row)
        date_count[row["trade_date"]] += 1
    market_dates = {d for d, n in date_count.items() if n >= MIN_DATE_COVERAGE * len(universe)}
    return {t: rows[-n_rows:] for t, rows in by_ticker.items()}, market_dates


# ============================================================ model + orchestration
def _load_model():
    from serving.model import get_model

    return get_model()


def make_predict_fn(model, cols: dict):
    from serving.inference import run_inference

    def predict(ticker, encoder_df):
        return run_inference(ticker, encoder_df, model, cols["historical"], cols["future"], cols["static"])

    return predict


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip() or None
    except Exception:
        return None


def run_batch(dsn: str, asof: date | None = None, out: str | None = None) -> dict:
    from serving.model import MODEL_PATH, get_feature_columns, get_model_version, load_champion_config

    t0 = time.perf_counter()
    cols = get_feature_columns()
    universe = load_universe(dsn)
    asof = resolve_asof(dsn, universe, asof)
    rows_by_ticker, market_dates = load_universe_rows(dsn, universe, asof, cols)
    model = _load_model()
    res = score_universe(universe, rows_by_ticker, market_dates, asof, make_predict_fn(model, cols), cols)
    result = {
        "asof": asof.isoformat(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": MODEL_PATH, "model_version": get_model_version(),
        "champion_config_version": load_champion_config().get("version"),
        "git_commit": _git_commit(),
        "score_definition": "score = p_buy - p_sell; rank_pct in [0,1] within this as-of date (1 = most buy-leaning)",
        "n_universe": len(universe), "n_scored": len(res["scores"]),
        "n_excluded": len(res["excluded"]), "n_failed": len(res["failed"]),
        **res,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    out = out or default_out_path(asof)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    logger.info("as-of %s: %d/%d scored, %d excluded, %d failed -> %s (%.1fs)", asof, result["n_scored"],
                len(universe), result["n_excluded"], result["n_failed"], out, result["elapsed_sec"])
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Batch-score the ticker universe on the latest completed trading day.")
    p.add_argument("--asof", type=date.fromisoformat, default=None,
                   help="as-of trade date YYYY-MM-DD (default: latest well-covered feature_pool date)")
    p.add_argument("--out", default=None,
                   help="output JSON path (default: training/artifacts/universe_scores_<asof>.json)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    dsn = os.environ.get("STOCK_DB_V2_DSN")
    if not dsn:
        raise ValueError("STOCK_DB_V2_DSN environment variable not set")
    run_batch(dsn, asof=args.asof, out=args.out)


if __name__ == "__main__":
    main()
