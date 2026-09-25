"""Task 15 Step 5: data loading, caching, and class-weight helpers for the live Stage-1
Optuna search + ablation run. Kept separate from run_stage1_search.py so the DB-load /
caching logic can be reasoned about (and reused) independently of the search orchestration.
"""
import logging
import os
import pickle

import pandas as pd
import psycopg2
import psycopg2.extras
import torch

from training.config import HISTORICAL_COLS_DEFAULT, STATIC_COLS
from training.dataset import target_offset_of

logger = logging.getLogger(__name__)

# training.config.KNOWN_FUTURE_COLS = ["time_progress", "is_bok", "is_fomc", "is_witching_kr",
# "is_witching_us"]. The latter four are real feature_pool columns (confirmed via
# information_schema query, 2026-09-16); "time_progress" is NOT -- it is synthetic per Task 12/13.
DB_FUTURE_COLS = ["is_bok", "is_fomc", "is_witching_kr", "is_witching_us"]

# This project predicts exactly one trading day ahead: TickerDayDataset (Task 12) reshapes the
# target row's future features to (1, -1) -- a single known-future step, not a multi-step
# horizon -- so there is no meaningful "fraction of the way through the horizon" to compute.
# Use a constant, matching the precedent already set by Task 13's own smoke test
# (task-13-brief.md Step 5: `'time_progress': 1.0`).
TIME_PROGRESS_CONSTANT = 1.0


# Label columns present on feature_pool that may be selected as the training target.
LABEL_COLS = ("label", "label_vn")


def label_select_expr(label_col: str) -> str:
    """SQL select expression yielding the chosen label column under the name `label`.
    Whitelisted (LABEL_COLS) because the name is interpolated into SQL."""
    if label_col not in LABEL_COLS:
        raise ValueError(f"unknown label_col {label_col!r}; expected one of {LABEL_COLS}")
    return "label" if label_col == "label" else f"{label_col} AS label"


def resolve_cache_path(cache_path: str, label_col: str = "label") -> str:
    """Default label keeps the existing cache filename (existing caches stay valid); any other
    label column gets a distinct '<stem>__<label_col><ext>' so caches can never be confused."""
    if label_col == "label":
        return cache_path
    stem, ext = os.path.splitext(cache_path)
    return f"{stem}__{label_col}{ext}"


def query_feature_pool(dsn: str, start_date: str, end_date: str, label_col: str = "label") -> pd.DataFrame:
    """Query feature_pool for HISTORICAL_COLS_DEFAULT + the DB-backed future cols + STATIC_COLS
    + label, across the full ticker universe, for one date range.

    Deliberately does NOT filter out NULL labels: those rows are still needed as historical
    context inside 60-day encoder windows near a ticker's early history / the tail of the range.
    TickerDayDataset (Task 12) already excludes them as prediction *targets* via its own
    `pd.notna(df.loc[t + encoder_len, "label"])` check.
    """
    cols = HISTORICAL_COLS_DEFAULT + DB_FUTURE_COLS + STATIC_COLS
    col_sql = ", ".join(cols)
    query = f"""
        SELECT ticker, trade_date, {col_sql}, {label_select_expr(label_col)}
        FROM feature_pool
        WHERE trade_date >= %s AND trade_date <= %s
        ORDER BY ticker, trade_date
    """
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(query, (start_date, end_date))
            rows = cur.fetchall()
            colnames = [d[0] for d in cur.description]
    finally:
        conn.close()
    return pd.DataFrame([list(r) for r in rows], columns=colnames)


def build_ticker_dfs(df: pd.DataFrame, label_col: str = "label") -> dict[str, pd.DataFrame]:
    """Split the flat feature_pool query result into a per-ticker dict of DataFrames, sorted by
    trade_date, with a synthetic time_progress column added and NaNs filled to 0.0.

    NaN sources confirmed via direct DB query (2026-09-16) on the 2019-2023 train range:
    - log_ret / kospi_ret / kosdaq_ret / snp500_ret / nasdaq_ret / phlx_semi_ret / vix_chg /
      usd_krw_chg / us_10y_yield_chg / wti_ret / gold_ret: NULL only on each ticker's first
      observed day (no prior close to compute a change from) -> 0.0 (no change) is the correct
      fill.
    - is_dividend / is_bonus_issue / is_rights_offering / is_split / is_bok / is_fomc /
      is_witching_kr / is_witching_us: stored NULL (not 0) when no matching event row exists for
      that ticker/date -> 0.0 (event did not occur) is the correct fill.
    - volatility_20d: a small number of NULLs (edge of window); 0.0 fill.
    disparity_60d/rsi_14/etc. had zero NULLs in the checked range (build_features.py already
    handles the 60-day rolling-window warmup), so this fillna is a safety net, not load-bearing.
    """
    if label_col != "label":
        # query_feature_pool already aliases the chosen column to `label`; this also accepts a raw
        # frame that still carries the chosen column under its own name.
        if "label" not in df.columns:
            if label_col not in df.columns:
                raise ValueError(f"df has neither 'label' nor {label_col!r}")
            df = df.rename(columns={label_col: "label"})
        elif label_col in df.columns:
            df = df.drop(columns=[label_col])
    numeric_cols = [c for c in df.columns if c not in ("ticker", "trade_date", "label")]
    ticker_dfs: dict[str, pd.DataFrame] = {}
    for ticker, group in df.groupby("ticker"):
        g = group.sort_values("trade_date").reset_index(drop=True)
        g[numeric_cols] = g[numeric_cols].astype(float).fillna(0.0)
        g["time_progress"] = TIME_PROGRESS_CONSTANT
        g["label"] = g["label"].astype(float)  # keep NaN (float) for unlabeled rows; Dataset checks notna()
        ticker_dfs[ticker] = g
    return ticker_dfs


def load_or_build_ticker_dfs(dsn: str, start_date: str, end_date: str, cache_path: str,
                             label_col: str = "label") -> dict[str, pd.DataFrame]:
    """DB-load + per-ticker frame construction is the expensive, trial-independent part of the
    pipeline (unlike TickerDayDataset construction, which is cheap and must be rebuilt per
    ablation config since the column subset changes) -- cache it to disk so 20 Optuna trials +
    7 ablation runs don't each re-run the same ~250k-row query."""
    cache_path = resolve_cache_path(cache_path, label_col)
    if os.path.exists(cache_path):
        logger.info("Loading cached ticker dataframes from %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    logger.info("Querying feature_pool for %s..%s (no cache at %s)", start_date, end_date, cache_path)
    df = query_feature_pool(dsn, start_date, end_date, label_col=label_col)
    ticker_dfs = build_ticker_dfs(df, label_col=label_col)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(ticker_dfs, f)
    logger.info("Cached %d tickers, %d total rows to %s", len(ticker_dfs), len(df), cache_path)
    return ticker_dfs


def compute_label_distribution(dataset) -> dict[int, int]:
    """Actual label distribution over the dataset's *usable* (windowed) targets -- not raw
    feature_pool row counts -- since windowing drops the first encoder_len rows of each ticker's
    history (no full lookback window available yet). This is the distribution the model actually
    trains against, so it's what class weights should be derived from."""
    counts = {0: 0, 1: 0, 2: 0}
    for ticker, t in dataset.index:
        label = int(dataset.ticker_dfs[ticker].loc[t + target_offset_of(dataset), "label"])
        counts[label] += 1
    return counts


def compute_class_weights(label_counts: dict[int, int]) -> torch.Tensor:
    """Inverse-frequency class weights (sklearn's 'balanced' formula:
    n_samples / (n_classes * n_samples_for_class)), computed from the ACTUAL train-split label
    distribution rather than assumed/uniform -- this project's redesign motivation explicitly
    includes fixing the original capstone's '관망(hold) signal never fires' bug, and uniform
    weights on an imbalanced 3-class target works against that."""
    total = sum(label_counts.values())
    n_classes = len(label_counts)
    weights = [total / (n_classes * label_counts[c]) for c in sorted(label_counts)]
    return torch.tensor(weights, dtype=torch.float32)


def compute_uniform_class_weights(label_counts: dict[int, int]) -> torch.Tensor:
    """Always [1.0, 1.0, 1.0], regardless of label_counts -- the "no correction at all" scheme,
    for the classweight-sweep experiment (see docs/model_versions.md's Stage-2 buy-skew note and
    the classweight-sweep plan doc). Still takes label_counts as a parameter, unused, for
    interface consistency with compute_class_weights/compute_mild_class_weights and to make its
    independence from label_counts explicit and directly testable."""
    del label_counts  # deliberately unused -- see docstring
    return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)


def compute_mild_class_weights(label_counts: dict[int, int]) -> torch.Tensor:
    """Geometric mean between uniform (1.0) and compute_class_weights' full-balanced weight, per
    class: mild_weight_c = sqrt(balanced_weight_c). A dampened middle ground between "no
    correction" (compute_uniform_class_weights) and "full inverse-frequency correction"
    (compute_class_weights), which the classweight-sweep experiment suspects may have overshot
    into over-predicting the down-weighted majority class's complement (buy-skew bug). Calls
    compute_class_weights internally rather than duplicating the balanced-weight formula."""
    balanced = compute_class_weights(label_counts)
    return torch.sqrt(balanced)


# Name -> weight-function registry for the classweight-sweep experiment and
# run_stage2_final_live.py's --weight-scheme flag, so both share one source of truth for the
# scheme name <-> function mapping instead of duplicating a dict in each caller.
WEIGHT_SCHEMES = {
    "balanced": compute_class_weights,
    "uniform": compute_uniform_class_weights,
    "mild": compute_mild_class_weights,
}
