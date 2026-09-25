"""Helpers mapping each TickerDayDataset sample to (ticker, target date, features/returns) for the
label-agnostic signal metrics (plan S3). Nothing here changes training behavior: the dataset's
frames do NOT carry next_day_return (stage1_data selects only feature/label columns), so it is
fetched separately (read-only) and joined by (ticker, target date)."""
import numpy as np
import psycopg2

from training.dataset import target_offset_of


def _fmt_date(d) -> str:
    return d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)


def sample_meta(ds) -> list[tuple[str, str, float]]:
    """Per sample, in ds.index order (== shuffle=False DataLoader order):
    (ticker, target trade_date 'YYYY-MM-DD', log_ret on the target row). The target row's log_ret
    is the return realised up to that day's close (yesterday-to-today), i.e. information known
    when predicting the NEXT day -- used for the reversal/momentum reference heuristic."""
    out = []
    for ticker, t in ds.index:
        row = ds.ticker_dfs[ticker].loc[t + target_offset_of(ds)]
        out.append((ticker, _fmt_date(row["trade_date"]), float(row["log_ret"])))
    return out


def filter_index_by_target_date(ds, min_target_date: str) -> int:
    """Keep only samples whose TARGET date >= min_target_date (rows before it stay in the frames
    as 60-day lookback context). Mutates ds.index in place, preserving order; returns #dropped."""
    before = len(ds.index)
    ds.index = [
        (ticker, t) for ticker, t in ds.index
        if _fmt_date(ds.ticker_dfs[ticker].loc[t + target_offset_of(ds), "trade_date"]) >= min_target_date
    ]
    return before - len(ds.index)


def fetch_next_day_returns(dsn: str, start_date: str, end_date: str) -> dict:
    """Read-only: {(ticker, 'YYYY-MM-DD'): next_day_return or None} from feature_pool."""
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ticker, trade_date, next_day_return FROM feature_pool "
                "WHERE trade_date >= %s AND trade_date <= %s", (start_date, end_date))
            return {(t, _fmt_date(d)): (None if r is None else float(r)) for t, d, r in cur.fetchall()}
    finally:
        conn.close()


def next_day_returns_for_samples(meta, lookup: dict) -> np.ndarray:
    """Float array aligned with `meta`; NaN where the lookup has no / NULL value."""
    return np.array([
        (lookup.get((tk, d)) if lookup.get((tk, d)) is not None else np.nan) for tk, d, _ in meta
    ], dtype=float)
