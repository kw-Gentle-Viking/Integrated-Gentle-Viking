"""
serving/feature_builder.py
===========================
Task 18 Step 5. Assembles a COMPLETE, correct 60-row encoder_df for one ticker:
    - 59 most recent real trading days, loaded as-is from stock_db_v2.feature_pool
      (already-correct engineered feature values straight from the DB)
    - 1 assembled "today" row, computed from today's live intraday 5-min bars
      (production stock_db.intraday_5min, read-only)

This module is the fix for the two Important issues Task 18 Steps 1-4's review flagged and
deferred to this step:
  1. `run_inference` used to guess today's opening price via `encoder_df.iloc[-1]["close"]`,
     silently falling back to 0 -- a real bug, since the real encoder_df (built from
     feature_pool's engineered columns) never has a raw "close" column, so the `else 0` branch
     was the production path. Fixed here: `open_price` for today's incomplete bar is either the
     first live intraday bar's `open` (the real, live opening print), or -- if no intraday bars
     exist yet (e.g. before/after market hours) -- yesterday's real close_price, never a bare 0.
  2. "Folding in today's live bar" used to only overwrite open/high/low/close/volume columns in
     encoder_df IF they already existed there -- since they never do in the real (engineered)
     column set, this was a complete no-op in production. Fixed here: this module is now the
     sole owner of "what does today's row look like", and every one of the champion's 33
     historical columns gets an explicit, documented value for today (see policy table below) --
     `serving/inference.py` no longer does any of this and just consumes the finished frame.

Column policy for "today"'s row (one of exactly three policies per column; see the constants
LIVE_COLS / FFILL_COLS / ZERO_DEFAULT_COLS below, and `test_column_policy_partition_covers_all_
historical_cols_with_no_overlap` in the test file, which pins that every champion column is
classified into exactly one bucket):

  Column                                                          | Policy | Why
  -----------------------------------------------------------------+--------+---------------------------------
  log_ret                                                          | LIVE   | ln(today_incomplete_close / yesterday_real_close)
  disparity_5d / disparity_20d / disparity_60d                     | LIVE   | (today_close - trailing_N_day_SMA_of_close_INCLUSIVE) / SMA
  rsi_14                                                           | LIVE   | standard 14-period RSI over trailing closes incl. today
  volatility_20d                                                   | LIVE   | rolling(20).std() of the log_ret series incl. today's live log_ret
  day_of_week                                                      | LIVE   | today's real calendar date .weekday() (Mon=0 -- confirmed against
                                                                    |        | stock_db_v2.calendar's own convention, see module test)
  sector_ret_1d/5d/20d, sector_ma_ratio_20d, sector_volatility,    | FFILL  | no live sector-index feed in this task's scope; carry forward
  sector_volume_ratio                                              |        | most recent real feature_pool value (project's established
  kospi_ret, kosdaq_ret, snp500_ret, nasdaq_ret, phlx_semi_ret,     | FFILL  | ffill-over-drop missing-value policy, same as build_features.py's
  vix_chg, usd_krw_chg, us_10y_yield_chg, rate_spread_us_kr,        |        | own macro_cols ffill)
  wti_ret, gold_ret                                                |        |
  lev_total_aum, lev_aum_to_mktcap, est_rebalancing_flow            | FFILL  | same -- no live leverage-ETF feed in scope
  vi_count_recent5d                                                 | FFILL  | trailing rolling stat, not a same-day event -- carrying it
                                                                    |        | forward one day is a reasonable approximation, unlike is_vi_triggered below
  is_dividend, is_bonus_issue, is_rights_offering, is_split         | ZERO   | event flags mean "this happened ON this specific day" --
  is_vi_triggered                                                   |        | carrying yesterday's "yes" forward as today's would be actively
                                                                    |        | wrong, not just stale. No live event/VI feed exists in this
                                                                    |        | task's scope, so 0 (honest "unknown/no") is the default.

KNOWN_FUTURE_COLS handling (not one of the 33 champion "historical" columns, but still part of
encoder_df since serving.inference.run_inference reads future_cols/static_cols off the same
frame):
  - time_progress: synthetic, not a stored column. EVERY row -- historical and today -- uses the
    constant 1.0 (training.stage1_data.TIME_PROGRESS_CONSTANT: training feeds 1.0 on all rows
    including the target row, so any other value would be out-of-distribution input). (Before
    2026-09-28 today's row used `compute_time_progress(now)` in [0, 1]; that was a train/serve
    mismatch and was removed. The helper is kept for informational use only.)
  - is_bok / is_fomc / is_witching_kr / is_witching_us: these ARE real feature_pool columns for
    the 59 historical days (joined from market_events at build_features.py time). For "today",
    the honest answer would be a real calendar lookup -- these are nominally "known in advance"
    events (BOK/FOMC schedules, witching days). However, a live check against this stock_db_v2
    confirmed market_events has no rows beyond 2025-12-19 -- the backfill was a historical
    snapshot, not a forward-populated calendar -- so there is no live source to look these up
    from for "today" in this task's scope, and ffilling from a >6-month-old row would be actively
    misleading (the same problem as the event flags above). These therefore also default to 0
    for today, not ffill. Static cols (sector_id, market_id) are effectively constant per ticker
    and are simply carried forward from the most recent real row (ffill).

Design note on trailing-average inputs (disparity_Nd / rsi_14 / volatility_20d): these need raw
close prices / log-return history, not just the ratio-shaped 33 champion columns. A live schema
check confirmed stock_db_v2.feature_pool already stores `close_price` directly (joined in from
price_daily at build_features.py time) -- so this module simply includes `close_price` in the
same feature_pool SELECT used for the 59-day history, rather than issuing a second query against
price_daily or reconstructing prices by cumulative-multiplying log_ret (which would compound
floating-point error for no benefit, since the exact value is already one column away). This
close_price is used purely as a computation input for today's row -- it is not one of the 33
model input columns and is dropped from the columns actually fed to the model.
"""

import math
from datetime import date, datetime, time as dtime

import pandas as pd

from training.dataset import build_incomplete_today_bar

MARKET_OPEN = dtime(9, 0)
MARKET_CLOSE = dtime(15, 30)

# time_progress constant used for ALL rows fed to the model (past days AND today) -- must equal
# training.stage1_data.TIME_PROGRESS_CONSTANT (pinned by a test; not imported here to keep
# psycopg2/training imports off the serving import path).
HISTORICAL_TIME_PROGRESS = 1.0

# The 4 real feature_pool columns backing 4 of the 5 KNOWN_FUTURE_COLS (time_progress is
# synthetic, handled separately). See module docstring for why "today" defaults these to 0.
FUTURE_DB_COLS = ["is_bok", "is_fomc", "is_witching_kr", "is_witching_us"]

# --- Column policy partition for the 33 champion historical columns (see docstring table) ---
LIVE_COLS = [
    "log_ret", "disparity_5d", "disparity_20d", "disparity_60d",
    "rsi_14", "volatility_20d", "day_of_week",
]
ZERO_DEFAULT_COLS = [
    "is_dividend", "is_bonus_issue", "is_rights_offering", "is_split", "is_vi_triggered",
]
# A column must be explicitly listed in exactly one of LIVE_COLS / ZERO_DEFAULT_COLS /
# FFILL_COLS to be served -- _make_today_row() raises ValueError for anything unclassified,
# rather than silently defaulting a future champion-column-list change to ffill without review
# (see 2026-09-18 review finding: the old unconditional "else -> ffill" branch would have let
# an unreviewed new column through silently).
FFILL_COLS = [
    "sector_ret_1d", "sector_ret_5d", "sector_ret_20d", "sector_ma_ratio_20d",
    "sector_volatility", "sector_volume_ratio",
    "lev_total_aum", "lev_aum_to_mktcap", "est_rebalancing_flow", "vi_count_recent5d",
    "kospi_ret", "kosdaq_ret", "snp500_ret", "nasdaq_ret", "phlx_semi_ret",
    "vix_chg", "usd_krw_chg", "us_10y_yield_chg", "rate_spread_us_kr", "wti_ret", "gold_ret",
]


def compute_time_progress(now: datetime) -> float:
    """(now - market_open) / (market_close - market_open), clamped to [0, 1].

    INFORMATIONAL ONLY -- deliberately NOT used as a model input: training always feeds the
    constant 1.0 (see HISTORICAL_TIME_PROGRESS), so this 0~1 value would be out-of-distribution."""
    open_dt = datetime.combine(now.date(), MARKET_OPEN)
    close_dt = datetime.combine(now.date(), MARKET_CLOSE)
    if now <= open_dt:
        return 0.0
    if now >= close_dt:
        return 1.0
    return (now - open_dt).total_seconds() / (close_dt - open_dt).total_seconds()


# ---------------------------------------------------------------------------
# History integrity guard (2026-09-28)
# ---------------------------------------------------------------------------
# Default max calendar-day distance between the last history row and "today" when no market
# trading-date list is available (weekend + one holiday fits; a week+ gap means stale data).
MAX_HISTORY_GAP_DAYS = 5


class HistoryIntegrityError(ValueError):
    """feature_pool history is not a trustworthy, gap-free run of days ending right before today."""


def _as_date(d) -> date:
    return d.date() if isinstance(d, datetime) else d


def validate_history_rows(
    rows: list[dict],
    today: date | None = None,
    market_dates: set[date] | None = None,
    market_covered_until: date | None = None,
    max_gap_days: int = MAX_HISTORY_GAP_DAYS,
) -> list[dict]:
    """Validate a ticker's feature_pool history and return the rows to actually use.

    (c) rows must be in strictly ascending trade_date order with no duplicate dates.
    (a) a row already dated `today` (e.g. the day's batch already landed in feature_pool) is
        dropped -- "today" is assembled separately, so keeping it would duplicate the day.
        Rows dated AFTER today are look-ahead and raise.
    (b) the history must end right before today: if `market_dates` (the market's real trading
        dates, from feature_pool itself) is given, the ticker must not lack any of the market's
        trading dates in (last_row, today); in all cases the distance between the newest known
        date (the ticker's last row, or `market_covered_until` if later) and today must not
        exceed `max_gap_days` calendar days. Without `market_dates` only the calendar-day rule
        applies. (stock_db_v2.calendar.is_market_open is NOT used: it is weekday-based and misses
        Korean holidays -- 119 days disagree with feature_pool's actual trading dates.)
    Raises HistoryIntegrityError explicitly instead of letting a stale/holey window through.
    """
    if not rows:
        raise HistoryIntegrityError("no feature_pool history rows")
    dates = [_as_date(r["trade_date"]) for r in rows]
    for prev, cur in zip(dates, dates[1:]):
        if cur == prev:
            raise HistoryIntegrityError(f"duplicate trade_date {cur} in history")
        if cur < prev:
            raise HistoryIntegrityError(
                f"history rows must be in ascending trade_date order ({cur} follows {prev})")
    if today is None:
        return list(rows)
    today = _as_date(today)
    if dates[-1] > today:
        raise HistoryIntegrityError(
            f"history contains rows dated after today ({dates[-1]} > {today}) -- look-ahead")
    kept = [r for r, d in zip(rows, dates) if d < today]
    if not kept:
        raise HistoryIntegrityError(f"no history rows before today ({today})")
    last = _as_date(kept[-1]["trade_date"])
    if market_dates is not None:
        missing = sorted(d for d in market_dates if last < d < today)
        if missing:
            shown = ", ".join(str(d) for d in missing[:5])
            raise HistoryIntegrityError(
                f"history ends {last} but the market traded on {len(missing)} later day(s) before "
                f"today ({today}); missing trading days: {shown}")
    newest_known = max(last, market_covered_until) if (market_dates is not None and market_covered_until) else last
    gap = (today - newest_known).days
    if gap > max_gap_days:
        raise HistoryIntegrityError(
            f"stale history: last known trading date {newest_known} is {gap} calendar days before "
            f"today ({today}) (gap limit {max_gap_days})")
    return kept


def _sma_inclusive(values: list[float], window: int) -> float:
    """Trailing SMA over the last `window` values (fewer if unavailable) -- matches
    build_features.py's `rolling(window=N, min_periods=1).mean()` semantics, where the window
    includes the current (last) value."""
    w = values[-window:]
    return sum(w) / len(w) if w else 0.0


def _rsi(closes: list[float], window: int = 14) -> float:
    """Standard RSI, matching build_features.py's calculate_rsi: avg_gain/avg_loss are means
    over the window (zeros included for non-gain/non-loss days), not means-of-nonzero-only."""
    diffs = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    w = diffs[-window:]
    if not w:
        return 50.0  # neutral default -- no diff history available at all
    avg_gain = sum(d for d in w if d > 0) / len(w)
    avg_loss = sum(-d for d in w if d < 0) / len(w)
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - 100 / (1 + rs)


def _rolling_std(values: list[float], window: int) -> float:
    """Sample std (ddof=1) over the trailing `window` values -- matches pandas'
    `.rolling(window).std()` default."""
    w = values[-window:]
    if len(w) < 2:
        return 0.0
    mean = sum(w) / len(w)
    var = sum((x - mean) ** 2 for x in w) / (len(w) - 1)
    return var ** 0.5


def build_encoder_df(
    ticker: str,
    history_rows: list[dict],
    today_intraday_rows: list[dict],
    now: datetime,
    historical_cols: list[str],
    future_cols: list[str],
    static_cols: list[str],
    market_dates: set[date] | None = None,
    market_covered_until: date | None = None,
) -> pd.DataFrame:
    """Pure computation, no DB access -- unit-testable with synthetic rows (see
    serving/test_feature_builder.py), matching the pattern already established by
    serving/test_inference.py.

    history_rows: the 59 most recent real trading days for `ticker`, oldest-first, each a dict
        with at least trade_date/close_price plus every column in historical_cols,
        FUTURE_DB_COLS, and static_cols (i.e. exactly what a `SELECT ... FROM feature_pool`
        row looks like).
    today_intraday_rows: today's live 5-min bars so far, each a dict with
        datetime/open/high/low/close/volume (production stock_db.intraday_5min shape).
    now: current wall-clock datetime (KST), used for day_of_week (time_progress is the constant
        1.0, see module docstring).
    market_dates / market_covered_until: optional market trading-date info for the gap guard
        (see validate_history_rows). Rows already dated today are dropped; ordering/duplicate/
        stale-gap violations raise HistoryIntegrityError.
    """
    if not history_rows:
        raise ValueError(f"build_encoder_df: no feature_pool history rows for ticker={ticker!r}")

    try:
        hist_sorted = validate_history_rows(
            history_rows, today=now.date(), market_dates=market_dates,
            market_covered_until=market_covered_until)
    except HistoryIntegrityError as e:
        raise HistoryIntegrityError(f"ticker={ticker!r}: {e}") from e
    closes = [float(r["close_price"]) for r in hist_sorted]
    last_row = hist_sorted[-1]

    # --- Fix for flagged bug #1: never silently default open_price to 0. ---
    if today_intraday_rows:
        open_price = float(min(today_intraday_rows, key=lambda r: r["datetime"])["open"])
    else:
        open_price = closes[-1]  # best available reference: yesterday's real close, not 0

    today_bar = build_incomplete_today_bar(today_intraday_rows, open_price=open_price)
    today_close = float(today_bar["close"])
    all_closes = closes + [today_close]

    live_values: dict[str, float] = {}
    live_values["log_ret"] = math.log(today_close / closes[-1]) if closes[-1] > 0 else 0.0
    for n, col in [(5, "disparity_5d"), (20, "disparity_20d"), (60, "disparity_60d")]:
        sma = _sma_inclusive(all_closes, n)
        live_values[col] = (today_close - sma) / sma if sma else 0.0
    live_values["rsi_14"] = _rsi(all_closes, 14)
    hist_log_rets = [float(r.get("log_ret") or 0.0) for r in hist_sorted]
    live_values["volatility_20d"] = _rolling_std(hist_log_rets + [live_values["log_ret"]], 20)
    live_values["day_of_week"] = float(now.date().weekday())

    def _make_today_row() -> dict:
        row = {}
        for col in historical_cols:
            if col in LIVE_COLS:
                row[col] = live_values[col]
            elif col in ZERO_DEFAULT_COLS:
                row[col] = 0.0
            elif col in FFILL_COLS:
                row[col] = float(last_row.get(col) or 0.0)
            else:
                # Not classified into any of the three policies -- e.g. a future champion
                # feature-set change adds a column nobody has reviewed yet for whether ffill
                # is actually the right default for it. Silently ffilling here (as the old
                # unconditional "else" branch did) would let it happen by accident rather than
                # by a reviewed decision. Fail loudly instead.
                raise ValueError(
                    f"build_encoder_df: historical column {col!r} is not classified into "
                    f"LIVE_COLS, ZERO_DEFAULT_COLS, or FFILL_COLS in serving/feature_builder.py "
                    f"-- add it to the appropriate list (with a reasoned policy, see module "
                    f"docstring) before it can be served."
                )
        for col in future_cols:
            if col == "time_progress":
                # Training feeds the constant 1.0 on EVERY row, including the target row
                # (training.stage1_data.TIME_PROGRESS_CONSTANT). Feeding compute_time_progress(now)
                # (0~1 during the session) would hand the model a value it never saw in training,
                # so today's row uses the same constant as history. compute_time_progress is kept
                # only as an informational helper -- it must not reach the model input.
                row[col] = HISTORICAL_TIME_PROGRESS
            elif col in FUTURE_DB_COLS:
                row[col] = 0.0  # see docstring: no forward-populated calendar in this task's scope
            else:
                row[col] = float(last_row.get(col) or 0.0)
        for col in static_cols:
            row[col] = last_row.get(col)
        return row

    def _make_history_row(r: dict) -> dict:
        row = {}
        for col in historical_cols:
            row[col] = float(r.get(col) or 0.0)
        for col in future_cols:
            row[col] = HISTORICAL_TIME_PROGRESS if col == "time_progress" else float(r.get(col) or 0.0)
        for col in static_cols:
            row[col] = r.get(col)
        return row

    rows = [_make_history_row(r) for r in hist_sorted]
    rows.append(_make_today_row())
    return pd.DataFrame(rows)


# ============================================================
# Thin DB wrappers (kept separately testable from the pure logic above)
# ============================================================

def fetch_feature_pool_history(dsn: str, ticker: str, historical_cols: list[str],
                                static_cols: list[str], n_days: int = 59,
                                today: date | None = None) -> list[dict]:
    """SELECT-only read from stock_db_v2.feature_pool: the ticker's most recent `n_days` real
    trading-day rows, oldest-first, including close_price (see module docstring) and
    FUTURE_DB_COLS. If `today` is given, rows dated today or later are excluded in SQL (today is
    assembled separately; a same-day feature_pool row would otherwise duplicate it)."""
    import psycopg2

    cols = ["trade_date", "close_price"] + historical_cols + FUTURE_DB_COLS + static_cols
    seen = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]  # dedupe, preserve order
    col_sql = ", ".join(cols)
    date_filter = "AND trade_date < %s" if today is not None else ""
    query = f"""
        SELECT {col_sql} FROM feature_pool
        WHERE ticker = %s {date_filter}
        ORDER BY trade_date DESC
        LIMIT %s
    """
    params = (ticker, today, n_days) if today is not None else (ticker, n_days)
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            colnames = [d[0] for d in cur.description]
    finally:
        conn.close()
    result = [dict(zip(colnames, r)) for r in rows]
    result.sort(key=lambda r: r["trade_date"])
    return result


def fetch_market_trading_dates(dsn: str, start: date, end: date) -> tuple[set[date] | None, date | None]:
    """SELECT-only: the market's real trading dates strictly between `start` and `end`, taken from
    feature_pool itself (distinct trade_date), plus the newest trade_date feature_pool has at all.
    Returns (None, None) if it cannot be read, so callers fall back to the calendar-day rule.
    Deliberately not stock_db_v2.calendar: its is_market_open ignores Korean holidays."""
    import psycopg2

    try:
        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT trade_date FROM feature_pool "
                            "WHERE trade_date > %s AND trade_date < %s", (start, end))
                dates = {r[0] for r in cur.fetchall()}
                cur.execute("SELECT MAX(trade_date) FROM feature_pool")
                covered = cur.fetchone()[0]
        finally:
            conn.close()
    except Exception:
        return None, None
    return dates, covered


def fetch_today_intraday_rows(prod_dsn: str, ticker: str, today: date) -> list[dict]:
    """SELECT-only read from PRODUCTION stock_db.intraday_5min (never written here -- that
    table is populated by production's own already-running collector_realtime.py cron job)."""
    import psycopg2

    query = """
        SELECT datetime, open, high, low, close, volume
        FROM intraday_5min
        WHERE ticker = %s AND datetime::date = %s
        ORDER BY datetime
    """
    conn = psycopg2.connect(prod_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(query, (ticker, today))
            rows = cur.fetchall()
    finally:
        conn.close()
    return [
        {"datetime": r[0], "open": float(r[1]), "high": float(r[2]),
         "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])}
        for r in rows
    ]


def build_encoder_df_for_ticker(
    ticker: str, v2_dsn: str, prod_dsn: str,
    historical_cols: list[str], future_cols: list[str], static_cols: list[str],
    now: datetime | None = None, n_days: int = 59,
) -> pd.DataFrame:
    """Orchestrates the two read-only DB fetches above and calls the pure build_encoder_df."""
    now = now or datetime.now()
    history_rows = fetch_feature_pool_history(v2_dsn, ticker, historical_cols, static_cols,
                                              n_days=n_days, today=now.date())
    if not history_rows:
        raise ValueError(f"no feature_pool history found for ticker={ticker!r}")
    if len(history_rows) < n_days:
        # A short history (e.g. a newly-listed ticker) would silently produce a shorter-than-
        # expected encoder_df -- tft-torch's LSTM-based encoder doesn't error on a short
        # sequence, so this would otherwise surface as a plausible-looking but out-of-
        # distribution prediction rather than a loud failure (2026-09-18 review finding).
        raise ValueError(
            f"build_encoder_df_for_ticker: only {len(history_rows)}/{n_days} real trading-day "
            f"rows found in feature_pool for ticker={ticker!r} -- too little history to serve "
            f"a reliable prediction (e.g. a newly-listed ticker)."
        )
    last_date = _as_date(history_rows[-1]["trade_date"])
    market_dates, covered_until = fetch_market_trading_dates(v2_dsn, last_date, now.date())
    today_intraday_rows = fetch_today_intraday_rows(prod_dsn, ticker, now.date())
    return build_encoder_df(ticker, history_rows, today_intraday_rows, now,
                             historical_cols, future_cols, static_cols,
                             market_dates=market_dates, market_covered_until=covered_until)
