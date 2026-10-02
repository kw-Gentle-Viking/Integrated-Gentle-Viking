"""Chronos-2 foundation-model scorer: pure logic for the Chronos-2 zero-shot experiment
(docs/chronos2_experiment.md). Kept separate from training/run_chronos2_experiment.py (DB access,
model loading, orchestration) so the context-window / long-format-conversion / scoring / date-stride
logic can be unit tested with synthetic data and no DB or model weights.

Design (per the experiment brief): one Chronos-2 forecast call per sample DATE, covering the full
cross-section of tickers trading that day. Each ticker is one `id_column` item with TWO target
columns (`close`, `volume_log1p`) in the SAME row -> Chronos-2's predict_df groups a single item_id's
target columns into one multivariate task (native group attention between close and volume of that
ticker); different tickers are independent tasks in the same batched call (no cross-ticker attention
unless cross_learning=True, which is not used here). Context = raw close price (model does its own
internal scaling) and log1p(volume) (reference pattern: timeseries_benchmark/models/chronos2/run.py).
Timestamps are synthetic, one-minute-spaced integer steps per ticker (not real trading dates): KRX
trading-day gaps (weekends, holidays, per-ticker halts) are irregular, but Chronos-2 requires a single
regular inferred frequency shared by every series in a call, so real dates cannot be used directly.
"""
import numpy as np
import pandas as pd

CONTEXT_LEN = 512
TARGET_CLOSE = "close"
TARGET_VOLUME = "volume_log1p"
TARGET_COLS = [TARGET_CLOSE, TARGET_VOLUME]
SYNTH_ORIGIN = pd.Timestamp("2020-01-01")
MIN_SERIES_LEN = 3   # Chronos-2 predict_df requires >= 3 points per series


def make_long_context_df(context: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    """context[ticker] = {"close": 1D array (ascending, last = asof day), "volume": 1D array, raw}.
    -> long-format df with columns [id, timestamp, close, volume_log1p], ready for
    Chronos2Pipeline.predict_df(..., id_column="id", timestamp_column="timestamp",
    target=["close", "volume_log1p"]). Row order does not matter: predict_df sorts by
    [id_column, timestamp_column] itself when validate_inputs=True (the default)."""
    parts = []
    for tk, s in context.items():
        close = np.asarray(s["close"], dtype=float)
        vol = np.asarray(s["volume"], dtype=float)
        if len(close) != len(vol):
            raise ValueError(f"{tk}: close/volume length mismatch ({len(close)} vs {len(vol)})")
        n = len(close)
        parts.append(pd.DataFrame({
            "id": tk,
            "timestamp": pd.to_datetime(np.arange(n), unit="m", origin=SYNTH_ORIGIN),
            TARGET_CLOSE: close,
            TARGET_VOLUME: np.log1p(vol),
        }))
    if not parts:
        return pd.DataFrame(columns=["id", "timestamp", TARGET_CLOSE, TARGET_VOLUME])
    return pd.concat(parts, ignore_index=True)


def extract_predicted_close(pred_df: pd.DataFrame, id_column: str = "id") -> dict[str, float]:
    """{ticker: predicted next-step close} from a Chronos2Pipeline.predict_df(...) output
    (long format with a `target_name` column and a `predictions` point-forecast column)."""
    sub = pred_df[pred_df["target_name"] == TARGET_CLOSE]
    return dict(zip(sub[id_column], sub["predictions"].astype(float)))


def predicted_log_returns(pred_close: dict[str, float], last_close: dict[str, float]) -> dict[str, float]:
    """log(pred_close / last_close) per ticker present in BOTH dicts; non-positive prices are
    floored at 1e-8 (mirrors the reference script) so a (near-)zero price cannot produce +/-inf."""
    out = {}
    for tk, pc in pred_close.items():
        if tk not in last_close:
            continue
        lc = max(float(last_close[tk]), 1e-8)
        pc = max(float(pc), 1e-8)
        out[tk] = float(np.log(pc / lc))
    return out


def select_strided_dates(sorted_unique_dates: list[str], stride: int) -> list[str]:
    """Every `stride`-th date of a sorted, de-duplicated date list, keeping index 0 (and therefore
    the first date) and preserving order. stride=1 keeps every date (no reduction)."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    return list(sorted_unique_dates[::stride])


def context_window(dates: np.ndarray, values: np.ndarray, asof_date: str,
                   context_len: int = CONTEXT_LEN) -> np.ndarray | None:
    """`values` for the up-to-`context_len` most recent rows of `dates` that are <= asof_date,
    ending at (and including) asof_date itself. `dates` must be sorted ascending. None if asof_date
    is not present in `dates`, or fewer than MIN_SERIES_LEN rows are available up to it."""
    dates = np.asarray(dates)
    pos = np.searchsorted(dates, asof_date)
    if pos >= len(dates) or dates[pos] != asof_date:
        return None
    end = pos + 1
    start = max(0, end - context_len)
    if end - start < MIN_SERIES_LEN:
        return None
    return np.asarray(values)[start:end]


def build_day_context(price_history: dict[str, dict[str, np.ndarray]], tickers: list[str], asof_date: str,
                      context_len: int = CONTEXT_LEN) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float]]:
    """For one sample date: context windows (close, volume) + last close, for every ticker in
    `tickers` that has enough history. `price_history[ticker]` = {"dates": sorted ascending array
    (strings, "YYYY-MM-DD"), "close": array, "volume": array} (full, un-windowed history).
    Returns (context, last_close) where context is ready for make_long_context_df and last_close
    is the asof-day close used to turn a predicted close into a log return. Tickers with no usable
    window (missing the date, or < MIN_SERIES_LEN prior rows) are silently dropped from both."""
    context: dict[str, dict[str, np.ndarray]] = {}
    last_close: dict[str, float] = {}
    for tk in tickers:
        hist = price_history.get(tk)
        if hist is None:
            continue
        c = context_window(hist["dates"], hist["close"], asof_date, context_len)
        if c is None:
            continue
        v = context_window(hist["dates"], hist["volume"], asof_date, context_len)
        context[tk] = {"close": c, "volume": v}
        last_close[tk] = float(c[-1])
    return context, last_close


def score_day(pred_df: pd.DataFrame, last_close: dict[str, float]) -> dict[str, float]:
    """Chronos-2 predict_df output for one sample date -> {ticker: score}, score = predicted
    next-day log return (the Chronos-2 signal, used directly as the cross-sectional rank score)."""
    pred_close = extract_predicted_close(pred_df)
    return predicted_log_returns(pred_close, last_close)
