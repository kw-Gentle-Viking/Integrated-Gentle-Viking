"""Train/serve parity: does the serving path (feature_pool 59 rows + a completed day's bar ->
serving.feature_builder.build_encoder_df -> serving.inference.assemble_model_inputs) produce the
same model inputs as training.dataset.TickerDayDataset(align="today") for the same (ticker, day)?

Runs only when STOCK_DB_V2_DSN is set AND the adj1 val cache exists (else skipped). Read-only DB
access. See docs/serving_parity.md for the list of INTENDED differences and their measured size.
"""
import os
import pickle
import random
from datetime import datetime, time as dtime

import numpy as np
import pytest

from serving.feature_builder import (
    FFILL_COLS, FUTURE_DB_COLS, LIVE_COLS, ZERO_DEFAULT_COLS,
    build_encoder_df, fetch_feature_pool_history,
)
from serving.inference import assemble_model_inputs
from serving.model import get_feature_columns
from training.dataset import TickerDayDataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(REPO_ROOT, "training", "artifacts",
                          "stage1_cache_val_2024-01-01_2024-12-31__adj1.pkl")
DSN = os.environ.get("STOCK_DB_V2_DSN")

pytestmark = pytest.mark.skipif(
    not DSN or not os.path.exists(CACHE_PATH),
    reason="needs STOCK_DB_V2_DSN and the adj1 val stage1 cache",
)

ATOL = 1e-5  # float32 tensors; columns are O(1)-O(1e3) so also use rtol
RTOL = 1e-5
ENCODER_LEN = 60


def _load_cache():
    with open(CACHE_PATH, "rb") as f:
        return pickle.load(f)


def _pick_samples(cache, n_tickers=3, n_dates=3, seed=20260928):
    """Deterministic pseudo-random (ticker, target_date) pairs whose 59-day history lies inside
    the val cache (so the training-side window is complete)."""
    rng = random.Random(seed)
    tickers = rng.sample(sorted(cache), n_tickers)
    out = []
    for tk in tickers:
        df = cache[tk].reset_index(drop=True)
        labeled = [i for i in range(ENCODER_LEN - 1, len(df)) if df.loc[i, "label"] == df.loc[i, "label"]]
        for i in rng.sample(labeled, n_dates):
            out.append((tk, df.loc[i, "trade_date"]))
    return out


def _train_sample(cache, ticker, day, cols):
    df = cache[ticker].reset_index(drop=True)
    ds = TickerDayDataset({ticker: df.copy()}, cols["historical"], cols["future"], cols["static"],
                          encoder_len=ENCODER_LEN, align="today")
    t = int(df.index[df["trade_date"] == day][0]) - (ENCODER_LEN - 1)
    idx = ds.index.index((ticker, t))
    return ds[idx]


def _day_bar(dsn, ticker, day):
    import psycopg2
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT open_price, high_price, low_price, close_price, volume "
                        "FROM feature_pool WHERE ticker=%s AND trade_date=%s", (ticker, day))
            o, h, l, c, v = cur.fetchone()
    finally:
        conn.close()
    return {"datetime": datetime.combine(day, dtime(15, 25)), "open": float(o), "high": float(h),
            "low": float(l), "close": float(c), "volume": float(v)}


def _serve_inputs(dsn, ticker, day, cols):
    """Serving path for `day` as if the day's bar were complete (close of session)."""
    history = fetch_feature_pool_history(dsn, ticker, cols["historical"], cols["static"],
                                         n_days=ENCODER_LEN - 1, today=day)
    df = build_encoder_df(ticker, history, [_day_bar(dsn, ticker, day)],
                          datetime.combine(day, dtime(15, 30)),
                          cols["historical"], cols["future"], cols["static"],
                          max_gap_days=15)  # parity is about values; holiday blocks must not skip samples
    return assemble_model_inputs(df, cols["historical"], cols["future"], cols["static"])


POLICY_COLS = set(FFILL_COLS) | set(ZERO_DEFAULT_COLS) | set(FUTURE_DB_COLS)
EXACT_TODAY_HINT = (
    "LIVE_COLS are recomputed from the completed day's close and must equal feature_pool's own "
    "value; a diff here means a formula mismatch with features/build_features.py, i.e. a serving "
    "bug (see docs/serving_parity.md).")


@pytest.fixture(scope="module")
def pairs():
    cols = get_feature_columns()
    cache = _load_cache()
    out = []
    for ticker, day in _pick_samples(cache):
        out.append({
            "ticker": ticker, "day": day,
            "train": _train_sample(cache, ticker, day, cols),
            "serve": _serve_inputs(DSN, ticker, day, cols),
            "prev_day_row": cache[ticker].reset_index(drop=True).pipe(
                lambda df: df.loc[int(df.index[df["trade_date"] == day][0]) - 1]),
        })
    return cols, out


def _msg(p, col, kind, tv, sv):
    return (f"{p['ticker']} {p['day']}: column {col!r} ({kind}) train={tv!r} serve={sv!r} -- "
            f"see docs/serving_parity.md")


def test_sample_selection_is_3_tickers_x_3_dates(pairs):
    _, out = pairs
    assert len(out) == 9 and len({p["ticker"] for p in out}) == 3


def test_59_history_rows_match_training_for_every_column(pairs):
    cols, out = pairs
    for p in out:
        th = p["train"]["historical_ts_numeric"].numpy()[:-1]
        sh = p["serve"]["historical_ts_numeric"][0].numpy()[:-1]
        for j, c in enumerate(cols["historical"]):
            assert np.allclose(th[:, j], sh[:, j], rtol=RTOL, atol=ATOL), _msg(
                p, c, "history rows 0..58", th[:, j].tolist()[-3:], sh[:, j].tolist()[-3:])


def test_today_row_columns_without_a_documented_policy_match_training(pairs):
    """Everything not in FFILL/ZERO/FUTURE_DB policy -- the LIVE_COLS (recomputed from the day's
    close), day_of_week, time_progress and static ids -- must equal training's target-row value."""
    cols, out = pairs
    for p in out:
        th = p["train"]["historical_ts_numeric"].numpy()[-1]
        sh = p["serve"]["historical_ts_numeric"][0].numpy()[-1]
        for j, c in enumerate(cols["historical"]):
            if c in POLICY_COLS:
                continue
            assert np.isclose(th[j], sh[j], rtol=RTOL, atol=ATOL), _msg(p, c, "LIVE/exact", th[j], sh[j]) + " " + EXACT_TODAY_HINT
        tf = p["train"]["future_ts_numeric"].numpy().ravel()
        sf = p["serve"]["future_ts_numeric"].numpy().ravel()
        for j, c in enumerate(cols["future"]):
            if c in POLICY_COLS:
                continue
            assert np.isclose(tf[j], sf[j]), _msg(p, c, "future", tf[j], sf[j])
        assert (p["train"]["static_feats_categorical"].numpy() == p["serve"]["static_feats_categorical"].numpy()).all()


def test_today_row_policy_columns_follow_the_documented_policy(pairs):
    """FFILL cols == previous trading day's value; ZERO cols == 0; DB future flags == previous
    day's flags. These are the INTENDED divergences from training's same-day values (no live
    sector/macro/leverage/event feed in scope); their measured size is in docs/serving_parity.md."""
    cols, out = pairs
    for p in out:
        th_prev = p["train"]["historical_ts_numeric"].numpy()[-2]  # D-1 real row as training sees it
        sh = p["serve"]["historical_ts_numeric"][0].numpy()[-1]
        for j, c in enumerate(cols["historical"]):
            if c in FFILL_COLS:
                assert np.isclose(sh[j], th_prev[j], rtol=RTOL, atol=ATOL), _msg(p, c, "FFILL", th_prev[j], sh[j])
            elif c in ZERO_DEFAULT_COLS:
                assert sh[j] == 0.0, _msg(p, c, "ZERO", 0.0, sh[j])
        sf = p["serve"]["future_ts_numeric"].numpy().ravel()
        for j, c in enumerate(cols["future"]):
            if c in FUTURE_DB_COLS:
                assert sf[j] == pytest.approx(float(p["prev_day_row"][c])), _msg(
                    p, c, "FUTURE_DB carried forward", float(p["prev_day_row"][c]), sf[j])


def test_intended_divergences_are_documented():
    doc = os.path.join(REPO_ROOT, "docs", "serving_parity.md")
    assert os.path.exists(doc), "docs/serving_parity.md must list the intended train/serve divergences"
    text = open(doc, encoding="utf-8").read()
    missing = [c for c in sorted(POLICY_COLS) if c not in text]
    assert not missing, f"undocumented policy columns: {missing}"
