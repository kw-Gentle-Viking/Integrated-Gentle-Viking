import numpy as np
import pandas as pd
import pytest

from training.chronos2_scorer import (
    MIN_SERIES_LEN, TARGET_CLOSE, TARGET_VOLUME, build_day_context, context_window,
    extract_predicted_close, make_long_context_df, predicted_log_returns, score_day,
    select_strided_dates,
)


# ---- make_long_context_df ----

def test_make_long_context_df_columns_and_values():
    ctx = {"005930": {"close": np.array([100.0, 101.0, 102.0]), "volume": np.array([10.0, 20.0, 30.0])},
           "000660": {"close": np.array([50.0, 51.0]), "volume": np.array([5.0, 0.0])}}
    df = make_long_context_df(ctx)
    assert set(df.columns) == {"id", "timestamp", TARGET_CLOSE, TARGET_VOLUME}
    a = df[df.id == "005930"].sort_values("timestamp")
    np.testing.assert_allclose(a[TARGET_CLOSE].values, [100.0, 101.0, 102.0])
    np.testing.assert_allclose(a[TARGET_VOLUME].values, np.log1p([10.0, 20.0, 30.0]))
    b = df[df.id == "000660"].sort_values("timestamp")
    assert len(b) == 2
    np.testing.assert_allclose(b[TARGET_VOLUME].values, np.log1p([5.0, 0.0]))


def test_make_long_context_df_timestamps_strictly_increasing_per_ticker():
    ctx = {"X": {"close": np.arange(5.0), "volume": np.arange(5.0)}}
    df = make_long_context_df(ctx)
    ts = df.sort_values("timestamp")["timestamp"].values
    assert (np.diff(ts.astype("int64")) > 0).all()


def test_make_long_context_df_empty():
    df = make_long_context_df({})
    assert len(df) == 0
    assert set(df.columns) == {"id", "timestamp", TARGET_CLOSE, TARGET_VOLUME}


def test_make_long_context_df_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        make_long_context_df({"X": {"close": np.array([1.0, 2.0]), "volume": np.array([1.0])}})


# ---- extract_predicted_close / predicted_log_returns ----

def _fake_pred_df():
    return pd.DataFrame({
        "id": ["A", "A", "B", "B"],
        "target_name": ["close", "volume_log1p", "close", "volume_log1p"],
        "predictions": [110.0, 3.1, 48.0, 2.0],
    })


def test_extract_predicted_close_only_close_rows():
    out = extract_predicted_close(_fake_pred_df())
    assert out == {"A": 110.0, "B": 48.0}


def test_predicted_log_returns_matches_manual_log():
    pred_close = {"A": 110.0, "B": 48.0}
    last_close = {"A": 100.0, "B": 50.0}
    out = predicted_log_returns(pred_close, last_close)
    assert out["A"] == pytest.approx(np.log(1.1))
    assert out["B"] == pytest.approx(np.log(48.0 / 50.0))


def test_predicted_log_returns_drops_tickers_missing_from_either_side():
    out = predicted_log_returns({"A": 110.0, "C": 10.0}, {"A": 100.0, "B": 50.0})
    assert set(out) == {"A"}


def test_predicted_log_returns_floors_nonpositive_price():
    out = predicted_log_returns({"A": -5.0}, {"A": 100.0})
    assert np.isfinite(out["A"])
    out2 = predicted_log_returns({"A": 100.0}, {"A": 0.0})
    assert np.isfinite(out2["A"])


def test_score_day_end_to_end():
    pred_df = pd.DataFrame({
        "id": ["A", "A"], "target_name": ["close", "volume_log1p"], "predictions": [105.0, 1.0]})
    out = score_day(pred_df, {"A": 100.0})
    assert out == {"A": pytest.approx(np.log(1.05))}


# ---- select_strided_dates ----

def test_select_strided_dates_stride_1_keeps_all():
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert select_strided_dates(dates, 1) == dates


def test_select_strided_dates_stride_keeps_first_and_spacing():
    dates = [f"d{i:02d}" for i in range(10)]
    out = select_strided_dates(dates, 3)
    assert out == ["d00", "d03", "d06", "d09"]
    assert out[0] == dates[0]


def test_select_strided_dates_invalid_stride_raises():
    with pytest.raises(ValueError):
        select_strided_dates(["a"], 0)


# ---- context_window ----

def test_context_window_ends_at_asof_inclusive():
    dates = np.array(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    out = context_window(dates, vals, "2024-01-03", context_len=10)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_context_window_truncates_to_context_len():
    dates = np.array([f"2024-01-{d:02d}" for d in range(1, 21)])
    vals = np.arange(20.0)
    out = context_window(dates, vals, "2024-01-20", context_len=5)
    np.testing.assert_allclose(out, [15.0, 16.0, 17.0, 18.0, 19.0])


def test_context_window_missing_asof_returns_none():
    dates = np.array(["2024-01-01", "2024-01-02"])
    vals = np.array([1.0, 2.0])
    assert context_window(dates, vals, "2024-01-05", context_len=10) is None


def test_context_window_too_few_rows_returns_none():
    dates = np.array(["2024-01-01", "2024-01-02"])
    vals = np.array([1.0, 2.0])
    assert MIN_SERIES_LEN == 3
    assert context_window(dates, vals, "2024-01-02", context_len=10) is None


# ---- build_day_context ----

def _price_history():
    dates = np.array([f"2024-01-{d:02d}" for d in range(1, 11)])
    return {
        "A": {"dates": dates, "close": np.linspace(100, 109, 10), "volume": np.linspace(1000, 1900, 10)},
        "B": {"dates": dates[:2], "close": np.array([5.0, 6.0]), "volume": np.array([1.0, 2.0])},  # too short
        "C": {"dates": dates, "close": np.linspace(10, 19, 10), "volume": np.linspace(100, 190, 10)},
    }


def test_build_day_context_drops_tickers_with_insufficient_history():
    ph = _price_history()
    context, last_close = build_day_context(ph, ["A", "B", "C", "D"], "2024-01-05", context_len=512)
    assert set(context) == {"A", "C"}   # B too short, D not in price_history
    assert set(last_close) == {"A", "C"}
    assert last_close["A"] == ph["A"]["close"][4]


def test_build_day_context_window_matches_context_window():
    ph = _price_history()
    context, last_close = build_day_context(ph, ["A"], "2024-01-10", context_len=3)
    np.testing.assert_allclose(context["A"]["close"], ph["A"]["close"][-3:])
    np.testing.assert_allclose(context["A"]["volume"], ph["A"]["volume"][-3:])
    assert last_close["A"] == ph["A"]["close"][-1]
