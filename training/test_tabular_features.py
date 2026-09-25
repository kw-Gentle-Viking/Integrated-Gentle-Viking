import numpy as np
import pandas as pd
import pytest

from training.tabular_features import (
    add_lags, add_cs_ranks, make_z_target, Preprocessor, per_ticker_ts_metrics)


def _df():
    return pd.DataFrame({
        "ticker": ["A", "A", "A", "B", "B", "B"],
        "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
        "x": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
    })


def test_add_lags_is_per_ticker_and_strictly_past():
    # deliberately unsorted input: function must sort by (ticker, date) itself
    df = _df().sample(frac=1, random_state=0)
    out = add_lags(df, "x", [1, 2])
    a = out[out.ticker == "A"].sort_values("trade_date")
    b = out[out.ticker == "B"].sort_values("trade_date")
    assert np.isnan(a["x_lag1"].iloc[0]) and a["x_lag1"].iloc[1] == 1.0 and a["x_lag1"].iloc[2] == 2.0
    assert np.isnan(a["x_lag2"].iloc[1]) and a["x_lag2"].iloc[2] == 1.0
    # no leakage across tickers: first B row must be NaN, not A's last value
    assert np.isnan(b["x_lag1"].iloc[0]) and b["x_lag1"].iloc[1] == 10.0


def test_add_lags_does_not_mutate_input():
    df = _df()
    add_lags(df, "x", [1])
    assert "x_lag1" not in df.columns


def test_add_cs_ranks_per_date_percentile():
    df = pd.DataFrame({
        "ticker": list("ABC") * 2,
        "trade_date": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3),
        "x": [3.0, 1.0, 2.0, 5.0, 5.0, 9.0],
    })
    out = add_cs_ranks(df, ["x"])
    d1 = out[out.trade_date == "2024-01-01"].set_index("ticker")["x_csr"]
    assert d1["B"] < d1["C"] < d1["A"]
    assert d1.max() == pytest.approx(1.0) and d1.min() == pytest.approx(1 / 3)
    d2 = out[out.trade_date == "2024-01-02"].set_index("ticker")["x_csr"]
    assert d2["A"] == d2["B"] < d2["C"]  # ties share rank


def test_add_cs_ranks_nan_stays_nan_and_ignored():
    df = pd.DataFrame({"ticker": list("ABC"), "trade_date": pd.to_datetime(["2024-01-01"] * 3),
                       "x": [1.0, np.nan, 3.0]})
    out = add_cs_ranks(df, ["x"])
    assert np.isnan(out["x_csr"].iloc[1])
    assert out["x_csr"].iloc[0] < out["x_csr"].iloc[2]


def test_make_z_target_floor_and_clip():
    ret = np.array([0.02, 0.5, -0.5, 0.01, np.nan])
    vol = np.array([0.02, 0.01, 0.01, 0.0, 0.01])
    z = make_z_target(ret, vol, floor=0.005, clip=5.0)
    assert z[0] == pytest.approx(1.0)
    assert z[1] == 5.0 and z[2] == -5.0        # clipped
    assert z[3] == pytest.approx(2.0)           # vol 0 -> floor
    assert np.isnan(z[4])


def test_make_z_target_nan_vol_gives_nan():
    z = make_z_target(np.array([0.01]), np.array([np.nan]), floor=0.005)
    assert np.isnan(z[0])


def test_preprocessor_fit_on_train_only_winsorises_and_fills():
    rng = np.random.default_rng(0)
    tr = rng.standard_normal((1000, 2))
    tr[0, 0] = 1000.0
    tr[1, 1] = np.nan
    pp = Preprocessor(lo=0.5, hi=99.5).fit(tr)
    te = np.array([[1e6, np.nan], [-1e6, 0.0]])
    out = pp.transform(te)
    assert np.isfinite(out).all()
    assert out[0, 0] < 10 and out[1, 0] > -10          # winsorised to train percentiles
    assert out[0, 1] == pytest.approx((np.nanmedian(tr[:, 1]) - pp.mean_[1]) / pp.std_[1], abs=1e-6)
    # train mean ~0 after standardisation (stats from the train fit)
    assert abs(pp.transform(tr)[:, 1].mean()) < 0.05


def test_preprocessor_constant_column_no_div_zero():
    pp = Preprocessor().fit(np.ones((10, 1)))
    assert np.isfinite(pp.transform(np.ones((3, 1)))).all()


def test_per_ticker_ts_metrics():
    n = 50
    ret = np.linspace(-1, 1, n)
    m = per_ticker_ts_metrics(ret * 2, ret)
    assert m["spearman"] == pytest.approx(1.0) and m["hit_rate"] == 1.0 and m["n"] == n
    m = per_ticker_ts_metrics(-ret, ret)
    assert m["spearman"] == pytest.approx(-1.0)
    m = per_ticker_ts_metrics(np.array([np.nan, 1.0]), np.array([1.0, 1.0]))
    assert m["n"] == 1 and m["spearman"] is None
