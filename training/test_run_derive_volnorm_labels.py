import numpy as np
import pandas as pd
import pytest

from training.run_derive_volnorm_labels import compute_volnorm_labels


def _df():
    return pd.DataFrame({
        "trade_date": pd.to_datetime(["2019-01-02", "2019-01-03", "2019-01-04", "2024-01-02", "2024-01-03"]),
        "next_day_return": [0.02, 0.001, -0.02, 0.5, None],
        "volatility_20d": [0.01, 0.01, 0.01, 0.01, 0.01],
    })


def test_params_use_train_rows_only_not_the_huge_post_train_return():
    params, _ = compute_volnorm_labels(_df(), "2019-01-02", "2019-01-04", floor_quantile=0.0)
    # train |z| = [2, 0.1, 2] -> median 2; the 2024 row (|z|=50) must not influence it
    assert params["k"] == pytest.approx(2.0)
    assert params["floor"] == pytest.approx(0.01)


def test_labels_cover_all_rows_and_none_for_missing_return():
    _, labels = compute_volnorm_labels(_df(), "2019-01-02", "2019-01-04", floor_quantile=0.0)
    assert labels == [0, 1, 2, 0, None]


def test_nan_vol_gives_none_label():
    df = _df()
    df.loc[1, "volatility_20d"] = np.nan
    _, labels = compute_volnorm_labels(df, "2019-01-02", "2019-01-04", floor_quantile=0.0)
    assert labels[1] is None
