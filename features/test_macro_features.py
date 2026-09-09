import pandas as pd
from features.macro_features import compute_return, compute_change, compute_rate_spread


def test_compute_return_is_pct_change():
    s = pd.Series([100.0, 110.0, 99.0])
    result = compute_return(s)
    assert result.iloc[0] != result.iloc[0]  # NaN for first row
    assert abs(result.iloc[1] - 0.10) < 1e-9
    assert abs(result.iloc[2] - (-0.1)) < 1e-9


def test_compute_change_is_absolute_diff():
    s = pd.Series([20.0, 22.5, 21.0])
    result = compute_change(s)
    assert result.iloc[0] != result.iloc[0]  # NaN for first row
    assert abs(result.iloc[1] - 2.5) < 1e-9
    assert abs(result.iloc[2] - (-1.5)) < 1e-9


def test_compute_rate_spread_is_same_day_level_diff():
    a = pd.Series([4.5, 4.6])
    b = pd.Series([3.5, 3.5])
    result = compute_rate_spread(a, b)
    assert abs(result.iloc[0] - 1.0) < 1e-9
    assert abs(result.iloc[1] - 1.1) < 1e-9
