import numpy as np
import pandas as pd
from features.clip_scale import fit_clip_bounds, apply_clip, fit_scaler


def test_fit_clip_bounds_uses_quantiles_not_hardcoded_values():
    train_values = np.array([-100.0] + list(np.linspace(-1, 1, 998)) + [100.0])
    lower, upper = fit_clip_bounds(train_values, lower_q=0.005, upper_q=0.995)
    assert lower > -100.0 and upper < 100.0
    assert lower < 0 < upper


def test_apply_clip_bounds_values():
    result = apply_clip(np.array([-10.0, -0.5, 0.0, 0.5, 10.0]), bounds=(-1.0, 1.0))
    np.testing.assert_array_equal(result, [-1.0, -0.5, 0.0, 0.5, 1.0])


def test_fit_scaler_only_uses_provided_columns():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0], "c": ["x", "y", "z"]})
    scaler = fit_scaler(df, columns=["a", "b"])
    assert list(scaler.feature_names_in_) == ["a", "b"]
    assert scaler.mean_[0] == 2.0
