import numpy as np
import pytest
from training.label import (compute_next_day_return, derive_threshold, assign_label, build_label_rows,
                            derive_volnorm_params, assign_label_volnorm)


def test_compute_next_day_return_shifts_forward_one_day():
    closes = [100.0, 110.0, 99.0, None]
    returns = compute_next_day_return(closes)
    assert returns[0] == pytest.approx(0.10)
    assert returns[1] == pytest.approx((99.0 - 110.0) / 110.0)
    assert returns[2] is None  # 마지막 날은 다음날 데이터 없음
    assert returns[3] is None


def test_derive_threshold_targets_hold_ratio():
    # -0.02~0.02 안에 정확히 50%가 들어가도록 구성
    returns = [-0.10, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10, -0.5]
    threshold = derive_threshold(returns, target_hold_ratio=0.5)
    hold_count = sum(1 for r in returns if abs(r) < threshold)
    assert hold_count / len(returns) == pytest.approx(0.5, abs=0.15)


def test_assign_label_buy_hold_sell():
    labels = assign_label([0.02, 0.001, -0.02, None], threshold=0.015)
    assert labels == [0, 1, 2, None]


def test_build_label_rows_pairs_ticker_dates_with_labels():
    dates = ["2019-01-02", "2019-01-03", "2019-01-04"]
    closes = [100.0, 110.0, 99.0]
    rows = build_label_rows(ticker="005930", trade_dates=dates, close_prices=closes, threshold=0.015)
    assert rows[0] == {"ticker": "005930", "trade_date": "2019-01-02",
                         "next_day_return": pytest.approx(0.10), "label": 0}
    assert rows[2] == {"ticker": "005930", "trade_date": "2019-01-04",
                         "next_day_return": None, "label": None}


# ---- S1: volatility-normalised label -------------------------------------------------

def test_derive_volnorm_params_floor_is_vol_quantile_and_k_is_abs_z_quantile():
    vols = [0.01, 0.02, 0.03, 0.04, 0.05]
    returns = [0.01, -0.04, 0.03, 0.0, -0.10]
    p = derive_volnorm_params(returns, vols, target_hold_ratio=0.5, floor_quantile=0.0)
    assert p["floor"] == pytest.approx(0.01)  # 0-quantile = min vol
    z_abs = [abs(r) / max(v, 0.01) for r, v in zip(returns, vols)]
    assert p["k"] == pytest.approx(float(np.quantile(z_abs, 0.5)))


def test_derive_volnorm_params_skips_none_and_nan_pairs():
    vols = [0.01, None, float("nan"), 0.02, 0.03]
    returns = [0.01, 0.5, 0.5, None, -0.03]
    p = derive_volnorm_params(returns, vols, floor_quantile=0.0)
    assert p["floor"] == pytest.approx(0.01)  # only rows 0 and 4 are complete
    assert p["k"] == pytest.approx(float(np.quantile([1.0, 1.0], 0.5)))


def test_derive_volnorm_params_tiny_vol_uses_floor_when_deriving_k():
    # 1st-percentile floor of these vols is well above 1e-9, so |z| for the tiny-vol row
    # must be divided by the floor, not by 1e-9.
    vols = [1e-9] + [0.02] * 99
    returns = [0.01] + [0.02] * 99
    p = derive_volnorm_params(returns, vols, floor_quantile=0.05)
    assert p["floor"] > 1e-9
    assert np.isfinite(p["k"]) and p["k"] < 10


def test_derive_volnorm_params_empty_raises():
    with pytest.raises(ValueError):
        derive_volnorm_params([None], [None])


def test_assign_label_volnorm_buy_hold_sell_and_boundaries():
    # vol 0.01, k=0.5 -> band is +-0.005 (inclusive at the edges, like assign_label)
    labels = assign_label_volnorm([0.02, 0.005, 0.001, -0.005, -0.02],
                                  [0.01] * 5, k=0.5, floor=0.001)
    assert labels == [0, 0, 1, 2, 2]


def test_assign_label_volnorm_zero_or_tiny_vol_uses_floor():
    # vol 0 -> divisor is floor 0.01; ret 0.004 -> z=0.4 < k=0.5 -> hold (no div-by-zero)
    labels = assign_label_volnorm([0.004, 0.006, -0.006], [0.0, 1e-12, 0.0], k=0.5, floor=0.01)
    assert labels == [1, 0, 2]


def test_assign_label_volnorm_vol_above_floor_is_used_as_is():
    # same ret, larger vol -> smaller z -> hold
    labels = assign_label_volnorm([0.02, 0.02], [0.01, 0.10], k=0.5, floor=0.001)
    assert labels == [0, 1]


def test_assign_label_volnorm_none_and_nan_inputs_give_none():
    nan = float("nan")
    labels = assign_label_volnorm([None, 0.02, nan, 0.02], [0.01, None, 0.01, nan],
                                  k=0.5, floor=0.001)
    assert labels == [None, None, None, None]


def test_volnorm_roundtrip_hold_share_close_to_target_on_train():
    rng = np.random.default_rng(0)
    vols = list(rng.uniform(0.005, 0.05, 5000))
    returns = [float(rng.normal(0, v)) for v in vols]
    p = derive_volnorm_params(returns, vols, target_hold_ratio=0.5)
    labels = assign_label_volnorm(returns, vols, p["k"], p["floor"])
    assert labels.count(1) / len(labels) == pytest.approx(0.5, abs=0.01)
