import pytest
from evaluation.evaluate import compute_metrics, split_by_regime


def test_compute_metrics_perfect_prediction():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["mcc"] == pytest.approx(1.0)
    assert m["per_class"][0]["f1"] == 1.0
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_compute_metrics_detects_class_collapse():
    # 관망(1)과 매도(2) 클래스를 전혀 못 맞추고 항상 매수(0)만 예측하는 경우
    # — macro_f1이 accuracy보다 뚜렷하게 낮아야 함
    y_true = [0, 0, 1, 1, 1, 2]
    y_pred = [0, 0, 0, 0, 0, 0]
    m = compute_metrics(y_true, y_pred)
    assert m["per_class"][1]["recall"] == 0.0
    assert m["macro_f1"] < m["accuracy"]


def test_split_by_regime_separates_pre_and_post_leverage():
    dates = ["2025-01-01", "2025-06-01", "2026-06-01", "2026-07-01"]
    y_true = [0, 1, 2, 0]
    y_pred = [0, 1, 2, 2]
    result = split_by_regime(dates, y_true, y_pred, leverage_start="2026-05-27")
    assert result["pre_leverage"]["accuracy"] == 1.0
    assert result["leverage_era"]["accuracy"] == 0.5


def test_compute_metrics_empty_input_does_not_crash():
    m = compute_metrics([], [])
    assert m["accuracy"] == 0.0
    assert m["macro_f1"] == 0.0
    assert m["mcc"] == 0.0
    assert m["confusion_matrix"] == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


def test_split_by_regime_handles_one_sided_regime():
    # 전부 leverage_era 이후인 경우 — pre_leverage가 빈 서브셋이 되어도 크래시하면 안 됨
    dates = ["2026-06-01", "2026-06-02"]
    y_true = [0, 1]
    y_pred = [0, 1]
    result = split_by_regime(dates, y_true, y_pred, leverage_start="2026-05-27")
    assert result["pre_leverage"]["accuracy"] == 0.0
    assert result["leverage_era"]["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# compute_signal_metrics (label-agnostic signal metrics, plan S3)
# ---------------------------------------------------------------------------
import math
import numpy as np
from evaluation.evaluate import compute_signal_metrics


def _probs_from_score(scores):
    """Build [N,3] probs whose (p_buy - p_sell) equals `scores` (all in [-1, 1]), p_hold = 0."""
    s = np.asarray(scores, dtype=float)
    return np.stack([(1 + s) / 2, np.zeros_like(s), (1 - s) / 2], axis=1)


def test_signal_metrics_perfect_rank_ic_single_day():
    dates = ["2024-01-02"] * 4
    rets = [0.04, 0.01, -0.01, -0.04]
    probs = _probs_from_score([0.9, 0.3, -0.3, -0.9])
    m = compute_signal_metrics(dates, rets, probs, min_names_per_day=4)
    assert m["n_days_used"] == 1
    assert m["mean_daily_rank_ic"] == pytest.approx(1.0)
    assert m["ic_std"] is None and m["ic_ir"] is None  # one day: sample std undefined


def test_signal_metrics_ic_mean_std_ir_two_days():
    dates = ["d1"] * 4 + ["d2"] * 4
    rets = [4, 3, 2, 1] + [4, 3, 2, 1]
    probs = _probs_from_score([0.8, 0.4, -0.4, -0.8] + [-0.8, -0.4, 0.4, 0.8])  # IC=+1, IC=-1
    m = compute_signal_metrics(dates, rets, probs, min_names_per_day=4)
    assert m["n_days_used"] == 2
    assert m["mean_daily_rank_ic"] == pytest.approx(0.0)
    assert m["ic_std"] == pytest.approx(math.sqrt(2))  # ddof=1
    assert m["ic_ir"] == pytest.approx(0.0)


def test_signal_metrics_spearman_hand_computed():
    # score ranks 1,2,3,4 ; return ranks 2,1,4,3 -> d=[-1,1,-1,1], sum d^2 = 4
    # rho = 1 - 6*4/(4*15) = 0.6
    dates = ["d"] * 4
    probs = _probs_from_score([-0.6, -0.2, 0.2, 0.6])
    rets = [0.02, 0.01, 0.04, 0.03]
    m = compute_signal_metrics(dates, rets, probs, min_names_per_day=4)
    assert m["mean_daily_rank_ic"] == pytest.approx(0.6)


def test_signal_metrics_days_below_min_names_are_excluded():
    dates = ["d1"] * 4 + ["d2"] * 2
    rets = [4, 3, 2, 1, 1, 2]
    probs = _probs_from_score([0.8, 0.4, -0.4, -0.8, 0.5, -0.5])
    m = compute_signal_metrics(dates, rets, probs, min_names_per_day=3)
    assert m["n_days_used"] == 1
    assert m["mean_daily_rank_ic"] == pytest.approx(1.0)


def test_signal_metrics_argmax_long_short_hand_computed():
    dates = ["d"] * 5
    rets = [0.03, 0.01, 0.00, -0.02, -0.04]
    # argmax: buy, buy, hold, sell, sell
    probs = np.array([[.7, .2, .1], [.5, .3, .2], [.2, .6, .2], [.1, .3, .6], [.1, .1, .8]])
    m = compute_signal_metrics(dates, rets, probs, min_names_per_day=2)
    ls = m["argmax_long_short"]
    assert ls["n_buy"] == 2 and ls["n_hold"] == 1 and ls["n_sell"] == 2
    assert ls["mean_ret_buy"] == pytest.approx(0.02)
    assert ls["mean_ret_hold"] == pytest.approx(0.0)
    assert ls["mean_ret_sell"] == pytest.approx(-0.03)
    assert ls["spread"] == pytest.approx(0.05)


def test_signal_metrics_argmax_spread_none_when_a_group_is_empty():
    dates = ["d"] * 3
    probs = np.array([[.8, .1, .1]] * 3)  # everything predicted buy
    m = compute_signal_metrics(dates, [0.01, 0.02, 0.03], probs, min_names_per_day=2)
    ls = m["argmax_long_short"]
    assert ls["n_buy"] == 3 and ls["n_sell"] == 0
    assert ls["mean_ret_sell"] is None and ls["spread"] is None


def test_signal_metrics_quantile_long_short_hand_computed():
    # day1: 5 names, q=0.4 -> k=2 ; top2 by score = rets 0.05,0.03 (avg .04); bottom2 = -0.01,-0.03 (avg -.02) -> .06
    # day2: same scores, returns reversed -> -.06 ; mean over days = 0
    d1 = [0.05, 0.03, 0.00, -0.01, -0.03]
    d2 = list(reversed(d1))
    dates = ["d1"] * 5 + ["d2"] * 5
    probs = _probs_from_score([0.9, 0.5, 0.0, -0.5, -0.9] * 2)
    m = compute_signal_metrics(dates, d1 + d2, probs, min_names_per_day=5, quantile=0.4)
    q = m["quantile_long_short"]
    assert q["n_days"] == 2
    assert q["mean_spread"] == pytest.approx(0.0)
    m1 = compute_signal_metrics(dates[:5], d1, probs[:5], min_names_per_day=5, quantile=0.4)
    assert m1["quantile_long_short"]["mean_spread"] == pytest.approx(0.06)


def test_signal_metrics_empty_input_does_not_crash():
    m = compute_signal_metrics([], [], np.zeros((0, 3)))
    assert m["n_days_used"] == 0
    assert m["mean_daily_rank_ic"] is None
    assert m["argmax_long_short"]["spread"] is None
    assert m["quantile_long_short"]["mean_spread"] is None


def test_signal_metrics_constant_scores_do_not_crash_and_are_skipped():
    dates = ["d"] * 4
    probs = np.array([[.4, .2, .4]] * 4)  # p_buy - p_sell == 0 for all
    m = compute_signal_metrics(dates, [0.01, 0.02, 0.03, 0.04], probs, min_names_per_day=4)
    assert m["n_days_used"] == 0
    assert m["mean_daily_rank_ic"] is None
    assert m["quantile_long_short"]["mean_spread"] is None


def test_signal_metrics_nan_returns_are_dropped():
    dates = ["d"] * 5
    rets = [0.04, float("nan"), 0.01, -0.01, -0.04]
    probs = _probs_from_score([0.9, 0.5, 0.3, -0.3, -0.9])
    m = compute_signal_metrics(dates, rets, probs, min_names_per_day=4)
    assert m["n_days_used"] == 1
    assert m["mean_daily_rank_ic"] == pytest.approx(1.0)
    assert m["n_samples"] == 4


def test_signal_metrics_ties_in_returns_use_average_ranks():
    # returns tie -> average ranks; scores strictly increasing: ranks 1..4 vs return ranks [1, 2.5, 2.5, 4]
    # pearson of [1,2,3,4] and [1,2.5,2.5,4] = 0.9486832980505138
    dates = ["d"] * 4
    probs = _probs_from_score([-0.6, -0.2, 0.2, 0.6])
    m = compute_signal_metrics(dates, [0.0, 0.01, 0.01, 0.02], probs, min_names_per_day=4)
    assert m["mean_daily_rank_ic"] == pytest.approx(0.9486832980505138)
