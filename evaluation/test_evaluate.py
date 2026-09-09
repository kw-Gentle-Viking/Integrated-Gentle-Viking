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
