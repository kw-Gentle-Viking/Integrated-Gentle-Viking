import pytest
from training.label import compute_next_day_return, derive_threshold, assign_label, build_label_rows


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
