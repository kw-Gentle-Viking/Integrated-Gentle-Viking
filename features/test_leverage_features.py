import pytest
from features.leverage_features import compute_rebalancing_flow, aggregate_leverage_signals


def test_rebalancing_flow_positive_when_leveraged_fund_tracks_rising_stock():
    # 배율 2x, 전일 AUM 1000억, 기초자산 당일 +2% → 추가 매수 필요(양수)
    flow = compute_rebalancing_flow(prev_aum=100_000_000_000, underlying_return=0.02, multiple=2.0)
    assert flow == pytest.approx(100_000_000_000 * 0.02 * (2.0 - 1))


def test_rebalancing_flow_for_inverse_fund_on_down_day():
    # 배율 -2x, 기초자산 당일 -3% → (배율-1) = -3, return -0.03 → flow 양수(추가 공매도 유사 매도 압력)
    flow = compute_rebalancing_flow(prev_aum=50_000_000_000, underlying_return=-0.03, multiple=-2.0)
    assert flow == pytest.approx(50_000_000_000 * -0.03 * (-2.0 - 1))
    assert flow > 0


def test_aggregate_sums_across_all_products_for_one_underlying():
    product_rows = [
        {"code": "A", "multiple": 2.0, "close_price": 10000, "volume": 100_000, "aum": 200_000_000_000},
        {"code": "B", "multiple": -2.0, "close_price": 5000, "volume": 50_000, "aum": 50_000_000_000},
    ]
    result = aggregate_leverage_signals(product_rows, underlying_return=0.01,
                                          underlying_market_cap=400_000_000_000_000)
    assert result["lev_total_volume"] == pytest.approx(10000 * 100_000 + 5000 * 50_000)
    assert result["lev_total_aum"] == pytest.approx(250_000_000_000)
    assert result["lev_aum_to_mktcap"] == pytest.approx(250_000_000_000 / 400_000_000_000_000)
    expected_flow = (200_000_000_000 * 0.01 * (2.0 - 1)) + (50_000_000_000 * 0.01 * (-2.0 - 1))
    assert result["est_rebalancing_flow"] == pytest.approx(expected_flow)


def test_aggregate_returns_zeros_when_no_products_yet_listed():
    result = aggregate_leverage_signals([], underlying_return=0.01, underlying_market_cap=1e14)
    assert result == {"lev_total_volume": 0.0, "lev_total_aum": 0.0,
                        "lev_aum_to_mktcap": 0.0, "est_rebalancing_flow": 0.0}
