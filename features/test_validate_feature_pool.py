import pandas as pd
import pytest
from features.validate_feature_pool import (
    check_market_cap_consistency, check_value_ranges, check_null_rates, check_trading_day_gaps,
)


def test_market_cap_cross_check_flags_large_discrepancy():
    # ticker_universe(price*shares)와 daily_valuation(hts_avls*1e6)이 독립적으로 계산된 시총 —
    # 단위 버그가 있으면 이 둘이 자릿수 단위로 어긋난다
    df = pd.DataFrame({
        "ticker": ["005930", "000660"],
        "universe_market_cap": [4.5e14, 9.0e13],
        "valuation_market_cap": [4.51e14, 9.0e7],  # 000660은 1e6배 축소된 버그 상황 가정
    })
    issues = check_market_cap_consistency(df, tolerance=0.1)
    assert issues == ["000660"]


def test_market_cap_cross_check_passes_when_close():
    df = pd.DataFrame({"ticker": ["005930"], "universe_market_cap": [4.5e14],
                         "valuation_market_cap": [4.52e14]})
    assert check_market_cap_consistency(df, tolerance=0.1) == []


def test_check_value_ranges_flags_negative_price_or_volume():
    df = pd.DataFrame({"close_price": [50000.0, -100.0], "volume": [1000, -5]})
    issues = check_value_ranges(df)
    assert "close_price" in issues
    assert "volume" in issues


def test_check_null_rates_flags_columns_over_threshold():
    df = pd.DataFrame({"per": [None, None, None, 1.0], "close_price": [1, 2, 3, 4]})
    issues = check_null_rates(df, max_null_ratio=0.5)
    assert issues == {"per": pytest.approx(0.75)}


def test_check_trading_day_gaps_flags_ticker_with_missing_days():
    df = pd.DataFrame({
        "ticker": ["005930"] * 3 + ["000660"] * 5,
        "trade_date": pd.to_datetime(["2019-01-02", "2019-01-03", "2019-01-04"] +
                                       list(pd.bdate_range("2019-01-02", periods=5))),
    })
    # 삼성전자는 3일치뿐인데 SK하이닉스는 5일치 — 같은 기간 대비 종목별 행 수 편차 검출
    issues = check_trading_day_gaps(df, expected_min_days=5)
    assert issues == ["005930"]
