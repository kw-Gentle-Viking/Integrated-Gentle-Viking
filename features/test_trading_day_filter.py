import pandas as pd
import pytest

from features.build_features import closed_market_days, drop_closed_days


def _cal(rows):
    return pd.DataFrame(rows, columns=["trade_date", "is_market_open"])


def test_closed_days_come_from_calendar():
    cal = _cal([(pd.Timestamp("2026-09-23"), 1), (pd.Timestamp("2026-09-24"), 0), (pd.Timestamp("2026-09-25"), 0)])
    assert closed_market_days(cal) == {pd.Timestamp("2026-09-24"), pd.Timestamp("2026-09-25")}


def test_empty_calendar_refuses_to_build():
    with pytest.raises(RuntimeError):
        closed_market_days(_cal([]))


def test_holiday_rows_removed_so_row_shift_is_trading_day_shift():
    closed = {pd.Timestamp("2026-09-24")}
    px = pd.DataFrame({"ticker": ["A"] * 4,
                       "trade_date": pd.to_datetime(["2026-09-23", "2026-09-24", "2026-09-25", "2026-09-28"]),
                       "close_price": [100.0, 999.0, 999.0, 110.0]})
    kept = drop_closed_days(px, closed)
    # 9/24 가짜 행이 있으면 9/23의 익일 수익률이 999 기준으로 계산됨. 제거하면 9/23 -> 9/25(영업일 다음) 기준
    kept = kept.assign(next_close=kept["close_price"].shift(-1))
    assert kept.iloc[0]["next_close"] == 999.0 and len(kept) == 3
    assert 999.0 not in kept[kept["trade_date"] == pd.Timestamp("2026-09-24")]["close_price"].tolist()


def test_lookback_counts_trading_days_not_calendar_days():
    from features.build_features import lookback_start
    # 9/24~25 가 휴장이면 9/28 직전 2영업일은 9/23, 9/22 (달력 기준이면 9/26, 9/25)
    open_days = [pd.Timestamp(d) for d in ["2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-28"]]
    assert lookback_start(open_days, pd.Timestamp("2026-09-28"), n=2) == pd.Timestamp("2026-09-22")


def test_lookback_fails_loudly_without_enough_history():
    from features.build_features import lookback_start
    with pytest.raises(RuntimeError):
        lookback_start([pd.Timestamp("2026-09-23")], pd.Timestamp("2026-09-28"), n=5)
