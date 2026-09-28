import math
from datetime import date, datetime

import pytest

from serving.feature_builder import (
    FFILL_COLS,
    LIVE_COLS,
    ZERO_DEFAULT_COLS,
    build_encoder_df,
    compute_time_progress,
)

HISTORICAL_COLS = [
    "log_ret", "disparity_5d", "disparity_20d", "disparity_60d", "rsi_14", "volatility_20d",
    "sector_ret_1d", "sector_ret_5d", "sector_ret_20d", "sector_ma_ratio_20d",
    "sector_volatility", "sector_volume_ratio",
    "is_dividend", "is_bonus_issue", "is_rights_offering", "is_split",
    "day_of_week",
    "lev_total_aum", "lev_aum_to_mktcap", "est_rebalancing_flow",
    "is_vi_triggered", "vi_count_recent5d",
    "kospi_ret", "kosdaq_ret", "snp500_ret", "nasdaq_ret", "phlx_semi_ret",
    "vix_chg", "usd_krw_chg", "us_10y_yield_chg", "rate_spread_us_kr",
    "wti_ret", "gold_ret",
]
FUTURE_COLS = ["time_progress", "is_bok", "is_fomc", "is_witching_kr", "is_witching_us"]
STATIC_COLS = ["sector_id", "market_id"]


def _make_history_rows(n=59, start_close=100.0):
    """Synthetic 59-day history mimicking a real feature_pool query result: one dict per row,
    with trade_date/close_price plus every historical/future(non-time_progress)/static column."""
    rows = []
    close = start_close
    for i in range(n):
        close *= 1.001  # gentle uptrend so disparity/log_ret have non-trivial signs
        row = {
            "trade_date": date(2026, 1, 1).toordinal(),  # placeholder, overwritten below
            "close_price": close,
            "log_ret": 0.001,
            "sector_ret_1d": 0.01, "sector_ret_5d": 0.02, "sector_ret_20d": 0.03,
            "sector_ma_ratio_20d": 1.01, "sector_volatility": 0.02, "sector_volume_ratio": 1.1,
            "is_dividend": 0, "is_bonus_issue": 0, "is_rights_offering": 0, "is_split": 0,
            "day_of_week": i % 5,
            "lev_total_aum": 1000.0, "lev_aum_to_mktcap": 0.05, "est_rebalancing_flow": 10.0,
            "is_vi_triggered": 0, "vi_count_recent5d": 1.0,
            "kospi_ret": 0.005, "kosdaq_ret": 0.006, "snp500_ret": 0.004, "nasdaq_ret": 0.007,
            "phlx_semi_ret": 0.008, "vix_chg": -0.01, "usd_krw_chg": 0.002,
            "us_10y_yield_chg": 0.001, "rate_spread_us_kr": 1.5, "wti_ret": 0.01, "gold_ret": 0.003,
            "disparity_5d": 0.0, "disparity_20d": 0.0, "disparity_60d": 0.0, "rsi_14": 55.0,
            "volatility_20d": 0.02,
            "is_bok": 0, "is_fomc": 0, "is_witching_kr": 0, "is_witching_us": 0,
            "sector_id": 3, "market_id": 1,
        }
        # real trade_date, oldest to newest, ending 2026-09-08 (last real trading day before "today")
        from datetime import timedelta
        row["trade_date"] = date(2026, 9, 8) - timedelta(days=(n - 1 - i))
        rows.append(row)
    return rows


def test_build_encoder_df_has_60_rows_and_all_columns():
    history = _make_history_rows(59)
    today_rows = [
        {"datetime": datetime(2026, 9, 9, 9, 0), "open": 105.0, "high": 106.0, "low": 104.5, "close": 105.5, "volume": 1000},
        {"datetime": datetime(2026, 9, 9, 9, 5), "open": 105.5, "high": 106.5, "low": 105.0, "close": 106.0, "volume": 800},
    ]
    now = datetime(2026, 9, 9, 10, 0)  # 1 hour into the session
    df = build_encoder_df("005930", history, today_rows, now, HISTORICAL_COLS, FUTURE_COLS, STATIC_COLS)
    assert len(df) == 60
    for col in HISTORICAL_COLS + FUTURE_COLS + STATIC_COLS:
        assert col in df.columns


def test_live_cols_computed_from_todays_intraday_not_carried_forward():
    history = _make_history_rows(59)
    today_rows = [
        {"datetime": datetime(2026, 9, 9, 9, 0), "open": 105.0, "high": 106.0, "low": 104.5, "close": 105.5, "volume": 1000},
        {"datetime": datetime(2026, 9, 9, 9, 5), "open": 105.5, "high": 106.5, "low": 105.0, "close": 106.0, "volume": 800},
    ]
    now = datetime(2026, 9, 9, 10, 0)
    df = build_encoder_df("005930", history, today_rows, now, HISTORICAL_COLS, FUTURE_COLS, STATIC_COLS)
    today_row = df.iloc[-1]

    yesterday_close = history[-1]["close_price"]
    today_close = 106.0  # latest intraday close
    expected_log_ret = math.log(today_close / yesterday_close)
    assert today_row["log_ret"] == pytest.approx(expected_log_ret)

    # today's row must NOT equal yesterday's carried-forward value for a live-computed column
    assert today_row["log_ret"] != pytest.approx(history[-1]["log_ret"])
    assert today_row["rsi_14"] != history[-1]["rsi_14"]

    # day_of_week comes from today's real date (2026-09-09 is a Wednesday -> weekday()==2)
    assert today_row["day_of_week"] == 2


def test_zero_default_cols_are_zero_for_today_not_ffilled():
    history = _make_history_rows(59)
    # make yesterday's event flags nonzero to prove today does NOT just copy them forward
    history[-1]["is_dividend"] = 1
    history[-1]["is_vi_triggered"] = 1
    today_rows = [{"datetime": datetime(2026, 9, 9, 9, 0), "open": 105.0, "high": 106.0,
                    "low": 104.5, "close": 105.5, "volume": 1000}]
    now = datetime(2026, 9, 9, 9, 10)
    df = build_encoder_df("005930", history, today_rows, now, HISTORICAL_COLS, FUTURE_COLS, STATIC_COLS)
    today_row = df.iloc[-1]
    for col in ZERO_DEFAULT_COLS:
        assert today_row[col] == 0.0, f"{col} should default to 0 for today, got {today_row[col]}"


def test_ffill_cols_carry_forward_most_recent_real_value():
    history = _make_history_rows(59)
    history[-1]["sector_ret_1d"] = 0.0777
    history[-1]["lev_total_aum"] = 12345.0
    history[-1]["vi_count_recent5d"] = 3.0
    today_rows = [{"datetime": datetime(2026, 9, 9, 9, 0), "open": 105.0, "high": 106.0,
                    "low": 104.5, "close": 105.5, "volume": 1000}]
    now = datetime(2026, 9, 9, 9, 10)
    df = build_encoder_df("005930", history, today_rows, now, HISTORICAL_COLS, FUTURE_COLS, STATIC_COLS)
    today_row = df.iloc[-1]
    assert today_row["sector_ret_1d"] == pytest.approx(0.0777)
    assert today_row["lev_total_aum"] == pytest.approx(12345.0)
    assert today_row["vi_count_recent5d"] == pytest.approx(3.0)


def test_open_price_falls_back_to_yesterdays_close_not_zero_when_no_intraday_rows():
    """Regression test for the Step 1-4 review's flagged bug #1: silently defaulting the
    reference open price to 0 when there is no OHLCV column to guess it from. With zero
    intraday rows, feature_builder must fall back to yesterday's real close, not 0."""
    history = _make_history_rows(59)
    now = datetime(2026, 9, 9, 9, 0)
    df = build_encoder_df("005930", history, [], now, HISTORICAL_COLS, FUTURE_COLS, STATIC_COLS)
    today_row = df.iloc[-1]
    # today's close (== open, since no intraday data) should equal yesterday's real close,
    # producing log_ret == 0 (flat carry), NOT some NaN/0-price artifact.
    assert today_row["log_ret"] == pytest.approx(0.0)


def test_static_cols_carried_forward_from_most_recent_real_row():
    history = _make_history_rows(59)
    history[-1]["sector_id"] = 7
    history[-1]["market_id"] = 0
    today_rows = [{"datetime": datetime(2026, 9, 9, 9, 0), "open": 105.0, "high": 106.0,
                    "low": 104.5, "close": 105.5, "volume": 1000}]
    now = datetime(2026, 9, 9, 9, 10)
    df = build_encoder_df("005930", history, today_rows, now, HISTORICAL_COLS, FUTURE_COLS, STATIC_COLS)
    today_row = df.iloc[-1]
    assert today_row["sector_id"] == 7
    assert today_row["market_id"] == 0


def test_time_progress_zero_at_open_one_at_close_clamped():
    d = date(2026, 9, 9)
    from datetime import time
    assert compute_time_progress(datetime.combine(d, time(9, 0))) == pytest.approx(0.0)
    assert compute_time_progress(datetime.combine(d, time(15, 30))) == pytest.approx(1.0)
    assert compute_time_progress(datetime.combine(d, time(8, 0))) == pytest.approx(0.0)  # before open, clamped
    assert compute_time_progress(datetime.combine(d, time(16, 0))) == pytest.approx(1.0)  # after close, clamped
    mid = compute_time_progress(datetime.combine(d, time(12, 15)))
    assert 0.0 < mid < 1.0


def test_historical_rows_use_constant_time_progress_one():
    history = _make_history_rows(59)
    today_rows = [{"datetime": datetime(2026, 9, 9, 9, 0), "open": 105.0, "high": 106.0,
                    "low": 104.5, "close": 105.5, "volume": 1000}]
    now = datetime(2026, 9, 9, 9, 10)
    df = build_encoder_df("005930", history, today_rows, now, HISTORICAL_COLS, FUTURE_COLS, STATIC_COLS)
    # first 59 rows (real trading days) must carry the constant historical time_progress=1.0,
    # matching training.stage1_data.TIME_PROGRESS_CONSTANT
    assert (df["time_progress"].iloc[:59] == 1.0).all()


def test_column_policy_partition_covers_all_historical_cols_with_no_overlap():
    all_classified = set(LIVE_COLS) | set(ZERO_DEFAULT_COLS)
    ffill_expected = [c for c in HISTORICAL_COLS if c not in all_classified]
    assert set(FFILL_COLS) & set(LIVE_COLS) == set()
    assert set(FFILL_COLS) & set(ZERO_DEFAULT_COLS) == set()
    assert set(LIVE_COLS) & set(ZERO_DEFAULT_COLS) == set()
    # every champion historical column must be classified into exactly one bucket
    for col in HISTORICAL_COLS:
        assert col in LIVE_COLS or col in ZERO_DEFAULT_COLS or col in ffill_expected


def test_unclassified_historical_column_raises_instead_of_silently_ffilling():
    # 2026-09-18 review finding: an unreviewed future champion-column-list addition must fail
    # loudly, not silently fall through to ffill via an unconditional "else" branch.
    history = _make_history_rows(59)
    for row in history:
        row["totally_new_unclassified_column"] = 1.0
    with pytest.raises(ValueError, match="not classified"):
        build_encoder_df(
            "005930", history, [], datetime(2026, 9, 9, 9, 10),
            HISTORICAL_COLS + ["totally_new_unclassified_column"], FUTURE_COLS, STATIC_COLS,
        )


def test_build_encoder_df_for_ticker_raises_on_insufficient_history(monkeypatch):
    # 2026-09-18 review finding: a newly-listed ticker with too little real history must fail
    # loudly rather than silently producing a shorter-than-expected encoder_df.
    import serving.feature_builder as fb

    short_history = _make_history_rows(10)  # far short of the requested 59
    monkeypatch.setattr(fb, "fetch_feature_pool_history", lambda *a, **k: short_history)
    monkeypatch.setattr(fb, "fetch_today_intraday_rows", lambda *a, **k: [])
    with pytest.raises(ValueError, match=r"10/59"):
        fb.build_encoder_df_for_ticker(
            "005930", "dummy_v2_dsn", "dummy_prod_dsn",
            HISTORICAL_COLS, FUTURE_COLS, STATIC_COLS,
            now=datetime(2026, 9, 9, 9, 10), n_days=59,
        )


# ---------------------------------------------------------------------------
# time_progress parity with training (2026-09-28): training feeds the constant 1.0 on EVERY row
# (training.stage1_data.TIME_PROGRESS_CONSTANT); serving must not feed the model a 0~1 value it
# never saw during training.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hour,minute", [(9, 0), (9, 10), (12, 15), (15, 30)])
def test_today_time_progress_equals_training_constant_at_any_time_of_day(hour, minute):
    from training.stage1_data import TIME_PROGRESS_CONSTANT

    history = _make_history_rows(59)
    today_rows = [{"datetime": datetime(2026, 9, 9, 9, 0), "open": 105.0, "high": 106.0,
                    "low": 104.5, "close": 105.5, "volume": 1000}]
    now = datetime(2026, 9, 9, hour, minute)
    df = build_encoder_df("005930", history, today_rows, now, HISTORICAL_COLS, FUTURE_COLS, STATIC_COLS)
    assert df["time_progress"].iloc[-1] == TIME_PROGRESS_CONSTANT
    assert (df["time_progress"] == TIME_PROGRESS_CONSTANT).all()


def test_historical_time_progress_constant_matches_training():
    from serving.feature_builder import HISTORICAL_TIME_PROGRESS
    from training.stage1_data import TIME_PROGRESS_CONSTANT

    assert HISTORICAL_TIME_PROGRESS == TIME_PROGRESS_CONSTANT
