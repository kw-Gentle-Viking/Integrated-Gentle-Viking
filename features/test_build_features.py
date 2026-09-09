import pandas as pd
import pytest
from features.build_features import compute_sector_derived_features, compute_vi_features, SECTOR_CODES


def _sector_df(closes, highs=None, lows=None, volumes=None, sector_code="0013"):
    n = len(closes)
    highs = highs or [c * 1.01 for c in closes]
    lows = lows or [c * 0.99 for c in closes]
    volumes = volumes or [1000.0] * n
    return pd.DataFrame({
        "sector_code": [sector_code] * n,
        "trade_date": pd.bdate_range("2024-01-02", periods=n),
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def test_sector_ret_1d_matches_day_over_day_change():
    df = _sector_df([100.0, 110.0, 99.0])
    out = compute_sector_derived_features(df)
    assert out["sector_ret_1d"].iloc[1] == pytest.approx(110.0 / 100.0 - 1)
    assert out["sector_ret_1d"].iloc[2] == pytest.approx(99.0 / 110.0 - 1)
    assert pd.isna(out["sector_ret_1d"].iloc[0])


def test_sector_ret_5d_and_20d_are_cumulative_over_n_days():
    closes = [100.0 + i for i in range(25)]  # linear 100..124
    df = _sector_df(closes)
    out = compute_sector_derived_features(df)
    assert out["sector_ret_5d"].iloc[5] == pytest.approx(closes[5] / closes[0] - 1)
    assert out["sector_ret_20d"].iloc[20] == pytest.approx(closes[20] / closes[0] - 1)
    assert pd.isna(out["sector_ret_5d"].iloc[4])
    assert pd.isna(out["sector_ret_20d"].iloc[19])


def test_sector_ma_ratio_20d_matches_hand_computed_rolling_mean():
    closes = [100.0, 102.0, 98.0, 101.0, 105.0]
    df = _sector_df(closes)
    out = compute_sector_derived_features(df)
    # min_periods=1 (matches this codebase's ticker-level sma convention): at index 2, the
    # 20d window only has 3 observations so far -> mean(100, 102, 98).
    expected_ma = sum(closes[:3]) / 3
    assert out["sector_ma_ratio_20d"].iloc[2] == pytest.approx(closes[2] / expected_ma)


def test_sector_volatility_is_intraday_high_low_range_ratio():
    df = _sector_df([100.0], highs=[105.0], lows=[95.0])
    out = compute_sector_derived_features(df)
    assert out["sector_volatility"].iloc[0] == pytest.approx((105.0 - 95.0) / 100.0)


def test_sector_volume_ratio_matches_hand_computed_rolling_mean():
    volumes = [1000.0, 2000.0, 3000.0]
    df = _sector_df([100.0, 100.0, 100.0], volumes=volumes)
    out = compute_sector_derived_features(df)
    expected_ma = sum(volumes[:3]) / 3
    assert out["sector_volume_ratio"].iloc[2] == pytest.approx(volumes[2] / expected_ma, rel=1e-6)


def test_forward_fills_gap_within_sector_series_instead_of_dropping():
    df = _sector_df([100.0, 110.0, 121.0])
    df.loc[1, ["open", "high", "low", "close", "volume"]] = [None, None, None, None, None]
    out = compute_sector_derived_features(df)
    assert len(out) == 3  # gap row is forward-filled, not dropped
    # row 1 forward-filled to row 0's close (100.0) -> flat day, ret == 0
    assert out["sector_ret_1d"].iloc[1] == pytest.approx(0.0)
    # row 2 computed against the forward-filled row 1 (100.0), not the original missing value
    assert out["sector_ret_1d"].iloc[2] == pytest.approx(121.0 / 100.0 - 1)


def test_derived_features_computed_independently_per_sector_code():
    df = pd.concat([
        _sector_df([100.0, 110.0], sector_code="0005"),
        _sector_df([200.0, 190.0], sector_code="0006"),
    ], ignore_index=True)
    out = compute_sector_derived_features(df)
    a = out[out["sector_code"] == "0005"].sort_values("trade_date")
    b = out[out["sector_code"] == "0006"].sort_values("trade_date")
    assert a["sector_ret_1d"].iloc[1] == pytest.approx(110.0 / 100.0 - 1)
    assert b["sector_ret_1d"].iloc[1] == pytest.approx(190.0 / 200.0 - 1)


def test_output_columns_are_exactly_key_plus_six_derived_features():
    df = _sector_df([100.0, 101.0])
    out = compute_sector_derived_features(df)
    assert list(out.columns) == [
        "sector_code", "trade_date", "sector_ret_1d", "sector_ret_5d", "sector_ret_20d",
        "sector_ma_ratio_20d", "sector_volatility", "sector_volume_ratio",
    ]


def _trading_days(ticker, n, start="2020-01-02"):
    return pd.DataFrame({
        "ticker": [ticker] * n,
        "trade_date": pd.bdate_range(start, periods=n),
    })


def test_vi_count_recent5d_does_not_bleed_across_widely_separated_events():
    # Regression test for the "sparse rolling window" bug: the old implementation rolled
    # window=5 over vi_daily rows that only existed for days with an actual VI event, so for
    # a ticker with infrequent VI events the "recent 5" window silently summed the last 5
    # EVENT-days regardless of how many real trading days separated them. Here two VI events
    # for the same ticker are 210 trading days apart -- the true trailing-5-trading-day window
    # around the later event must not include the earlier one.
    ticker = "005930"
    trading_days = _trading_days(ticker, 220)
    early_event_date = trading_days["trade_date"].iloc[0]
    late_event_date = trading_days["trade_date"].iloc[210]

    vi_raw = pd.DataFrame({
        "ticker": [ticker, ticker],
        "triggered_at": [
            pd.Timestamp(early_event_date) + pd.Timedelta(hours=9, minutes=5),
            pd.Timestamp(late_event_date) + pd.Timedelta(hours=9, minutes=5),
        ],
        "vi_type": ["정적", "정적"],
    })

    out = compute_vi_features(vi_raw, trading_days)

    late_row = out.loc[out["trade_date"] == late_event_date].iloc[0]
    assert late_row["is_vi_triggered"] == 1
    # Only the late event falls within the trailing 5 trading days ending on late_event_date --
    # the day-0 event (210 trading days earlier) must NOT bleed into this window.
    assert late_row["vi_count_recent5d"] == pytest.approx(1.0)

    early_row = out.loc[out["trade_date"] == early_event_date].iloc[0]
    assert early_row["is_vi_triggered"] == 1
    assert early_row["vi_count_recent5d"] == pytest.approx(1.0)


def test_vi_count_recent5d_sums_events_within_true_trailing_5_trading_day_window():
    ticker = "000660"
    trading_days = _trading_days(ticker, 30)
    d = trading_days["trade_date"]

    # Two events 3 trading days apart -- both fall inside a real trailing-5-trading-day window.
    vi_raw = pd.DataFrame({
        "ticker": [ticker, ticker],
        "triggered_at": [
            pd.Timestamp(d.iloc[10]) + pd.Timedelta(hours=9, minutes=5),
            pd.Timestamp(d.iloc[13]) + pd.Timedelta(hours=10),
        ],
        "vi_type": ["정적", "동적"],
    })

    out = compute_vi_features(vi_raw, trading_days)
    row13 = out.loc[out["trade_date"] == d.iloc[13]].iloc[0]
    assert row13["vi_count_recent5d"] == pytest.approx(2.0)


def test_vi_features_cover_every_trading_day_not_just_event_days():
    # Non-event trading days must appear with is_vi_triggered=0 / vi_count_recent5d=0, not be
    # absent from the output (the merge back into feature_pool relies on full coverage).
    ticker = "005930"
    trading_days = _trading_days(ticker, 10)
    vi_raw = pd.DataFrame({
        "ticker": [ticker],
        "triggered_at": [pd.Timestamp(trading_days["trade_date"].iloc[0]) + pd.Timedelta(hours=9)],
        "vi_type": ["정적"],
    })

    out = compute_vi_features(vi_raw, trading_days)
    assert len(out) == 10
    no_event_row = out.loc[out["trade_date"] == trading_days["trade_date"].iloc[5]].iloc[0]
    assert no_event_row["is_vi_triggered"] == 0
    assert no_event_row["vi_count_recent5d"] == pytest.approx(0.0)


def test_vi_features_empty_input_returns_full_trading_day_calendar_all_zero():
    ticker = "005930"
    trading_days = _trading_days(ticker, 5)
    empty_vi = pd.DataFrame(columns=["ticker", "triggered_at", "vi_type"])

    out = compute_vi_features(empty_vi, trading_days)
    assert len(out) == 5
    assert (out["is_vi_triggered"] == 0).all()
    assert (out["vi_count_recent5d"] == 0).all()


def test_sector_codes_has_20_entries_matching_documented_mapping():
    # Must stay in sync with data_collection/run_dart_calendar_sector_backfill.py's SECTOR_CODES
    # and data_collection/backfill_sector_id.py's SECTOR_NAMES (same index order).
    assert len(SECTOR_CODES) == 20
    assert SECTOR_CODES[0] == "0005"
    assert SECTOR_CODES[8] == "0013"  # index 8 == "전기·전자" per backfill_sector_id.SECTOR_NAMES
    assert SECTOR_CODES[-1] == "0026"
    assert "0022" not in SECTOR_CODES and "0023" not in SECTOR_CODES
