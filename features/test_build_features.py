import pandas as pd
import pytest
from features.build_features import compute_sector_derived_features, SECTOR_CODES


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


def test_sector_codes_has_20_entries_matching_documented_mapping():
    # Must stay in sync with data_collection/run_dart_calendar_sector_backfill.py's SECTOR_CODES
    # and data_collection/backfill_sector_id.py's SECTOR_NAMES (same index order).
    assert len(SECTOR_CODES) == 20
    assert SECTOR_CODES[0] == "0005"
    assert SECTOR_CODES[8] == "0013"  # index 8 == "전기·전자" per backfill_sector_id.SECTOR_NAMES
    assert SECTOR_CODES[-1] == "0026"
    assert "0022" not in SECTOR_CODES and "0023" not in SECTOR_CODES
