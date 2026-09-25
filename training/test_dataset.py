import numpy as np
import pandas as pd
import torch
from training.dataset import TickerDayDataset, build_incomplete_today_bar

HIST_COLS = ["log_ret", "disparity_20"]
FUT_COLS = ["time_progress", "is_bok"]
STATIC_COLS = ["sector_id", "market_id"]


def make_ticker_df(n_days: int) -> pd.DataFrame:
    return pd.DataFrame({
        "trade_date": pd.date_range("2019-01-02", periods=n_days, freq="B"),
        "log_ret": np.random.randn(n_days) * 0.01,
        "disparity_20": 1.0 + np.random.randn(n_days) * 0.02,
        "time_progress": 1.0, "is_bok": 0,
        "sector_id": 3, "market_id": 1,
        "label": np.random.randint(0, 3, n_days),
    })


def test_returns_none_for_insufficient_history():
    df = make_ticker_df(n_days=30)  # encoder_len=60보다 짧음
    ds = TickerDayDataset({"005930": df}, HIST_COLS, FUT_COLS, STATIC_COLS, encoder_len=60)
    assert len(ds) == 0


def test_produces_correct_tensor_shapes():
    df = make_ticker_df(n_days=65)
    ds = TickerDayDataset({"005930": df}, HIST_COLS, FUT_COLS, STATIC_COLS, encoder_len=60)
    assert len(ds) == 5  # 65 - 60
    sample = ds[0]
    assert sample["historical_ts_numeric"].shape == (60, 2)
    assert sample["future_ts_numeric"].shape == (1, 2)
    assert sample["static_feats_categorical"].shape == (1, 2)
    assert sample["label"].shape == (1,)


def test_static_features_are_int_dtype_for_embedding():
    df = make_ticker_df(n_days=65)
    ds = TickerDayDataset({"005930": df}, HIST_COLS, FUT_COLS, STATIC_COLS, encoder_len=60)
    assert ds[0]["static_feats_categorical"].dtype == torch.long


def test_build_incomplete_today_bar_computes_ohlc_from_intraday():
    rows = [
        {"datetime": "2026-09-08 09:00:00", "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000},
        {"datetime": "2026-09-08 09:05:00", "open": 100.5, "high": 102, "low": 100, "close": 101.5, "volume": 1500},
        {"datetime": "2026-09-08 09:10:00", "open": 101.5, "high": 101.8, "low": 101.0, "close": 101.2, "volume": 800},
    ]
    bar = build_incomplete_today_bar(rows, open_price=100)
    assert bar["open"] == 100
    assert bar["high"] == 102
    assert bar["low"] == 99
    assert bar["close"] == 101.2  # 가장 최근 종가
    assert bar["volume"] == 3300


# ---- alignment: legacy (target = row after the encoder) vs today (target = last encoder row) ----
def _marked_df(n, nan_label_rows=()):
    """Every row carries its own index in each column so we can see which row a tensor came from."""
    idx = np.arange(n, dtype=float)
    lab = (np.arange(n) % 3).astype(float)
    for r in nan_label_rows:
        lab[r] = np.nan
    return pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "log_ret": idx, "disparity_20": idx * 10,
        "time_progress": idx * 100, "is_bok": idx * 1000,
        "sector_id": np.arange(n) % 21, "market_id": np.arange(n) % 2,
        "label": lab,
    })


def test_legacy_alignment_is_default_and_unchanged():
    df = _marked_df(65)
    ds = TickerDayDataset({"A": df}, HIST_COLS, FUT_COLS, STATIC_COLS, encoder_len=6)
    assert ds.align == "legacy" and ds.target_offset == 6 and len(ds) == 65 - 6
    s = ds[0]                                            # window rows 0..5, target row 6
    assert s["historical_ts_numeric"][:, 0].tolist() == [0, 1, 2, 3, 4, 5]
    assert s["future_ts_numeric"][0, 0].item() == 600 and s["label"].item() == 6 % 3


def test_today_alignment_last_hist_row_is_the_label_row():
    df = _marked_df(20)
    ds = TickerDayDataset({"A": df}, HIST_COLS, FUT_COLS, STATIC_COLS, encoder_len=6, align="today")
    assert ds.target_offset == 5
    s = ds[0]                                            # window rows 0..5, target row 5 (== last hist row)
    assert s["historical_ts_numeric"][:, 0].tolist() == [0, 1, 2, 3, 4, 5]
    assert s["historical_ts_numeric"][-1, 0].item() == 5.0
    assert s["future_ts_numeric"][0].tolist() == [500.0, 5000.0]          # from row 5, not row 6
    assert s["static_feats_categorical"][0].tolist() == [5, 1]            # sector 5 % 21, market 5 % 2
    assert s["label"].item() == 5 % 3
    last = ds[len(ds) - 1]                                                # last window ends at final row 19
    assert last["historical_ts_numeric"][-1, 0].item() == 19.0 and last["label"].item() == 19 % 3


def test_today_alignment_window_count_is_one_more_per_ticker():
    dfs = lambda: {"A": _marked_df(20), "B": _marked_df(20)}
    legacy = TickerDayDataset(dfs(), HIST_COLS, FUT_COLS, STATIC_COLS, encoder_len=6)
    today = TickerDayDataset(dfs(), HIST_COLS, FUT_COLS, STATIC_COLS, encoder_len=6, align="today")
    assert len(legacy) == 2 * (20 - 6) and len(today) == 2 * (20 - 6 + 1)
    only = TickerDayDataset({"A": _marked_df(6)}, HIST_COLS, FUT_COLS, STATIC_COLS, encoder_len=6, align="today")
    assert len(only) == 1                                                  # exactly one full window
    assert len(TickerDayDataset({"A": _marked_df(5)}, HIST_COLS, FUT_COLS, STATIC_COLS, encoder_len=6, align="today")) == 0


def test_today_alignment_skips_nan_label_at_last_encoder_row():
    ds = TickerDayDataset({"A": _marked_df(12, nan_label_rows=(5, 11))}, HIST_COLS, FUT_COLS, STATIC_COLS,
                          encoder_len=6, align="today")
    assert ds.index == [("A", t) for t in range(1, 6)]    # window t targets row t+5: rows 5 and 11 skipped
    legacy = TickerDayDataset({"A": _marked_df(12, nan_label_rows=(5, 11))}, HIST_COLS, FUT_COLS, STATIC_COLS,
                              encoder_len=6)
    assert legacy.index == [("A", t) for t in range(0, 5)]  # legacy targets row t+6: only row 11 (t=5) skipped


def test_unknown_alignment_rejected():
    import pytest
    with pytest.raises(ValueError):
        TickerDayDataset({"A": _marked_df(10)}, HIST_COLS, FUT_COLS, STATIC_COLS, encoder_len=6, align="tomorrow")
