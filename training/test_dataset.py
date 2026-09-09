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
