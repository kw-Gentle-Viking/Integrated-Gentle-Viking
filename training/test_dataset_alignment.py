"""Alignment consumers + train/serving consistency for TickerDayDataset(align="today")."""
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from serving.feature_builder import build_encoder_df
from serving.inference import run_inference
from serving.test_feature_builder import FUTURE_COLS, HISTORICAL_COLS, STATIC_COLS, _make_history_rows
from training.dataset import TickerDayDataset
from training.signal_data import filter_index_by_target_date, sample_meta
from training.stage1_data import compute_label_distribution
from training.train import _to_device_batch


def _frame(n=12):
    return pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "log_ret": np.arange(n, dtype=float), "time_progress": 1.0, "is_bok": 0.0,
        "sector_id": 3, "market_id": 1,
        "label": (np.arange(n) % 3).astype(float),
    })


def _ds(align, n=12):
    return TickerDayDataset({"A": _frame(n)}, ["log_ret"], ["time_progress", "is_bok"],
                            ["sector_id", "market_id"], encoder_len=6, align=align)


def test_sample_meta_uses_the_last_encoder_row_date_for_today_alignment():
    d = list(_frame()["trade_date"])
    today, legacy = _ds("today"), _ds("legacy")
    assert [m[1] for m in sample_meta(today)][0] == d[5].strftime("%Y-%m-%d")      # window 0's last encoder row
    assert [m[1] for m in sample_meta(legacy)][0] == d[6].strftime("%Y-%m-%d")     # legacy: row after encoder
    assert sample_meta(today)[0][2] == 5.0                                          # log_ret of that same row


def test_filter_index_by_target_date_uses_target_row_for_both_alignments():
    for align, first_target_row in (("today", 5), ("legacy", 6)):
        ds = _ds(align)
        cut = _frame()["trade_date"][first_target_row + 2].strftime("%Y-%m-%d")
        dropped = filter_index_by_target_date(ds, cut)
        assert dropped == 2, align
        assert sample_meta(ds)[0][1] == cut


def test_class_counts_read_the_target_row_label():
    # labels are row % 3; today targets rows 5..11 -> {2,0,1,2,0,1,2}; legacy rows 6..11 -> {0,1,2,0,1,2}
    assert compute_label_distribution(_ds("today")) == {0: 2, 1: 2, 2: 3}
    assert compute_label_distribution(_ds("legacy")) == {0: 2, 1: 2, 2: 2}


def test_dates_for_dataset_uses_target_offset():
    from training.run_stage1_champion_selection import dates_for_dataset
    d = list(_frame()["trade_date"])
    assert dates_for_dataset(_ds("today"))[0] == d[5].strftime("%Y-%m-%d")
    assert dates_for_dataset(_ds("legacy"))[0] == d[6].strftime("%Y-%m-%d")


def test_training_sample_equals_serving_tensors_for_the_same_60_rows():
    """The 60-row encoder_df laid out by serving.feature_builder (59 real days + today last) must give
    the same hist/future/static tensors from align="today" as serving.inference.run_inference builds."""
    history = _make_history_rows(59)
    today_rows = [{"datetime": datetime(2026, 9, 9, 9, 0), "open": 105.0, "high": 106.0, "low": 104.5,
                   "close": 105.5, "volume": 1000}]
    encoder_df = build_encoder_df("005930", history, today_rows, datetime(2026, 9, 9, 10, 0),
                                  HISTORICAL_COLS, FUTURE_COLS, STATIC_COLS)
    assert len(encoder_df) == 60
    # A training frame is the same 60 rows plus trade_date and a label for "today" (row 59).
    frame = encoder_df.copy()
    frame["trade_date"] = pd.date_range("2026-06-15", periods=60, freq="B")
    frame["label"] = 1.0
    ds = TickerDayDataset({"005930": frame}, HISTORICAL_COLS, FUTURE_COLS, STATIC_COLS, 60, align="today")
    assert len(ds) == 1                                        # exactly the serving window
    batch = next(iter(DataLoader(ds, batch_size=1)))
    batch.pop("label")
    train_batch = _to_device_batch(batch, torch.device("cpu"))

    captured = {}

    class Recorder:
        def eval(self):
            return self

        def __call__(self, b):
            captured.update(b)
            return {"class_logits": torch.zeros(1, 1, 3)}

    run_inference("005930", encoder_df, Recorder(), HISTORICAL_COLS, FUTURE_COLS, STATIC_COLS)
    for key in ("historical_ts_numeric", "future_ts_numeric", "static_feats_categorical"):
        assert captured[key].shape == train_batch[key].shape, key
        assert captured[key].dtype == train_batch[key].dtype, key
        assert torch.equal(captured[key], train_batch[key]), key
    # and the legacy alignment would NOT have matched (it reads the row after the window)
    legacy = TickerDayDataset({"005930": frame}, HISTORICAL_COLS, FUTURE_COLS, STATIC_COLS, 60)
    assert len(legacy) == 0
