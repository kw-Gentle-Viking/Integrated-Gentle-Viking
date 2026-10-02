import numpy as np
import pandas as pd
import pytest

from training.signal_data import (
    filter_index_by_target_date, sample_meta, next_day_returns_for_samples,
)


class _FakeDS:
    encoder_len = 2

    def __init__(self):
        dates = pd.to_datetime(["2025-12-29", "2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05"])
        self.ticker_dfs = {
            "AAA": pd.DataFrame({"trade_date": dates, "log_ret": [0.0, .01, .02, .03, .04]}),
            "BBB": pd.DataFrame({"trade_date": dates, "log_ret": [0.0, -.01, -.02, -.03, -.04]}),
        }
        # samples: target row is t + encoder_len
        self.index = [("AAA", 0), ("AAA", 1), ("AAA", 2), ("BBB", 1), ("BBB", 2)]


def test_sample_meta_lists_ticker_date_and_todays_logret_in_index_order():
    meta = sample_meta(_FakeDS())
    assert meta == [
        ("AAA", "2025-12-31", .02), ("AAA", "2026-01-02", .03), ("AAA", "2026-01-05", .04),
        ("BBB", "2026-01-02", -.03), ("BBB", "2026-01-05", -.04),
    ]


def test_filter_index_by_target_date_drops_earlier_targets_and_keeps_order():
    ds = _FakeDS()
    n_dropped = filter_index_by_target_date(ds, "2026-01-01")
    assert n_dropped == 1
    assert ds.index == [("AAA", 1), ("AAA", 2), ("BBB", 1), ("BBB", 2)]
    assert min(d for _, d, _ in sample_meta(ds)) >= "2026-01-01"


def test_next_day_returns_for_samples_lookup_and_missing_is_nan():
    meta = [("AAA", "2026-01-02", 0.0), ("BBB", "2026-01-02", 0.0), ("ZZZ", "2026-01-02", 0.0)]
    lookup = {("AAA", "2026-01-02"): 0.05, ("BBB", "2026-01-02"): None}
    out = next_day_returns_for_samples(meta, lookup)
    assert out[0] == 0.05
    assert np.isnan(out[1]) and np.isnan(out[2])
    assert out.dtype == float


def test_filter_on_empty_index():
    ds = _FakeDS()
    ds.index = []
    assert filter_index_by_target_date(ds, "2026-01-01") == 0
    assert sample_meta(ds) == []
