import logging

import numpy as np
import pandas as pd
import pytest

from training.gbm_meta_data import assert_join_coverage, join_embeddings_with_tabular


def test_join_keeps_only_matching_keys_and_adds_emb_columns():
    keys = [("A", "2024-01-02"), ("B", "2024-01-02"), ("A", "2024-01-03")]
    emb = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    tab = pd.DataFrame({
        "ticker": ["A", "A", "C"], "date_s": ["2024-01-02", "2024-01-03", "2024-01-02"],
        "label": [0, 1, 2],
    })
    out = join_embeddings_with_tabular(keys, emb, tab)
    assert len(out) == 2  # only (A, 01-02) and (A, 01-03) match
    assert set(out.columns) >= {"emb_0", "emb_1", "label", "ticker", "date_s"}
    row = out[(out.ticker == "A") & (out.date_s == "2024-01-02")].iloc[0]
    assert row["emb_0"] == 1.0 and row["emb_1"] == 2.0 and row["label"] == 0


def test_join_raises_on_duplicate_embedding_keys():
    keys = [("A", "2024-01-02"), ("A", "2024-01-02")]
    emb = np.zeros((2, 1))
    tab = pd.DataFrame({"ticker": ["A"], "date_s": ["2024-01-02"], "label": [0]})
    with pytest.raises(ValueError, match="duplicate"):
        join_embeddings_with_tabular(keys, emb, tab)


def test_join_raises_on_duplicate_tabular_keys():
    keys = [("A", "2024-01-02")]
    emb = np.zeros((1, 1))
    tab = pd.DataFrame({"ticker": ["A", "A"], "date_s": ["2024-01-02", "2024-01-02"], "label": [0, 1]})
    with pytest.raises(ValueError, match="duplicate"):
        join_embeddings_with_tabular(keys, emb, tab)


def test_assert_join_coverage_passes_when_ratio_met():
    assert_join_coverage(n_joined=95, n_emb=100, n_tab=200, min_ratio=0.9, label="train")


def test_assert_join_coverage_raises_when_below_ratio():
    with pytest.raises(ValueError, match="coverage"):
        assert_join_coverage(n_joined=50, n_emb=100, n_tab=200, min_ratio=0.9, label="train")


def test_assert_join_coverage_logs_the_ratio_even_when_it_passes(caplog):
    # Code review finding (Important): a train join allowed to drop up to 5% (min_ratio=0.95) never
    # logged how much was actually dropped -- a passing call was silent, so nobody running
    # run_gbm_meta.py could tell whether 0% or 4.9% of train rows were lost to the join.
    with caplog.at_level(logging.INFO, logger="training.gbm_meta_data"):
        assert_join_coverage(n_joined=95, n_emb=100, n_tab=200, min_ratio=0.9, label="train")
    assert any("train" in r.message and "0.95" in r.message for r in caplog.records)
