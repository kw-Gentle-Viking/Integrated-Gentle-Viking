import math

import torch

from training.stage1_data import (
    compute_class_weights,
    compute_mild_class_weights,
    compute_uniform_class_weights,
)


def test_uniform_class_weights_always_ones_regardless_of_counts():
    balanced_counts = {0: 100, 1: 100, 2: 100}
    skewed_counts = {0: 85714, 1: 168032, 2: 90969}

    assert compute_uniform_class_weights(balanced_counts).tolist() == [1.0, 1.0, 1.0]
    assert compute_uniform_class_weights(skewed_counts).tolist() == [1.0, 1.0, 1.0]


def test_uniform_class_weights_returns_float_tensor():
    weights = compute_uniform_class_weights({0: 1, 1: 2, 2: 3})
    assert isinstance(weights, torch.Tensor)
    assert weights.dtype == torch.float32


def test_mild_class_weights_equals_sqrt_of_balanced_weights():
    # Real Stage-2 live-log label distribution from the buy-skew bug report.
    label_counts = {0: 85714, 1: 168032, 2: 90969}

    balanced = compute_class_weights(label_counts)
    mild = compute_mild_class_weights(label_counts)

    expected = torch.sqrt(balanced)
    assert torch.allclose(mild, expected, atol=1e-6)


def test_mild_class_weights_strictly_between_uniform_and_balanced_per_class():
    label_counts = {0: 85714, 1: 168032, 2: 90969}

    balanced = compute_class_weights(label_counts)
    mild = compute_mild_class_weights(label_counts)

    for c in range(3):
        b = balanced[c].item()
        m = mild[c].item()
        lo, hi = sorted([1.0, b])
        # Dampened toward 1.0, not equal to either extreme (skip degenerate case b == 1.0).
        if not math.isclose(b, 1.0, abs_tol=1e-9):
            assert lo < m < hi, f"class {c}: mild={m} not strictly between 1.0 and balanced={b}"


def test_mild_class_weights_returns_float_tensor():
    weights = compute_mild_class_weights({0: 10, 1: 20, 2: 30})
    assert isinstance(weights, torch.Tensor)
    assert weights.dtype == torch.float32


def test_weight_schemes_registry_maps_names_to_the_three_functions():
    from training.stage1_data import WEIGHT_SCHEMES

    assert set(WEIGHT_SCHEMES) == {"balanced", "uniform", "mild"}
    assert WEIGHT_SCHEMES["balanced"] is compute_class_weights
    assert WEIGHT_SCHEMES["uniform"] is compute_uniform_class_weights
    assert WEIGHT_SCHEMES["mild"] is compute_mild_class_weights


# ---- S2: label_col switch ------------------------------------------------------------
import pandas as pd
import pytest
from training import stage1_data as sd


def test_resolve_cache_path_default_label_keeps_existing_filename():
    p = "training/artifacts/stage1_cache_train_2019-01-02_2023-12-31.pkl"
    assert sd.resolve_cache_path(p, "label") == p


def test_resolve_cache_path_other_label_gets_distinct_filename():
    p = "training/artifacts/stage1_cache_train_2019-01-02_2023-12-31.pkl"
    q = sd.resolve_cache_path(p, "label_vn")
    assert q != p
    assert q == "training/artifacts/stage1_cache_train_2019-01-02_2023-12-31__label_vn.pkl"


def test_label_select_expr_default_and_alias():
    assert sd.label_select_expr("label") == "label"
    assert sd.label_select_expr("label_vn") == "label_vn AS label"


def test_label_select_expr_rejects_unknown_column():
    with pytest.raises(ValueError):
        sd.label_select_expr("label; DROP TABLE feature_pool")


def test_query_feature_pool_aliases_chosen_label_column(monkeypatch):
    seen = {}

    class _Cur:
        description = [("ticker",), ("label",)]
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def execute(self, q, params): seen["q"] = q
        def fetchall(self): return []

    class _Conn:
        def cursor(self, cursor_factory=None): return _Cur()
        def close(self): pass

    monkeypatch.setattr(sd.psycopg2, "connect", lambda dsn: _Conn())
    sd.query_feature_pool("dsn", "2019-01-02", "2019-12-31", label_col="label_vn")
    assert "label_vn AS label" in seen["q"]
    sd.query_feature_pool("dsn", "2019-01-02", "2019-12-31")
    assert "label_vn" not in seen["q"] and ", label\n" in seen["q"]


def test_load_or_build_uses_label_specific_cache_and_default_cache(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sd, "query_feature_pool",
                        lambda dsn, s, e, label_col="label": calls.append(label_col) or
                        pd.DataFrame({"ticker": ["A"], "trade_date": ["2019-01-02"], "label": [1]}))
    monkeypatch.setattr(sd, "build_ticker_dfs", lambda df, label_col="label": {"A": df})
    base = str(tmp_path / "c.pkl")
    sd.load_or_build_ticker_dfs("dsn", "s", "e", base)                      # writes c__<ver>.pkl
    sd.load_or_build_ticker_dfs("dsn", "s", "e", base, label_col="label_vn")  # writes c__label_vn__<ver>.pkl
    sd.load_or_build_ticker_dfs("dsn", "s", "e", base)                      # cache hit
    sd.load_or_build_ticker_dfs("dsn", "s", "e", base, label_col="label_vn")  # cache hit
    assert calls == ["label", "label_vn"]
    v = sd.DATA_VERSION
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([f"c__{v}.pkl", f"c__label_vn__{v}.pkl"])


def test_build_ticker_dfs_renames_raw_label_col_to_label():
    df = pd.DataFrame({"ticker": ["A", "A"], "trade_date": ["2019-01-02", "2019-01-03"],
                       "x": [1.0, None], "label_vn": [2, None]})
    out = sd.build_ticker_dfs(df, label_col="label_vn")["A"]
    assert "label" in out.columns and "label_vn" not in out.columns
    assert out["label"].iloc[0] == 2.0 and pd.isna(out["label"].iloc[1])
    assert out["x"].iloc[1] == 0.0  # label NaN preserved, features filled


def test_build_ticker_dfs_default_unchanged_and_aliased_frame_ok():
    df = pd.DataFrame({"ticker": ["A"], "trade_date": ["2019-01-02"], "x": [1.0], "label": [1]})
    assert sd.build_ticker_dfs(df)["A"]["label"].iloc[0] == 1.0
    assert sd.build_ticker_dfs(df, label_col="label_vn")["A"]["label"].iloc[0] == 1.0


def test_data_version_is_set_and_folded_into_effective_cache_path():
    assert sd.DATA_VERSION == "adj1"
    p = "training/artifacts/stage1_cache_train_2019-01-02_2023-12-31.pkl"
    assert sd.versioned_cache_path(p, "label") == "training/artifacts/stage1_cache_train_2019-01-02_2023-12-31__adj1.pkl"
    assert sd.versioned_cache_path(p, "label_vn") == "training/artifacts/stage1_cache_train_2019-01-02_2023-12-31__label_vn__adj1.pkl"


def test_unversioned_legacy_cache_is_never_served(tmp_path, monkeypatch):
    """A pre-adjustment pickle at the bare path must be ignored: the DB is re-queried."""
    import pickle
    base = tmp_path / "c.pkl"
    with open(base, "wb") as f:
        pickle.dump({"STALE": None}, f)
    monkeypatch.setattr(sd, "query_feature_pool", lambda dsn, s, e, label_col="label":
                        pd.DataFrame({"ticker": ["A"], "trade_date": ["2019-01-02"], "label": [1]}))
    monkeypatch.setattr(sd, "build_ticker_dfs", lambda df, label_col="label": {"FRESH": df})
    out = sd.load_or_build_ticker_dfs("dsn", "s", "e", str(base))
    assert list(out) == ["FRESH"]
