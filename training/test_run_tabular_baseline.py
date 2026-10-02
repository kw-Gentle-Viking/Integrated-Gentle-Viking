"""E0v2: the tabular baseline must score exactly the (ticker, date) sample set the TFT runner scores."""
import numpy as np

import training.run_tabular_baseline as tb
import training.run_tfx_experiments as tfx
from training.config import HISTORICAL_COLS_DEFAULT
from training.test_run_tfx_experiments import FakeData


def _independent_keys(fr, off, min_target):
    """Definition, spelled out: sample date = date of row index `off` (59 = last encoder row of the
    first window), label not NaN, date >= min_target. Order: ticker (dict order), then time."""
    out = []
    for tk, df in fr.items():
        for i in range(off, len(df)):
            row = df.iloc[i]
            d = row["trade_date"].strftime("%Y-%m-%d")
            if not np.isnan(row["label"]) and d >= min_target:
                out.append((tk, d))
    return out


def test_today_sample_set_equals_tfx_runner_and_definition():
    data = FakeData()
    for w in tfx.EVAL_WINDOWS:
        sp = tfx.SPLITS[w]
        keys = tb.enumerate_samples(data.frames(w, "label"), HISTORICAL_COLS_DEFAULT, sp["min_target"], "today")
        # TFT runner path: recipe label = label_vn (extra NULL rows) but the eval sample set must stay the fixed-label one
        default, recipe = data.frames(w, "label"), data.frames(w, "label_vn")
        eval_frames = tfx.build_eval_frames(default, recipe)
        _, meta = tfx.build_window_dataset(w, eval_frames, HISTORICAL_COLS_DEFAULT, {"align": "today"})
        assert keys == [(t, d) for t, d, _ in meta]
        assert keys == _independent_keys(data.frames(w, "label"), 59, sp["min_target"])
        assert len(keys) > 0 and len(set(keys)) == len(keys)


def test_legacy_option_keeps_old_enumeration():
    data = FakeData()
    w = "val_2024"
    sp = tfx.SPLITS[w]
    today = tb.enumerate_samples(data.frames(w, "label"), HISTORICAL_COLS_DEFAULT, sp["min_target"], "today")
    legacy = tb.enumerate_samples(data.frames(w, "label"), HISTORICAL_COLS_DEFAULT, sp["min_target"], "legacy")
    assert legacy == _independent_keys(data.frames(w, "label"), 60, sp["min_target"])
    assert len(today) > len(legacy)            # today adds each ticker's first window
    assert set(legacy) - set(today) == set()   # dates present in legacy are a subset here (same rows, offset by one)


def test_v2_paths_do_not_overwrite_legacy_e0_outputs():
    assert tb.V2_JSON != tb.RESULT_JSON and tb.V2_LEGACY_JSON != tb.RESULT_JSON
    assert tb.V2_DOC != tb.DOC_PATH
