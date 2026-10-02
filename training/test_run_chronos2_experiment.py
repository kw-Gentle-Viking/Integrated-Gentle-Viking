import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

from training.chronos2_scorer import MIN_SERIES_LEN
from training.run_chronos2_experiment import (
    group_tickers_by_date, reduce_samples, render_doc, run_window, scores_to_arrays,
)

DSN = os.environ.get("STOCK_DB_V2_DSN")
HAS_CHRONOS = importlib.util.find_spec("chronos") is not None


# ---- reduce_samples ----

def _samples():
    dates = np.array(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"])
    tickers = np.array(["A", "B", "A", "B", "A", "B"])
    ret = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    return {"val_2024": {"tickers": tickers, "dates": dates, "n": 6, "ret": ret}}


def test_reduce_samples_stride_1_is_identity():
    s = _samples()
    out = reduce_samples(s, 1)
    assert out["val_2024"]["n"] == 6
    np.testing.assert_array_equal(out["val_2024"]["dates"], s["val_2024"]["dates"])
    np.testing.assert_array_equal(out["val_2024"]["ret"], s["val_2024"]["ret"])


def test_reduce_samples_stride_2_keeps_full_cross_section_on_kept_dates():
    s = _samples()
    out = reduce_samples(s, 2)
    w = out["val_2024"]
    assert set(w["dates"].tolist()) == {"2024-01-01", "2024-01-03"}
    assert w["n"] == 4   # both tickers on both kept dates
    assert w["n_dates_total"] == 3
    assert w["n_dates_kept"] == 2
    # ret stays aligned with tickers/dates after filtering
    for tk, d, r in zip(w["tickers"], w["dates"], w["ret"]):
        orig_idx = np.where((s["val_2024"]["tickers"] == tk) & (s["val_2024"]["dates"] == d))[0][0]
        assert r == s["val_2024"]["ret"][orig_idx]


def test_reduce_samples_large_stride_keeps_first_date_only():
    s = _samples()
    out = reduce_samples(s, 10)
    assert set(out["val_2024"]["dates"].tolist()) == {"2024-01-01"}


# ---- group_tickers_by_date ----

def test_group_tickers_by_date():
    tickers = np.array(["A", "B", "A"])
    dates = np.array(["d1", "d1", "d2"])
    out = group_tickers_by_date(tickers, dates)
    assert out == {"d1": ["A", "B"], "d2": ["A"]}


# ---- run_window (fake pipeline, no DB / model) ----

class FakePipeline:
    """predict_df stub: predicted close = last context close * 1.01 (deterministic +1% move),
    predicted volume_log1p = last context volume_log1p (irrelevant, unused by the scorer)."""

    def __init__(self):
        self.calls = []

    def predict_df(self, df, prediction_length, quantile_levels, id_column, timestamp_column, target,
                   context_length, batch_size):
        self.calls.append(df["id"].nunique())
        rows = []
        for tk, g in df.groupby(id_column):
            g = g.sort_values(timestamp_column)
            last_close = g["close"].iloc[-1]
            last_vol = g["volume_log1p"].iloc[-1]
            rows.append({"id": tk, "target_name": "close", "predictions": last_close * 1.01})
            rows.append({"id": tk, "target_name": "volume_log1p", "predictions": last_vol})
        return pd.DataFrame(rows)


def _price_history():
    dates = np.array([f"2024-01-{d:02d}" for d in range(1, 11)])
    return {
        "A": {"dates": dates, "close": np.linspace(100, 109, 10), "volume": np.linspace(1000, 1900, 10)},
        "B": {"dates": dates, "close": np.linspace(50, 59, 10), "volume": np.linspace(500, 950, 10)},
        "SHORT": {"dates": dates[:2], "close": np.array([1.0, 2.0]), "volume": np.array([1.0, 2.0])},
    }


def test_run_window_scores_match_deterministic_fake_pipeline():
    ph = _price_history()
    window_samples = {"tickers": np.array(["A", "B", "A", "B"]),
                      "dates": np.array(["2024-01-05", "2024-01-05", "2024-01-06", "2024-01-06"])}
    pipeline = FakePipeline()
    scores, meta = run_window(pipeline, ph, window_samples, context_len=512, batch_size=1024)
    assert meta["n_dates"] == 2
    assert meta["n_requested"] == 4
    assert meta["n_scored"] == 4
    assert meta["n_dropped_insufficient_history"] == 0
    # +1% predicted move -> score == log(1.01) for every sample (fake pipeline is deterministic)
    for v in scores.values():
        assert v == pytest.approx(np.log(1.01))


def test_run_window_drops_tickers_with_insufficient_history():
    ph = _price_history()
    window_samples = {"tickers": np.array(["A", "SHORT"]), "dates": np.array(["2024-01-05", "2024-01-05"])}
    pipeline = FakePipeline()
    scores, meta = run_window(pipeline, ph, window_samples, context_len=512, batch_size=1024)
    assert meta["n_requested"] == 2
    assert meta["n_scored"] == 1
    assert meta["n_dropped_insufficient_history"] == 1
    assert ("A", "2024-01-05") in scores
    assert ("SHORT", "2024-01-05") not in scores


def test_run_window_skips_predict_df_call_when_no_ticker_has_history():
    ph = {"SHORT": _price_history()["SHORT"]}
    window_samples = {"tickers": np.array(["SHORT"]), "dates": np.array(["2024-01-05"])}
    pipeline = FakePipeline()
    scores, meta = run_window(pipeline, ph, window_samples, context_len=512, batch_size=1024)
    assert scores == {}
    assert pipeline.calls == []   # never called predict_df for an empty context


# ---- scores_to_arrays ----

def test_scores_to_arrays_aligns_and_fills_nan_for_missing():
    scores = {("A", "d1"): 0.5, ("B", "d1"): -0.2}
    tickers = np.array(["A", "B", "C"])
    dates = np.array(["d1", "d1", "d1"])
    out = scores_to_arrays(scores, tickers, dates)
    np.testing.assert_allclose(out[:2], [0.5, -0.2])
    assert np.isnan(out[2])


# ---- render_doc (pure rendering, synthetic results dict) ----

def _fake_results():
    sig = {"mean_daily_rank_ic": 0.03, "ic_ir": 0.2, "quantile_long_short": {"mean_spread": 0.001}}
    cfg = {"signal": sig, "ic_se": 0.01, "timing": {"fixed_effect_ic": 0.01, "timing_ic": 0.02, "timing_ic_se": 0.01}}
    return {"code_commit": "abc123", "model_id": "amazon/chronos-2", "context_len": 512, "stride": 5,
           "device": "cpu", "total_wall_seconds": 120.0,
           "windows": {"val_2024": {"n_samples_full": 100, "n_samples_scored": 20, "n_dates_total": 50,
                                    "n_dates_kept": 10, "n_scored": 20, "n_dropped_insufficient_history": 0,
                                    "wall_seconds": 60.0, "seconds_per_date": 6.0}},
           "configs": {"chronos2": {"val_2024": cfg}}}


def test_render_doc_runs_and_contains_key_numbers():
    doc = render_doc(_fake_results())
    assert "Chronos-2" in doc
    assert "amazon/chronos-2" in doc
    assert "0.0300" in doc or "0.03" in doc
    assert "stride" in doc.lower() or "5" in doc


def test_render_doc_handles_missing_reference_artifacts(tmp_path, monkeypatch):
    # point the reference-artifact paths at files that don't exist -> render_doc must not raise
    import training.run_chronos2_experiment as mod
    monkeypatch.setattr(mod, "TABULAR_V2_JSON", str(tmp_path / "nope1.json"))
    monkeypatch.setattr(mod, "TFX_JSON", str(tmp_path / "nope2.json"))
    monkeypatch.setattr(mod, "E4_JSON", str(tmp_path / "nope3.json"))
    doc = mod.render_doc(_fake_results())
    assert "Chronos-2" in doc


# ---- integration: real DB + real Chronos-2 weights (skipped unless both are available) ----

@pytest.mark.skipif(not DSN or not HAS_CHRONOS, reason="needs STOCK_DB_V2_DSN and the chronos package "
                    "(run under chronosbolt_env: /home/user/miniconda3/envs/chronosbolt_env/bin/python)")
def test_main_tiny_end_to_end_real_db_and_model(tmp_path):
    """A huge stride keeps only the first sample date of each window -> one real predict_df call per
    window (~200 tickers), a couple of minutes on CPU. Checks the whole pipeline wires together on
    real data/weights, not a specific IC value (zero-shot IC on 1 day is not a meaningful claim)."""
    from training.run_chronos2_experiment import main

    out_json = str(tmp_path / "chronos2_results.json")
    doc_path = str(tmp_path / "chronos2_experiment.md")
    results = main(stride=100_000, device="cpu", out_json=out_json, doc_path=doc_path)

    assert os.path.exists(out_json)
    assert os.path.exists(doc_path)
    for w in ("val_2024", "oot_2026"):
        wv = results["windows"][w]
        assert wv["n_dates_kept"] == 1
        assert wv["n_scored"] > 0
        cfg = results["configs"]["chronos2"][w]
        assert cfg["signal"] is not None
        assert "timing" in cfg
