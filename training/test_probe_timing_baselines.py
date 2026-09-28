import numpy as np

from training.probe_timing_baselines import render, summarize


def _samples(seed=0, n_days=30, n_tk=25):
    rng = np.random.default_rng(seed)
    dates = np.repeat([f"2024-01-{d + 1:02d}" for d in range(n_days)], n_tk)
    tickers = np.tile([f"T{i}" for i in range(n_tk)], n_days)
    return {"dates": dates, "tickers": tickers, "ret": rng.standard_normal(len(dates)) * 0.01}


def test_static_scorer_has_no_timing_component_and_pure_noise_has_none_of_either():
    val, oot = _samples(1), _samples(2)
    samples = {"val_2024": val, "oot_2026": oot}
    tk_effect = {f"T{i}": i / 25 for i in range(25)}
    static = {w: np.array([tk_effect[t] for t in samples[w]["tickers"]]) for w in samples}
    out = summarize({"static": static}, samples)
    # a pure per-ticker constant: score minus its ticker mean is identically 0 -> timing IC undefined (None) or ~0
    t = out["static"]["val_2024"]["timing_ic"]
    assert t is None or abs(t) < 1e-9
    assert out["static"]["val_2024"]["fixed_effect_ic"] == out["static"]["val_2024"]["raw_ic"]


def test_oot_uses_val_ticker_means_as_prior_and_render_lists_every_scorer():
    val, oot = _samples(3), _samples(4)
    samples = {"val_2024": val, "oot_2026": oot}
    rng = np.random.default_rng(5)
    scores = {"noise": {w: rng.standard_normal(len(samples[w]["dates"])) for w in samples}}
    out = summarize(scores, samples)
    assert set(out["noise"]) == {"val_2024", "oot_2026"}
    doc = render(out, {"code_commit": "abc", "data_version": "adj1"})
    assert "noise" in doc and "timing IC" in doc and "oot_2026" in doc
