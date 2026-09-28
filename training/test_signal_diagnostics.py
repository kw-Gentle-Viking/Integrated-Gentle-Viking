"""timing_decomposition: fixed-effect / timing split of a per-(ticker, day) score's rank IC."""
import os

import numpy as np
import pytest
from scipy.stats import spearmanr

from evaluation.evaluate import compute_signal_metrics
from training.signal_diagnostics import timing_decomposition

V3_DUMP = "/tmp/claude-1000/-home-user/1de18e8e-f244-4909-8b34-b508a4a09cc1/scratchpad/v3_scores.npz"


def _brute(dates, tickers, scores, rets, means, min_names=20):
    """Independent implementation (scipy Spearman, python loops) of the same definition."""
    dates, tickers = np.asarray(dates), np.asarray(tickers)
    scores, rets = np.asarray(scores, float), np.asarray(rets, float)
    out = {"raw": [], "fixed": [], "timing": []}
    for d in sorted(set(dates.tolist())):
        idx = [i for i in range(len(dates)) if dates[i] == d and np.isfinite(scores[i]) and np.isfinite(rets[i])]
        if len(idx) < min_names:
            continue
        s, r = scores[idx], rets[idx]
        f = np.array([means[tickers[i]] for i in idx])
        for key, x in (("raw", s), ("fixed", f), ("timing", s - f)):
            if np.ptp(x) == 0:
                continue
            out[key].append(spearmanr(x, r)[0])
    return {k: float(np.mean(v)) for k, v in out.items()}


def _synthetic(n_days=60, n_tk=40, seed=0, fixed_scale=1.0, timing_scale=0.3):
    """Scores = ticker fixed effect + noise; returns correlated with the fixed effect and anti-correlated
    with the timing part (mimics the V3 diagnosis: raw IC comes from the fixed effect, timing IC < 0)."""
    rng = np.random.default_rng(seed)
    fe = rng.standard_normal(n_tk) * fixed_scale
    tks = np.array([f"T{i:03d}" for i in range(n_tk)])
    dates, tickers, scores, rets = [], [], [], []
    for d in range(n_days):
        tim = rng.standard_normal(n_tk) * timing_scale
        r = 0.05 * fe - 0.05 * tim + rng.standard_normal(n_tk) * 0.1
        dates += [f"2024-01-{d + 1:03d}"] * n_tk
        tickers += list(tks)
        scores += list(fe + tim)
        rets += list(r)
    return np.array(dates), np.array(tickers), np.array(scores), np.array(rets)


def test_matches_brute_force_and_window_mean_is_default():
    d, t, s, r = _synthetic()
    means = {tk: s[t == tk].mean() for tk in set(t)}
    res = timing_decomposition(d, t, s, r)
    exp = _brute(d, t, s, r, means)
    assert res["raw_ic"] == pytest.approx(exp["raw"], abs=1e-12)
    assert res["fixed_effect_ic"] == pytest.approx(exp["fixed"], abs=1e-12)
    assert res["timing_ic"] == pytest.approx(exp["timing"], abs=1e-12)
    # the construction: raw IC is carried by the fixed effect, timing IC is negative
    assert res["fixed_effect_ic"] > 0.1 and res["timing_ic"] < -0.05
    assert res["raw_ic"] > 0
    for k in ("raw_ic_se", "fixed_effect_ic_se", "timing_ic_se"):
        assert res[k] is not None and res[k] > 0
    assert res["n_days"] == 60 and set(res["ticker_means"]) == set(t)


def test_raw_ic_equals_evaluate_rule():
    """raw_ic must be the evaluate.compute_signal_metrics daily Spearman of p_buy - p_sell."""
    d, t, s, r = _synthetic(seed=3)
    probs = np.stack([np.clip(s, 0, None), np.zeros_like(s), np.clip(-s, 0, None)], axis=1)  # p_buy - p_sell = s
    sig = compute_signal_metrics(list(d), r, probs, min_names_per_day=20)
    res = timing_decomposition(d, t, s, r)
    assert res["raw_ic"] == pytest.approx(sig["mean_daily_rank_ic"], abs=1e-12)
    assert res["n_days"] == sig["n_days_used"]
    assert res["raw_ic_se"] == pytest.approx(sig["ic_std"] / np.sqrt(sig["n_days_used"]), abs=1e-12)


def test_prior_ticker_means_are_used_for_the_split_not_window_means():
    d, t, s, r = _synthetic(seed=1)
    rng = np.random.default_rng(9)
    prior = {tk: float(rng.standard_normal()) for tk in set(t)}       # arbitrary prior, unrelated to the scores
    res = timing_decomposition(d, t, s, r, prior_ticker_means=prior)
    exp = _brute(d, t, s, r, prior)
    assert res["fixed_effect_ic"] == pytest.approx(exp["fixed"], abs=1e-12)
    assert res["timing_ic"] == pytest.approx(exp["timing"], abs=1e-12)
    assert res["raw_ic"] == pytest.approx(exp["raw"], abs=1e-12)
    assert res["ticker_means"] == prior or set(res["ticker_means"]) == set(prior)


def test_only_scores_enter_the_fixed_effect_never_returns():
    d, t, s, r = _synthetic(seed=2)
    a = timing_decomposition(d, t, s, r)
    b = timing_decomposition(d, t, s, np.random.default_rng(5).standard_normal(len(r)))
    assert a["ticker_means"] == b["ticker_means"]                       # means independent of the returns
    # also: rows without a finite return do not change the window means
    r2 = r.copy()
    r2[::7] = np.nan
    assert timing_decomposition(d, t, s, r2)["ticker_means"] == a["ticker_means"]


def test_min_names_and_constant_days_are_skipped_like_evaluate():
    d, t, s, r = _synthetic(n_days=10, n_tk=25, seed=4)
    d, t, s, r = list(d), list(t), s.copy(), r.copy()
    # day 0: constant raw score -> raw IC skipped (evaluate rule)
    s[:25] = 1.0
    # day 1: only 10 names -> below min_names, skipped for everything
    keep = np.ones(len(d), bool)
    keep[25 + 10:50] = False
    dd, tt = np.array(d)[keep], np.array(t)[keep]
    ss, rr = s[keep], r[keep]
    res = timing_decomposition(dd, tt, ss, rr, min_names=20)
    assert (res["n_days_raw"], res["n_days_fixed"], res["n_days_timing"]) == (8, 9, 9) and res["n_days"] == 8
    assert res["timing_ic"] is not None
    tiny = timing_decomposition(dd[:5], tt[:5], ss[:5], rr[:5])
    assert tiny["raw_ic"] is None and tiny["timing_ic"] is None and tiny["fixed_effect_ic"] is None
    assert tiny["timing_ic_se"] is None


def test_tickers_missing_from_prior_are_excluded_from_fixed_and_timing_only():
    d, t, s, r = _synthetic(seed=6, n_tk=40)
    prior = {tk: float(s[t == tk].mean()) for tk in set(t) if tk not in ("T000", "T001")}
    res = timing_decomposition(d, t, s, r, prior_ticker_means=prior)
    full = timing_decomposition(d, t, s, r)
    assert res["raw_ic"] == pytest.approx(full["raw_ic"], abs=1e-12)   # raw uses every finite row
    keep = ~np.isin(t, ["T000", "T001"])
    exp = _brute(d[keep], t[keep], s[keep], r[keep], prior)
    assert res["timing_ic"] == pytest.approx(exp["timing"], abs=1e-12)
    assert res["fixed_effect_ic"] == pytest.approx(exp["fixed"], abs=1e-12)


def test_quantile_long_short_of_timing_score():
    d, t, s, r = _synthetic(seed=7)
    res = timing_decomposition(d, t, s, r)
    assert res["raw_quantile_ls"] is not None and res["timing_quantile_ls"] is not None
    # timing is anti-correlated with returns by construction -> negative long/short
    assert res["timing_quantile_ls"] < 0
    # hand-computed first day: top-k minus bottom-k mean return of the timing score (k = int(n*0.2))
    day = d == d[0]
    tim = s[day] - np.array([res["ticker_means"][x] for x in t[day]])
    o = np.argsort(tim, kind="mergesort")
    exp0 = float(r[day][o[-8:]].mean() - r[day][o[:8]].mean())
    one = timing_decomposition(d[day], t[day], s[day], r[day], prior_ticker_means=res["ticker_means"])
    assert one["timing_quantile_ls"] == pytest.approx(exp0, abs=1e-12)


@pytest.mark.skipif(not os.path.exists(V3_DUMP), reason="V3 score dump (scratchpad v3_scores.npz) not present")
def test_reproduces_v3_signal_diagnosis_numbers():
    """docs/signal_diagnosis.md section 3: V3 raw 0.0412/0.0392, fixed 0.0409/0.0392, timing -0.0253/-0.0393."""
    z = np.load(V3_DUMP, allow_pickle=True)
    sc = {w: (z[f"{w}_dt"], z[f"{w}_tk"], z[f"{w}_p"][:, 0].astype(float) - z[f"{w}_p"][:, 2].astype(float),
              z[f"{w}_ret"]) for w in ("val_2024", "oot_2026")}
    v = timing_decomposition(*sc["val_2024"])
    o = timing_decomposition(*sc["oot_2026"], prior_ticker_means=v["ticker_means"])
    assert v["raw_ic"] == pytest.approx(0.0412, abs=1e-3) and o["raw_ic"] == pytest.approx(0.0392, abs=1e-3)
    assert v["fixed_effect_ic"] == pytest.approx(0.0409, abs=1e-3) and o["fixed_effect_ic"] == pytest.approx(0.0392, abs=1e-3)
    assert v["timing_ic"] == pytest.approx(-0.0253, abs=1e-3) and o["timing_ic"] == pytest.approx(-0.0393, abs=1e-3)
    assert o["timing_ic_se"] == pytest.approx(0.018, abs=1e-3)
