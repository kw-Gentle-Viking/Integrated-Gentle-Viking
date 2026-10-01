import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

import training.run_e4_experiments as e4
import training.run_tfx_experiments as tfx
from training.config import STATIC_COLS
from training.test_run_tfx_experiments import FakeData, _expected_counts


@pytest.fixture
def small_min_names(monkeypatch):
    monkeypatch.setattr(tfx, "MIN_NAMES_PER_DAY", 2)


def _no_budget(monkeypatch):
    import training.run_stage1_search as rss
    monkeypatch.setattr(rss, "Budget", lambda *_: type("B", (), {"exceeded": lambda self: False})())


def _frames(rets_by_date, tickers=None, glitch=None):
    """{ticker: frame} with next_day_return / label(=1.0 dummy fixed label) for a hand-made panel.
    rets_by_date: {date_str: [ret per ticker]} (None -> NaN return and NaN label)."""
    dates = sorted(rets_by_date)
    n = len(next(iter(rets_by_date.values())))
    tickers = tickers or [f"T{i:02d}" for i in range(n)]
    out = {}
    for i, tk in enumerate(tickers):
        r = [rets_by_date[d][i] for d in dates]
        r = [np.nan if x is None else x for x in r]
        out[tk] = pd.DataFrame({
            "ticker": tk, "trade_date": [dt.date.fromisoformat(d) for d in dates], "next_day_return": r,
            "label": [np.nan if np.isnan(x) else 1.0 for x in r], "volatility_20d": 0.0, "sector_volatility": 0.0,
            "sector_id": i % 3, "market_id": i % 2})
    return out


# ---------------- recipe table / opts / fingerprint ----------------
ORIGINAL_4 = ["x1_cslabel", "x2_cslabel_nostatic", "x3_vn_nostatic", "x4_cslabel_vnfeat"]
COLLAPSE_DIAG_6 = ["x2_nostatic_nowd", "x3_vn_nostatic_nowd", "x2_nostatic_seed1", "x3_vn_nostatic_seed1",
                   "x2_nostatic_seed2", "x3_vn_nostatic_seed2"]
V3_STRUCTURE_NOWD = ["v3_structure_nowd"]
V3_STRUCTURE_FOLLOWUPS_8 = ["v3_structure_nowd_seed1", "v3_structure_nowd_seed2", "v3_structure_nowd_vn",
                            "v3_structure_nowd_cs", "v3_structure_wd1e4", "v3_structure_wd3e4",
                            "v3_structure_nowd_patience8", "v3_structure_nowd_dropout30"]
ALL_11 = ORIGINAL_4 + COLLAPSE_DIAG_6 + V3_STRUCTURE_NOWD
ALL_19 = ALL_11 + V3_STRUCTURE_FOLLOWUPS_8


def test_recipe_table_and_order():
    assert e4.DEFAULT_ORDER == ALL_19                 # original 4, then the 6 reruns, then V3wd0, then its 8 follow-ups
    assert e4.DEFAULT_ORDER[4:6] == ["x2_nostatic_nowd", "x3_vn_nostatic_nowd"]        # weight_decay hypothesis first
    assert e4.DEFAULT_ORDER[6:10] == ["x2_nostatic_seed1", "x3_vn_nostatic_seed1", "x2_nostatic_seed2", "x3_vn_nostatic_seed2"]
    assert e4.DEFAULT_ORDER[10] == "v3_structure_nowd"                         # V3-structure weight_decay control
    assert e4.DEFAULT_ORDER[11:19] == V3_STRUCTURE_FOLLOWUPS_8                 # seed1 -> seed2 -> vn -> cs -> wd1e4 -> wd3e4 -> patience8 -> dropout30
    assert set(e4.RECIPES) == set(e4.DEFAULT_ORDER) == set(ALL_19)
    assert len({e4.RECIPES[n]["tag"] for n in e4.RECIPES}) == 19               # every tag unique, old and new
    r = e4.RECIPES
    assert (r["x1_cslabel"]["label_source"], r["x1_cslabel"]["static_const"], r["x1_cslabel"]["vol_rank"]) == ("cs", False, False)
    assert (r["x2_cslabel_nostatic"]["label_source"], r["x2_cslabel_nostatic"]["static_const"]) == ("cs", True)
    assert (r["x3_vn_nostatic"]["label_source"], r["x3_vn_nostatic"]["static_const"]) == ("label_vn", True)
    assert (r["x4_cslabel_vnfeat"]["label_source"], r["x4_cslabel_vnfeat"]["static_const"], r["x4_cslabel_vnfeat"]["vol_rank"]) == ("cs", False, True)
    assert (r["v3_structure_nowd"]["label_source"], r["v3_structure_nowd"]["static_const"],
            r["v3_structure_nowd"]["vol_rank"]) == ("label", False, False)
    assert r["v3_structure_nowd"]["weight_decay"] == 0.0 and "seed" not in r["v3_structure_nowd"]
    assert e4.PATIENCE == 4 and e4.MAX_EPOCHS == 12 and e4.SELECT_KEY == "val_timing_ic"
    # the original 4 carry no seed/weight_decay override (module defaults apply)
    for n in ORIGINAL_4:
        assert "seed" not in r[n] and "weight_decay" not in r[n]


def test_v3_structure_followups_8_each_change_exactly_one_variable_off_v3_structure_nowd():
    r = e4.RECIPES
    base = r["v3_structure_nowd"]
    same_spec_keys = ("static_const", "vol_rank")                            # label_source differs for vn/cs only
    for n in V3_STRUCTURE_FOLLOWUPS_8:
        for k in same_spec_keys:
            assert r[n][k] == base[k], (n, k)
    # label_source: identical to base except vn/cs
    for n in set(V3_STRUCTURE_FOLLOWUPS_8) - {"v3_structure_nowd_vn", "v3_structure_nowd_cs"}:
        assert r[n]["label_source"] == "label", n
    assert r["v3_structure_nowd_vn"]["label_source"] == "label_vn"
    assert r["v3_structure_nowd_cs"]["label_source"] == "cs"
    # weight_decay: 0.0 for every follow-up except the two wd variants
    for n in set(V3_STRUCTURE_FOLLOWUPS_8) - {"v3_structure_wd1e4", "v3_structure_wd3e4"}:
        assert r[n]["weight_decay"] == 0.0, n
    assert r["v3_structure_wd1e4"]["weight_decay"] == pytest.approx(1e-4)
    assert r["v3_structure_wd3e4"]["weight_decay"] == pytest.approx(3e-4)
    # seed: only the two seed variants override it
    assert r["v3_structure_nowd_seed1"]["seed"] == 1 and r["v3_structure_nowd_seed2"]["seed"] == 2
    for n in set(V3_STRUCTURE_FOLLOWUPS_8) - {"v3_structure_nowd_seed1", "v3_structure_nowd_seed2"}:
        assert "seed" not in r[n], n
    # patience: only patience8 overrides it
    assert r["v3_structure_nowd_patience8"]["patience"] == 8
    for n in set(V3_STRUCTURE_FOLLOWUPS_8) - {"v3_structure_nowd_patience8"}:
        assert "patience" not in r[n], n
    # dropout: only dropout30 overrides it
    assert r["v3_structure_nowd_dropout30"]["dropout"] == pytest.approx(0.30)
    for n in set(V3_STRUCTURE_FOLLOWUPS_8) - {"v3_structure_nowd_dropout30"}:
        assert "dropout" not in r[n], n


def test_collapse_diagnosis_recipes_reuse_x2_x3_spec_with_only_seed_or_weight_decay_overridden():
    r = e4.RECIPES
    x2, x3 = r["x2_cslabel_nostatic"], r["x3_vn_nostatic"]
    same_spec_keys = ("label_source", "static_const", "vol_rank")
    for n, parent in (("x2_nostatic_nowd", x2), ("x2_nostatic_seed1", x2), ("x2_nostatic_seed2", x2),
                      ("x3_vn_nostatic_nowd", x3), ("x3_vn_nostatic_seed1", x3), ("x3_vn_nostatic_seed2", x3)):
        for k in same_spec_keys:
            assert r[n][k] == parent[k], (n, k)
    assert r["x2_nostatic_nowd"]["weight_decay"] == 0.0 and "seed" not in r["x2_nostatic_nowd"]
    assert r["x3_vn_nostatic_nowd"]["weight_decay"] == 0.0 and "seed" not in r["x3_vn_nostatic_nowd"]
    assert r["x2_nostatic_seed1"]["seed"] == 1 and "weight_decay" not in r["x2_nostatic_seed1"]
    assert r["x2_nostatic_seed2"]["seed"] == 2 and "weight_decay" not in r["x2_nostatic_seed2"]
    assert r["x3_vn_nostatic_seed1"]["seed"] == 1 and "weight_decay" not in r["x3_vn_nostatic_seed1"]
    assert r["x3_vn_nostatic_seed2"]["seed"] == 2 and "weight_decay" not in r["x3_vn_nostatic_seed2"]


def test_recipe_opts_common_settings_and_fingerprints_differ():
    base = {"hparams": {"lr": 3e-4, "dropout": 0.17, "state_size": 32}, "batch_size": 8, "seed": 5, "align": "today"}
    for n in e4.RECIPES:
        o = e4.recipe_opts(n, base)
        exp_patience = 8 if n == "v3_structure_nowd_patience8" else 4          # only patience8 overrides it
        exp_dropout = 0.30 if n == "v3_structure_nowd_dropout30" else 0.17     # only dropout30 overrides it
        assert o["patience"] == exp_patience and o["select"] == "val_timing_ic" and o["align"] == "today"
        assert o["hparams"]["dropout"] == pytest.approx(exp_dropout)
    for n in ORIGINAL_4:                                                       # module defaults: wd 1e-3, seed 0
        o = e4.recipe_opts(n, base)
        assert o["hparams"]["weight_decay"] == 1e-3 and o["seed"] == 0
    assert "weight_decay" not in base["hparams"] and base["seed"] == 5          # base not mutated
    assert base["hparams"]["dropout"] == pytest.approx(0.17)                    # base not mutated by dropout override either
    assert e4.recipe_opts("x1_cslabel", {**base, "state_size_override": 8})["hparams"]["state_size"] == 8
    fps = {n: e4.cheap_fingerprint(n, e4.recipe_opts(n, base)) for n in e4.RECIPES}
    assert len({json.dumps(f, sort_keys=True) for f in fps.values()}) == 19     # all 19 fingerprints distinct
    assert fps["x3_vn_nostatic"]["label_col"] == "label_vn" and fps["x1_cslabel"]["label_col"] == "cs_quantile"
    assert fps["v3_structure_nowd"]["label_col"] == "label"                     # the SAME fixed label as V3 (not cs_quantile)
    assert fps["x4_cslabel_vnfeat"]["e4"]["vol_rank_cols"] == ["volatility_20d", "sector_volatility"]
    assert fps["x2_cslabel_nostatic"]["e4"]["static_const"] is True and fps["x1_cslabel"]["e4"]["static_const"] is False
    assert fps["x1_cslabel"]["selection"] == {"mode": "val_timing_ic", "patience": 4}
    assert fps["v3_structure_nowd_patience8"]["selection"] == {"mode": "val_timing_ic", "patience": 8}
    assert fps["x1_cslabel"]["preprocessed"] is False
    full = e4.full_fingerprint("x1_cslabel", e4.recipe_opts("x1_cslabel", base), ["a"], [1, 2, 3])
    assert full["columns_hash"] and full["class_weights"] == [1.0, 2.0, 3.0]


def test_patience_and_dropout_overrides_change_only_their_own_fingerprint_field():
    base = {"hparams": {"lr": 3e-4, "dropout": 0.1658, "state_size": 32}, "batch_size": 8, "align": "today"}
    o_base = e4.recipe_opts("v3_structure_nowd", base)
    o_p8 = e4.recipe_opts("v3_structure_nowd_patience8", base)
    o_d30 = e4.recipe_opts("v3_structure_nowd_dropout30", base)
    assert o_base["patience"] == 4 and o_base["hparams"]["dropout"] == pytest.approx(0.1658)
    assert o_p8["patience"] == 8 and o_p8["hparams"]["dropout"] == pytest.approx(0.1658)    # dropout untouched
    assert o_d30["patience"] == 4 and o_d30["hparams"]["dropout"] == pytest.approx(0.30)    # patience untouched
    fp_base = e4.cheap_fingerprint("v3_structure_nowd", o_base)
    fp_p8 = e4.cheap_fingerprint("v3_structure_nowd_patience8", o_p8)
    fp_d30 = e4.cheap_fingerprint("v3_structure_nowd_dropout30", o_d30)
    assert fp_p8 != fp_base and fp_d30 != fp_base and fp_p8 != fp_d30
    # same label/static/vol_rank spec as v3_structure_nowd -> identical "e4" section; only hparams/selection differ
    assert fp_p8["e4"] == fp_base["e4"] == fp_d30["e4"]
    assert fp_p8["selection"]["patience"] == 8 and fp_base["selection"]["patience"] == fp_d30["selection"]["patience"] == 4
    assert fp_d30["hparams"]["dropout"] == pytest.approx(0.30)
    assert fp_p8["hparams"]["dropout"] == pytest.approx(fp_base["hparams"]["dropout"])


def test_collapse_diagnosis_recipe_opts_and_fingerprints_differ_only_by_seed_or_weight_decay():
    base = {"hparams": {"lr": 3e-4, "dropout": 0.17, "state_size": 32}, "batch_size": 8, "align": "today"}
    o_x2 = e4.recipe_opts("x2_cslabel_nostatic", base)
    assert e4.recipe_opts("x2_nostatic_nowd", base)["hparams"]["weight_decay"] == 0.0
    assert e4.recipe_opts("x2_nostatic_nowd", base)["seed"] == 0                # weight_decay override only
    assert e4.recipe_opts("x2_nostatic_seed1", base)["seed"] == 1
    assert e4.recipe_opts("x2_nostatic_seed1", base)["hparams"]["weight_decay"] == 1e-3   # seed override only
    assert e4.recipe_opts("x2_nostatic_seed2", base)["seed"] == 2
    o_x3 = e4.recipe_opts("x3_vn_nostatic", base)
    assert e4.recipe_opts("x3_vn_nostatic_nowd", base)["hparams"]["weight_decay"] == 0.0
    assert e4.recipe_opts("x3_vn_nostatic_seed1", base)["seed"] == 1
    assert e4.recipe_opts("x3_vn_nostatic_seed2", base)["seed"] == 2
    # cheap_fingerprint: "e4" section (label_source/static_const/vol_rank_cols) is IDENTICAL to the parent
    # (same spec), but the outer "seed"/"hparams" differ, so the overall fingerprint still differs (no collision).
    fp_x2 = e4.cheap_fingerprint("x2_cslabel_nostatic", o_x2)
    for n in ("x2_nostatic_nowd", "x2_nostatic_seed1", "x2_nostatic_seed2"):
        fp_n = e4.cheap_fingerprint(n, e4.recipe_opts(n, base))
        assert fp_n["e4"] == fp_x2["e4"]
        assert fp_n != fp_x2
    fp_x3 = e4.cheap_fingerprint("x3_vn_nostatic", o_x3)
    for n in ("x3_vn_nostatic_nowd", "x3_vn_nostatic_seed1", "x3_vn_nostatic_seed2"):
        fp_n = e4.cheap_fingerprint(n, e4.recipe_opts(n, base))
        assert fp_n["e4"] == fp_x3["e4"]
        assert fp_n != fp_x3


def test_min_train_loss_helper():
    assert e4._min_train_loss({"epoch_curve": []}) is None
    assert e4._min_train_loss({}) is None
    rec = {"epoch_curve": [{"epoch": 0, "train_loss": 1.15}, {"epoch": 1, "train_loss": 1.0986},
                           {"epoch": 2, "train_loss": 1.0986}]}
    assert e4._min_train_loss(rec) == pytest.approx(1.0986)


def test_vol_columns_are_named_and_part_of_the_champion_33():
    import json as _j
    champ = _j.load(open(tfx.CHAMPION_CONFIG_PATH))["columns"]
    assert e4.VOL_RANK_COLS == ["volatility_20d", "sector_volatility"] and set(e4.VOL_RANK_COLS) <= set(champ)
    assert len(champ) == 33


# ---------------- cross-sectional quantile labels ----------------
def test_cs_labels_top_bottom_30pct_hold_middle_and_min_names():
    # date A: 10 names, returns 1..10 (shuffled by ticker) -> BUY = 3 highest, SELL = 3 lowest, HOLD = 4
    perm = [4, 9, 1, 7, 2, 10, 5, 3, 8, 6]
    a = [p / 100 for p in perm]
    b = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    fr = _frames({"2024-01-02": a, "2024-01-03": b})
    out = e4.cs_label_frames(fr, min_names=5)
    lab = np.array([out[f"T{i:02d}"]["label"].iloc[0] for i in range(10)])
    exp = np.array([0 if p >= 8 else (2 if p <= 3 else 1) for p in perm])
    assert (lab == exp).all()
    assert (lab == 0).sum() == 3 and (lab == 2).sum() == 3 and (lab == 1).sum() == 4
    lab_b = np.array([out[f"T{i:02d}"]["label"].iloc[1] for i in range(10)])
    assert list(lab_b) == [2, 2, 2, 1, 1, 1, 1, 0, 0, 0]                        # ordered returns -> ordered labels
    assert set(out["T00"].columns) == set(fr["T00"].columns)                    # label REPLACED in place, no new column
    assert (fr["T00"]["label"] == 1.0).all()                                    # input frames untouched


def test_cs_labels_min_names_nan_return_and_ineligible_rows():
    n = 6
    d1 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    d2 = [0.01, 0.02, None, None, None, None]                                   # only 2 finite names
    d3 = [0.01, 0.02, 0.03, 0.04, 0.05, None]                                   # one NaN return
    fr = _frames({"2024-01-02": d1, "2024-01-03": d2, "2024-01-04": d3})
    fr["T00"].loc[0, "label"] = np.nan                                          # gate: row without a base label is not labelled
    out = e4.cs_label_frames(fr, min_names=3)
    col = lambda tk: out[tk]["label"].to_numpy()
    assert np.isnan(col("T00")[0])                                              # ineligible (label gate)
    assert all(np.isnan(col(f"T{i:02d}")[1]) for i in range(n))                 # < min_names -> NaN for everybody
    assert np.isnan(col("T05")[2]) and not np.isnan(col("T04")[2])
    # day 3 has 5 eligible names: ranks 1..5 -> pct (r-.5)/5 = .1 .3 .5 .7 .9 -> SELL, SELL, HOLD, BUY, BUY
    assert [col(f"T{i:02d}")[2] for i in range(5)] == [2, 2, 1, 0, 0]


def test_cs_labels_ties_are_symmetric_midrank_and_order_independent():
    r = [0.0] * 4 + [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    fr = _frames({"2024-01-02": r})
    out = e4.cs_label_frames(fr, min_names=2)
    lab = [out[f"T{i:02d}"]["label"].iloc[0] for i in range(10)]
    assert lab[:4] == [lab[0]] * 4                                              # a tied block gets ONE label
    fr2 = _frames({"2024-01-02": r[::-1]})
    lab2 = [e4.cs_label_frames(fr2, min_names=2)[f"T{i:02d}"]["label"].iloc[0] for i in range(10)]
    assert lab2[::-1] == lab                                                    # ticker order does not matter


def test_cs_labels_match_bruteforce_on_random_panel_and_only_use_same_day():
    rng = np.random.default_rng(3)
    days = [f"2024-02-{d:02d}" for d in range(1, 15)]
    panel = {d: list(rng.standard_normal(23) * 0.02) for d in days}
    fr = _frames(panel)
    out = e4.cs_label_frames(fr, min_names=20)
    for d_i, d in enumerate(days):
        r = np.array(panel[d])
        order = np.argsort(np.argsort(r)) + 1                                    # ranks 1..23 (no ties)
        pct = (order - 0.5) / 23
        exp = np.where(pct >= 0.7, 0, np.where(pct <= 0.3, 2, 1))
        got = np.array([out[f"T{i:02d}"]["label"].iloc[d_i] for i in range(23)])
        assert (got == exp).all()
        assert (exp == 0).sum() == 7 and (exp == 2).sum() == 7                   # round(.3 * 23)
    # perturbing another day's returns never changes this day's labels
    panel2 = dict(panel)
    panel2[days[5]] = list(rng.standard_normal(23))
    out2 = e4.cs_label_frames(_frames(panel2), min_names=20)
    for i in range(23):
        a, b = out[f"T{i:02d}"]["label"].to_numpy(), out2[f"T{i:02d}"]["label"].to_numpy()
        assert (np.delete(a, 5) == np.delete(b, 5)).all()


# ---------------- nostatic / vol rank transforms ----------------
def test_constant_static_only_touches_static_columns():
    fr = _frames({"2024-01-02": [0.1, 0.2, 0.3, 0.4], "2024-01-03": [0.1, 0.2, 0.3, 0.4]})
    assert any(fr[t][c].nunique() > 1 for t in fr for c in STATIC_COLS) or len({fr[t]["sector_id"].iloc[0] for t in fr}) > 1
    out = e4.constant_static_frames(fr)
    for tk, df in out.items():
        assert (df[list(STATIC_COLS)] == 0).all().all()
        assert set(df.columns) == set(fr[tk].columns)
        for c in df.columns:
            if c not in STATIC_COLS:
                assert df[c].equals(fr[tk][c])
    assert fr["T01"]["sector_id"].iloc[0] == 1                                   # input untouched


def test_vol_rank_is_same_day_cross_section_only_and_leaves_other_columns():
    rng = np.random.default_rng(1)
    days = [f"2024-03-{d:02d}" for d in range(1, 8)]
    fr = _frames({d: list(rng.standard_normal(8)) for d in days})
    for i, tk in enumerate(fr):
        fr[tk]["volatility_20d"] = rng.random(7) * (i + 1)
        fr[tk]["sector_volatility"] = rng.random(7) * 3
        fr[tk]["rsi_14"] = rng.random(7)
    out = e4.vol_rank_frames(fr, ["volatility_20d", "sector_volatility"])
    tks = list(fr)
    for j in range(7):
        for c in ("volatility_20d", "sector_volatility"):
            vals = np.array([fr[t][c].iloc[j] for t in tks])
            exp = pd.Series(vals).rank(method="average", pct=True).to_numpy()
            got = np.array([out[t][c].iloc[j] for t in tks])
            assert np.allclose(got, exp) and got.min() > 0 and got.max() <= 1.0     # (0, 1] percentile rank
    assert all(out[t]["rsi_14"].equals(fr[t]["rsi_14"]) for t in tks)
    # NO future information: changing every later day's values leaves the ranks of earlier days untouched
    fr2 = {t: df.copy() for t, df in fr.items()}
    for t in tks:
        fr2[t].loc[4:, "volatility_20d"] = rng.random(3) * 100
    out2 = e4.vol_rank_frames(fr2, ["volatility_20d"])
    for t in tks:
        assert (out2[t]["volatility_20d"].iloc[:4].to_numpy() == out[t]["volatility_20d"].iloc[:4].to_numpy()).all()
    # NaN stays NaN
    fr3 = {t: df.copy() for t, df in fr.items()}
    fr3["T00"].loc[2, "volatility_20d"] = np.nan
    assert np.isnan(e4.vol_rank_frames(fr3, ["volatility_20d"])["T00"]["volatility_20d"].iloc[2])


# ---------------- prepare (synthetic data) ----------------
def _prep(name, monkeypatch, **kw):
    monkeypatch.setattr(tfx, "MIN_NAMES_PER_DAY", 2)
    data = FakeData()
    from training import run_tfx_experiments as t
    import json as _j
    champ = _j.load(open(t.CHAMPION_CONFIG_PATH))
    opts = {"champion": champ, "hparams": {k: champ[k] for k in ("state_size", "attention_heads", "lstm_layers", "dropout", "lr")},
            "batch_size": 8, "align": "today", **kw}
    return data, e4.prepare_e4(name, data, e4.recipe_opts(name, opts)), opts


def test_prepare_x1_train_labels_glitch_excluded_before_ranking(monkeypatch):
    data, prep, _ = _prep("x1_cslabel", monkeypatch)
    raw = tfx.attach_next_day_return(data.frames("train", "label"), data.next_day_returns("train"))
    tks = list(raw)
    n = len(raw[tks[0]])
    ret = np.array([raw[t]["next_day_return"].to_numpy() for t in tks])         # [tickers, days]
    assert (np.abs(ret[:, 20]) > 0.31).all()                                     # the glitch day (all tickers)
    for j in (5, 19, 21, 150):
        r = ret[:, j]
        elig = np.array([raw[t]["label"].iloc[j] == raw[t]["label"].iloc[j] for t in tks]) & (np.abs(r) <= 0.31)
        exp = np.full(len(tks), np.nan)
        if elig.sum() >= 2:
            rk = pd.Series(np.where(elig, r, np.nan)).rank(method="average").to_numpy()
            pct = (rk - 0.5) / elig.sum()
            exp = np.where(elig, np.where(pct >= 0.7 - 1e-9, 0, np.where(pct <= 0.3 + 1e-9, 2, 1)), np.nan)
        got = np.array([prep["train_frames"][t]["label"].iloc[j] for t in tks], dtype=float)
        assert np.array_equal(np.nan_to_num(got, nan=-1), np.nan_to_num(exp, nan=-1)), (j, got, exp)
    assert all(np.isnan(prep["train_frames"][t]["label"].iloc[20]) for t in tks)  # glitch rows: NaN label
    assert prep["n_glitch_masked"] == len(tks)
    assert set(np.unique(np.concatenate([f["label"].dropna().to_numpy() for f in prep["train_frames"].values()]))) <= {0.0, 1.0, 2.0}
    assert prep["kept_columns"] == list(prep["kept_columns"]) and len(prep["kept_columns"]) == 33
    for t in tks:                                                                # x1 keeps the static ids as they are
        assert prep["train_frames"][t][list(STATIC_COLS)].equals(raw[t][list(STATIC_COLS)])
    assert len({prep["train_frames"][t]["sector_id"].iloc[0] for t in tks}) > 1


def test_prepare_x1_eval_sample_set_is_the_fixed_label_gate_with_label_ok(monkeypatch):
    data, prep, _ = _prep("x1_cslabel", monkeypatch)
    for w in tfx.EVAL_WINDOWS:
        base = data.frames(w, "label")
        for tk, df in prep["eval_frames"][w].items():
            assert (df["label"].notna().to_numpy() == base[tk]["label"].notna().to_numpy()).all()   # same rows as V3
            assert df["label_ok"].any()
    v = prep["val_loss_frames"]["T1"]["label"].dropna()
    assert set(v.unique()) <= {0.0, 1.0, 2.0} and len(v) > 0


def test_prepare_x2_static_are_constant_everywhere_and_x1_labels_identical(monkeypatch):
    _, p1, _ = _prep("x1_cslabel", monkeypatch)
    _, p2, _ = _prep("x2_cslabel_nostatic", monkeypatch)
    frames = [p2["train_frames"], p2["val_loss_frames"], *p2["eval_frames"].values()]
    for fr in frames:
        for df in fr.values():
            assert (df[list(STATIC_COLS)] == 0).all().all()
    assert any(p1["train_frames"][t]["sector_id"].nunique() > 1 or p1["train_frames"][t]["sector_id"].iloc[0] != 0 for t in p1["train_frames"])
    for t in p1["train_frames"]:
        a, b = p1["train_frames"][t], p2["train_frames"][t]
        assert a["label"].equals(b["label"])
        cols = [c for c in a.columns if c not in STATIC_COLS]
        assert a[cols].equals(b[cols])


def test_prepare_x3_uses_label_vn_and_constant_static(monkeypatch):
    data, p3, _ = _prep("x3_vn_nostatic", monkeypatch)
    assert ("train", "label_vn") in data.calls and ("val_2024", "label_vn") in data.calls
    ref = data.frames("train", "label_vn")
    lab = p3["train_frames"]["T2"]["label"].to_numpy()
    exp = ref["T2"]["label"].to_numpy().copy()
    assert np.array_equal(np.nan_to_num(lab[:20], nan=-1), np.nan_to_num(exp[:20], nan=-1))   # DB label_vn, glitch row masked
    assert np.isnan(lab[20])
    assert (p3["train_frames"]["T2"][list(STATIC_COLS)] == 0).all().all()
    # eval sample set still the fixed-label gate (label_vn NULL rows -> placeholder + label_ok False)
    df = p3["eval_frames"]["val_2024"]["T2"]
    assert (~df["label_ok"]).sum() >= 1


def test_prepare_x4_vol_columns_are_ranks_others_raw_labels_cs(monkeypatch):
    data, p1, _ = _prep("x1_cslabel", monkeypatch)
    _, p4, _ = _prep("x4_cslabel_vnfeat", monkeypatch)
    for t in p1["train_frames"]:
        a, b = p1["train_frames"][t], p4["train_frames"][t]
        assert a["label"].equals(b["label"])                                     # x4 = x1 labels
        other = [c for c in a.columns if c not in e4.VOL_RANK_COLS]
        assert a[other].equals(b[other])
        assert b["volatility_20d"].between(0, 1).all() and not a["volatility_20d"].between(0, 1).all()
    tks = list(p4["train_frames"])
    raw = data.frames("train", "label")
    for j in (0, 50, 199):
        vals = pd.Series([raw[t]["volatility_20d"].iloc[j] for t in tks])
        got = np.array([p4["train_frames"][t]["volatility_20d"].iloc[j] for t in tks])
        assert np.allclose(got, vals.rank(method="average", pct=True).to_numpy())
    for w in tfx.EVAL_WINDOWS:                                                    # applied to eval windows too
        assert p4["eval_frames"][w]["T1"]["sector_volatility"].between(0, 1).all()


def test_prepare_v3_structure_nowd_label_matches_v3_exactly(monkeypatch):
    """The whole point of v3_structure_nowd is "V3 with weight_decay=0 and nothing else changed" -- this is the
    load-bearing check that label_source="label" really does give it V3's own fixed-threshold label (not some
    new E4 label), and that static_const=False / vol_rank=False leave every input exactly as V3 saw it. It
    compares prepare_e4's output frame-for-frame against tfx.prepare_recipe("aligned", ...), which is the exact
    code path e2e3's v3_wd (V3) was built from (BASE_RECIPE = "aligned" = R0, label_col="label", no preprocessing,
    no rank inputs -- see training/run_e2e3_experiments.py BASE_RECIPE / RECIPES["v3_wd"])."""
    monkeypatch.setattr(tfx, "MIN_NAMES_PER_DAY", 2)
    data = FakeData()
    champ = json.load(open(tfx.CHAMPION_CONFIG_PATH))
    opts = {"champion": champ, "hparams": {k: champ[k] for k in ("state_size", "attention_heads", "lstm_layers", "dropout", "lr")},
            "batch_size": 8, "align": "today"}

    v3_prep = tfx.prepare_recipe("aligned", data, opts)                       # V3's own data prep (BASE_RECIPE)
    assert data.calls == [("train", "label"), ("val_2024", "label"), ("oot_2026", "label")]   # label_col="label" only

    data2 = FakeData()                                                        # fresh instance: FakeData is deterministic
    e4_prep = e4.prepare_e4("v3_structure_nowd", data2, e4.recipe_opts("v3_structure_nowd", opts))
    assert data2.calls == [("train", "label"), ("val_2024", "label"), ("oot_2026", "label")]  # same label source, nothing else

    assert e4.label_col_of("v3_structure_nowd") == "label"
    assert e4_prep["kept_columns"] == v3_prep["kept_columns"] and len(e4_prep["kept_columns"]) == 33
    assert e4_prep["n_glitch_masked"] == v3_prep["n_glitch_masked"]
    for tk in v3_prep["train_frames"]:
        assert v3_prep["train_frames"][tk].equals(e4_prep["train_frames"][tk])
        assert v3_prep["train_frames"][tk]["label"].equals(e4_prep["train_frames"][tk]["label"])
    for tk in v3_prep["val_loss_frames"]:
        assert v3_prep["val_loss_frames"][tk].equals(e4_prep["val_loss_frames"][tk])
    for w in tfx.EVAL_WINDOWS:
        for tk in v3_prep["eval_frames"][w]:
            assert v3_prep["eval_frames"][w][tk].equals(e4_prep["eval_frames"][w][tk])
    # and it is NOT the x1 cross-sectional label (sanity: the fixed label and the cs label actually differ here)
    _, p1, _ = _prep("x1_cslabel", monkeypatch)
    some_differ = any(not p1["train_frames"][tk]["label"].equals(e4_prep["train_frames"][tk]["label"])
                      for tk in e4_prep["train_frames"])
    assert some_differ


def test_prepare_v3_structure_nowd_keeps_static_ids_as_is(monkeypatch):
    data, p, _ = _prep("v3_structure_nowd", monkeypatch)
    raw = tfx.attach_next_day_return(data.frames("train", "label"), data.next_day_returns("train"))
    for tk in p["train_frames"]:
        assert p["train_frames"][tk][list(STATIC_COLS)].equals(raw[tk][list(STATIC_COLS)])
    assert len({p["train_frames"][tk]["sector_id"].iloc[0] for tk in p["train_frames"]}) > 1    # not all constant 0


def test_v3_structure_nowd_opts_and_fingerprint():
    base = {"hparams": {"lr": 3e-4, "dropout": 0.17, "state_size": 32}, "batch_size": 8, "align": "today"}
    o = e4.recipe_opts("v3_structure_nowd", base)
    assert o["hparams"]["weight_decay"] == 0.0 and o["seed"] == 0 and o["select"] == "val_timing_ic"
    fp = e4.cheap_fingerprint("v3_structure_nowd", o)
    # "label" is not label_vn, so the "label_vn"-only threshold-dict wrapping in e4.cheap_fingerprint never fires:
    # fp["threshold"] is read_threshold("label", ...) straight from tfx, byte-identical in shape to V3's own fingerprint.
    assert fp["label_col"] == "label" and fp["threshold"] == tfx.read_threshold("label")
    assert fp["e4"]["label_source"] == "label" and fp["e4"]["static_const"] is False and fp["e4"]["vol_rank_cols"] == []
    # distinct from every other recipe's fingerprint, in particular x1 (same static/vol_rank spec, different label)
    fp_x1 = e4.cheap_fingerprint("x1_cslabel", e4.recipe_opts("x1_cslabel", base))
    assert fp != fp_x1 and fp["label_col"] != fp_x1["label_col"]


# ---------------- per-epoch metric hook ----------------
def test_timing_fn_uses_window_mean_and_returns_plain_floats(tmp_path):
    import torch
    from training.signal_diagnostics import timing_decomposition
    rng = np.random.default_rng(0)
    tickers = [f"T{i}" for i in range(30)]
    meta, rets, scores = [], [], []
    for d in range(12):
        for tk in tickers:
            meta.append((tk, f"2024-01-{d + 1:02d}", 0.0))
    scores = rng.standard_normal(len(meta))
    rets = rng.standard_normal(len(meta))

    def fake_scores(model, ds, device):
        return scores
    fn = e4.make_timing_fn(None, meta, rets, device="cpu", score_fn=fake_scores)
    m = fn(None, 3)
    dec = timing_decomposition([x[1] for x in meta], [x[0] for x in meta], scores, rets, None, 20)
    assert m["val_timing_ic"] == pytest.approx(dec["timing_ic"]) and m["val_ic"] == pytest.approx(dec["raw_ic"])
    assert m["val_fixed_ic"] == pytest.approx(dec["fixed_effect_ic"])
    assert all(type(v) is float or v is None for v in m.values())               # torch.save(weights_only) safe
    p = tmp_path / "ck.pt"
    torch.save({"epoch_history": [m]}, p)
    assert torch.load(p, weights_only=True)["epoch_history"][0]["val_timing_ic"] == m["val_timing_ic"]


# ---------------- CPU dry run on synthetic data ----------------
def _args(tmp, *extra):
    return ["--artifacts-dir", str(tmp / "art"), "--doc-path", str(tmp / "doc.md"),
            "--model-versions-path", str(tmp / "mv.md"), "--ref-dir", str(tmp / "noref"),
            "--epochs", "2", "--batch-size", "64", "--state-size", "8", "--no-wandb", "--device", "cpu", *extra]


def test_dry_run_interrupt_resume_idempotent(tmp_path, monkeypatch, small_min_names):
    import training.run_stage1_search as rss
    import training.train as train_mod
    data = FakeData()
    data.expected_samples = _expected_counts(data)
    art = tmp_path / "art"
    art.mkdir()
    tfx.main(["--artifacts-dir", str(art), "--doc-path", str(tmp_path / "tfxdoc.md"), "--model-versions-path",
              str(tmp_path / "tfxmv.md"), "--ref-dir", str(tmp_path / "noref"), "--epochs", "2", "--batch-size", "64",
              "--state-size", "8", "--no-wandb", "--device", "cpu", "--recipes", "aligned"], data=data)
    (art / "e2e3_results.json").write_text(json.dumps({"v3_wd": json.load(open(art / "tfx_results.json"))["aligned"]}))

    seq = iter([False, True])

    class ScriptedBudget:
        def __init__(self, *_):
            pass

        def exceeded(self):
            return next(seq, False)

    monkeypatch.setattr(rss, "Budget", ScriptedBudget)
    a = _args(tmp_path, "--recipes", "x1_cslabel", "x2_cslabel_nostatic", "--ref-dir", str(art))
    assert e4.main(a, data=data) == 0
    assert (art / "checkpoints" / "e4-x1_cslabel_inprogress.pt").exists()
    assert not (art / "e4_results.json").exists() and not (tmp_path / "mv.md").exists()

    _no_budget(monkeypatch)
    real_eval = train_mod.evaluate_loss
    monkeypatch.setattr(train_mod, "evaluate_loss", lambda *x, **k: (_ for _ in ()).throw(RuntimeError("kill")))
    with pytest.raises(RuntimeError, match="kill"):
        e4.main(a, data=data)
    monkeypatch.setattr(train_mod, "evaluate_loss", real_eval)

    real_train, calls = train_mod.train_one_epoch, {"n": 0}
    monkeypatch.setattr(train_mod, "train_one_epoch",
                        lambda *x, **k: (calls.__setitem__("n", calls["n"] + 1), real_train(*x, **k))[1])
    assert e4.main(a, data=data) == 0
    assert calls["n"] == 1 + 2                                                  # x1 resumes for 1 epoch, x2 runs 2 fresh
    res = json.load(open(art / "e4_results.json"))
    assert list(res) == ["x1_cslabel", "x2_cslabel_nostatic"]
    x1, x2 = res["x1_cslabel"], res["x2_cslabel_nostatic"]
    assert x1["meta"]["resumed"] is True and x1["meta"]["resumed_from_epoch"] == 1 and x2["meta"]["resumed"] is False
    for r in (x1, x2):
        assert r["selection"] == "val_timing_ic" and r["seed"] == 0 and r["align"] == "today"
        assert r["hparams"]["weight_decay"] == 1e-3 and r["patience"] == 4 and r["max_epochs"] == 2
        assert [h["epoch"] for h in r["epoch_curve"]] == [0, 1]
        assert all({"val_timing_ic", "val_ic", "val_fixed_ic", "val_loss"} <= set(h) for h in r["epoch_curve"])
        assert "not a selection criterion" in r["oot_scored"]
        for w in tfx.EVAL_WINDOWS:
            d = r[w]["decomposition"]
            assert {"raw_ic", "fixed_effect_ic", "timing_ic", "raw_ic_se", "timing_ic_se", "timing_ic_ir",
                    "raw_quantile_ls", "timing_quantile_ls"} <= set(d) and "ticker_means" not in d
            assert r[w]["signal"]["n_days_used"] > 0 and "macro_f1" in r[w]["metrics"]
            assert r[w]["decomposition"]["raw_ic"] == pytest.approx(r[w]["signal"]["mean_daily_rank_ic"], abs=1e-9)
        assert r["oot_2026"]["decomposition"]["fixed_effect_source"] == "val_2024 ticker means of the final checkpoint"
        assert r["val_2024"]["decomposition"]["fixed_effect_source"] == "window mean"
        best = max(r["epoch_curve"], key=lambda h: h["val_timing_ic"] if h["val_timing_ic"] is not None else -9)["epoch"]
        assert r["best_epoch"] == best and r["best_val_timing_ic"] == r["epoch_curve"][best]["val_timing_ic"]
    # the recorded val decomposition of the FINAL checkpoint equals the curve value of the selected epoch
    assert x2["val_2024"]["decomposition"]["timing_ic"] == pytest.approx(x2["best_val_timing_ic"], abs=1e-6)
    assert x2["hparams"]["dropout"] == x1["hparams"]["dropout"]
    doc = (tmp_path / "doc.md").read_text()
    assert "V3 seed 0" in doc and "X1" in doc and "X2" in doc and "timing" in doc
    mv = (tmp_path / "mv.md").read_text()
    assert [l.split("|")[1].strip() for l in mv.splitlines() if l.startswith("| e4-")] == ["e4-x1_cslabel", "e4-x2_cslabel_nostatic"]

    calls["n"] = 0
    before = (art / "e4_results.json").read_text()
    assert e4.main(a, data=data) == 0
    assert calls["n"] == 0 and (tmp_path / "mv.md").read_text() == mv and (art / "e4_results.json").read_text() == before

    a2 = _args(tmp_path, "--recipes", "x1_cslabel", "--ref-dir", str(art), "--batch-size", "32")
    e4.main(a2, data=data)
    assert (art / "e4_results.json.stale.x1_cslabel.json").exists() and calls["n"] == 2


def test_x2_x3_x4_end_to_end_dry_run_static_constant_reaches_the_model(tmp_path, monkeypatch, small_min_names):
    _no_budget(monkeypatch)
    import training.train as train_mod
    seen = {}
    real = train_mod.train_one_epoch

    def spy(model, loader, *a, **k):
        b = next(iter(loader))
        seen.setdefault("static", []).append(b["static_feats_categorical"].unique().tolist())
        seen.setdefault("hist_vol", []).append(float(b["historical_ts_numeric"][..., 5].max()))
        return real(model, loader, *a, **k)
    monkeypatch.setattr(train_mod, "train_one_epoch", spy)
    data = FakeData()
    data.expected_samples = _expected_counts(data)
    a = _args(tmp_path, "--recipes", "x2_cslabel_nostatic", "x3_vn_nostatic", "x4_cslabel_vnfeat", "--epochs", "1")
    assert e4.main(a, data=data) == 0
    assert seen["static"][0] == [0] and seen["static"][1] == [0]                 # x2, x3: every static id is 0
    assert seen["static"][2] != [0]                                              # x4 keeps sector/market ids
    assert seen["hist_vol"][2] > 0.5 > seen["hist_vol"][0]                       # x4: percentile ranks; x2: raw ~N(0, 0.02)
    res = json.load(open(tmp_path / "art" / "e4_results.json"))
    assert list(res) == ["x2_cslabel_nostatic", "x3_vn_nostatic", "x4_cslabel_vnfeat"]
    assert res["x3_vn_nostatic"]["label_source"] == "label_vn" and res["x4_cslabel_vnfeat"]["vol_rank_cols"] == e4.VOL_RANK_COLS
    assert res["x2_cslabel_nostatic"]["train"]["n_inputs"] == 33


def test_collapse_diagnosis_six_recipes_weight_decay_and_seed_reach_training_idempotent_no_collision(
        tmp_path, monkeypatch, small_min_names):
    """CPU dry run of all 6 new recipes: weight_decay=0 must actually reach torch.optim.Adam as "no weight_decay
    kwarg" (train.py's own convention: falsy weight_decay => omitted => Adam's true default of 0, see
    training/train.py `opt_kwargs`), a positive weight_decay must reach it as that kwarg, each recipe's seed
    must reach torch.manual_seed, a second invocation must retrain nothing (idempotent skip), and running the
    x2/x3 parents afterwards in the same results file must not collide with (or overwrite) these 6 results."""
    _no_budget(monkeypatch)
    import torch
    import training.train as train_mod

    seeds_seen = []
    real_manual_seed = torch.manual_seed

    def spy_seed(s):
        seeds_seen.append(s)
        return real_manual_seed(s)
    monkeypatch.setattr(torch, "manual_seed", spy_seed)

    adam_kwargs = []
    real_adam_init = torch.optim.Adam.__init__

    def spy_adam_init(self, params, **kw):
        adam_kwargs.append(kw)
        return real_adam_init(self, params, **kw)
    monkeypatch.setattr(torch.optim.Adam, "__init__", spy_adam_init)

    real_train, calls = train_mod.train_one_epoch, {"n": 0}
    monkeypatch.setattr(train_mod, "train_one_epoch",
                        lambda *x, **k: (calls.__setitem__("n", calls["n"] + 1), real_train(*x, **k))[1])

    data = FakeData()
    data.expected_samples = _expected_counts(data)
    new6 = ["x2_nostatic_nowd", "x3_vn_nostatic_nowd", "x2_nostatic_seed1", "x3_vn_nostatic_seed1",
            "x2_nostatic_seed2", "x3_vn_nostatic_seed2"]
    a = _args(tmp_path, "--recipes", *new6, "--epochs", "1")
    assert e4.main(a, data=data) == 0
    assert calls["n"] == 6
    # torch.manual_seed is called twice per recipe (once in recipe_opts/run_recipe setup, once again inside
    # run_training's per-epoch reseed -- with --epochs 1 that's seed+0 == the same value), in recipe order:
    # nowd(seed 0), nowd(seed 0), seed1, seed1, seed2, seed2. Adjacent recipes share a seed value (both nowd
    # variants default to seed 0, etc.), so a naive groupby over-collapses across the recipe boundary; the
    # only value-based check that actually verifies per-recipe seeding is the full 12-call sequence.
    assert seeds_seen == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    # train.py always passes lr as a kwarg (torch.optim.Adam(model.parameters(), lr=..., **opt_kwargs)); only
    # weight_decay is conditionally included. weight_decay=0 is falsy -> opt_kwargs omits it entirely.
    assert "weight_decay" not in adam_kwargs[0] and "weight_decay" not in adam_kwargs[1]
    assert [kw["weight_decay"] for kw in adam_kwargs[2:]] == [pytest.approx(1e-3)] * 4   # seed recipes keep 1e-3

    res = json.load(open(tmp_path / "art" / "e4_results.json"))
    assert set(res) == set(new6)                                              # exactly the 6, no stray keys
    exp_seed = {"x2_nostatic_nowd": 0, "x3_vn_nostatic_nowd": 0, "x2_nostatic_seed1": 1, "x3_vn_nostatic_seed1": 1,
                "x2_nostatic_seed2": 2, "x3_vn_nostatic_seed2": 2}
    exp_wd = {n: (0.0 if "nowd" in n else 1e-3) for n in new6}
    for n in new6:
        assert res[n]["seed"] == exp_seed[n] and res[n]["hparams"]["weight_decay"] == pytest.approx(exp_wd[n])

    # idempotent: a second invocation retrains nothing and leaves the results file byte-identical
    calls["n"] = 0
    before = (tmp_path / "art" / "e4_results.json").read_text()
    assert e4.main(a, data=data) == 0
    assert calls["n"] == 0
    assert (tmp_path / "art" / "e4_results.json").read_text() == before

    # running the x2/x3 parents afterwards adds two more keys, none of the 6 are touched or overwritten
    a2 = _args(tmp_path, "--recipes", "x2_cslabel_nostatic", "x3_vn_nostatic", "--epochs", "1")
    assert e4.main(a2, data=data) == 0
    res2 = json.load(open(tmp_path / "art" / "e4_results.json"))
    assert set(res2) == set(new6) | {"x2_cslabel_nostatic", "x3_vn_nostatic"}
    for n in new6:
        assert res2[n] == res[n]                                              # untouched by the parent runs
    assert res2["x2_cslabel_nostatic"]["fingerprint"] != res2["x2_nostatic_nowd"]["fingerprint"]
    assert res2["x2_cslabel_nostatic"]["fingerprint"] != res2["x2_nostatic_seed1"]["fingerprint"]
    assert res2["x3_vn_nostatic"]["fingerprint"] != res2["x3_vn_nostatic_nowd"]["fingerprint"]


def test_v3_structure_nowd_end_to_end_dry_run_static_kept_wd_omitted_and_idempotent(tmp_path, monkeypatch, small_min_names):
    """CPU dry run of v3_structure_nowd alone: static ids must reach the model UNCHANGED (unlike x2/x3), Adam
    must see no weight_decay kwarg at all (train.py's falsy-omits convention, same as the nowd collapse-diagnosis
    recipes), the recorded fingerprint's label_col must be "label" (V3's own fixed label, not cs_quantile /
    label_vn), docs/e4_experiments.md must render it correctly, docs/model_versions.md's row must say the
    correct weight_decay (not the old hardcoded "1e-3" text bug), and a second invocation must retrain nothing."""
    _no_budget(monkeypatch)
    import torch
    import training.train as train_mod

    seen = {}
    real = train_mod.train_one_epoch

    def spy(model, loader, *a, **k):
        b = next(iter(loader))
        seen.setdefault("static", []).append(b["static_feats_categorical"].unique().tolist())
        return real(model, loader, *a, **k)
    monkeypatch.setattr(train_mod, "train_one_epoch", spy)

    adam_kwargs = []
    real_adam_init = torch.optim.Adam.__init__

    def spy_adam_init(self, params, **kw):
        adam_kwargs.append(kw)
        return real_adam_init(self, params, **kw)
    monkeypatch.setattr(torch.optim.Adam, "__init__", spy_adam_init)

    real_train, calls = train_mod.train_one_epoch, {"n": 0}
    data = FakeData()
    data.expected_samples = _expected_counts(data)
    a = _args(tmp_path, "--recipes", "v3_structure_nowd", "--epochs", "1")
    assert e4.main(a, data=data) == 0
    assert seen["static"][0] != [0]                                           # static ids kept as-is, NOT constant 0
    assert "weight_decay" not in adam_kwargs[0]                               # wd=0 -> omitted, same as x2/x3_..._nowd

    res = json.load(open(tmp_path / "art" / "e4_results.json"))
    assert list(res) == ["v3_structure_nowd"]
    rec = res["v3_structure_nowd"]
    assert rec["label_source"] == "label" and rec["static_const"] is False and rec["vol_rank_cols"] == []
    assert rec["hparams"]["weight_decay"] == 0.0 and rec["seed"] == 0
    assert rec["fingerprint"]["label_col"] == "label"

    doc = (tmp_path / "doc.md").read_text()
    assert "v3_structure_nowd" in doc and "V3wd0" in doc and "fixed label (V3)" in doc

    mv = (tmp_path / "mv.md").read_text()
    row = next(l for l in mv.splitlines() if l.startswith("| e4-v3_structure_nowd "))
    assert "weight_decay=0," in row and "weight_decay=1e-3" not in row
    assert "고정 임계값 라벨" in row

    # idempotent: a second invocation retrains nothing and leaves both outputs byte-identical
    before_res, before_mv = (tmp_path / "art" / "e4_results.json").read_text(), mv
    calls["n"] = 0
    monkeypatch.setattr(train_mod, "train_one_epoch",
                        lambda *x, **k: (calls.__setitem__("n", calls["n"] + 1), real_train(*x, **k))[1])
    assert e4.main(a, data=data) == 0
    assert calls["n"] == 0
    assert (tmp_path / "art" / "e4_results.json").read_text() == before_res
    assert (tmp_path / "mv.md").read_text() == before_mv


def test_patience_and_dropout_overrides_reach_run_training_and_model_construction(tmp_path, monkeypatch, small_min_names):
    """CPU dry run of v3_structure_nowd plus its patience8 / dropout30 follow-ups: patience must actually reach
    train.run_training's early_stopping_patience config key, and dropout must actually reach
    config.build_tft_config's dropout kwarg (i.e. the model), not just sit in the recorded hparams dict. The
    base recipe (no override) must see the unmodified champion patience/dropout, proving the override is
    opt-in and does not leak into neighbouring recipes."""
    _no_budget(monkeypatch)
    import training.config as config_mod
    import training.train as train_mod

    patience_seen, dropout_seen = [], []
    real_run_training = train_mod.run_training
    real_build_tft_config = config_mod.build_tft_config

    def spy_run_training(config):
        patience_seen.append(config["early_stopping_patience"])
        return real_run_training(config)

    def spy_build_tft_config(*a, **k):
        dropout_seen.append(k["dropout"])
        return real_build_tft_config(*a, **k)
    monkeypatch.setattr(train_mod, "run_training", spy_run_training)
    monkeypatch.setattr(config_mod, "build_tft_config", spy_build_tft_config)

    data = FakeData()
    data.expected_samples = _expected_counts(data)
    a = _args(tmp_path, "--recipes", "v3_structure_nowd", "v3_structure_nowd_patience8",
              "v3_structure_nowd_dropout30", "--epochs", "1")
    assert e4.main(a, data=data) == 0
    assert patience_seen == [4, 8, 4]                                          # base, patience8, dropout30(unchanged)
    assert dropout_seen[0] == pytest.approx(dropout_seen[1])                   # base vs patience8: same (champion) dropout
    assert dropout_seen[2] == pytest.approx(0.30)                              # dropout30 recipe: overridden

    res = json.load(open(tmp_path / "art" / "e4_results.json"))
    assert res["v3_structure_nowd"]["patience"] == 4
    assert res["v3_structure_nowd_patience8"]["patience"] == 8
    assert res["v3_structure_nowd_patience8"]["hparams"]["dropout"] == pytest.approx(res["v3_structure_nowd"]["hparams"]["dropout"])
    assert res["v3_structure_nowd_dropout30"]["patience"] == 4
    assert res["v3_structure_nowd_dropout30"]["hparams"]["dropout"] == pytest.approx(0.30)
    assert res["v3_structure_nowd_dropout30"]["hparams"]["dropout"] != pytest.approx(res["v3_structure_nowd"]["hparams"]["dropout"])


def test_eight_v3_structure_followups_end_to_end_dry_run_idempotent_no_collision(tmp_path, monkeypatch, small_min_names):
    """CPU dry run of all 8 v3_structure_nowd follow-ups: static ids must stay AS-IS (static_const=False, unlike
    x2/x3) for every one of them, label_source must route to the right column (label / label_vn / cs), seed must
    reach torch.manual_seed, weight_decay must reach torch.optim.Adam exactly as train.py's falsy-omits
    convention dictates, a second invocation must retrain nothing (idempotent skip), and running v3_structure_nowd
    itself afterwards must not collide with or overwrite any of the 8 (all fingerprints distinct)."""
    _no_budget(monkeypatch)
    import torch
    import training.train as train_mod

    seeds_seen = []
    real_manual_seed = torch.manual_seed

    def spy_manual_seed(s):
        seeds_seen.append(s)
        return real_manual_seed(s)
    monkeypatch.setattr(torch, "manual_seed", spy_manual_seed)

    adam_kwargs = []
    real_adam_init = torch.optim.Adam.__init__

    def spy_adam_init(self, params, **kw):
        adam_kwargs.append(kw)
        return real_adam_init(self, params, **kw)
    monkeypatch.setattr(torch.optim.Adam, "__init__", spy_adam_init)

    static_vals = []
    real_train_one_epoch = train_mod.train_one_epoch

    def spy_train_one_epoch(model, loader, *a, **k):
        b = next(iter(loader))
        static_vals.append(b["static_feats_categorical"].unique().tolist())
        return real_train_one_epoch(model, loader, *a, **k)
    monkeypatch.setattr(train_mod, "train_one_epoch", spy_train_one_epoch)

    data = FakeData()
    data.expected_samples = _expected_counts(data)
    new8 = list(V3_STRUCTURE_FOLLOWUPS_8)
    a = _args(tmp_path, "--recipes", *new8, "--epochs", "1")
    assert e4.main(a, data=data) == 0

    res = json.load(open(tmp_path / "art" / "e4_results.json"))
    assert set(res) == set(new8)
    for i, n in enumerate(new8):
        assert res[n]["static_const"] is False
        assert static_vals[i] != [0]                                          # static ids kept, never forced to 0

    exp_label = {"v3_structure_nowd_seed1": "label", "v3_structure_nowd_seed2": "label",
                 "v3_structure_nowd_vn": "label_vn", "v3_structure_nowd_cs": "cs",
                 "v3_structure_wd1e4": "label", "v3_structure_wd3e4": "label",
                 "v3_structure_nowd_patience8": "label", "v3_structure_nowd_dropout30": "label"}
    for n, lab in exp_label.items():
        assert res[n]["label_source"] == lab, n

    exp_wd = {"v3_structure_nowd_seed1": 0.0, "v3_structure_nowd_seed2": 0.0, "v3_structure_nowd_vn": 0.0,
              "v3_structure_nowd_cs": 0.0, "v3_structure_wd1e4": 1e-4, "v3_structure_wd3e4": 3e-4,
              "v3_structure_nowd_patience8": 0.0, "v3_structure_nowd_dropout30": 0.0}
    for n, wd in exp_wd.items():
        assert res[n]["hparams"]["weight_decay"] == pytest.approx(wd), n

    # one torch.optim.Adam init per recipe (epochs=1, no resume) -- wd=0 is falsy -> opt_kwargs omits it entirely
    for i, n in enumerate(new8):
        if exp_wd[n] == 0.0:
            assert "weight_decay" not in adam_kwargs[i], n
        else:
            assert adam_kwargs[i]["weight_decay"] == pytest.approx(exp_wd[n]), n

    assert res["v3_structure_nowd_seed1"]["seed"] == 1 and res["v3_structure_nowd_seed2"]["seed"] == 2
    for n in set(new8) - {"v3_structure_nowd_seed1", "v3_structure_nowd_seed2"}:
        assert res[n]["seed"] == 0, n
    assert res["v3_structure_nowd_patience8"]["patience"] == 8
    for n in set(new8) - {"v3_structure_nowd_patience8"}:
        assert res[n]["patience"] == 4, n
    assert res["v3_structure_nowd_dropout30"]["hparams"]["dropout"] == pytest.approx(0.30)

    # idempotent: a second invocation retrains nothing and leaves the results file byte-identical
    before = (tmp_path / "art" / "e4_results.json").read_text()
    calls = {"n": 0}
    monkeypatch.setattr(train_mod, "train_one_epoch",
                        lambda *x, **k: (calls.__setitem__("n", calls["n"] + 1), real_train_one_epoch(*x, **k))[1])
    assert e4.main(a, data=data) == 0
    assert calls["n"] == 0
    assert (tmp_path / "art" / "e4_results.json").read_text() == before

    # running v3_structure_nowd afterwards adds one more key, none of the 8 are touched, overwritten, or collide
    a2 = _args(tmp_path, "--recipes", "v3_structure_nowd", "--epochs", "1")
    assert e4.main(a2, data=data) == 0
    res2 = json.load(open(tmp_path / "art" / "e4_results.json"))
    assert set(res2) == set(new8) | {"v3_structure_nowd"}
    for n in new8:
        assert res2[n] == res[n]                                              # untouched by the parent run
    fps = {n: json.dumps(res2[n]["fingerprint"], sort_keys=True) for n in res2}
    assert len(set(fps.values())) == len(fps)                                 # every fingerprint distinct, no collisions

    doc = (tmp_path / "doc.md").read_text()
    for n in new8:
        assert n in doc and e4.RECIPES[n]["tag"] in doc


def test_oot_never_scored_per_epoch(tmp_path, monkeypatch, small_min_names):
    _no_budget(monkeypatch)
    windows = []
    real = tfx.score_window
    monkeypatch.setattr(tfx, "score_window", lambda w, *a, **k: (windows.append(w), real(w, *a, **k))[1])
    data = FakeData()
    data.expected_samples = _expected_counts(data)
    e4.main(_args(tmp_path, "--recipes", "x1_cslabel"), data=data)
    assert windows == ["val_2024", "oot_2026"]


def test_epoch_hook_never_sees_oot_frames(tmp_path, monkeypatch, small_min_names):
    _no_budget(monkeypatch)
    built = []
    real = e4.make_timing_fn

    def spy(ds, meta, rets, device, score_fn=None):
        built.append(sorted({m[1][:4] for m in meta}))
        return real(ds, meta, rets, device, score_fn)
    monkeypatch.setattr(e4, "make_timing_fn", spy)
    data = FakeData()
    data.expected_samples = _expected_counts(data)
    e4.main(_args(tmp_path, "--recipes", "x1_cslabel", "--epochs", "1"), data=data)
    assert built == [["2024"]]


def test_max_tickers_requires_artifacts_dir_and_defaults():
    with pytest.raises(SystemExit):
        e4.parse_args(["--max-tickers", "3"])
    a = e4.parse_args([])
    assert a.recipes == e4.DEFAULT_ORDER and a.epochs == 12 and a.max_minutes == 600.0


def test_render_doc_baseline_row_and_empty_states():
    v3 = {"val_2024": {"signal": {"mean_daily_rank_ic": 0.0412, "ic_std": 0.2, "n_days_used": 100, "ic_ir": 0.2,
                                  "quantile_long_short": {"mean_spread": 0.0005},
                                  "argmax_long_short": {"spread": 0.0}}, "metrics": {"macro_f1": 0.35}},
          "oot_2026": {"signal": {"mean_daily_rank_ic": 0.0392, "ic_std": 0.2, "n_days_used": 100, "ic_ir": 0.2,
                                  "quantile_long_short": {"mean_spread": 0.0016},
                                  "argmax_long_short": {"spread": 0.0}}, "metrics": {"macro_f1": 0.33}},
          "best_epoch": 3, "train": {"last_epoch": 7}}
    md = e4.render_doc({}, {"v3_wd": v3}, "abc123")
    assert "V3 seed 0" in md and "0.0412" in md and "_pending_" in md and "abc123" in md
    assert "volatility_20d" in md and "sector_volatility" in md and "ONCE" in md and "2025" in md
    assert "-0.0253" in md and "-0.0393" in md                                  # V3 timing IC quoted from the diagnosis doc
    assert "min train_loss" in md and "1.0986" in md                            # collapse-diagnosis column + ln(3) reference
    for n in COLLAPSE_DIAG_6:                                                   # pending rows for all 6 new recipes too
        assert n in md
    assert "v3_structure_nowd" in md and "V3wd0" in md
    for n in V3_STRUCTURE_FOLLOWUPS_8:                                         # pending rows for the 8 follow-ups too
        assert n in md and e4.RECIPES[n]["tag"] in md
    # weight_decay/patience/dropout overrides are visible in the table's description cell when non-default
    assert "weight_decay=1e-4" in md and "weight_decay=3e-4" in md
    assert "patience=8" in md
    assert "dropout=0.30" in md
    # the recipe-table row's label cell must say it's V3's fixed label, NOT be mislabeled "label_vn" (the old
    # two-way "cs" vs "label_vn" ternary would have silently mislabeled any non-cs, non-label_vn recipe).
    row = next(l for l in md.splitlines() if l.startswith("| v3_structure_nowd |"))
    assert "| fixed label (V3) |" in row and "label_vn" not in row
    md2 = e4.render_doc({}, {}, "abc")
    assert "V3 seed 0" in md2 and "not found" in md2


def test_render_doc_shows_min_train_loss_and_flags_a_collapsed_curve():
    results = {"x2_cslabel_nostatic": {
        "best_epoch": 0, "best_val_timing_ic": 0.0153,
        "train": {"last_epoch": 4, "stopped_reason": "early_stopping", "class_weights": [1.11, 0.83, 1.11],
                  "n_train_samples": 100, "n_glitch_masked": 0},
        "epoch_curve": [{"epoch": 0, "train_loss": 1.15, "val_loss": 1.0993, "val_timing_ic": 0.0153, "val_ic": -0.0003},
                        {"epoch": 1, "train_loss": 1.0986, "val_loss": 1.1006, "val_timing_ic": 0.0132, "val_ic": -0.0539},
                        {"epoch": 2, "train_loss": 1.0986, "val_loss": 1.0999, "val_timing_ic": 0.0051, "val_ic": -0.0539}],
        "val_2024": {"decomposition": {"raw_ic": -0.0003, "raw_ic_se": 0.0067, "raw_ic_ir": -0.004,
                                       "fixed_effect_ic": -0.008, "timing_ic": 0.0153, "timing_ic_se": 0.0059,
                                       "timing_ic_ir": 0.19, "n_days_raw": 185, "timing_quantile_ls": 0.001},
                    "signal": {"quantile_long_short": {"mean_spread": -0.0004}}, "metrics": {"macro_f1": 0.24}, "n_label_ok": 36929},
        "oot_2026": {"decomposition": {"raw_ic": 0.0314, "raw_ic_se": 0.0091, "raw_ic_ir": 0.267,
                                       "fixed_effect_ic": 0.0044, "timing_ic": 0.0237, "timing_ic_se": 0.0123,
                                       "timing_ic_ir": 0.149, "n_days_raw": 167, "timing_quantile_ls": 0.0004},
                    "signal": {"quantile_long_short": {"mean_spread": 0.0028}}, "metrics": {"macro_f1": 0.15}, "n_label_ok": 33364},
    }}
    md = e4.render_doc(results, {}, "abc123", recipes=["x2_cslabel_nostatic"])
    assert "1.0986" in md                                                       # min train_loss column shows the collapse
    assert "tloss 1.1500" in md and "tloss 1.0986" in md                        # per-epoch curve carries train_loss too
    assert e4._min_train_loss(results["x2_cslabel_nostatic"]) == pytest.approx(1.0986)
