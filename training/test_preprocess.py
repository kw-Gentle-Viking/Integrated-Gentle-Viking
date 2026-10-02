import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from training import preprocess as pp


def _frames(seed=0, n=400, tickers=("A", "B")):
    rng = np.random.default_rng(seed)
    out = {}
    for tk in tickers:
        dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)]
        out[tk] = pd.DataFrame({
            "ticker": tk, "trade_date": dates,
            "cont": rng.standard_normal(n) * 3 + 10,
            "flag": (rng.random(n) < 0.004).astype(float),   # rare binary: 0.5/99.5 clip would kill it
            "dead": np.zeros(n),
            "const1": np.ones(n),
            "label": 1.0,
        })
    return out


COLS = ["cont", "flag", "dead", "const1"]


def test_binary_untouched_and_rare_flag_survives():
    fr = _frames()
    art = pp.fit_preprocessor(fr, COLS)
    assert art["columns"]["flag"]["kind"] == "binary"
    out = pp.apply_preprocessor(fr, art)
    for tk in fr:
        assert np.array_equal(out[tk]["flag"].values, fr[tk]["flag"].values)
    assert sum(o["flag"].sum() for o in out.values()) > 0


def test_constant_columns_dropped_including_constant_one():
    art = pp.fit_preprocessor(_frames(), COLS)
    assert art["columns"]["dead"]["kind"] == "dropped_constant"
    assert art["columns"]["const1"]["kind"] == "dropped_constant"
    assert art["kept_columns"] == ["cont", "flag"]
    assert art["dropped_columns"] == ["dead", "const1"]


def test_train_posttransform_mean0_std1_for_continuous():
    fr = _frames()
    art = pp.fit_preprocessor(fr, COLS)
    out = pp.apply_preprocessor(fr, art)
    x = np.concatenate([o["cont"].values for o in out.values()])
    assert abs(x.mean()) < 1e-9
    assert abs(x.std() - 1.0) < 1e-9


def test_oot_beyond_bounds_clipped_no_blowup():
    fr = _frames()
    art = pp.fit_preprocessor(fr, COLS)
    oot = {"A": pd.DataFrame({"ticker": "A", "trade_date": [dt.date(2026, 1, 1)] * 3,
                              "cont": [1e9, -1e9, 10.0], "flag": [0.0, 1.0, 0.0],
                              "dead": 0.0, "const1": 1.0, "label": 1.0})}
    z = pp.apply_preprocessor(oot, art)["A"]["cont"].values
    lo, hi = art["columns"]["cont"]["lo"], art["columns"]["cont"]["hi"]
    m, s = art["columns"]["cont"]["mean"], art["columns"]["cont"]["std"]
    assert z[0] == pytest.approx((hi - m) / s) and z[1] == pytest.approx((lo - m) / s)
    assert np.abs(z).max() < 10


def test_fit_idempotent_and_json_serialisable():
    fr = _frames()
    a1, a2 = pp.fit_preprocessor(fr, COLS), pp.fit_preprocessor(fr, COLS)
    assert a1 == a2
    assert json.loads(json.dumps(a1)) == a1


def test_fit_depends_only_on_passed_data():
    fr = _frames()
    a_before = pp.fit_preprocessor(fr, COLS)
    other = _frames(seed=99)
    for f in other.values():
        f["cont"] = f["cont"] * 1000          # wildly different "val/OOT" data
    pp.apply_preprocessor(other, a_before)    # applying must not change artifact
    assert pp.fit_preprocessor(fr, COLS) == a_before
    assert pp.fit_preprocessor(other, COLS) != a_before


def test_fit_cutoff_guard_rejects_rows_after_cutoff():
    fr = _frames(n=400)   # dates span into 2021
    with pytest.raises(ValueError, match="cutoff"):
        pp.fit_preprocessor(fr, COLS, max_date="2020-06-01")
    art = pp.fit_preprocessor(fr, COLS, max_date="2021-12-31")
    assert art["fit_max_date"] <= "2021-12-31"


def test_apply_does_not_mutate_input_and_fills_nan_with_zero():
    fr = _frames()
    art = pp.fit_preprocessor(fr, COLS)
    test = {"A": fr["A"].iloc[:5].copy()}
    test["A"].loc[test["A"].index[0], "cont"] = np.nan
    test["A"].loc[test["A"].index[1], "flag"] = np.nan
    snap = test["A"].copy()
    out = pp.apply_preprocessor(test, art)["A"]
    pd.testing.assert_frame_equal(test["A"], snap)      # input untouched
    assert out["cont"].iloc[0] == 0.0 and out["flag"].iloc[1] == 0.0


def test_sparse_continuous_degenerate_quantiles_fall_back_not_dropped():
    n = 2000
    x = np.zeros(n); x[:3] = [1, 2, 5]           # >99.5% zeros: quantile clip would flatten it
    fr = {"A": pd.DataFrame({"ticker": "A", "trade_date": [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)],
                             "sp": x})}
    art = pp.fit_preprocessor(fr, ["sp"])
    assert art["columns"]["sp"]["kind"] == "continuous"
    assert art["columns"]["sp"]["std"] > 0
    assert art["columns"]["sp"]["hi"] == 5.0    # fell back to unclipped max


def test_save_load_roundtrip(tmp_path):
    art = pp.fit_preprocessor(_frames(), COLS)
    p = tmp_path / "preproc_x.json"
    pp.save_preprocessor(art, str(p))
    assert pp.load_preprocessor(str(p)) == art


# ---- cross-sectional rank inputs (R3) -------------------------------------------------------

def _xs_frames():
    d = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(3)]
    def mk(tk, vals, dates=d):
        return pd.DataFrame({"ticker": tk, "trade_date": dates, "log_ret": vals, "other": 1.0})
    return {"A": mk("A", [1.0, 5.0, 3.0]), "B": mk("B", [2.0, 4.0, 3.0]), "C": mk("C", [3.0, 3.0, 3.0])}


def test_cs_rank_range_and_centering():
    out = pp.add_cs_rank_inputs(_xs_frames(), ["log_ret"])
    r = pd.concat(out.values())
    assert (r["log_ret_csr"].abs() < 0.5).all()
    for _, g in r.groupby("trade_date"):
        assert g["log_ret_csr"].mean() == pytest.approx(0.0, abs=1e-12)
    d0 = r[r.trade_date == dt.date(2024, 1, 1)].set_index("ticker")["log_ret_csr"]
    assert d0["A"] < d0["B"] < d0["C"]
    # ties (all 3.0 on last date) -> identical rank 0
    d2 = r[r.trade_date == dt.date(2024, 1, 3)]["log_ret_csr"]
    assert (d2 == 0).all()


def test_cs_rank_per_date_independence():
    base = pp.add_cs_rank_inputs(_xs_frames(), ["log_ret"])
    fr = _xs_frames()
    fr["A"].loc[0, "log_ret"] = 1e6       # change date 0 only
    fr["B"].loc[2, "log_ret"] = -1e6      # change date 2 only
    chg = pp.add_cs_rank_inputs(fr, ["log_ret"])
    for tk in "ABC":
        assert base[tk]["log_ret_csr"].iloc[1] == chg[tk]["log_ret_csr"].iloc[1]   # date 1 unaffected


def test_cs_rank_missing_dates_and_nan_and_input_untouched():
    fr = _xs_frames()
    fr["B"] = fr["B"].iloc[[0, 2]].reset_index(drop=True)      # B lacks date index 1
    fr["C"].loc[0, "log_ret"] = np.nan
    snap = {k: v.copy() for k, v in fr.items()}
    out = pp.add_cs_rank_inputs(fr, ["log_ret"])
    for k in fr:
        pd.testing.assert_frame_equal(fr[k], snap[k])
        assert len(out[k]) == len(fr[k])
    # date 1: B absent -> A(5.0) and C(3.0) ranked among 2 names only
    assert out["A"]["log_ret_csr"].iloc[1] == pytest.approx(0.25)
    assert out["C"]["log_ret_csr"].iloc[1] == pytest.approx(-0.25)
    # NaN input -> NaN rank on that row; other names ranked among 2
    assert np.isnan(out["C"]["log_ret_csr"].iloc[0])
    assert out["A"]["log_ret_csr"].iloc[0] == pytest.approx(-0.25) and out["B"]["log_ret_csr"].iloc[0] == pytest.approx(0.25)


def test_cs_rank_uses_tabular_features_builder():
    """Ranks must equal E0's add_cs_ranks percentile shifted to be centred (same builder)."""
    from training.tabular_features import add_cs_ranks
    fr = _xs_frames()
    out = pp.add_cs_rank_inputs(fr, ["log_ret"])
    flat = add_cs_ranks(pd.concat(fr.values()), ["log_ret"])
    n = flat.groupby("trade_date")["log_ret"].transform("count")
    expect = (flat["log_ret_csr"] - (n + 1) / (2 * n)).values
    got = pd.concat(out.values())["log_ret_csr"].values
    assert np.allclose(got, expect)
