import datetime as dt
import json
import os

import numpy as np
import pandas as pd
import pytest

import training.run_tfx_experiments as tfx
from training.config import HISTORICAL_COLS_DEFAULT, STATIC_COLS

DEAD = ["lev_total_aum", "lev_aum_to_mktcap", "est_rebalancing_flow"]


def _bdays(start, n):
    d, out = dt.date.fromisoformat(start), []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


class FakeData:
    """Synthetic stand-in for RealData: same interface, real column names, dates placed so the
    S3 windows (val 2024 targets, OOT 2026 targets) are exercised by the real window logic."""
    SPEC = {"train": ("2023-03-01", 200), "val_2024": ("2024-01-01", 130), "oot_2026": ("2025-09-01", 150)}
    TICKERS = ["T1", "T2", "T3", "T4", "T5", "T6"]

    def __init__(self):
        self.expected_samples = {}
        self.calls = []

    def frames(self, split, label_col):
        self.calls.append((split, label_col))
        start, n = self.SPEC[split]
        dates = _bdays(start, n)
        out = {}
        for i, tk in enumerate(self.TICKERS):
            rng = np.random.default_rng(1000 * i + len(split))     # same across label_col (same features)
            df = pd.DataFrame({"ticker": tk, "trade_date": dates})
            for c in HISTORICAL_COLS_DEFAULT:
                df[c] = rng.standard_normal(n) * 0.02
            for c in DEAD:
                df[c] = 0.0
            df["is_dividend"] = (rng.random(n) < 0.05).astype(float)
            df["day_of_week"] = [d.weekday() for d in dates]
            for c in ("is_bok", "is_fomc", "is_witching_kr", "is_witching_us"):
                df[c] = 0.0
            df["sector_id"], df["market_id"], df["time_progress"] = i % 3, i % 2, 1.0
            lrng = np.random.default_rng(7 * i + (0 if label_col == "label" else 1))
            lab = lrng.integers(0, 3, n).astype(float)
            lab[-1] = np.nan                                       # last row: no next day
            if label_col == "label_vn":
                lab[100:104] = np.nan                                # vol-missing rows: NULL only for label_vn
            df["label"] = lab
            out[tk] = df
        return out

    def next_day_returns(self, split):
        start, n = self.SPEC[split]
        lk = {}
        for i, tk in enumerate(self.TICKERS):
            rng = np.random.default_rng(5 + i + len(split))
            r = rng.standard_normal(n) * 0.02
            if split == "train":
                r[20] = 0.5       # glitch row -> label masked in training
            for d, v in zip(_bdays(start, n), r):
                lk[(tk, d.strftime("%Y-%m-%d"))] = float(v)
        return lk


def _expected_counts(data, off=59):
    """Independent count of samples: target row index >= off (59 = align today, 60 = legacy), fixed label
    not NaN, target date >= min_target."""
    out = {}
    for w in tfx.EVAL_WINDOWS:
        fr = data.frames(w, "label")
        n = 0
        for df in fr.values():
            for t in range(len(df) - off):
                row = df.iloc[t + off]
                if not np.isnan(row["label"]) and row["trade_date"].strftime("%Y-%m-%d") >= tfx.SPLITS[w]["min_target"]:
                    n += 1
        out[w] = n
    return out


# ---------------- pure helpers ----------------
def test_attach_and_mask_glitch_labels_only_touches_large_moves():
    df = pd.DataFrame({"ticker": "A", "trade_date": [dt.date(2020, 1, 1), dt.date(2020, 1, 2), dt.date(2020, 1, 3)],
                       "label": [0.0, 1.0, 2.0]})
    fr = tfx.attach_next_day_return({"A": df}, {("A", "2020-01-01"): 0.05, ("A", "2020-01-02"): -0.35})
    assert np.isnan(fr["A"]["next_day_return"].iloc[2])
    out, n = tfx.mask_glitch_labels(fr)
    assert n == 1 and out["A"]["label"].isna().tolist() == [False, True, False]   # NaN return not masked
    assert fr["A"]["label"].notna().all()                                        # input untouched


def test_build_eval_frames_sample_set_equals_fixed_label_set():
    d = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(4)]
    base = pd.DataFrame({"ticker": "A", "trade_date": d, "label": [0.0, 1.0, np.nan, 2.0]})
    rec = pd.DataFrame({"ticker": "A", "trade_date": d, "label": [2.0, np.nan, 0.0, 1.0]})
    out = tfx.build_eval_frames({"A": base}, {"A": rec})["A"]
    assert out["label"].notna().tolist() == [True, True, False, True]        # gate = fixed label
    assert out["label"].tolist()[0] == 2.0 and out["label"].tolist()[3] == 1.0
    assert out["label_ok"].tolist() == [True, False, False, True]           # NaN vn label / gated-out rows not ok
    with pytest.raises(ValueError):
        tfx.build_eval_frames({"A": base}, {"B": rec})


def test_clip_returns_and_pending_and_results_roundtrip(tmp_path):
    assert tfx.clip_returns([0.9, -0.9, 0.1]).tolist() == [0.3, -0.3, 0.1]
    assert tfx.pending_recipes({"std": {}}, ["std", "std_vn"]) == ["std_vn"]
    p = str(tmp_path / "r.json")
    tfx.save_results_atomic(p, {"a": 1})
    assert tfx.load_results(p) == {"a": 1} and not os.path.exists(p + ".tmp")
    assert tfx.load_results(str(tmp_path / "missing.json")) == {}


def test_model_versions_row_exists(tmp_path):
    p = tmp_path / "mv.md"
    p.write_text("| tfx-std | x |\n")
    assert tfx.model_versions_row_exists(str(p), "tfx-std")
    assert not tfx.model_versions_row_exists(str(p), "tfx-std_vn")
    assert not tfx.model_versions_row_exists(str(tmp_path / "none.md"), "tfx-std")


def test_read_reference_rows_real_files_if_present_and_tolerates_missing(tmp_path):
    assert tfx.read_reference_rows(str(tmp_path)) == {"val_2024": [], "oot_2026": []}
    if os.path.exists("training/artifacts/signal_baseline.json"):
        r = tfx.read_reference_rows("training/artifacts")
        names = [n for n, _, _ in r["val_2024"]]
        assert any(n.startswith("TFT champion (S3") for n in names) and any(n.startswith("1-day reversal (S3") for n in names) and any(n.startswith("HGB reg (z), F3 (E0") for n in names)


def test_render_doc_with_no_results_has_selection_note():
    md = tfx.render_doc({}, {"val_2024": [], "oot_2026": []}, "abc123")
    assert "Selection by val IC; OOT is confirmation only" in md and "abc123" in md and "_pending_" in md


# ---------------- end-to-end dry run on synthetic data (CPU) ----------------
def _args(tmp, *extra):
    return ["--artifacts-dir", str(tmp / "art"), "--doc-path", str(tmp / "doc.md"),
            "--model-versions-path", str(tmp / "mv.md"), "--ref-dir", str(tmp / "noref"),
            "--epochs", "2", "--patience", "3", "--batch-size", "64", "--state-size", "8",
            "--no-wandb", "--device", "cpu", *extra]


@pytest.fixture
def small_min_names(monkeypatch):
    monkeypatch.setattr(tfx, "MIN_NAMES_PER_DAY", 2)


def test_window_assertion_fails_before_training_on_count_mismatch(tmp_path, monkeypatch, small_min_names):
    data = FakeData()
    data.expected_samples = {"val_2024": 1, "oot_2026": 1}
    data.force_legacy_assert = True
    import training.train as train_mod
    monkeypatch.setattr(train_mod, "run_training", lambda *a, **k: pytest.fail("training must not start"))
    with pytest.raises(AssertionError, match="expected 1"):
        tfx.main(_args(tmp_path, "--recipes", "std", "--align", "legacy"), data=data)


def test_check_windows_aligned_does_not_assert_legacy_counts(tmp_path, small_min_names):
    data = FakeData()
    data.expected_samples = {"val_2024": 1, "oot_2026": 1}       # would fail if compared for align=today
    assert tfx.main(_args(tmp_path, "--check-windows"), data=data) == 0


def test_check_windows_legacy_reproduces_expected_counts_and_rejects_mismatch(tmp_path, small_min_names):
    data = FakeData()
    data.expected_samples = _expected_counts(data, off=60)
    assert min(data.expected_samples.values()) > 50
    assert tfx.main(_args(tmp_path, "--check-windows", "--align", "legacy"), data=data) == 0
    aligned = _expected_counts(data, off=59)
    # aligned sets gain the first window per ticker where it falls inside the window (val), never fewer
    assert aligned["val_2024"] > data.expected_samples["val_2024"] and aligned["oot_2026"] >= data.expected_samples["oot_2026"]
    data.expected_samples = {"val_2024": 1, "oot_2026": 1}
    with pytest.raises(AssertionError, match="expected 1"):
        tfx.main(_args(tmp_path, "--check-windows", "--align", "legacy"), data=data)


def test_aligned_sample_dates_are_last_encoder_row_dates_inside_windows(tmp_path, small_min_names):
    from training.config import KNOWN_FUTURE_COLS, STATIC_COLS
    data = FakeData()
    opts = {"align": "today", "max_tickers": None}
    prep = tfx.prepare_recipe("aligned", data, {"champion": {"columns": HISTORICAL_COLS_DEFAULT}, "max_tickers": None})
    for w in tfx.EVAL_WINDOWS:
        ds, meta = tfx.build_window_dataset(w, prep["eval_frames"][w], prep["kept_columns"], opts)
        for (tk, t), (_, d, _) in zip(ds.index, meta):
            row = ds.ticker_dfs[tk].iloc[t + 59]
            assert d == row["trade_date"].strftime("%Y-%m-%d")           # last encoder row
        assert min(m[1] for m in meta) >= tfx.SPLITS[w]["min_target"]
        assert not any("2025-01-01" <= m[1] <= "2025-12-31" for m in meta)


def test_full_dry_run_interrupt_resume_and_skip(tmp_path, monkeypatch, small_min_names):
    import training.run_stage1_search as rss
    import training.train as train_mod
    data = FakeData()
    data.expected_samples = _expected_counts(data)

    # --- invocation 1: budget expires after epoch 0 of the first recipe -> resumable, nothing recorded
    seq = iter([False, True])          # top-of-loop check, then the after-epoch-0 check

    class ScriptedBudget:
        def __init__(self, *_):
            pass

        def exceeded(self):
            return next(seq, False)

    monkeypatch.setattr(rss, "Budget", ScriptedBudget)
    assert tfx.main(_args(tmp_path), data=data) == 0
    art = tmp_path / "art"
    assert (art / "checkpoints" / "tfx-aligned_inprogress.pt").exists()
    assert not (art / "tfx_results.json").exists()
    assert not (art / "preproc_aligned.json").exists()        # R0: raw inputs, no artifact
    assert not (tmp_path / "mv.md").exists()

    # --- invocation 2: hard crash during validation of the resumed epoch -> exception, state kept
    monkeypatch.setattr(rss, "Budget", lambda *_: type("B", (), {"exceeded": lambda self: False})())
    real_eval = train_mod.evaluate_loss
    monkeypatch.setattr(train_mod, "evaluate_loss", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kill")))
    with pytest.raises(RuntimeError, match="kill"):
        tfx.main(_args(tmp_path), data=data)
    monkeypatch.setattr(train_mod, "evaluate_loss", real_eval)

    # --- invocation 3: resumes aligned from epoch 1 (only ONE epoch trains), then runs the other 3 recipes
    real_train, calls = train_mod.train_one_epoch, {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return real_train(*a, **k)

    monkeypatch.setattr(train_mod, "train_one_epoch", counting)
    assert tfx.main(_args(tmp_path), data=data) == 0
    assert calls["n"] == 1 + 2 + 2 + 2      # aligned resumed (1 epoch) + 3 fresh recipes x 2
    res = json.load(open(art / "tfx_results.json"))
    assert set(res) == {"aligned", "std", "std_vn", "std_vn_csr"}
    for name, r in res.items():
        assert r["train"]["align"] == "today" and r["align"] == "today"
        assert r["train"]["dropped_columns"] == ([] if name == "aligned" else DEAD)
        assert r["train"]["n_inputs"] == {"aligned": 33, "std": 30, "std_vn": 30, "std_vn_csr": 36}[name]
        assert r["train"]["n_glitch_masked"] >= 1
        for w in tfx.EVAL_WINDOWS:
            assert r[w]["n_samples"] == data.expected_samples[w]
            assert r[w]["signal"]["n_days_used"] > 0 and r[w]["signal_clip30"] is not None
    # label_vn recipes: eval sample set unchanged but fewer samples carry a genuine label
    assert res["std"]["val_2024"]["n_label_ok"] == res["std"]["val_2024"]["n_samples"]
    assert res["aligned"]["val_2024"]["n_samples"] == res["std"]["val_2024"]["n_samples"] == res["std_vn"]["val_2024"]["n_samples"]
    assert res["std_vn"]["val_2024"]["n_label_ok"] < res["std_vn"]["val_2024"]["n_samples"]
    # preprocessing artifacts: R3 has rank cols, all fitted only on train (max date <= cutoff)
    from training.preprocess import load_preprocessor
    a3 = load_preprocessor(str(art / "preproc_std_vn_csr.json"))
    assert "log_ret_csr" in a3["kept_columns"] and a3["fit_max_date"] <= tfx.FIT_CUTOFF
    doc = (tmp_path / "doc.md").read_text()
    for n in ("aligned", "std", "std_vn", "std_vn_csr"):
        assert f"**{tfx.RECIPES[n]['tag']} {n}**" in doc
    assert "OOT is confirmation only" in doc and "lev_total_aum" in doc and "align" in doc and "not like-for-like" in doc
    mv = (tmp_path / "mv.md").read_text()
    assert [l.split("|")[1].strip() for l in mv.splitlines() if l.startswith("| tfx-")] == ["tfx-aligned", "tfx-std", "tfx-std_vn", "tfx-std_vn_csr"]

    # --- invocation 4: everything recorded -> nothing trains, no duplicate rows
    calls["n"] = 0
    assert tfx.main(_args(tmp_path), data=data) == 0
    assert calls["n"] == 0
    assert (tmp_path / "mv.md").read_text() == mv


def test_recipes_filter_only_runs_requested(tmp_path, monkeypatch, small_min_names):
    data = FakeData()
    data.expected_samples = _expected_counts(data)
    tfx.main(_args(tmp_path, "--recipes", "std_vn"), data=data)
    assert set(json.load(open(tmp_path / "art" / "tfx_results.json"))) == {"std_vn"}


# ---------------- fingerprint / meta / dry-run guard ----------------
def test_max_tickers_requires_explicit_artifacts_dir(tmp_path):
    with pytest.raises(SystemExit):
        tfx.parse_args(["--max-tickers", "3"])
    assert tfx.parse_args(["--max-tickers", "3", "--artifacts-dir", str(tmp_path)]).artifacts_dir == str(tmp_path)
    assert tfx.parse_args([]).artifacts_dir == tfx.ARTIFACTS_DIR


def test_fingerprint_helpers():
    opts = {"hparams": {"lr": 1e-3}, "batch_size": 8, "seed": 3, "align": "today"}
    fp = tfx.full_fingerprint("std_vn", opts, ["a", "b"], [1.0, 2.0, 3.0])
    assert fp["label_col"] == "label_vn" and fp["threshold"]["file"] == "threshold_vn.json" and "k" in fp["threshold"]
    assert tfx.full_fingerprint("std", opts, ["a", "b"], [1, 2, 3])["threshold"]["file"] == "threshold.json"
    assert fp["columns_hash"] != tfx.full_fingerprint("std_vn", opts, ["b", "a"], [1, 2, 3])["columns_hash"]
    assert tfx.fingerprint_mismatch(fp, fp) == []
    assert tfx.fingerprint_mismatch(None, fp) == sorted(fp)
    assert tfx.fingerprint_mismatch(fp, {**fp, "seed": 4}) == ["seed"]


def test_meta_and_fingerprint_recorded_and_stale_inprogress_and_result_are_handled(tmp_path, monkeypatch, small_min_names):
    import hashlib
    data = FakeData()
    data.expected_samples = _expected_counts(data)
    assert tfx.main(_args(tmp_path, "--recipes", "std", "--seed", "3"), data=data) == 0
    art = tmp_path / "art"
    res = json.load(open(art / "tfx_results.json"))
    r = res["std"]
    m = r["meta"]
    assert m["seed"] == 3 and m["data_version"] == r["fingerprint"]["data_version"] and m["threshold"]["file"] == "threshold.json"
    assert m["columns_hash"] == tfx.columns_hash(r["train"]["kept_columns"]) == r["fingerprint"]["columns_hash"]
    assert m["preproc"]["sha256"] == hashlib.sha256((art / "preproc_std.json").read_bytes()).hexdigest()
    assert m["resumed"] is False and m["resume_count"] == 0 and m["torch"] and "cuda" in m
    assert r["fingerprint"]["class_weights"] == [round(w, 6) for w in r["train"]["class_weights"]]

    # same setup -> skipped (no training)
    import training.train as train_mod
    real, calls = train_mod.train_one_epoch, {"n": 0}
    monkeypatch.setattr(train_mod, "train_one_epoch", lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), real(*a, **k))[1])
    tfx.main(_args(tmp_path, "--recipes", "std", "--seed", "3"), data=data)
    assert calls["n"] == 0
    # different seed -> recorded result is stale: moved aside and re-run
    tfx.main(_args(tmp_path, "--recipes", "std", "--seed", "4"), data=data)
    assert calls["n"] == 2
    assert (art / "tfx_results.json.stale.std.json").exists()
    assert json.load(open(art / "tfx_results.json"))["std"]["meta"]["seed"] == 4


def test_resume_meta_and_stale_inprogress_on_changed_columns(tmp_path, monkeypatch, small_min_names):
    import training.run_stage1_search as rss
    data = FakeData()
    data.expected_samples = _expected_counts(data)
    seq = iter([False, True])
    monkeypatch.setattr(rss, "Budget", type("SB", (), {"__init__": lambda s, *_: None, "exceeded": lambda s: next(seq, False)}))
    tfx.main(_args(tmp_path, "--recipes", "aligned"), data=data)
    ck = tmp_path / "art" / "checkpoints" / "tfx-aligned_inprogress.pt"
    assert ck.exists()
    # resume with an identical setup -> meta says resumed
    monkeypatch.setattr(rss, "Budget", lambda *_: type("B", (), {"exceeded": lambda self: False})())
    tfx.main(_args(tmp_path, "--recipes", "aligned"), data=data)
    m = json.load(open(tmp_path / "art" / "tfx_results.json"))["aligned"]["meta"]
    assert m["resumed"] is True and m["resume_count"] == 1 and m["resumed_from_epoch"] == 1

    # interrupted again, but the data version changes between runs -> in-progress ignored (.stale)
    (tmp_path / "art" / "tfx_results.json").unlink()
    monkeypatch.setattr(rss, "Budget", type("SB", (), {"__init__": lambda s, *_: None, "exceeded": lambda s: next(seq, False)}))
    seq2 = iter([False, True])
    monkeypatch.setattr(rss, "Budget", type("SB2", (), {"__init__": lambda s, *_: None, "exceeded": lambda s: next(seq2, False)}))
    tfx.main(_args(tmp_path, "--recipes", "aligned"), data=data)
    assert ck.exists()
    import training.stage1_data as sd
    monkeypatch.setattr(sd, "DATA_VERSION", "adj2")
    monkeypatch.setattr(rss, "Budget", lambda *_: type("B", (), {"exceeded": lambda self: False})())
    tfx.main(_args(tmp_path, "--recipes", "aligned"), data=data)
    assert (ck.parent / "tfx-aligned_inprogress.pt.stale").exists()
    m = json.load(open(tmp_path / "art" / "tfx_results.json"))["aligned"]["meta"]
    assert m["resumed"] is False and m["data_version"] == "adj2" and m["stale_inprogress_ignored"]


def test_git_dirty_check_covers_data_and_label_files():
    import inspect
    src = inspect.getsource(tfx._git_commit)
    for f in ("dataset.py", "config.py", "label.py", "preprocess.py", "train.py"):
        assert f"training/{f}" in src
