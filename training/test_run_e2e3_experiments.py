import json
import math

import pytest

import training.run_e2e3_experiments as e2
import training.run_tfx_experiments as tfx
from training.test_run_tfx_experiments import FakeData, _expected_counts


# ---------------- pure helpers ----------------
def test_recipe_table_and_priority_order():
    assert e2.DEFAULT_ORDER == ["e3_seed1", "e3_seed2", "v4_lr_do", "v1_lr", "v2_do", "v3_wd", "v5_lr_do_wd",
                               "v3_wd_seed1", "v4_lr_do_seed1", "v3_wd_seed2", "v4_lr_do_seed2"]
    assert e2.RECIPES["v4_lr_do"]["overrides"] == {"lr": 1e-4, "dropout": 0.3}
    assert e2.RECIPES["v5_lr_do_wd"]["overrides"] == {"lr": 1e-4, "dropout": 0.3, "weight_decay": 1e-3}
    assert e2.RECIPES["v6_state16"]["overrides"] == {"state_size": 16} and "v6_state16" not in e2.DEFAULT_ORDER
    assert [e2.RECIPES[n]["seed"] for n in ("e3_seed1", "e3_seed2")] == [1, 2]
    assert all(e2.RECIPES[n]["seed"] == 0 and e2.RECIPES[n]["select"] == "val_ic" for n in e2.DEFAULT_ORDER[2:7])
    assert all(e2.RECIPES[n]["select"] == "val_loss" for n in ("e3_seed1", "e3_seed2"))
    # v3_wd / v4_lr_do seed-1/2 repeats: same overrides as the seed-0 recipe, seed only differs, no
    # collision with the existing (already-run, do-not-retrain) seed-0 recipes or the E3 seed rows.
    assert e2.RECIPES["v3_wd_seed1"]["overrides"] == e2.RECIPES["v3_wd"]["overrides"] == {"weight_decay": 1e-3}
    assert e2.RECIPES["v3_wd_seed2"]["overrides"] == e2.RECIPES["v3_wd"]["overrides"]
    assert e2.RECIPES["v4_lr_do_seed1"]["overrides"] == e2.RECIPES["v4_lr_do"]["overrides"] == {"lr": 1e-4, "dropout": 0.3}
    assert e2.RECIPES["v4_lr_do_seed2"]["overrides"] == e2.RECIPES["v4_lr_do"]["overrides"]
    assert [e2.RECIPES[n]["seed"] for n in ("v3_wd_seed1", "v4_lr_do_seed1", "v3_wd_seed2", "v4_lr_do_seed2")] == [1, 1, 2, 2]
    assert all(e2.RECIPES[n]["group"] == "E2" and e2.RECIPES[n]["select"] == "val_ic" and e2.RECIPES[n]["patience"] == e2.E2_PATIENCE
              for n in ("v3_wd_seed1", "v4_lr_do_seed1", "v3_wd_seed2", "v4_lr_do_seed2"))
    assert e2.DEFAULT_ORDER[7:] == ["v3_wd_seed1", "v4_lr_do_seed1", "v3_wd_seed2", "v4_lr_do_seed2"]
    assert len({e2.RECIPES[n]["tag"] for n in e2.RECIPES}) == len(e2.RECIPES)          # tags unique (doc table)
    assert e2.RECIPES["v3_wd"]["seed"] == 0 and e2.RECIPES["v4_lr_do"]["seed"] == 0    # untouched, already run


def test_recipe_opts_merge_and_fingerprint_distinguishes_variants():
    base = {"hparams": {"lr": 3e-4, "dropout": 0.17, "state_size": 32}, "batch_size": 8, "seed": 0, "align": "today"}
    v = e2.recipe_opts("v4_lr_do", base)
    assert v["hparams"] == {"lr": 1e-4, "dropout": 0.3, "state_size": 32} and v["seed"] == 0 and v["patience"] == 4
    assert base["hparams"]["lr"] == 3e-4                                        # base not mutated
    s1 = e2.recipe_opts("e3_seed1", base)
    assert s1["hparams"] == base["hparams"] and s1["seed"] == 1 and s1["patience"] == 3 and s1["select"] == "val_loss"
    assert e2.recipe_opts("v1_lr", {**base, "state_size_override": 8})["hparams"]["state_size"] == 8
    fps = {n: e2.cheap_fingerprint(n, e2.recipe_opts(n, base)) for n in e2.RECIPES}
    assert len({json.dumps(f, sort_keys=True) for f in fps.values()}) == len(e2.RECIPES)
    assert fps["v3_wd"]["hparams"]["weight_decay"] == 1e-3 and fps["v3_wd"]["selection"]["mode"] == "val_ic"
    assert fps["e3_seed1"]["seed"] == 1 and fps["e3_seed1"]["label_col"] == "label" and fps["e3_seed1"]["preprocessed"] is False
    full = e2.full_fingerprint("v1_lr", e2.recipe_opts("v1_lr", base), ["a"], [1, 2, 3])
    assert full["columns_hash"] and full["class_weights"] == [1.0, 2.0, 3.0]


def test_mean_std_best_epoch_and_seed_rows():
    assert e2.mean_std([]) == (None, None) and e2.mean_std([0.5]) == (0.5, None)
    m, s = e2.mean_std([0.02, 0.04, None, float("nan")])
    assert m == pytest.approx(0.03) and s == pytest.approx(math.sqrt(0.0002))
    assert e2.best_epoch_of({"best_epoch": 2, "train": {}}) == 2
    assert e2.best_epoch_of({"patience": 3, "train": {"stopped_reason": "early_stopping", "last_epoch": 3}}) == 0
    assert e2.best_epoch_of({"patience": 3, "train": {"stopped_reason": "max_epochs", "last_epoch": 11}}) is None
    rows = e2.e3_seed_rows({"aligned": {"x": 0}}, {"e3_seed2": {"x": 2}, "v1_lr": {"x": 9}})
    assert [(s, r["x"]) for s, r in rows] == [(0, 0), (2, 2)]
    # v3_wd/v4_lr_do seed-1/2 repeats are group E2, not E3: they must not leak into the R0 seed-variance table.
    rows2 = e2.e3_seed_rows({"aligned": {"x": 0}},
                            {"v3_wd_seed1": {"x": 1}, "v4_lr_do_seed1": {"x": 1}, "e3_seed1": {"x": 1}})
    assert [s for s, _ in rows2] == [0, 1]


def test_render_doc_empty_states_discipline():
    md = e2.render_doc({}, {}, {"val_2024": [], "oot_2026": []}, "abc123")
    assert "ONLY" in md and "ONCE" in md and "never scored" in md and "abc123" in md and "_pending_" in md


# ---------------- CPU dry run on synthetic data ----------------
def _args(tmp, *extra):
    return ["--artifacts-dir", str(tmp / "art"), "--doc-path", str(tmp / "doc.md"),
            "--model-versions-path", str(tmp / "mv.md"), "--ref-dir", str(tmp / "noref"),
            "--epochs", "2", "--batch-size", "64", "--state-size", "8", "--no-wandb", "--device", "cpu", *extra]


@pytest.fixture
def small_min_names(monkeypatch):
    monkeypatch.setattr(tfx, "MIN_NAMES_PER_DAY", 2)


def _no_budget(monkeypatch):
    import training.run_stage1_search as rss
    monkeypatch.setattr(rss, "Budget", lambda *_: type("B", (), {"exceeded": lambda self: False})())


def test_dry_run_interrupt_resume_idempotent(tmp_path, monkeypatch, small_min_names):
    import training.run_stage1_search as rss
    import training.train as train_mod
    data = FakeData()
    data.expected_samples = _expected_counts(data)
    art = tmp_path / "art"
    # a stand-in R0 seed-0 record so the doc has its baseline row
    (art).mkdir()
    tfx.main(["--artifacts-dir", str(art), "--doc-path", str(tmp_path / "tfxdoc.md"), "--model-versions-path",
              str(tmp_path / "tfxmv.md"), "--ref-dir", str(tmp_path / "noref"), "--epochs", "2", "--batch-size", "64",
              "--state-size", "8", "--no-wandb", "--device", "cpu", "--recipes", "aligned"], data=data)
    assert (art / "tfx_results.json").exists()

    # --- invocation 1: budget expires after epoch 0 of the first recipe -> resumable, nothing recorded
    seq = iter([False, True])

    class ScriptedBudget:
        def __init__(self, *_):
            pass

        def exceeded(self):
            return next(seq, False)

    monkeypatch.setattr(rss, "Budget", ScriptedBudget)
    a = _args(tmp_path, "--recipes", "e3_seed1", "v4_lr_do", "--ref-dir", str(art))
    assert e2.main(a, data=data) == 0
    assert (art / "checkpoints" / "e2e3-e3_seed1_inprogress.pt").exists()
    assert not (art / "e2e3_results.json").exists() and not (tmp_path / "mv.md").exists()

    # --- invocation 2: crash while validating the resumed epoch -> exception, state kept
    _no_budget(monkeypatch)
    real_eval = train_mod.evaluate_loss
    monkeypatch.setattr(train_mod, "evaluate_loss", lambda *x, **k: (_ for _ in ()).throw(RuntimeError("kill")))
    with pytest.raises(RuntimeError, match="kill"):
        e2.main(a, data=data)
    monkeypatch.setattr(train_mod, "evaluate_loss", real_eval)

    # --- invocation 3: resumes seed1 from epoch 1 (one epoch), then v4 fresh (val-IC selection, 2 epochs)
    real_train, calls = train_mod.train_one_epoch, {"n": 0}
    monkeypatch.setattr(train_mod, "train_one_epoch",
                        lambda *x, **k: (calls.__setitem__("n", calls["n"] + 1), real_train(*x, **k))[1])
    assert e2.main(a, data=data) == 0
    assert calls["n"] == 1 + 2
    res = json.load(open(art / "e2e3_results.json"))
    assert list(res) == ["e3_seed1", "v4_lr_do"]
    s1, v4 = res["e3_seed1"], res["v4_lr_do"]
    assert s1["meta"]["seed"] == 1 and s1["meta"]["resumed"] is True and s1["meta"]["resumed_from_epoch"] == 1
    assert s1["selection"] == "val_loss" and v4["selection"] == "val_ic" and v4["meta"]["seed"] == 0
    assert v4["hparams"]["lr"] == 1e-4 and v4["hparams"]["dropout"] == 0.3
    for r in (s1, v4):
        assert [h["epoch"] for h in r["epoch_curve"]] == [0, 1] and all("val_ic" in h and "val_loss" in h for h in r["epoch_curve"])
        assert r["train"]["align"] == "today" and r["train"]["n_inputs"] == 33 and r["train"]["preprocessed"] is False
        assert r["val_2024"]["signal"]["n_days_used"] > 0 and r["oot_2026"]["signal"]["n_days_used"] > 0
        assert "not a selection criterion" in r["oot_scored"]
    best_ic_epoch = max(v4["epoch_curve"], key=lambda h: h["val_ic"] if h["val_ic"] is not None else -9)["epoch"]
    assert v4["best_epoch"] == best_ic_epoch and v4["best_val_ic"] == v4["epoch_curve"][best_ic_epoch]["val_ic"]
    doc = (tmp_path / "doc.md").read_text()
    assert "R0 seed 0 (tfx)" in doc and "E3-s1" in doc and "V4" in doc and "Across 2 seed(s)" in doc
    mv = (tmp_path / "mv.md").read_text()
    assert [l.split("|")[1].strip() for l in mv.splitlines() if l.startswith("| e2e3-")] == ["e2e3-e3_seed1", "e2e3-v4_lr_do"]

    # --- invocation 4: everything recorded -> nothing trains, no duplicate rows, results untouched
    calls["n"] = 0
    before = (art / "e2e3_results.json").read_text()
    assert e2.main(a, data=data) == 0
    assert calls["n"] == 0 and (tmp_path / "mv.md").read_text() == mv and (art / "e2e3_results.json").read_text() == before

    # --- a changed setup invalidates only the affected record (stale file kept)
    a2 = _args(tmp_path, "--recipes", "v4_lr_do", "--ref-dir", str(art), "--batch-size", "32")
    e2.main(a2, data=data)
    assert (art / "e2e3_results.json.stale.v4_lr_do.json").exists() and calls["n"] == 2


def test_weight_decay_recipe_reaches_optimizer(tmp_path, monkeypatch, small_min_names):
    import torch
    _no_budget(monkeypatch)
    seen = []
    real = torch.optim.Adam
    monkeypatch.setattr(torch.optim, "Adam", lambda p, **kw: (seen.append(kw), real(p, **kw))[1])
    data = FakeData()
    data.expected_samples = _expected_counts(data)
    e2.main(_args(tmp_path, "--recipes", "v3_wd", "--epochs", "1"), data=data)
    assert seen and seen[0]["weight_decay"] == 1e-3


def test_seed_variants_of_leading_e2_candidates_run_with_correct_seed_and_overrides(tmp_path, monkeypatch, small_min_names):
    """v3_wd_seed1/2 and v4_lr_do_seed1/2 reuse the seed-0 recipe's overrides with only the seed changed, and
    do not collide (fingerprint-wise or results-key-wise) with the already-run seed-0 v3_wd/v4_lr_do."""
    _no_budget(monkeypatch)
    data = FakeData()
    data.expected_samples = _expected_counts(data)
    art = tmp_path / "art"
    a = _args(tmp_path, "--recipes", "v3_wd_seed1", "v4_lr_do_seed1", "v3_wd_seed2", "v4_lr_do_seed2",
             "--epochs", "1")
    assert e2.main(a, data=data) == 0
    res = json.load(open(art / "e2e3_results.json"))
    assert list(res) == ["v3_wd_seed1", "v4_lr_do_seed1", "v3_wd_seed2", "v4_lr_do_seed2"]
    assert res["v3_wd_seed1"]["hparams"]["weight_decay"] == 1e-3 and res["v3_wd_seed1"]["seed"] == 1
    assert res["v4_lr_do_seed1"]["hparams"]["lr"] == 1e-4 and res["v4_lr_do_seed1"]["hparams"]["dropout"] == 0.3
    assert res["v3_wd_seed2"]["seed"] == 2 and res["v4_lr_do_seed2"]["seed"] == 2
    for n in ("v3_wd_seed1", "v4_lr_do_seed1", "v3_wd_seed2", "v4_lr_do_seed2"):
        assert res[n]["selection"] == "val_ic" and res[n]["group"] == "E2"

    # rerunning is a no-op (idempotent skip -- nothing retrained), and running the untouched seed-0
    # recipes never collides with these seed-1/2 results keys.
    before = (art / "e2e3_results.json").read_text()
    assert e2.main(a, data=data) == 0
    assert (art / "e2e3_results.json").read_text() == before
    doc = (tmp_path / "doc.md").read_text()
    assert "V3-s1 v3_wd_seed1" in doc and "V4-s2 v4_lr_do_seed2" in doc


def test_oot_never_scored_per_epoch_and_2025_guard(tmp_path, monkeypatch, small_min_names):
    """The per-epoch hook only ever sees the val window: score_window (which computes OOT) runs once per window."""
    _no_budget(monkeypatch)
    windows = []
    real = tfx.score_window
    monkeypatch.setattr(tfx, "score_window", lambda w, *a, **k: (windows.append(w), real(w, *a, **k))[1])
    data = FakeData()
    data.expected_samples = _expected_counts(data)
    e2.main(_args(tmp_path, "--recipes", "v1_lr"), data=data)
    assert windows == ["val_2024", "oot_2026"]


def test_max_tickers_requires_artifacts_dir(tmp_path):
    with pytest.raises(SystemExit):
        e2.parse_args(["--max-tickers", "3"])
    assert e2.parse_args([]).recipes == e2.DEFAULT_ORDER and e2.parse_args([]).epochs == 12
