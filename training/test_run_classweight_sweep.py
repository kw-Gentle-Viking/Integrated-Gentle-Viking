import json

from training.run_classweight_sweep import (
    Budget,
    append_model_version_row_at,
    format_note,
    model_versions_has_row,
    predicted_frequency,
    run_schemes,
    scheme_done,
)

CHAMPION = {
    "version": "stage1-remove_lev_total_volume", "columns": ["log_ret", "rsi_14"],
    "state_size": 32, "attention_heads": 8, "lstm_layers": 2,
    "dropout": 0.1657, "lr": 0.00029,
}


def _fake_metrics(macro_f1: float, confusion_matrix: list[list[int]]) -> dict:
    per_class = {c: {"precision": 0.5, "recall": 0.5, "f1": 0.5} for c in range(3)}
    return {"macro_f1": macro_f1, "accuracy": 0.5, "mcc": 0.1,
            "per_class": per_class, "confusion_matrix": confusion_matrix}


# --- pure helpers --------------------------------------------------------------------------

def test_predicted_frequency_is_column_sums_over_total():
    # rows=true, cols=predicted (sklearn confusion_matrix convention)
    cm = [[10, 0, 0], [40, 10, 40], [0, 0, 10]]  # predicted totals: 50, 10, 50 out of 110
    freq = predicted_frequency(cm)
    assert freq[0] == 50 / 110
    assert freq[1] == 10 / 110
    assert freq[2] == 50 / 110
    assert abs(sum(freq.values()) - 1.0) < 1e-9


def test_predicted_frequency_empty_matrix_is_all_zero():
    cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert predicted_frequency(cm) == {0: 0.0, 1: 0.0, 2: 0.0}


def test_format_note_includes_weights_per_class_and_predicted_frequency():
    metrics = _fake_metrics(0.40, [[10, 0, 0], [0, 10, 0], [0, 0, 10]])
    note = format_note([1.157, 0.819, 1.122], metrics, epochs=10, val_loss=1.05)
    assert "class_weights=[1.157, 0.819, 1.122]" in note
    assert "epochs=10" in note
    assert "val_loss=1.0500" in note
    assert "P=0.50 R=0.50" in note
    assert "예측빈도" in note


# --- scheme_done / model_versions_has_row dedup ---------------------------------------------

def test_model_versions_has_row_true_only_for_exact_prefix_match(tmp_path):
    mv_path = tmp_path / "model_versions.md"
    mv_path.write_text("| classweight-uniform | ... |\n| classweight-uniform-extra | ... |\n")
    assert model_versions_has_row("classweight-uniform", str(mv_path)) is True
    assert model_versions_has_row("classweight-mild", str(mv_path)) is False


def test_scheme_done_true_if_in_progress_dict(tmp_path):
    mv_path = tmp_path / "model_versions.md"
    progress = {"uniform": {"macro_f1": 0.4}}
    assert scheme_done("uniform", progress, str(mv_path)) is True
    assert scheme_done("mild", progress, str(mv_path)) is False


def test_scheme_done_true_if_model_versions_row_exists_even_without_progress_entry(tmp_path):
    mv_path = tmp_path / "model_versions.md"
    mv_path.write_text("| classweight-mild | ... |\n")
    assert scheme_done("mild", {}, str(mv_path)) is True


# --- append_model_version_row_at writes a real row to a temp path (no pollution of the real doc) --

def test_append_model_version_row_at_writes_header_once_and_appends_rows(tmp_path):
    mv_path = str(tmp_path / "model_versions.md")
    append_model_version_row_at(mv_path, "classweight-uniform", "클래스가중치실험", "33개",
                                 {"state_size": 32}, 0.41, "note1")
    append_model_version_row_at(mv_path, "classweight-mild", "클래스가중치실험", "33개",
                                 {"state_size": 32}, 0.42, "note2")
    text = open(mv_path).read()
    assert text.count("# 모델 버전 기록") == 1
    assert "| classweight-uniform |" in text
    assert "| classweight-mild |" in text


# --- CPU-only mocked-train-fn resumability simulation ----------------------------------------
# Real JSON progress + real model_versions.md-append machinery against TEMP paths, with a fake
# train_fn standing in for the actual GPU training call -- same technique as Task 15's own
# ablation-phase simulation (a fake train_fn closed over in-memory state instead of a real model).

def test_sweep_trains_both_schemes_in_order_and_records_progress_and_rows(tmp_path):
    progress_path = str(tmp_path / "progress.json")
    mv_path = str(tmp_path / "model_versions.md")
    calls = []

    def fake_train_fn(scheme, columns, hparams):
        calls.append(scheme)
        return {
            "macro_f1": {"uniform": 0.35, "mild": 0.40}[scheme],
            "metrics": _fake_metrics(0.35, [[10, 0, 0], [0, 10, 0], [0, 0, 10]]),
            "val_loss": 1.1, "checkpoint_path": f"ckpt-{scheme}.pt",
            "class_weights": [1.0, 1.0, 1.0],
        }

    progress = {}
    completed = run_schemes(["uniform", "mild"], epochs=1, budget=Budget(None), champion=CHAMPION,
                             train_fn=fake_train_fn, progress=progress,
                             progress_path=progress_path, model_versions_path=mv_path)

    assert calls == ["uniform", "mild"]
    assert completed == ["uniform", "mild"]
    assert set(progress) == {"uniform", "mild"}
    saved = json.load(open(progress_path))
    assert set(saved) == {"uniform", "mild"}
    text = open(mv_path).read()
    assert "| classweight-uniform |" in text
    assert "| classweight-mild |" in text


def test_sweep_skips_already_recorded_scheme_on_reinvocation(tmp_path):
    """Re-invoking run_schemes with a scheme already in `progress` must NOT call train_fn for it
    again -- proves the resumability contract without touching a real train fn."""
    progress_path = str(tmp_path / "progress.json")
    mv_path = str(tmp_path / "model_versions.md")
    calls = []

    def fake_train_fn(scheme, columns, hparams):
        calls.append(scheme)
        return {"macro_f1": 0.4, "metrics": _fake_metrics(0.4, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                "val_loss": 1.0, "checkpoint_path": f"ckpt-{scheme}.pt", "class_weights": [1, 1, 1]}

    progress = {"uniform": {"macro_f1": 0.35, "metrics": _fake_metrics(0.35, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                             "checkpoint_path": "old.pt", "class_weights": [1, 1, 1], "val_loss": 1.1}}
    completed = run_schemes(["uniform", "mild"], epochs=1, budget=Budget(None), champion=CHAMPION,
                             train_fn=fake_train_fn, progress=progress,
                             progress_path=progress_path, model_versions_path=mv_path)

    assert calls == ["mild"]  # uniform skipped entirely
    assert completed == ["mild"]


def test_sweep_skips_scheme_already_recorded_only_in_model_versions_md(tmp_path):
    """Belt-and-suspenders: a scheme present as a docs/model_versions.md row but NOT in the
    progress dict (e.g. progress JSON was lost) is still skipped, not retrained."""
    progress_path = str(tmp_path / "progress.json")
    mv_path = str(tmp_path / "model_versions.md")
    with open(mv_path, "w") as f:
        f.write("| classweight-uniform | ... |\n")
    calls = []

    def fake_train_fn(scheme, columns, hparams):
        calls.append(scheme)
        return {"macro_f1": 0.4, "metrics": _fake_metrics(0.4, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                "val_loss": 1.0, "checkpoint_path": f"ckpt-{scheme}.pt", "class_weights": [1, 1, 1]}

    completed = run_schemes(["uniform", "mild"], epochs=1, budget=Budget(None), champion=CHAMPION,
                             train_fn=fake_train_fn, progress={},
                             progress_path=progress_path, model_versions_path=mv_path)

    assert calls == ["mild"]
    assert completed == ["mild"]


def test_max_minutes_budget_stops_between_schemes_not_mid_run(tmp_path):
    """An already-exceeded budget must stop the loop BEFORE starting the next scheme's train_fn
    call -- never abort a scheme that has already started (there is no mid-run abort mechanism
    here; train_fn is called or it isn't)."""
    progress_path = str(tmp_path / "progress.json")
    mv_path = str(tmp_path / "model_versions.md")
    calls = []

    def fake_train_fn(scheme, columns, hparams):
        calls.append(scheme)
        return {"macro_f1": 0.4, "metrics": _fake_metrics(0.4, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                "val_loss": 1.0, "checkpoint_path": f"ckpt-{scheme}.pt", "class_weights": [1, 1, 1]}

    already_exceeded_budget = Budget(-1.0)  # deadline already in the past
    completed = run_schemes(["uniform", "mild"], epochs=1, budget=already_exceeded_budget,
                             champion=CHAMPION, train_fn=fake_train_fn, progress={},
                             progress_path=progress_path, model_versions_path=mv_path)

    assert calls == []
    assert completed == []


def test_kill_and_resume_does_not_redo_completed_schemes(tmp_path):
    """Simulates: invocation 1 completes "uniform" then gets killed before "mild" starts;
    invocation 2 (fresh `progress` loaded from the same progress_path, like main() would do via
    load_progress) must train only "mild"."""
    progress_path = str(tmp_path / "progress.json")
    mv_path = str(tmp_path / "model_versions.md")
    calls = []

    def fake_train_fn(scheme, columns, hparams):
        calls.append(scheme)
        return {"macro_f1": 0.4, "metrics": _fake_metrics(0.4, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                "val_loss": 1.0, "checkpoint_path": f"ckpt-{scheme}.pt", "class_weights": [1, 1, 1]}

    # Invocation 1: budget exhausted right after "uniform" finishes (simulated via a budget that
    # only allows one scheme -- easiest deterministic way to model a kill without real time.sleep).
    budget_gate = Budget(None)
    real_exceeded = budget_gate.exceeded

    def exceeded_after_first_call():
        return len(calls) >= 1

    budget_gate.exceeded = exceeded_after_first_call
    run_schemes(["uniform", "mild"], epochs=1, budget=budget_gate, champion=CHAMPION,
                train_fn=fake_train_fn, progress={}, progress_path=progress_path,
                model_versions_path=mv_path)
    assert calls == ["uniform"]

    from training.run_classweight_sweep import load_progress  # fresh load, like a restarted process
    resumed_progress = load_progress(progress_path)
    assert set(resumed_progress) == {"uniform"}

    # Invocation 2: fresh process, fresh Budget, loads progress from disk.
    completed2 = run_schemes(["uniform", "mild"], epochs=1, budget=Budget(None), champion=CHAMPION,
                              train_fn=fake_train_fn, progress=resumed_progress,
                              progress_path=progress_path, model_versions_path=mv_path)

    assert calls == ["uniform", "mild"]  # "uniform" never retrained
    assert completed2 == ["mild"]
    text = open(mv_path).read()
    assert "| classweight-uniform |" in text
    assert "| classweight-mild |" in text


def test_oom_leaves_scheme_unrecorded_for_retry(tmp_path, monkeypatch):
    progress_path = str(tmp_path / "progress.json")
    mv_path = str(tmp_path / "model_versions.md")
    calls = []

    def flaky_train_fn(scheme, columns, hparams):
        calls.append(scheme)
        if scheme == "uniform":
            raise RuntimeError("CUDA out of memory. Tried to allocate ...")
        return {"macro_f1": 0.4, "metrics": _fake_metrics(0.4, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                "val_loss": 1.0, "checkpoint_path": f"ckpt-{scheme}.pt", "class_weights": [1, 1, 1]}

    completed = run_schemes(["uniform", "mild"], epochs=1, budget=Budget(None), champion=CHAMPION,
                             train_fn=flaky_train_fn, progress={}, progress_path=progress_path,
                             model_versions_path=mv_path)

    assert calls == ["uniform", "mild"]  # OOM on uniform doesn't stop mild from being attempted
    assert completed == ["mild"]  # uniform not recorded -- eligible for retry next invocation
