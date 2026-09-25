import json

from data_collection.run_adjusted_price_backfill import (
    load_progress, pending_tickers, run_backfill,
)


def test_load_progress_missing_file_is_empty(tmp_path):
    assert load_progress(str(tmp_path / "nope.json")) == {"done": [], "failed": {}}


def test_pending_skips_done_and_keeps_order():
    assert pending_tickers(["a", "b", "c"], {"done": ["b"], "failed": {}}) == ["a", "c"]


def test_run_marks_done_and_persists(tmp_path):
    path = str(tmp_path / "p.json")
    seen = []
    res = run_backfill(["a", "b"], lambda t: seen.append(t), path, retry_rounds=0, sleep=lambda s: None)
    assert seen == ["a", "b"] and res["failed"] == {}
    assert json.load(open(path))["done"] == ["a", "b"]


def test_run_is_resumable(tmp_path):
    path = str(tmp_path / "p.json")
    json.dump({"done": ["a"], "failed": {}}, open(path, "w"))
    seen = []
    run_backfill(["a", "b"], lambda t: seen.append(t), path, retry_rounds=0, sleep=lambda s: None)
    assert seen == ["b"]


def test_error_isolated_then_retried_at_end(tmp_path):
    path = str(tmp_path / "p.json")
    attempts = {"b": 0}

    def fn(t):
        if t == "b":
            attempts["b"] += 1
            if attempts["b"] == 1:
                raise RuntimeError("boom")

    res = run_backfill(["a", "b", "c"], fn, path, retry_rounds=1, sleep=lambda s: None)
    assert res["failed"] == {} and set(res["done"]) == {"a", "b", "c"} and attempts["b"] == 2


def test_persistent_failure_reported_not_marked_done(tmp_path):
    path = str(tmp_path / "p.json")

    def fn(t):
        if t == "b":
            raise RuntimeError("always")

    res = run_backfill(["a", "b", "c"], fn, path, retry_rounds=2, sleep=lambda s: None)
    assert "b" in res["failed"] and "b" not in res["done"] and {"a", "c"} <= set(res["done"])
