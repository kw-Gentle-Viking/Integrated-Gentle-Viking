import numpy as np
import pytest

from evaluation.evaluate import compute_signal_metrics
from training.run_signal_baseline import score_to_probs, class_shares, render_markdown


def test_score_to_probs_preserves_order_and_argmax_sign_rule():
    s = np.array([-2.0, -0.1, 0.0, 0.3, 5.0])
    p = score_to_probs(s)
    assert p.shape == (5, 3)
    assert p.sum(axis=1) == pytest.approx(np.ones(5))
    diff = p[:, 0] - p[:, 2]
    assert list(np.argsort(diff)) == list(np.argsort(s))
    assert p.argmax(axis=1).tolist() == [2, 2, 1, 0, 0]  # >0 buy, <0 sell, ==0 hold


def test_score_to_probs_empty_and_constant():
    assert score_to_probs(np.array([])).shape == (0, 3)
    p = score_to_probs(np.zeros(4))
    assert np.isfinite(p).all()


def test_class_shares():
    assert class_shares([0, 0, 1, 2]) == {"buy": 0.5, "hold": 0.25, "sell": 0.25}
    assert class_shares([]) == {"buy": 0.0, "hold": 0.0, "sell": 0.0}


def test_render_markdown_handles_none_values_and_missing_windows():
    sig = compute_signal_metrics([], [], np.zeros((0, 3)))
    row = {"signal": sig, "metrics": {"macro_f1": 0.33, "accuracy": 0.4, "mcc": 0.0,
           "per_class": {str(c): {"precision": .3, "recall": .3, "f1": .3} for c in range(3)}},
           "pred_share": {"buy": .3, "hold": .3, "sell": .4}}
    results = {"val_2024": {"n_samples": 0, "window": "x", "model": row, "references": {"random": row}}}
    md = render_markdown(results, code_commit="abc123")
    assert "abc123" in md and "val_2024" in md and "n/a" in md
    assert "oot_2026" in md  # pending window mentioned


def test_render_markdown_accepts_int_per_class_keys_from_in_memory_metrics():
    sig = compute_signal_metrics([], [], np.zeros((0, 3)))
    row = {"signal": sig, "metrics": {"macro_f1": 0.33, "accuracy": 0.4, "mcc": 0.0,
           "per_class": {c: {"precision": .3, "recall": .3, "f1": .3} for c in range(3)}},
           "pred_share": {"buy": .3, "hold": .3, "sell": .4}}
    results = {"val_2024": {"n_samples": 0, "window": "x", "model": row, "references": {}}}
    assert "val_2024" in render_markdown(results, code_commit="abc")
