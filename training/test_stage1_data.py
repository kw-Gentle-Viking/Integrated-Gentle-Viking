import math

import torch

from training.stage1_data import (
    compute_class_weights,
    compute_mild_class_weights,
    compute_uniform_class_weights,
)


def test_uniform_class_weights_always_ones_regardless_of_counts():
    balanced_counts = {0: 100, 1: 100, 2: 100}
    skewed_counts = {0: 85714, 1: 168032, 2: 90969}

    assert compute_uniform_class_weights(balanced_counts).tolist() == [1.0, 1.0, 1.0]
    assert compute_uniform_class_weights(skewed_counts).tolist() == [1.0, 1.0, 1.0]


def test_uniform_class_weights_returns_float_tensor():
    weights = compute_uniform_class_weights({0: 1, 1: 2, 2: 3})
    assert isinstance(weights, torch.Tensor)
    assert weights.dtype == torch.float32


def test_mild_class_weights_equals_sqrt_of_balanced_weights():
    # Real Stage-2 live-log label distribution from the buy-skew bug report.
    label_counts = {0: 85714, 1: 168032, 2: 90969}

    balanced = compute_class_weights(label_counts)
    mild = compute_mild_class_weights(label_counts)

    expected = torch.sqrt(balanced)
    assert torch.allclose(mild, expected, atol=1e-6)


def test_mild_class_weights_strictly_between_uniform_and_balanced_per_class():
    label_counts = {0: 85714, 1: 168032, 2: 90969}

    balanced = compute_class_weights(label_counts)
    mild = compute_mild_class_weights(label_counts)

    for c in range(3):
        b = balanced[c].item()
        m = mild[c].item()
        lo, hi = sorted([1.0, b])
        # Dampened toward 1.0, not equal to either extreme (skip degenerate case b == 1.0).
        if not math.isclose(b, 1.0, abs_tol=1e-9):
            assert lo < m < hi, f"class {c}: mild={m} not strictly between 1.0 and balanced={b}"


def test_mild_class_weights_returns_float_tensor():
    weights = compute_mild_class_weights({0: 10, 1: 20, 2: 30})
    assert isinstance(weights, torch.Tensor)
    assert weights.dtype == torch.float32
