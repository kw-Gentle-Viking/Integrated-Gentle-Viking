import numpy as np
import pandas as pd
import torch
from serving.inference import run_inference, softmax_probs


def test_softmax_probs_sums_to_one_and_maps_labels():
    logits = torch.tensor([[2.0, 0.5, 0.1]])
    result = softmax_probs(logits)
    assert result["pred_label"] == 0
    assert result["pred_str"] == "매수"
    assert abs(result["prob_buy"] + result["prob_hold"] + result["prob_sell"] - 1.0) < 1e-6


def test_run_inference_returns_all_required_keys(monkeypatch):
    """run_inference no longer builds today's bar or guesses an open price itself (Task 18
    Step 5 fix) -- it consumes an already-complete encoder_df, exactly as
    serving.feature_builder.build_encoder_df[_for_ticker] produces (60 rows: 59 real days +
    1 assembled "today" row), and only does the forward pass + softmax."""
    class FakeModel:
        def eval(self): return self
        def __call__(self, batch):
            return {"class_logits": torch.tensor([[[1.0, 3.0, 0.5]]])}  # 관망이 최댓값

    encoder_df = pd.DataFrame({
        "log_ret": np.random.randn(60) * 0.01, "disparity_20": 1 + np.random.randn(60) * 0.02,
        "time_progress": [1.0] * 59 + [0.5], "is_bok": 0, "sector_id": 3, "market_id": 1,
    })
    result = run_inference(
        ticker="005930", encoder_df=encoder_df,
        model=FakeModel(),
        historical_cols=["log_ret", "disparity_20"], future_cols=["time_progress", "is_bok"],
        static_cols=["sector_id", "market_id"],
    )
    assert set(result.keys()) == {"pred_label", "pred_str", "prob_buy", "prob_hold", "prob_sell"}
    assert result["pred_str"] == "관망"


def test_run_inference_does_not_require_today_intraday_rows_argument():
    """The old (today_intraday_rows, open-price-guessing) signature is gone -- calling with it
    should fail with a clear TypeError, not silently ignore the argument."""
    import inspect
    sig = inspect.signature(run_inference)
    assert "today_intraday_rows" not in sig.parameters
    assert "encoder_df" in sig.parameters


def test_run_inference_builds_2d_static_tensor_not_3d():
    """Regression test: tft-torch's static input channel is NOT time-distributed (see
    training/train.py's _to_device_batch, which squeezes this exact singleton dim off the
    training batch before calling the model) -- static_feats_categorical must be 2D
    [batch, num_static], not [batch, 1, num_static]. A [batch, 1, num_static] tensor passes the
    FakeModel test above (which ignores batch contents) but crashes the real model with
    `IndexError: index 1 is out of bounds for dimension 1 with size 1` inside
    tft_torch's categorical embedding layer -- caught during this task's real-model smoke test."""
    captured = {}

    class RecordingModel:
        def eval(self): return self
        def __call__(self, batch):
            captured["static"] = batch["static_feats_categorical"]
            return {"class_logits": torch.tensor([[[1.0, 3.0, 0.5]]])}

    encoder_df = pd.DataFrame({
        "log_ret": np.random.randn(60) * 0.01, "disparity_20": 1 + np.random.randn(60) * 0.02,
        "time_progress": [1.0] * 59 + [0.5], "is_bok": 0, "sector_id": 3, "market_id": 1,
    })
    run_inference(
        ticker="005930", encoder_df=encoder_df, model=RecordingModel(),
        historical_cols=["log_ret", "disparity_20"], future_cols=["time_progress", "is_bok"],
        static_cols=["sector_id", "market_id"],
    )
    assert captured["static"].dim() == 2
    assert tuple(captured["static"].shape) == (1, 2)


def test_assemble_model_inputs_shapes_and_dtypes():
    """run_inference's tensor assembly is exposed so the train/serve parity test exercises the
    exact code path serving uses."""
    from serving.inference import assemble_model_inputs

    encoder_df = pd.DataFrame({
        "log_ret": np.arange(60, dtype=float), "disparity_20": np.ones(60),
        "time_progress": [1.0] * 60, "is_bok": 0, "sector_id": 3, "market_id": 1,
    })
    b = assemble_model_inputs(encoder_df, ["log_ret", "disparity_20"], ["time_progress", "is_bok"],
                              ["sector_id", "market_id"])
    assert tuple(b["historical_ts_numeric"].shape) == (1, 60, 2)
    assert tuple(b["future_ts_numeric"].shape) == (1, 1, 2)
    assert tuple(b["static_feats_categorical"].shape) == (1, 2)
    assert b["historical_ts_numeric"].dtype == torch.float32
    assert b["static_feats_categorical"].dtype == torch.int64
    assert b["historical_ts_numeric"][0, -1, 0].item() == 59.0
