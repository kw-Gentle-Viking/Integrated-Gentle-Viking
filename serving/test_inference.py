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
    class FakeModel:
        def eval(self): return self
        def __call__(self, batch):
            return {"class_logits": torch.tensor([[[1.0, 3.0, 0.5]]])}  # 관망이 최댓값

    encoder_df = pd.DataFrame({
        "log_ret": np.random.randn(59) * 0.01, "disparity_20": 1 + np.random.randn(59) * 0.02,
        "time_progress": 1.0, "is_bok": 0, "sector_id": 3, "market_id": 1,
    })
    result = run_inference(
        ticker="005930", encoder_df=encoder_df,
        today_intraday_rows=[{"datetime": "2026-09-08 09:00:00", "open": 100, "high": 101,
                                "low": 99, "close": 100.5, "volume": 1000}],
        model=FakeModel(),
        historical_cols=["log_ret", "disparity_20"], future_cols=["time_progress", "is_bok"],
        static_cols=["sector_id", "market_id"],
    )
    assert set(result.keys()) == {"pred_label", "pred_str", "prob_buy", "prob_hold", "prob_sell"}
    assert result["pred_str"] == "관망"
