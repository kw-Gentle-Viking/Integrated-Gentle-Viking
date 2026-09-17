import torch
import torch.nn.functional as F
import pandas as pd

LABEL_MAP = {0: "매수", 1: "관망", 2: "매도"}


def softmax_probs(logits: torch.Tensor) -> dict:
    probs = F.softmax(logits, dim=-1).squeeze().tolist()
    pred_label = int(torch.argmax(logits, dim=-1).item())
    return {
        "pred_label": pred_label, "pred_str": LABEL_MAP[pred_label],
        "prob_buy": probs[0], "prob_hold": probs[1], "prob_sell": probs[2],
    }


def run_inference(ticker: str, encoder_df: pd.DataFrame, model,
                   historical_cols: list[str], future_cols: list[str],
                   static_cols: list[str]) -> dict:
    """Forward pass + softmax only.

    Task 18 Step 5: this used to also guess today's opening price (silently defaulting to 0 --
    see feature_builder.py's docstring for why that was a real bug) and try to fold today's live
    intraday bar into encoder_df itself (a no-op against the real, engineered column set).
    `encoder_df` must now already be a COMPLETE, correct 60-row frame (59 real trading days +
    1 assembled "today" row) -- that is entirely serving.feature_builder's job
    (build_encoder_df / build_encoder_df_for_ticker), not this function's.
    """
    hist = torch.tensor(encoder_df[historical_cols].values.astype("float32")).unsqueeze(0)
    fut = torch.tensor(encoder_df[future_cols].iloc[[-1]].values.astype("float32")).unsqueeze(0)
    # NOTE: static must stay 2D [batch, num_static], NOT [batch, 1, num_static] -- tft-torch's
    # static input channel is not time-distributed (see training/train.py's _to_device_batch,
    # which squeezes the same singleton dim off the training batch before calling the model).
    # `.iloc[[-1]].values` already has shape (1, num_static) -- that IS the batch-of-1 shape, so
    # no extra unsqueeze here (unlike hist/fut, which start 2D per-sample and need a batch dim
    # added). Caught live during this task's real-model smoke test (a bug the Steps 1-4 FakeModel
    # unit test couldn't catch, since it doesn't touch tft-torch's static embedding layer).
    static = torch.tensor(encoder_df[static_cols].iloc[[-1]].values.astype("int64"))

    model.eval()
    with torch.no_grad():
        out = model({"historical_ts_numeric": hist, "future_ts_numeric": fut,
                      "static_feats_categorical": static})
    return softmax_probs(out["class_logits"])
