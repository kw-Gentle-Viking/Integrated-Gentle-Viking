import torch
import torch.nn.functional as F
import pandas as pd
from training.dataset import build_incomplete_today_bar

LABEL_MAP = {0: "매수", 1: "관망", 2: "매도"}


def softmax_probs(logits: torch.Tensor) -> dict:
    probs = F.softmax(logits, dim=-1).squeeze().tolist()
    pred_label = int(torch.argmax(logits, dim=-1).item())
    return {
        "pred_label": pred_label, "pred_str": LABEL_MAP[pred_label],
        "prob_buy": probs[0], "prob_hold": probs[1], "prob_sell": probs[2],
    }


def run_inference(ticker: str, encoder_df: pd.DataFrame, today_intraday_rows: list[dict],
                   model, historical_cols: list[str], future_cols: list[str],
                   static_cols: list[str]) -> dict:
    """encoder_df: 과거 59거래일(오늘 제외) 일봉 피처. 오늘 행은 today_intraday_rows로부터
    미완성 일봉으로 계산해 encoder 마지막(60번째) 행으로 붙인다 (설계 §3)."""
    open_price = encoder_df.iloc[-1]["close"] if len(encoder_df) and "close" in encoder_df.columns else 0
    today_bar = build_incomplete_today_bar(today_intraday_rows, open_price=open_price)
    today_row = encoder_df.iloc[-1].copy()
    for k in ["open", "high", "low", "close", "volume"]:
        if k in today_row.index:
            today_row[k] = today_bar[k]
    full_encoder = pd.concat([encoder_df, today_row.to_frame().T], ignore_index=True)

    hist = torch.tensor(full_encoder[historical_cols].values.astype("float32")).unsqueeze(0)
    fut = torch.tensor(full_encoder[future_cols].iloc[[-1]].values.astype("float32")).unsqueeze(0)
    static = torch.tensor(full_encoder[static_cols].iloc[[-1]].values.astype("int64")).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        out = model({"historical_ts_numeric": hist, "future_ts_numeric": fut,
                      "static_feats_categorical": static})
    return softmax_probs(out["class_logits"])
