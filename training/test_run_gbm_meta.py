import numpy as np
import pandas as pd
import torch

from training.config import build_tft_config
from training.run_gbm_meta import prepare_gbm_meta_data
from training.test_run_tfx_experiments import FakeData

# FakeData.frames() only ever produces columns from training.config.HISTORICAL_COLS_DEFAULT (+ label/
# static/future) -- it ignores champion["columns"] entirely, so the champion fixture used with FakeData
# MUST pick real names out of that list (not arbitrary "f1"/"f2"; those don't exist in FakeData's frames).
CHAMPION = {"columns": ["log_ret", "disparity_5d"], "state_size": 4, "attention_heads": 2,
           "lstm_layers": 1, "dropout": 0.0}


def _toy_model_checkpoint(tmp_path):
    from training.config import KNOWN_FUTURE_COLS, STATIC_COLS
    from tft_torch.tft import TemporalFusionTransformer
    cfg = build_tft_config(
        {"historical": CHAMPION["columns"], "future": KNOWN_FUTURE_COLS, "static_cardinalities": [21, 3]},
        num_classes=3, state_size=CHAMPION["state_size"], attention_heads=CHAMPION["attention_heads"],
        lstm_layers=CHAMPION["lstm_layers"], dropout=CHAMPION["dropout"])
    model = TemporalFusionTransformer(cfg)
    path = tmp_path / "toy.pt"
    torch.save(model.state_dict(), path)
    return str(path)


def _fake_tabular_df():
    """FakeData가 내는 모든 (ticker, date) 조합을 커버하는 합성 F3 표 피처. label이 NaN인 행
    (FakeData가 마지막 행 등에 일부러 넣는 결측)은 실제 feature_pool처럼 제외한다."""
    data = FakeData()
    rows = []
    for split in ("train", "val_2024", "oot_2026"):
        frames = data.frames(split, "label")
        for tk, df in frames.items():
            for _, r in df.iterrows():
                if pd.isna(r["label"]):
                    continue
                row = {c: r[c] for c in CHAMPION["columns"]}
                row.update({"ticker": tk, "date_s": r["trade_date"].strftime("%Y-%m-%d"),
                           "label": int(r["label"]), "next_day_return": 0.001})
                rows.append(row)
    return pd.DataFrame(rows).drop_duplicates(subset=["ticker", "date_s"])


def test_prepare_gbm_meta_data_returns_joined_frames_for_all_windows(tmp_path):
    ckpt = _toy_model_checkpoint(tmp_path)
    out = prepare_gbm_meta_data(
        dsn=None, champion=CHAMPION, checkpoint_path=ckpt, device=torch.device("cpu"),
        data=FakeData(), tabular_df=_fake_tabular_df())
    assert set(out) == {"train", "val_2024", "oot_2026"}
    for name, df in out.items():
        assert len(df) > 0, name
        assert "emb_0" in df.columns and "emb_3" in df.columns  # state_size=4
        assert "label" in df.columns and "next_day_return" in df.columns
        assert df["date_s"].min() >= ("2019-01-01" if name == "train" else
                                      "2024-01-01" if name == "val_2024" else "2026-01-01")
        assert not ((df["date_s"] >= "2025-01-01") & (df["date_s"] <= "2025-12-31")).any()
