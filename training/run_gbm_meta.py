#!/usr/bin/env python
"""TFT 임베딩(R0 체크포인트, forward hook) + F3 표 피처를 합쳐 GBM 메타 모델을 학습/평가한다.
spec: docs/superpowers/specs/2026-09-27-tft-embedding-gbm-design.md
plan: docs/superpowers/plans/2026-09-27-tft-embedding-gbm-meta.md

    set -a && source .env && set +a
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python training/run_gbm_meta.py
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from training.gbm_meta_data import assert_join_coverage, join_embeddings_with_tabular
from training.tft_embeddings import extract_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "training/artifacts"
CHAMPION_CONFIG_PATH = "training/champion_config.json"
CHECKPOINT_PATH = os.path.join(ARTIFACTS_DIR, "checkpoints", "tfx-aligned.pt")
RESULT_JSON = os.path.join(ARTIFACTS_DIR, "gbm_meta_results.json")
DOC_PATH = "docs/gbm_meta_results.md"
FIT_END, SELECT_START, SELECT_END, TRAIN_END = "2022-12-31", "2023-01-01", "2023-12-31", "2023-12-31"
MIN_JOIN_RATIO = {"train": 0.95, "val_2024": 1.0, "oot_2026": 1.0}


def _load_model(champion: dict, checkpoint_path: str, device):
    from training.config import KNOWN_FUTURE_COLS, STATIC_COLS, build_tft_config
    from training.run_stage1_search import STATIC_CARDINALITIES
    from tft_torch.tft import TemporalFusionTransformer
    tft_config = build_tft_config(
        {"historical": champion["columns"], "future": KNOWN_FUTURE_COLS,
         "static_cardinalities": STATIC_CARDINALITIES},
        num_classes=3, state_size=champion["state_size"], attention_heads=champion["attention_heads"],
        lstm_layers=champion["lstm_layers"], dropout=champion["dropout"])
    model = TemporalFusionTransformer(tft_config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def _embed_split(name: str, model, frames: dict, cols: list[str], device, align: str = "today"):
    from training.config import KNOWN_FUTURE_COLS, STATIC_COLS
    from training.dataset import TickerDayDataset
    ds = TickerDayDataset(frames, cols, KNOWN_FUTURE_COLS, STATIC_COLS, 60, align=align)
    keys, emb = extract_embeddings(model, ds, device)
    logger.info("[%s] embedded %d samples, dim=%d", name, len(keys), emb.shape[1] if emb.size else 0)
    return keys, emb


def prepare_gbm_meta_data(dsn: str | None, champion: dict, checkpoint_path: str, device,
                          data=None, tabular_df: pd.DataFrame | None = None) -> dict:
    """R0 임베딩(train/val_2024/oot_2026) + F3 표 피처를 (ticker, date) 조인. `data`/`tabular_df`는
    테스트 주입용(RealData/load_frame을 대체)."""
    from training.run_tfx_experiments import DEFAULT_ALIGN, EVAL_WINDOWS, RealData, build_window_dataset, prepare_recipe

    model = _load_model(champion, checkpoint_path, device)
    data = data or RealData(dsn)
    opts = {"champion": champion, "align": DEFAULT_ALIGN}
    prep = prepare_recipe("aligned", data, opts)  # R0: raw 33 cols, no preprocessing, fixed label
    cols = prep["kept_columns"]

    if tabular_df is None:
        from training.run_tabular_baseline import build_features, load_frame
        raw = load_frame(dsn, champion["columns"])
        tabular_df, _ = build_features(raw, champion["columns"])
        tabular_df["date_s"] = tabular_df["trade_date"].dt.strftime("%Y-%m-%d")

    out = {}
    # --- train ---
    keys, emb = _embed_split("train", model, prep["train_frames"], cols, device)
    joined = join_embeddings_with_tabular(keys, emb, tabular_df)
    joined = joined[joined["date_s"] <= TRAIN_END]
    assert_join_coverage(len(joined), len(keys), tabular_df["date_s"].le(TRAIN_END).sum(),
                         MIN_JOIN_RATIO["train"], "train")
    out["train"] = joined
    # --- val/oot: exact keys from aligned_sample_sets (already asserted == TFT runner's own list) ---
    for w in EVAL_WINDOWS:
        ds, meta = build_window_dataset(w, prep["eval_frames"][w], cols, opts)
        eval_keys, emb = extract_embeddings(model, ds, device)
        joined = join_embeddings_with_tabular(eval_keys, emb, tabular_df)
        assert_join_coverage(len(joined), len(eval_keys), len(tabular_df), MIN_JOIN_RATIO[w], w)
        out[w] = joined
    return out
