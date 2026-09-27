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


EXCLUDE_COLS = {"ticker", "date_s", "label", "label_vn", "next_day_return", "trade_date"}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def _staged_score(model, X):
    for p in model.staged_predict_proba(X):   # classes_는 fit 직전에 [0,1,2]로 검증됨
        yield p[:, 0] - p[:, 2]


def fit_and_score(prepared: dict) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from evaluation.evaluate import compute_metrics
    from training.run_tabular_baseline import HGB_PARAMS, bundle, select_hgb_iters

    train = prepared["train"]
    cols = _feature_cols(train)
    fit_df = train[train["date_s"] <= FIT_END]
    sel_df = train[(train["date_s"] >= SELECT_START) & (train["date_s"] <= SELECT_END)]
    full_df = train[train["date_s"] <= TRAIN_END]
    assert fit_df["date_s"].max() <= FIT_END and full_df["date_s"].max() <= TRAIN_END
    for df in (fit_df, sel_df, full_df):
        assert not ((df["date_s"] >= "2025-01-01") & (df["date_s"] <= "2025-12-31")).any()

    # Checked BEFORE fitting: select_hgb_iters's staged_predict_proba scoring (`_staged_score`)
    # indexes columns 0 and 2 of predict_proba's output, which silently means something else (or
    # raises a confusing IndexError) if a class is entirely absent from the fit rows -- catch it
    # here with a clear message instead.
    for name, df in (("fit", fit_df), ("full", full_df)):
        present = sorted(df["label"].dropna().astype(int).unique().tolist())
        if present != [0, 1, 2]:
            raise ValueError(f"fit_and_score: expected classes_ == [0,1,2] in {name} rows, got {present}")

    mk = lambda n: HistGradientBoostingClassifier(max_iter=n, **HGB_PARAMS)  # noqa: E731
    best_n, grid = select_hgb_iters(
        mk, fit_df[cols].values, fit_df["label"].astype(int).values,
        sel_df[cols].values, sel_df["date_s"].values, sel_df["next_day_return"].values, _staged_score)
    model = mk(best_n).fit(full_df[cols].values, full_df["label"].astype(int).values)
    assert list(model.classes_) == [0, 1, 2]  # guaranteed by the check above; belt-and-braces

    rec = {"chosen_n_iter": best_n, "select_ic_2023_by_iter": {str(k): v for k, v in grid.items()},
          "n_features": len(cols), "feature_cols": cols}
    for w in ("val_2024", "oot_2026"):
        df = prepared[w]
        probs = model.predict_proba(df[cols].values)
        score = probs[:, 0] - probs[:, 2]
        b, _ics = bundle(df["date_s"].values, df["next_day_return"].values, score, df["ticker"].values)
        preds = model.predict(df[cols].values)
        b["metrics"] = compute_metrics(df["label"].astype(int).tolist(), preds.astype(int).tolist())
        rec[w] = b
    return rec


def render_doc(rec: dict, code_commit: str) -> str:
    L = ["# TFT 임베딩 + GBM 메타 모델 (Phase 1, 오프라인)", "",
        f"code_commit={code_commit}, chosen_n_iter={rec['chosen_n_iter']}, n_features={rec['n_features']}", "",
        "| window | IC | IC IR | macro F1 |", "|---|---|---|---|"]
    for w in ("val_2024", "oot_2026"):
        s = rec[w]["signal"]
        L.append(f"| {w} | {s['mean_daily_rank_ic']:.4f} | {s.get('ic_ir') or float('nan'):.3f} | "
                f"{rec[w]['metrics']['macro_f1']:.4f} |")
    L += ["", "선택: HGB n_iter는 train 내부 2023 분할(fit<=2022, select 2023)로만 골랐다 -- "
         "val 2024/OOT 2026은 이 선택에 전혀 쓰이지 않았다.", ""]
    return "\n".join(L)


def record_model_version(rec: dict, path: str) -> None:
    from training import run_stage1_search as rss
    from training.run_tfx_experiments import model_versions_row_exists  # reuse, don't reimplement
    if model_versions_row_exists(path, "gbm-meta"):
        return
    old = rss.MODEL_VERSIONS_PATH
    rss.MODEL_VERSIONS_PATH = path
    try:
        rss.append_model_version_row(
            version="gbm-meta", stage="2단계-GBM메타",
            feature_desc=f"R0 TFT 임베딩(32차원) + F3 표 피처, 합계 {rec['n_features']}열 입력, "
                        f"HistGradientBoostingClassifier(n_iter={rec['chosen_n_iter']})",
            hparams={"n_iter": rec["chosen_n_iter"]}, macro_f1_val=rec["val_2024"]["metrics"]["macro_f1"],
            note=f"val IC={rec['val_2024']['signal']['mean_daily_rank_ic']:.4f}, "
                f"OOT IC={rec['oot_2026']['signal']['mean_daily_rank_ic']:.4f}(확인용). "
                f"상세: docs/gbm_meta_results.md")
    finally:
        rss.MODEL_VERSIONS_PATH = old


def main(dsn=None, champion=None, checkpoint_path=CHECKPOINT_PATH, data=None, tabular_df=None,
        out_json=RESULT_JSON, doc_path=DOC_PATH, model_versions_path="docs/model_versions.md") -> int:
    import subprocess
    champion = champion or json.load(open(CHAMPION_CONFIG_PATH))
    device = torch.device("cpu")
    prepared = prepare_gbm_meta_data(dsn, champion, checkpoint_path, device, data=data, tabular_df=tabular_df)
    rec = fit_and_score(prepared)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    rec["code_commit"] = commit
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    with open(doc_path, "w") as f:
        f.write(render_doc(rec, commit))
    record_model_version(rec, model_versions_path)
    logger.info("wrote %s, %s", out_json, doc_path)
    return 0


if __name__ == "__main__":
    dsn = os.environ.get("STOCK_DB_V2_DSN")
    if not dsn:
        raise SystemExit("STOCK_DB_V2_DSN environment variable not set")
    raise SystemExit(main(dsn=dsn))
