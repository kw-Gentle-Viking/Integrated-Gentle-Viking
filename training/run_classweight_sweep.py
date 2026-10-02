#!/usr/bin/env python
"""classweight-sweep: train the champion architecture (Task 16's frozen 33-column feature set +
hparams, training/champion_config.json) on the same 2019-2023 train / 2024 val split as Task 15,
once per NEW class-weight scheme ("uniform", "mild" -- training.stage1_data's
compute_uniform_class_weights / compute_mild_class_weights), to compare against the existing
"balanced" scheme (sklearn inverse-frequency, Task 15's stage1-remove_lev_total_volume row,
val macro F1=0.3823) whose per-class precision/recall was never recorded. See
.superpowers/sdd/2026-09-08-ai-model-redesign-plan/classweight-sweep-plan.md for the full design.

Two things happen per invocation, both resumable across separate GPU-availability windows:

1. Balanced-scheme baseline per-class breakdown (once, ever): if
   training/artifacts/checkpoints/stage1-ablation-remove_lev_total_volume.pt exists and this
   hasn't been recorded yet, load that checkpoint and run inference (no training) over the 2024
   val set to get its per-class precision/recall + predicted-class frequency, stored in the
   progress JSON under the "balanced" key. If the checkpoint is missing, the balanced comparison
   point stays macro-F1-only (0.3823) and this step is skipped with a warning -- no numbers are
   fabricated.
2. For each of "uniform" and "mild" not yet recorded (checked via the progress JSON AND a
   `| classweight-<scheme> |` row in docs/model_versions.md, belt-and-suspenders like
   run_stage2_final_live.py's _model_versions_has_row): train 10 epochs (default; override via
   --epochs), evaluate macro F1 + per-class P/R + predicted-class frequency on 2024 val, append a
   docs/model_versions.md row, and record progress immediately after training completes and
   before the row is appended (mirrors run_stage1_search.py's run_ablation_phase ordering) so a
   crash between those two steps never leaves an untracked-but-half-done state.

Usage (inside tmux for the real run -- see training/run_classweight_chain.sh):

    set -a && source .env && set +a
    PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python training/run_classweight_sweep.py \\
        [--epochs 10] [--max-minutes 180]

For a cheap end-to-end dry run (seconds, not the real ~1.6h for two 10-epoch runs):
    --epochs 1 --max-minutes 5
"""
import argparse
import gc
import json
import logging
import os
import sys
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from evaluation.evaluate import compute_metrics
from training.config import KNOWN_FUTURE_COLS, STATIC_COLS, build_tft_config
from training.dataset import TickerDayDataset
from training.run_stage1_search import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    ENCODER_LEN,
    MODEL_VERSIONS_HEADER,
    MODEL_VERSIONS_PATH,
    STATIC_CARDINALITIES,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    Budget,
    _is_oom,
    get_device,
    train_and_score,
)
from training.stage1_data import WEIGHT_SCHEMES, compute_label_distribution, load_or_build_ticker_dfs
from training.train import TemporalFusionTransformer, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHAMPION_CONFIG_PATH = "training/champion_config.json"
PROGRESS_PATH = f"{ARTIFACTS_DIR}/classweight_sweep_progress.json"
BALANCED_CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/stage1-ablation-remove_lev_total_volume.pt"
SCHEMES = ["uniform", "mild"]  # "balanced" is handled separately -- see compute_balanced_baseline
HPARAM_KEYS = ["state_size", "attention_heads", "lstm_layers", "dropout", "lr"]
CLASS_NAMES = {0: "buy(매수)", 1: "hold(관망)", 2: "sell(매도)"}


# --- progress JSON (resumability) -------------------------------------------------------------

def load_progress(path: str = PROGRESS_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_progress(progress: dict, path: str = PROGRESS_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


# --- docs/model_versions.md (parameterized path so tests never touch the real file) ------------

def ensure_model_versions_header_at(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(MODEL_VERSIONS_HEADER)


def model_versions_has_row(version: str, model_versions_path: str = MODEL_VERSIONS_PATH) -> bool:
    if not os.path.exists(model_versions_path):
        return False
    prefix = f"| {version} |"
    with open(model_versions_path) as f:
        return any(line.startswith(prefix) for line in f)


def append_model_version_row_at(path: str, version: str, stage: str, feature_desc: str,
                                 hparams: dict, macro_f1_val: float, note: str) -> None:
    ensure_model_versions_header_at(path)
    hparams_str = ", ".join(f"{k}={v}" for k, v in hparams.items())
    row = f"| {version} | {stage} | {feature_desc} | {hparams_str} | {macro_f1_val:.4f} |  | {note} |\n"
    with open(path, "a") as f:
        f.write(row)


def scheme_done(scheme: str, progress: dict, model_versions_path: str = MODEL_VERSIONS_PATH) -> bool:
    """Belt-and-suspenders dedup, matching run_stage2_final_live.py's _model_versions_has_row
    precedent: a scheme is considered done if EITHER the progress JSON already has it OR
    docs/model_versions.md already has a `| classweight-<scheme> |` row (e.g. because the progress
    JSON was lost/reset but the real row survived)."""
    return scheme in progress or model_versions_has_row(f"classweight-{scheme}", model_versions_path)


# --- pure reporting helpers ---------------------------------------------------------------------

def predicted_frequency(confusion_matrix: list[list[int]]) -> dict[int, float]:
    """Predicted-class share per class = column sum / total, from
    evaluation.evaluate.compute_metrics' sklearn confusion_matrix(y_true, y_pred) convention
    (rows=true label, columns=predicted label) -- i.e. "what fraction of all predictions landed
    in class c", used for the sweep's no-dominant-predicted-class balance check."""
    n = len(confusion_matrix)
    total = sum(sum(row) for row in confusion_matrix)
    if total == 0:
        return {c: 0.0 for c in range(n)}
    col_sums = [sum(confusion_matrix[r][c] for r in range(n)) for c in range(n)]
    return {c: col_sums[c] / total for c in range(n)}


def format_note(class_weights: list[float], metrics: dict, epochs: int, val_loss: float) -> str:
    """docs/model_versions.md note field: weight tensor values + full per-class precision/recall +
    predicted-class frequency shares + epochs/val_loss, matching the existing row style
    (`epochs=.., val_loss=..`) plus the extra detail this experiment specifically needs."""
    per_class = metrics["per_class"]
    freq = predicted_frequency(metrics["confusion_matrix"])
    weights_str = "class_weights=[" + ", ".join(f"{w:.3f}" for w in class_weights) + "]"
    per_class_str = ", ".join(
        f"{CLASS_NAMES[c]}: P={per_class[c]['precision']:.2f} R={per_class[c]['recall']:.2f}"
        for c in sorted(per_class))
    freq_str = ", ".join(f"{CLASS_NAMES[c]}={freq[c] * 100:.1f}%" for c in sorted(freq))
    return f"{weights_str}, epochs={epochs}, val_loss={val_loss:.4f}, {per_class_str}, 예측빈도: {freq_str}"


# --- balanced-scheme baseline (existing checkpoint, inference only, no training) ---------------

def compute_balanced_baseline(champion: dict, val_ds: TickerDayDataset, device: torch.device,
                               progress: dict, progress_path: str = PROGRESS_PATH,
                               checkpoint_path: str = BALANCED_CHECKPOINT_PATH) -> dict | None:
    """Recover the "balanced" scheme's per-class breakdown on 2024 val from the EXISTING
    checkpoint (Task 15's stage1-remove_lev_total_volume ablation run) via inference only -- no
    retraining, since re-running "balanced" is explicitly out of scope (the plan doc: "Do NOT
    re-run balanced -- reuse the existing 0.3823 number"). Returns None (and leaves "balanced" out
    of progress) if the checkpoint file isn't present, so the comparison stays macro-F1-only
    rather than fabricating numbers."""
    if "balanced" in progress:
        logger.info("balanced baseline already recorded (macro_f1=%.4f); skipping.",
                     progress["balanced"]["macro_f1"])
        return progress["balanced"]
    if not os.path.exists(checkpoint_path):
        logger.warning("balanced checkpoint not found at %s -- balanced comparison will be "
                        "macro-F1-only (0.3823 per docs/model_versions.md's existing row).",
                        checkpoint_path)
        return None

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    feature_columns = {
        "historical": champion["columns"], "future": KNOWN_FUTURE_COLS,
        "static_cardinalities": STATIC_CARDINALITIES,
    }
    tft_config = build_tft_config(
        feature_columns, num_classes=3, state_size=champion["state_size"],
        attention_heads=champion["attention_heads"], lstm_layers=champion["lstm_layers"],
        dropout=champion["dropout"],
    )
    model = TemporalFusionTransformer(tft_config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    y_true, y_pred = predict(model, val_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    del model

    entry = {"macro_f1": metrics["macro_f1"], "metrics": metrics, "checkpoint_path": checkpoint_path}
    progress["balanced"] = entry
    save_progress(progress, progress_path)
    logger.info("balanced baseline (existing checkpoint, inference only) 2024-val macro_f1=%.4f "
                "per_class=%s predicted_freq=%s", metrics["macro_f1"], metrics["per_class"],
                predicted_frequency(metrics["confusion_matrix"]))
    return entry


# --- core resumable sweep loop (pure orchestration -- train_fn is injected for testability, ----
# --- mirroring run_stage1_search.py's run_ablation_phase / train_fn closure pattern) -----------

def run_schemes(schemes: list[str], epochs: int, budget: Budget, champion: dict,
                 train_fn: Callable[[str, list[str], dict], dict],
                 progress: dict, progress_path: str = PROGRESS_PATH,
                 model_versions_path: str = MODEL_VERSIONS_PATH) -> list[str]:
    """For each scheme in `schemes` not already done: call `train_fn(scheme, columns, hparams)`,
    which must return a dict with "macro_f1", "metrics" (evaluation.evaluate.compute_metrics
    shape: per_class + confusion_matrix), "val_loss", "checkpoint_path", "class_weights" (list of
    float) -- or raise a RuntimeError for a CUDA OOM (caught via _is_oom, logged, left unrecorded
    for retry on the next invocation, never crashes the sweep). Progress is saved immediately
    after a scheme's training completes and BEFORE the model_versions.md row is appended, so a
    crash between those two never leaves untracked state. `budget.exceeded()` is checked only
    before starting a NEW scheme, never mid-training. Returns the list of scheme names actually
    completed (trained) during THIS call."""
    completed: list[str] = []
    for scheme in schemes:
        if scheme_done(scheme, progress, model_versions_path):
            logger.info("scheme=%s already recorded; skipping.", scheme)
            continue
        if budget.exceeded():
            logger.info("Time budget exhausted; not starting scheme=%s this invocation.", scheme)
            break

        hparams = {k: champion[k] for k in HPARAM_KEYS}
        try:
            result = train_fn(scheme, champion["columns"], hparams)
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            logger.warning("scheme=%s hit CUDA OOM: %s", scheme, e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        progress[scheme] = {
            "macro_f1": result["macro_f1"], "metrics": result["metrics"],
            "checkpoint_path": result["checkpoint_path"],
            "class_weights": result["class_weights"], "val_loss": result["val_loss"],
        }
        save_progress(progress, progress_path)

        note = format_note(result["class_weights"], result["metrics"], epochs, result["val_loss"])
        feature_desc = (f"1단계 챔피언({champion['version']}) 피처 그대로, "
                         f"{len(champion['columns'])}개 (2019-2023 train/2024 val)")
        append_model_version_row_at(
            model_versions_path, version=f"classweight-{scheme}", stage="클래스가중치실험",
            feature_desc=feature_desc, hparams=hparams, macro_f1_val=result["macro_f1"], note=note,
        )
        completed.append(scheme)
        logger.info("scheme=%s done: macro_f1=%.4f", scheme, result["macro_f1"])
    return completed


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=10,
                    help="10 by default, matching Task 15's ablation-phase epoch budget (the "
                         "task brief's requirement) -- NOT the 3-epoch Optuna-search budget.")
    p.add_argument("--max-minutes", type=float, default=None,
                    help="Wall-clock budget for this invocation. Checked only between schemes, "
                         "never mid-training. Default: no budget (both schemes run if not already "
                         "done).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    budget = Budget(args.max_minutes)
    device = get_device()
    logger.info("device=%s epochs=%d max_minutes=%s", device, args.epochs, args.max_minutes)

    dsn = os.environ.get("STOCK_DB_V2_DSN")
    if not dsn:
        raise ValueError("STOCK_DB_V2_DSN environment variable not set")

    with open(CHAMPION_CONFIG_PATH) as f:
        champion = json.load(f)
    logger.info("Loaded champion config: version=%s (%d columns)", champion["version"],
                len(champion["columns"]))

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    train_ticker_dfs = load_or_build_ticker_dfs(
        dsn, TRAIN_START, TRAIN_END,
        cache_path=f"{ARTIFACTS_DIR}/stage1_cache_train_{TRAIN_START}_{TRAIN_END}.pkl")
    val_ticker_dfs = load_or_build_ticker_dfs(
        dsn, VAL_START, VAL_END,
        cache_path=f"{ARTIFACTS_DIR}/stage1_cache_val_{VAL_START}_{VAL_END}.pkl")

    train_ds = TickerDayDataset(train_ticker_dfs, champion["columns"], KNOWN_FUTURE_COLS,
                                 STATIC_COLS, ENCODER_LEN)
    val_ds = TickerDayDataset(val_ticker_dfs, champion["columns"], KNOWN_FUTURE_COLS,
                               STATIC_COLS, ENCODER_LEN)
    logger.info("train samples=%d val samples=%d", len(train_ds), len(val_ds))

    ensure_model_versions_header_at(MODEL_VERSIONS_PATH)
    progress = load_progress(PROGRESS_PATH)

    compute_balanced_baseline(champion, val_ds, device, progress)

    def real_train_fn(scheme: str, columns: list[str], hparams: dict) -> dict:
        label_counts = compute_label_distribution(train_ds)
        class_weights = WEIGHT_SCHEMES[scheme](label_counts)
        run_name = f"classweight-{scheme}"
        logger.info("scheme=%s class_weights=%s", scheme, class_weights.tolist())
        # Epoch-level resume: each scheme is ~55 min of GPU time and the GPU is shared, so an
        # interruption mid-scheme should cost at most one epoch, not the whole scheme.
        result = train_and_score(columns, hparams, args.epochs, run_name, train_ds, val_ds,
                                  class_weights, device,
                                  epoch_checkpoint_path=f"{CHECKPOINT_DIR}/{run_name}_inprogress.pt")
        result["class_weights"] = class_weights.tolist()
        return result

    completed = run_schemes(SCHEMES, args.epochs, budget, champion, real_train_fn, progress,
                             PROGRESS_PATH, MODEL_VERSIONS_PATH)
    logger.info("run_classweight_sweep invocation complete. schemes trained this call: %s", completed)


if __name__ == "__main__":
    main()
