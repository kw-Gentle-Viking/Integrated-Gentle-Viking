#!/usr/bin/env python
"""Task 15 Step 5: live Stage-1 execution.

    Optuna hyperparameter search (default 20 trials x 3 epochs)
        -> best trial's hyperparameters
        -> leverage-feature ablation (baseline + 6 single-feature removals x 10 epochs)
        -> docs/model_versions.md rows appended automatically, one per completed trial/run.

Designed to be invoked repeatedly across short GPU-availability windows (see
.superpowers/sdd/2026-09-08-ai-model-redesign-plan/task-15-brief.md, "Step 5 amendment
(2026-09-16)"), not run start-to-finish in one sitting:

  - The Optuna study lives in a persistent SQLite DB
    (training/artifacts/stage1_optuna.db, study_name="stage1-tft-search"). Re-invoking this
    script with load_if_exists=True picks up exactly where a prior invocation left off --
    completed trials stay completed, only the remaining trials run. A trial abandoned by a
    killed process (state stuck RUNNING) is reclaimed via Optuna's heartbeat mechanism
    (heartbeat_interval=60s, grace_period=180s) + an explicit fail_stale_trials() call before
    counting existing trials, so it doesn't silently eat into the trial budget forever.
  - Ablation progress is tracked in training/artifacts/stage1_ablation_progress.json (7 keys:
    "baseline" + "remove_<feature>" x6). Any key already present is skipped on re-invocation.
  - CUDA OOM during a single trial/run is caught, logged, the trial reported to Optuna as
    TrialPruned (ablation: recorded as NaN, not written to progress, retried next invocation)
    -- never crashes the whole search.
  - `--max-minutes` bounds how long a single invocation runs: the current trial/run is always
    allowed to finish, but no *new* one starts once the budget is exceeded. Achieved via
    Optuna's own `timeout=` kwarg for the search phase (checked between trials, never mid-trial)
    and an explicit budget.exceeded() check before starting each ablation run.

Usage (inside tmux for real long runs -- see task-15-report.md):

    set -a && source .env && set +a
    PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python -m training.run_stage1_search \\
        [--max-minutes 600] [--n-trials 20] [--optuna-epochs 3] [--ablation-epochs 10] \\
        [--train-start 2019-01-02] [--train-end 2023-12-31] \\
        [--val-start 2024-01-01] [--val-end 2024-12-31]

For a cheap end-to-end dry run (seconds, not minutes), override the scope, e.g.:
    --n-trials 1 --optuna-epochs 1 --ablation-epochs 1 --max-minutes 5
"""
import argparse
import gc
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
import torch
from optuna.storages import RDBStorage, fail_stale_trials
from torch.utils.data import DataLoader

from evaluation.evaluate import compute_metrics
from training.ablation import run_ablation
from training.config import HISTORICAL_COLS_DEFAULT, KNOWN_FUTURE_COLS, STATIC_COLS, build_tft_config
from training.dataset import TickerDayDataset
from training.stage1_data import (
    compute_class_weights,
    compute_label_distribution,
    load_or_build_ticker_dfs,
)
from training.train import TemporalFusionTransformer, predict, run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "training/artifacts"
OPTUNA_DB_PATH = f"{ARTIFACTS_DIR}/stage1_optuna.db"
STUDY_NAME = "stage1-tft-search"
ABLATION_PROGRESS_PATH = f"{ARTIFACTS_DIR}/stage1_ablation_progress.json"
CHECKPOINT_DIR = f"{ARTIFACTS_DIR}/checkpoints"
MODEL_VERSIONS_PATH = "docs/model_versions.md"

TRAIN_START, TRAIN_END = "2019-01-02", "2023-12-31"
VAL_START, VAL_END = "2024-01-01", "2024-12-31"
ENCODER_LEN = 60
BATCH_SIZE = 128  # fixed per the 2026-09-16 benchmark (288.9s/epoch, ~5GB, RTX 2070 SUPER)
STATIC_CARDINALITIES = [21, 3]  # sector_id (0-20, incl. SECTOR_ID_UNCLASSIFIED=20), market_id (0-1)

# 레버리지 피처 6개 (training.config.HISTORICAL_COLS_DEFAULT 기준 확인, 2026-09-16)
LEVERAGE_FEATURES = [
    "lev_total_volume", "lev_total_aum", "lev_aum_to_mktcap",
    "est_rebalancing_flow", "is_vi_triggered", "vi_count_recent5d",
]

MODEL_VERSIONS_HEADER = (
    "# 모델 버전 기록\n\n"
    "| 버전 | 단계 | 피처 구성 | 하이퍼파라미터 | Macro F1 (val) | Macro F1 (test) | 비고 |\n"
    "|---|---|---|---|---|---|---|\n"
)


class Budget:
    """Wall-clock budget for a single invocation. `exceeded()` is checked only between
    trials/runs, never mid-training -- so the current trial/run always finishes cleanly."""

    def __init__(self, max_minutes: float | None):
        self.deadline = time.time() + max_minutes * 60 if max_minutes is not None else None

    def exceeded(self) -> bool:
        return self.deadline is not None and time.time() >= self.deadline

    def remaining_seconds(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.time())


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_model_versions_header() -> None:
    if not os.path.exists(MODEL_VERSIONS_PATH):
        os.makedirs(os.path.dirname(MODEL_VERSIONS_PATH) or ".", exist_ok=True)
        with open(MODEL_VERSIONS_PATH, "w") as f:
            f.write(MODEL_VERSIONS_HEADER)


def append_model_version_row(version: str, stage: str, feature_desc: str, hparams: dict,
                              macro_f1_val: float, note: str) -> None:
    ensure_model_versions_header()
    hparams_str = ", ".join(f"{k}={v}" for k, v in hparams.items())
    row = f"| {version} | {stage} | {feature_desc} | {hparams_str} | {macro_f1_val:.4f} |  | {note} |\n"
    with open(MODEL_VERSIONS_PATH, "a") as f:
        f.write(row)


def _is_oom(exc: Exception) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
    return isinstance(exc, oom_type) or "out of memory" in str(exc).lower()


def train_and_score(historical_cols: list[str], hparams: dict, epochs: int, run_name: str,
                     train_ds: TickerDayDataset, val_ds: TickerDayDataset,
                     class_weights: torch.Tensor, device: torch.device) -> dict:
    """Train `epochs` epochs via training.train.run_training (Task 13, reused as-is -- handles
    wandb + checkpointing), then load the best checkpoint and compute val macro-F1 via
    evaluation.evaluate.compute_metrics (Task 14) -- run_training itself only tracks val loss."""
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    feature_columns = {
        "historical": historical_cols, "future": KNOWN_FUTURE_COLS,
        "static_cardinalities": STATIC_CARDINALITIES,
    }
    tft_config = build_tft_config(
        feature_columns, num_classes=3, state_size=hparams["state_size"],
        attention_heads=hparams["attention_heads"], lstm_layers=hparams["lstm_layers"],
        dropout=hparams["dropout"],
    )
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    config = {
        "tft_config": tft_config, "class_weights": class_weights,
        "train_loader": train_loader, "val_loader": val_loader,
        "epochs": epochs, "lr": hparams["lr"], "device": device,
        "run_name": run_name, "checkpoint_dir": CHECKPOINT_DIR,
        "wandb_config": {**hparams, "epochs": epochs, "n_historical_features": len(historical_cols)},
    }
    result = run_training(config)

    model = TemporalFusionTransformer(tft_config).to(device)
    model.load_state_dict(torch.load(result["checkpoint_path"], map_location=device))
    y_true, y_pred = predict(model, val_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    del model
    return {
        "macro_f1": metrics["macro_f1"], "val_loss": result["best_val_loss"],
        "checkpoint_path": result["checkpoint_path"], "metrics": metrics,
    }


def make_objective(train_ds, val_ds, class_weights, device, epochs):
    def objective_fn(trial: optuna.Trial) -> float:
        hparams = {
            "state_size": trial.suggest_categorical("state_size", [16, 32, 64, 128]),
            # tft-torch's InterpretableMultiHeadAttention projects Q/K to embed_dim * num_heads
            # (each head gets the FULL state_size, not a state_size/num_heads slice -- verified
            # by reading tft_torch/tft.py's InterpretableMultiHeadAttention, 2026-09-16), so
            # there is no state_size % attention_heads == 0 constraint to enforce here.
            "attention_heads": trial.suggest_categorical("attention_heads", [1, 2, 4, 8]),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        }
        run_name = f"stage1-optuna-trial{trial.number}"
        try:
            result = train_and_score(HISTORICAL_COLS_DEFAULT, hparams, epochs, run_name,
                                      train_ds, val_ds, class_weights, device)
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            logger.warning("Trial %d hit CUDA OOM (state_size=%s, heads=%s): %s",
                            trial.number, hparams["state_size"], hparams["attention_heads"], e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise optuna.TrialPruned(f"CUDA OOM: {e}")

        trial.set_user_attr("hparams", hparams)
        trial.set_user_attr("checkpoint_path", result["checkpoint_path"])
        append_model_version_row(
            version=run_name, stage="1단계-optuna",
            feature_desc="전체(레버리지 6개 포함, ablation 이전 baseline 피처셋)",
            hparams=hparams, macro_f1_val=result["macro_f1"],
            note=f"epochs={epochs}, val_loss={result['val_loss']:.4f}",
        )
        return result["macro_f1"]

    return objective_fn


def run_optuna_phase(n_trials_target: int, epochs: int, budget: Budget,
                      train_ds, val_ds, class_weights, device) -> optuna.Study:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    storage = RDBStorage(
        url=f"sqlite:///{OPTUNA_DB_PATH}",
        heartbeat_interval=60, grace_period=180,  # reclaim trials orphaned by a killed process
    )
    study = optuna.create_study(
        storage=storage, study_name=STUDY_NAME, direction="maximize", load_if_exists=True,
    )
    fail_stale_trials(study)  # explicit, not relying solely on the automatic heartbeat sweep

    # Count only COMPLETE trials toward the target, not total attempts. n_trials= below is a cap
    # on how many NEW trials this invocation may start, not a cap on total attempts ever made --
    # if it counted total attempts, a PRUNED/FAILED trial (e.g. a CUDA OOM at a heavy
    # hyperparameter combo) would permanently consume one "slot" with nothing to show for it,
    # and once total attempts reached n_trials_target with any non-COMPLETE among them, no further
    # trial would ever be scheduled again on any future invocation -- silently stalling the
    # downstream ablation phase forever (its gate in main() requires n_complete >= n_trials_target,
    # which could then never be reached). Counting only COMPLETE trials means a pruned/failed
    # attempt gets backfilled by a fresh one on this or a later invocation instead.
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_total = len(study.trials)
    remaining = max(0, n_trials_target - n_complete)
    if remaining == 0:
        logger.info("Optuna study already has %d/%d COMPLETE trials (%d total attempts); "
                     "skipping search phase.", n_complete, n_trials_target, n_total)
        return study

    timeout = budget.remaining_seconds()
    if timeout is not None and timeout <= 0:
        logger.info("Time budget already exhausted before starting the Optuna phase.")
        return study

    logger.info("Running up to %d more Optuna trial(s) (complete=%d/%d, %d total attempts so "
                "far), timeout=%s", remaining, n_complete, n_trials_target, n_total, timeout)
    # catch=(Exception,): an unanticipated bug in one trial must not kill a many-hour unattended
    # search. KeyboardInterrupt/SIGTERM are BaseException, not Exception, so a deliberate kill
    # (e.g. the kill-and-resume validation) still stops the process immediately.
    study.optimize(make_objective(train_ds, val_ds, class_weights, device, epochs),
                    n_trials=remaining, timeout=timeout, catch=(Exception,))
    return study


def load_ablation_progress() -> dict:
    if not os.path.exists(ABLATION_PROGRESS_PATH):
        return {}
    import json
    with open(ABLATION_PROGRESS_PATH) as f:
        return json.load(f)


def save_ablation_progress(progress: dict) -> None:
    import json
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(ABLATION_PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def run_ablation_phase(best_hparams: dict, epochs: int, budget: Budget,
                        train_ticker_dfs: dict, val_ticker_dfs: dict, device: torch.device) -> list[dict]:
    progress = load_ablation_progress()

    def key_for(removed: str | None) -> str:
        return "baseline" if removed is None else f"remove_{removed}"

    def train_fn(columns: list[str]) -> float:
        removed = next((c for c in LEVERAGE_FEATURES if c not in columns), None)
        key = key_for(removed)

        if key in progress:
            logger.info("Ablation '%s' already recorded (macro_f1=%.4f); skipping.",
                         key, progress[key]["macro_f1"])
            return progress[key]["macro_f1"]

        if budget.exceeded():
            logger.info("Time budget exhausted; not starting ablation run '%s' this invocation.", key)
            return float("nan")

        train_ds = TickerDayDataset(train_ticker_dfs, columns, KNOWN_FUTURE_COLS, STATIC_COLS, ENCODER_LEN)
        val_ds = TickerDayDataset(val_ticker_dfs, columns, KNOWN_FUTURE_COLS, STATIC_COLS, ENCODER_LEN)
        class_weights = compute_class_weights(compute_label_distribution(train_ds))
        run_name = f"stage1-ablation-{key}"
        logger.info("Ablation '%s': %d historical features, class_weights=%s",
                     key, len(columns), class_weights.tolist())
        try:
            result = train_and_score(columns, best_hparams, epochs, run_name,
                                      train_ds, val_ds, class_weights, device)
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            logger.warning("Ablation '%s' hit CUDA OOM: %s", key, e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return float("nan")

        progress[key] = {
            "removed": removed, "macro_f1": result["macro_f1"],
            "checkpoint_path": result["checkpoint_path"], "hparams": best_hparams,
        }
        save_ablation_progress(progress)
        feature_desc = ("전체(레버리지 6개 포함)" if removed is None
                         else f"레버리지 피처 제거: {removed} (나머지 5개 포함)")
        append_model_version_row(
            version=f"stage1-{key}", stage="1단계-ablation", feature_desc=feature_desc,
            hparams=best_hparams, macro_f1_val=result["macro_f1"],
            note=f"epochs={epochs}, val_loss={result['val_loss']:.4f}",
        )
        return result["macro_f1"]

    return run_ablation(HISTORICAL_COLS_DEFAULT, LEVERAGE_FEATURES, train_fn)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-minutes", type=float, default=600.0,
                    help="Wall-clock budget for this invocation. Current trial/run always finishes; "
                         "no new one starts once exceeded. Default 600 (10h, comfortably above the "
                         "~10.4h full-search estimate so a full uninterrupted run finishes in one go "
                         "if the GPU stays free that long).")
    p.add_argument("--n-trials", type=int, default=20, help="Total Optuna trial target (cumulative across invocations).")
    p.add_argument("--optuna-epochs", type=int, default=3)
    p.add_argument("--ablation-epochs", type=int, default=10)
    p.add_argument("--train-start", default=TRAIN_START)
    p.add_argument("--train-end", default=TRAIN_END)
    p.add_argument("--val-start", default=VAL_START)
    p.add_argument("--val-end", default=VAL_END)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    budget = Budget(args.max_minutes)
    device = get_device()
    logger.info("device=%s max_minutes=%s n_trials=%d optuna_epochs=%d ablation_epochs=%d",
                device, args.max_minutes, args.n_trials, args.optuna_epochs, args.ablation_epochs)

    dsn = os.environ.get("STOCK_DB_V2_DSN")
    if not dsn:
        raise ValueError("STOCK_DB_V2_DSN environment variable not set")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    train_ticker_dfs = load_or_build_ticker_dfs(
        dsn, args.train_start, args.train_end,
        cache_path=f"{ARTIFACTS_DIR}/stage1_cache_train_{args.train_start}_{args.train_end}.pkl")
    val_ticker_dfs = load_or_build_ticker_dfs(
        dsn, args.val_start, args.val_end,
        cache_path=f"{ARTIFACTS_DIR}/stage1_cache_val_{args.val_start}_{args.val_end}.pkl")

    train_ds_full = TickerDayDataset(train_ticker_dfs, HISTORICAL_COLS_DEFAULT, KNOWN_FUTURE_COLS,
                                      STATIC_COLS, ENCODER_LEN)
    val_ds_full = TickerDayDataset(val_ticker_dfs, HISTORICAL_COLS_DEFAULT, KNOWN_FUTURE_COLS,
                                    STATIC_COLS, ENCODER_LEN)
    logger.info("train samples=%d val samples=%d", len(train_ds_full), len(val_ds_full))

    label_counts = compute_label_distribution(train_ds_full)
    class_weights = compute_class_weights(label_counts)
    logger.info("train label distribution=%s -> class_weights=%s", label_counts, class_weights.tolist())

    ensure_model_versions_header()

    study = run_optuna_phase(args.n_trials, args.optuna_epochs, budget,
                              train_ds_full, val_ds_full, class_weights, device)

    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    logger.info("Optuna phase: %d/%d COMPLETE trials in study.", n_complete, args.n_trials)
    if n_complete < args.n_trials:
        logger.info("Not enough completed trials yet to pick a champion for ablation. "
                     "Re-run this script when GPU time is next available.")
        return
    if budget.exceeded():
        logger.info("Time budget exhausted after the Optuna phase; skipping ablation this invocation.")
        return

    best_trial = study.best_trial
    best_hparams = best_trial.user_attrs.get("hparams") or {
        k: best_trial.params[k] for k in ["state_size", "attention_heads", "lstm_layers", "dropout", "lr"]
    }
    logger.info("Best trial=%d macro_f1=%.4f hparams=%s", best_trial.number, best_trial.value, best_hparams)

    run_ablation_phase(best_hparams, args.ablation_epochs, budget, train_ticker_dfs, val_ticker_dfs, device)
    logger.info("run_stage1_search invocation complete.")


if __name__ == "__main__":
    main()
