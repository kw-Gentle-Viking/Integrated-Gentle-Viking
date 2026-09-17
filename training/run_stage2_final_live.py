#!/usr/bin/env python
"""Task 17 Step 5: live Stage-2 final retrain.

    training/champion_config.json (Task 16's Stage-1 champion: architecture + hparams + the
    34-column feature subset that survived ablation)
        -> live-queried feature_pool max trade_date ("today")
        -> build_stage2_config() (Steps 1-4, already tested) -> full-range training config
           (2019-01-02..today, i.e. the entire history including the 2026-05-27+
           leveraged-ETF volatility regime)
        -> split_held_out_by_trading_days(): last 100 GLOBAL trading days held OUT of gradient
           training (checkpoint-selection + regime-split reporting slice, NOT a second
           Task-16-style one-touch test set -- there's no further held-out future data)
        -> class weights computed from the actual TRAINING portion's label distribution only
        -> 15 epochs via training.train.run_training (Task 13's train/val loop + best-checkpoint-
           on-val-loss mechanism), with epoch-level checkpoint/resume so a GPU-contention
           interruption mid-run just needs this script re-invoked
        -> held-out evaluation: compute_metrics + split_by_regime (leverage_start=2026-05-27,
           confirmed to match evaluation.evaluate.split_by_regime's own default)
        -> best checkpoint copied to serving/best_model_state_dict.pt (Task 18's inference.py
           consumes this)
        -> docs/model_versions.md gets one new "2단계-최종" row, recording the regime split
           honestly even if leverage_era underperforms pre_leverage (accepted per this project's
           design philosophy -- see the plan doc's "변명 가능한 국면" principle).

Usage (inside tmux for the real ~1.4-2.8h run -- see training/run_stage2_chain.sh):

    set -a && source .env && set +a
    PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python -m training.run_stage2_final_live \\
        [--epochs 15]

If interrupted (e.g. GPU contention from other shared work), just re-run the same command --
the in-progress epoch checkpoint (training/artifacts/checkpoints/<run_name>_inprogress.pt) is
detected automatically and training resumes from the next epoch, not epoch 0.
"""
import argparse
import json
import logging
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
import torch
from torch.utils.data import DataLoader

from evaluation.evaluate import compute_metrics, split_by_regime
from training.config import KNOWN_FUTURE_COLS, STATIC_COLS, build_tft_config
from training.dataset import TickerDayDataset
from training.run_stage1_champion_selection import dates_for_dataset
from training.run_stage1_search import (
    STATIC_CARDINALITIES,
    append_model_version_row,
    ensure_model_versions_header,
)
from training.run_stage2_final import build_stage2_config, split_held_out_by_trading_days
from training.stage1_data import compute_class_weights, compute_label_distribution, load_or_build_ticker_dfs
from training.train import TemporalFusionTransformer, predict, run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "training/artifacts"
CHECKPOINT_DIR = f"{ARTIFACTS_DIR}/checkpoints"
MODEL_VERSIONS_PATH = "docs/model_versions.md"
CHAMPION_CONFIG_PATH = "training/champion_config.json"
SERVING_DIR = "serving"
SERVING_CHECKPOINT_PATH = f"{SERVING_DIR}/best_model_state_dict.pt"

ENCODER_LEN = 60
BATCH_SIZE = 128
N_HELD_OUT_DAYS = 100
DEFAULT_EPOCHS = 15
# Confirmed to match evaluation.evaluate.split_by_regime's own default (2026-09-17).
LEVERAGE_START = "2026-05-27"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def query_max_trade_date(dsn: str) -> str:
    """Live-query feature_pool's actual latest trade_date -- Task 17's "today" must be the real
    latest date currently in the table, not the literal string "today" or a hardcoded guess."""
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(trade_date) FROM feature_pool")
            (max_date,) = cur.fetchone()
    finally:
        conn.close()
    if max_date is None:
        raise ValueError("feature_pool has no rows -- cannot determine 'today'")
    return max_date.strftime("%Y-%m-%d") if hasattr(max_date, "strftime") else str(max_date)


def _model_versions_has_row(version: str, model_versions_path: str = MODEL_VERSIONS_PATH) -> bool:
    if not os.path.exists(model_versions_path):
        return False
    prefix = f"| {version} |"
    with open(model_versions_path) as f:
        return any(line.startswith(prefix) for line in f)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                    help="15 by default: Stage-1's ablation phase used 10 epochs on the smaller "
                         "2019-2023-only train split; this is the final deployed model trained on "
                         "~1.53x more data (full 2019-present range), so somewhat more epochs is "
                         "reasonable.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = get_device()
    dsn = os.environ.get("STOCK_DB_V2_DSN")
    if not dsn:
        raise ValueError("STOCK_DB_V2_DSN environment variable not set")

    with open(CHAMPION_CONFIG_PATH) as f:
        champion = json.load(f)
    logger.info("Loaded champion config: version=%s state_size=%s attention_heads=%s "
                "lstm_layers=%s dropout=%s lr=%s (%d columns)",
                champion["version"], champion["state_size"], champion["attention_heads"],
                champion["lstm_layers"], champion["dropout"], champion["lr"], len(champion["columns"]))

    today = query_max_trade_date(dsn)
    logger.info("Live feature_pool max trade_date (today)=%s", today)

    cfg = build_stage2_config(champion, today)
    logger.info("Stage-2 config: train_start=%s train_end=%s run_name=%s",
                cfg["train_start"], cfg["train_end"], cfg["run_name"])

    if _model_versions_has_row(cfg["run_name"]):
        logger.warning("%s already has a row for %r in %s -- skipping to avoid a duplicate row. "
                        "If you genuinely need to redo this run, remove that row first.",
                        MODEL_VERSIONS_PATH, cfg["run_name"], MODEL_VERSIONS_PATH)
        return

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    full_ticker_dfs = load_or_build_ticker_dfs(
        dsn, cfg["train_start"], cfg["train_end"],
        cache_path=f"{ARTIFACTS_DIR}/stage2_cache_full_{cfg['train_start']}_{cfg['train_end']}.pkl")

    train_ticker_dfs, held_out_ticker_dfs, held_out_dates = split_held_out_by_trading_days(
        full_ticker_dfs, N_HELD_OUT_DAYS, ENCODER_LEN)
    n_leverage_days = sum(1 for d in held_out_dates if str(d) >= LEVERAGE_START)
    n_pre_leverage_days = len(held_out_dates) - n_leverage_days
    logger.info("Held-out window: %s..%s (%d global trading days: %d pre_leverage, %d leverage_era)",
                held_out_dates[0], held_out_dates[-1], len(held_out_dates),
                n_pre_leverage_days, n_leverage_days)

    train_ds = TickerDayDataset(train_ticker_dfs, cfg["columns"], KNOWN_FUTURE_COLS, STATIC_COLS, ENCODER_LEN)
    held_out_ds = TickerDayDataset(held_out_ticker_dfs, cfg["columns"], KNOWN_FUTURE_COLS, STATIC_COLS, ENCODER_LEN)
    logger.info("train samples=%d held_out samples=%d", len(train_ds), len(held_out_ds))

    # Class weights from the actual TRAINING portion only (excludes the held-out slice) -- the
    # full-range class balance may differ from Stage 1's 2019-2023-only balance, so this is
    # computed fresh rather than reused from champion_config.json.
    label_counts = compute_label_distribution(train_ds)
    class_weights = compute_class_weights(label_counts)
    logger.info("train (excl. held-out) label distribution=%s -> class_weights=%s",
                label_counts, class_weights.tolist())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    held_out_loader = DataLoader(held_out_ds, batch_size=BATCH_SIZE, shuffle=False)
    held_out_sample_dates = dates_for_dataset(held_out_ds)

    feature_columns = {
        "historical": cfg["columns"], "future": KNOWN_FUTURE_COLS,
        "static_cardinalities": STATIC_CARDINALITIES,
    }
    tft_config = build_tft_config(
        feature_columns, num_classes=3, state_size=cfg["state_size"],
        attention_heads=cfg["attention_heads"], lstm_layers=cfg["lstm_layers"], dropout=cfg["dropout"],
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    epoch_checkpoint_path = f"{CHECKPOINT_DIR}/{cfg['run_name']}_inprogress.pt"
    training_config = {
        "tft_config": tft_config, "class_weights": class_weights,
        "train_loader": train_loader, "val_loader": held_out_loader,
        "epochs": args.epochs, "lr": champion["lr"], "device": device,
        "run_name": cfg["run_name"], "checkpoint_dir": CHECKPOINT_DIR,
        "epoch_checkpoint_path": epoch_checkpoint_path,
        "wandb_config": {
            "state_size": cfg["state_size"], "attention_heads": cfg["attention_heads"],
            "lstm_layers": cfg["lstm_layers"], "dropout": cfg["dropout"], "lr": champion["lr"],
            "epochs": args.epochs, "n_historical_features": len(cfg["columns"]),
            "train_start": cfg["train_start"], "train_end": cfg["train_end"],
            "held_out_start": str(held_out_dates[0]), "held_out_end": str(held_out_dates[-1]),
        },
    }
    logger.info("Starting training: epochs=%d run_name=%s (epoch checkpoint=%s)",
                args.epochs, cfg["run_name"], epoch_checkpoint_path)
    result = run_training(training_config)
    logger.info("Training complete: best_val_loss=%.4f checkpoint=%s",
                result["best_val_loss"], result["checkpoint_path"])

    model = TemporalFusionTransformer(tft_config).to(device)
    model.load_state_dict(torch.load(result["checkpoint_path"], map_location=device))
    y_true, y_pred = predict(model, held_out_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    regime_split = split_by_regime(held_out_sample_dates, y_true, y_pred, leverage_start=LEVERAGE_START)
    logger.info("Held-out macro_f1=%.4f accuracy=%.4f", metrics["macro_f1"], metrics["accuracy"])
    logger.info("Regime split -- pre_leverage: macro_f1=%.4f accuracy=%.4f | leverage_era: macro_f1=%.4f accuracy=%.4f",
                regime_split["pre_leverage"]["macro_f1"], regime_split["pre_leverage"]["accuracy"],
                regime_split["leverage_era"]["macro_f1"], regime_split["leverage_era"]["accuracy"])
    logger.info("Per-class: %s", metrics["per_class"])
    logger.info("Confusion matrix: %s", metrics["confusion_matrix"])

    os.makedirs(SERVING_DIR, exist_ok=True)
    shutil.copyfile(result["checkpoint_path"], SERVING_CHECKPOINT_PATH)
    logger.info("Copied best checkpoint to %s", SERVING_CHECKPOINT_PATH)

    ensure_model_versions_header()
    hparams = {
        "state_size": cfg["state_size"], "attention_heads": cfg["attention_heads"],
        "lstm_layers": cfg["lstm_layers"], "dropout": cfg["dropout"], "lr": champion["lr"],
    }
    feature_desc = (
        f"1단계 챔피언({champion['version']}) 피처 그대로 승계, {len(cfg['columns'])}개 "
        f"({cfg['train_start']}~{cfg['train_end']} 전체, 레버리지 국면 포함)"
    )
    note = (
        f"epochs={args.epochs}, val_loss={result['best_val_loss']:.4f}, "
        f"held-out={held_out_dates[0]}~{held_out_dates[-1]}"
        f"({len(held_out_dates)}거래일: pre_leverage {n_pre_leverage_days}/leverage_era {n_leverage_days}), "
        f"held-out acc={metrics['accuracy']:.4f}, "
        f"pre_leverage macro_f1={regime_split['pre_leverage']['macro_f1']:.4f}, "
        f"leverage_era macro_f1={regime_split['leverage_era']['macro_f1']:.4f} "
        f"(참고용 checkpoint/reporting split, Task 16식 1회성 test set 아님)"
    )
    append_model_version_row(
        version=cfg["run_name"], stage="2단계-최종", feature_desc=feature_desc,
        hparams=hparams, macro_f1_val=metrics["macro_f1"], note=note,
    )
    logger.info("Appended %s row for %s (held-out macro_f1=%.4f)",
                MODEL_VERSIONS_PATH, cfg["run_name"], metrics["macro_f1"])

    logger.info("run_stage2_final_live complete.")


if __name__ == "__main__":
    main()
