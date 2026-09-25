#!/usr/bin/env python
"""Task 16 Step 5: live Stage-1 champion selection + one-shot 2025 test-set evaluation.

    Parse docs/model_versions.md's 7 ablation rows (Task 15's 10-epoch ablation phase --
    NOT the 20 Optuna search rows, which were only 3-epoch and not apples-to-apples)
        -> select_champion() picks the highest Macro F1 (val)
        -> build the champion's 2025 test set from feature_pool (its own column subset,
           encoder_len=60, same TickerDayDataset/stage1_data.py loader as Task 15)
        -> evaluate_on_test() EXACTLY ONCE, champion only
        -> write training/champion_config.json (Task 17 reuses this verbatim for Stage 2)
        -> fill in the champion's row's Macro F1 (test) cell in docs/model_versions.md
           (every other row's test cell stays blank -- they were never evaluated on test).

Usage:
    set -a && source .env && set +a
    PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python -m training.run_stage1_champion_selection
"""
import argparse
import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from training.config import HISTORICAL_COLS_DEFAULT, KNOWN_FUTURE_COLS, STATIC_COLS
from training.dataset import TickerDayDataset, target_offset_of
from training.run_stage1_search import LEVERAGE_FEATURES
from training.select_stage1_champion import evaluate_on_test, select_champion
from training.stage1_data import load_or_build_ticker_dfs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "training/artifacts"
CHECKPOINT_DIR = f"{ARTIFACTS_DIR}/checkpoints"
MODEL_VERSIONS_PATH = "docs/model_versions.md"
CHAMPION_CONFIG_PATH = "training/champion_config.json"

TEST_START, TEST_END = "2025-01-01", "2025-12-31"
ENCODER_LEN = 60
BATCH_SIZE = 128

# One row of docs/model_versions.md, e.g.:
# | stage1-remove_lev_total_volume | 1단계-ablation | 레버리지 피처 제거: lev_total_volume (나머지 5개 포함) | state_size=32, attention_heads=8, lstm_layers=2, dropout=0.16577574724588015, lr=0.0002905890080008017 | 0.3823 |  | epochs=10, val_loss=1.0908 |
ROW_RE = re.compile(
    r"^\|\s*(?P<version>stage1-[^\|]+?)\s*\|\s*(?P<stage>[^\|]+?)\s*\|\s*(?P<feature_desc>[^\|]+?)\s*\|"
    r"\s*(?P<hparams>[^\|]+?)\s*\|\s*(?P<f1_val>[^\|]+?)\s*\|\s*(?P<f1_test>[^\|]*?)\s*\|\s*(?P<note>[^\|]*?)\s*\|\s*$"
)


def _parse_hparams(hparams_str: str) -> dict:
    """'state_size=32, attention_heads=8, lstm_layers=2, dropout=0.1657..., lr=0.00029...' -> dict,
    with int fields cast to int and float fields cast to float."""
    int_fields = {"state_size", "attention_heads", "lstm_layers"}
    hparams: dict = {}
    for part in hparams_str.split(","):
        key, _, value = part.strip().partition("=")
        key = key.strip()
        hparams[key] = int(value) if key in int_fields else float(value)
    return hparams


def _columns_for_version(version: str) -> list[str]:
    """stage1-baseline -> all 34 HISTORICAL_COLS_DEFAULT columns.
    stage1-remove_<feature> -> HISTORICAL_COLS_DEFAULT minus that one leverage feature."""
    if version == "stage1-baseline":
        return list(HISTORICAL_COLS_DEFAULT)
    prefix = "stage1-remove_"
    if not version.startswith(prefix):
        raise ValueError(f"unrecognized ablation version name: {version!r}")
    removed = version[len(prefix):]
    if removed not in LEVERAGE_FEATURES:
        raise ValueError(f"version {version!r} names an unrecognized leverage feature: {removed!r}")
    return [c for c in HISTORICAL_COLS_DEFAULT if c != removed]


def load_ablation_versions(model_versions_path: str = MODEL_VERSIONS_PATH) -> list[dict]:
    """Parse docs/model_versions.md's 7 '1단계-ablation' rows (Task 15's 10-epoch ablation phase)
    into the list[dict] shape select_champion() consumes. Deliberately excludes the 20
    '1단계-optuna' rows (3-epoch hyperparameter search, not apples-to-apples with the 10-epoch
    ablation rows)."""
    with open(model_versions_path) as f:
        lines = f.readlines()

    versions = []
    for line in lines:
        m = ROW_RE.match(line.rstrip("\n"))
        if not m or m.group("stage") != "1단계-ablation":
            continue
        version = m.group("version")
        versions.append({
            "version": version,
            "macro_f1_val": float(m.group("f1_val")),
            "columns": _columns_for_version(version),
            "hparams": _parse_hparams(m.group("hparams")),
        })
    return versions


def update_test_f1_in_model_versions(version: str, macro_f1_test: float,
                                      model_versions_path: str = MODEL_VERSIONS_PATH) -> None:
    """Fill in the Macro F1 (test) cell for exactly one row (the champion's), leaving every other
    row's test cell untouched (blank) -- they were never evaluated on the held-out test set."""
    with open(model_versions_path) as f:
        lines = f.readlines()

    target_prefix = f"| {version} |"
    updated = False
    new_lines = []
    for line in lines:
        if line.startswith(target_prefix):
            m = ROW_RE.match(line.rstrip("\n"))
            if not m:
                raise ValueError(f"row for {version!r} did not match expected table format: {line!r}")
            if m.group("f1_test").strip():
                raise ValueError(f"row for {version!r} already has a non-blank test F1 -- "
                                  f"refusing to overwrite (test set must be touched exactly once)")
            new_line = (
                f"| {m.group('version')} | {m.group('stage')} | {m.group('feature_desc')} | "
                f"{m.group('hparams')} | {m.group('f1_val')} | {macro_f1_test:.4f} | {m.group('note')} |\n"
            )
            new_lines.append(new_line)
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        raise ValueError(f"no row found for version {version!r} in {model_versions_path}")

    with open(model_versions_path, "w") as f:
        f.writelines(new_lines)


def build_test_dataset(dsn: str, columns: list[str]) -> TickerDayDataset:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    test_ticker_dfs = load_or_build_ticker_dfs(
        dsn, TEST_START, TEST_END,
        cache_path=f"{ARTIFACTS_DIR}/stage1_cache_test_{TEST_START}_{TEST_END}.pkl")
    return TickerDayDataset(test_ticker_dfs, columns, KNOWN_FUTURE_COLS, STATIC_COLS, ENCODER_LEN)


def dates_for_dataset(test_ds: TickerDayDataset) -> list[str]:
    """Per-sample trade_date strings, in the same order TickerDayDataset.index (and therefore a
    shuffle=False DataLoader over it) iterates -- required for split_by_regime alignment."""
    dates = []
    for ticker, t in test_ds.index:
        row = test_ds.ticker_dfs[ticker].loc[t + target_offset_of(test_ds)]
        d = row["trade_date"]
        dates.append(d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d))
    return dates


def save_champion_config(champion: dict, macro_f1_test: float,
                          path: str = CHAMPION_CONFIG_PATH) -> dict:
    import json
    config = {
        "version": champion["version"],
        "state_size": champion["hparams"]["state_size"],
        "attention_heads": champion["hparams"]["attention_heads"],
        "lstm_layers": champion["hparams"]["lstm_layers"],
        "dropout": champion["hparams"]["dropout"],
        "lr": champion["hparams"]["lr"],
        "columns": champion["columns"],
        "macro_f1_val": champion["macro_f1_val"],
        "macro_f1_test": macro_f1_test,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return config


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-start", default=TEST_START)
    p.add_argument("--test-end", default=TEST_END)
    return p.parse_args(argv)


def main(argv=None) -> None:
    import torch

    args = parse_args(argv)
    dsn = os.environ.get("STOCK_DB_V2_DSN")
    if not dsn:
        raise ValueError("STOCK_DB_V2_DSN environment variable not set")

    versions = load_ablation_versions(MODEL_VERSIONS_PATH)
    logger.info("Loaded %d ablation versions: %s", len(versions),
                [(v["version"], v["macro_f1_val"]) for v in versions])
    if len(versions) != 7:
        raise ValueError(f"expected 7 ablation rows in {MODEL_VERSIONS_PATH}, found {len(versions)}")

    champion = select_champion(versions)
    logger.info("Champion selected: %s (macro_f1_val=%.4f, %d historical columns)",
                champion["version"], champion["macro_f1_val"], len(champion["columns"]))

    # Refuse to touch the test set at all if this version's row already has a test F1 recorded --
    # update_test_f1_in_model_versions() below also refuses to overwrite it, but that check alone
    # only protects the written record, not the real evaluation itself: a second accidental
    # invocation of this script would still rebuild the test set and re-run evaluate_on_test()
    # before ever reaching that guard. The whole point of a held-out test set is that it's used
    # exactly once, so the check has to happen before build_test_dataset()/evaluate_on_test(),
    # not just before the file write.
    with open(MODEL_VERSIONS_PATH) as f:
        existing_row = next((ln for ln in f if ln.startswith(f"| {champion['version']} |")), None)
    if existing_row is not None:
        m = ROW_RE.match(existing_row.rstrip("\n"))
        if m and m.group("f1_test").strip():
            raise RuntimeError(
                f"{champion['version']!r} already has a recorded test F1 "
                f"({m.group('f1_test').strip()!r}) -- refusing to re-run evaluate_on_test() "
                f"against the real 2025 test set a second time. The test set must be touched "
                f"exactly once; if you genuinely need to redo this (e.g. a bug fix upstream), "
                f"that is a deliberate decision a human should make explicitly, not something "
                f"this script should do silently on a routine re-invocation."
            )

    checkpoint_path = f"{CHECKPOINT_DIR}/stage1-ablation-{champion['version'][len('stage1-'):]}.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"champion checkpoint not found at {checkpoint_path}")

    logger.info("Building 2025 test set (%s..%s) with champion's %d columns...",
                args.test_start, args.test_end, len(champion["columns"]))
    test_ds = build_test_dataset(dsn, champion["columns"])
    logger.info("Test samples=%d", len(test_ds))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    dates = dates_for_dataset(test_ds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluating champion %s on 2025 test set EXACTLY ONCE (device=%s)...",
                champion["version"], device)
    result = evaluate_on_test(checkpoint_path, test_loader, champion["columns"],
                               champion["hparams"], dates)
    macro_f1_test = result["metrics"]["macro_f1"]
    logger.info("Test macro_f1=%.4f (val was %.4f) accuracy=%.4f",
                macro_f1_test, champion["macro_f1_val"], result["metrics"]["accuracy"])
    logger.info("Per-class: %s", result["metrics"]["per_class"])
    logger.info("Confusion matrix: %s", result["metrics"]["confusion_matrix"])
    logger.info("Regime split: %s", result["regime_split"])

    config = save_champion_config(champion, macro_f1_test)
    logger.info("Wrote %s: %s", CHAMPION_CONFIG_PATH, config)

    update_test_f1_in_model_versions(champion["version"], macro_f1_test)
    logger.info("Updated %s row for %s with Macro F1 (test)=%.4f",
                MODEL_VERSIONS_PATH, champion["version"], macro_f1_test)

    logger.info("run_stage1_champion_selection complete.")


if __name__ == "__main__":
    main()
