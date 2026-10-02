#!/usr/bin/env python
"""Task 15 pre-flight check: confirm the Optuna search space's heaviest point actually fits
and time it for real, before committing to the full 10+ hour unattended search.

training/run_stage1_search.py's Optuna search space includes state_size up to 128 with
attention_heads up to 8. tft-torch's InterpretableMultiHeadAttention does NOT split heads
across state_size (all_heads_dim = embed_dim * num_heads, each head gets a full-width
projection) -- so state_size=128/attention_heads=8 is ~8x wider than the 4.8min/epoch
benchmark's presumed config, and is also the point most likely to hit CUDA OOM. This script
trains exactly 1 epoch at that specific worst-case combination and reports timing/OOM status,
so the real search (kicked off separately) isn't the first place this gets discovered.

Usage:
    set -a && source .env && set +a
    PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python training/preflight_heavy_hparam_check.py

Writes training/artifacts/preflight_result.json: {"status": "PASS"|"OOM"|"ERROR", "epoch_seconds": float|null, "detail": str}
"""
import json
import os
import time
import traceback

import torch
from torch.utils.data import DataLoader

from training.config import build_tft_config, HISTORICAL_COLS_DEFAULT, KNOWN_FUTURE_COLS, STATIC_COLS
from training.dataset import TickerDayDataset
from training.stage1_data import (
    load_or_build_ticker_dfs, compute_label_distribution, compute_class_weights,
)
from training.run_stage1_search import ARTIFACTS_DIR, TRAIN_START, TRAIN_END, BATCH_SIZE
from training.train import train_one_epoch
from tft_torch.tft import TemporalFusionTransformer

RESULT_PATH = f"{ARTIFACTS_DIR}/preflight_result.json"

# The heaviest point in run_stage1_search.py's search space (state_size in [16,32,64,128],
# attention_heads in [1,2,4,8], lstm_layers in [1,3]).
HEAVY_HPARAMS = {"state_size": 128, "attention_heads": 8, "lstm_layers": 3,
                  "dropout": 0.1, "lr": 1e-3}


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    result = {"status": "ERROR", "epoch_seconds": None, "detail": "", "hparams": HEAVY_HPARAMS}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    try:
        dsn = os.environ["STOCK_DB_V2_DSN"]
        train_ticker_dfs = load_or_build_ticker_dfs(
            dsn, TRAIN_START, TRAIN_END,
            cache_path=f"{ARTIFACTS_DIR}/stage1_cache_train_{TRAIN_START}_{TRAIN_END}.pkl")
        train_ds = TickerDayDataset(train_ticker_dfs, HISTORICAL_COLS_DEFAULT, KNOWN_FUTURE_COLS,
                                     STATIC_COLS, 60)
        print(f"train samples={len(train_ds)}", flush=True)

        class_weights = compute_class_weights(compute_label_distribution(train_ds))
        loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        feature_columns = {"historical": HISTORICAL_COLS_DEFAULT, "future": KNOWN_FUTURE_COLS,
                            "static_cardinalities": [21, 3]}
        cfg = build_tft_config(feature_columns, num_classes=3, state_size=HEAVY_HPARAMS["state_size"],
                                attention_heads=HEAVY_HPARAMS["attention_heads"],
                                lstm_layers=HEAVY_HPARAMS["lstm_layers"],
                                dropout=HEAVY_HPARAMS["dropout"])
        model = TemporalFusionTransformer(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=HEAVY_HPARAMS["lr"])
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

        t0 = time.time()
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        elapsed = time.time() - t0

        result["status"] = "PASS"
        result["epoch_seconds"] = elapsed
        result["detail"] = f"1 epoch at heavy hparams completed, loss={loss:.4f}"
        print(f"PASS: {elapsed:.1f}s/epoch (vs 288.9s/epoch benchmark at defaults), loss={loss:.4f}",
              flush=True)

    except RuntimeError as e:
        oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
        is_oom = isinstance(e, oom_type) or "out of memory" in str(e).lower()
        result["status"] = "OOM" if is_oom else "ERROR"
        result["detail"] = str(e)
        print(f"{'OOM' if is_oom else 'ERROR'}: {e}", flush=True)
        if not is_oom:
            traceback.print_exc()
    except Exception as e:
        result["detail"] = str(e)
        print(f"ERROR: {e}", flush=True)
        traceback.print_exc()

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result written to {RESULT_PATH}: {result}", flush=True)


if __name__ == "__main__":
    main()
