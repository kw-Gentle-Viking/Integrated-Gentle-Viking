#!/bin/bash
# Task 17 Step 5: wait for the GPU to go idle, then launch the Stage-2 final retrain (full
# 2019-present, leverage-inclusive). This is a single training run, not a multi-trial search
# (unlike training/run_stage1_chain.sh's Optuna+ablation chain) -- no preflight
# heavy-hyperparameter check needed, since Stage 2 reuses Stage 1's champion
# architecture/hyperparameters verbatim (already proven to fit in GPU memory during Task 15/16).
# Epoch-level checkpoint/resume lives inside run_stage2_final_live.py itself (via
# training/train.py's epoch_checkpoint_path), so a GPU-contention interruption mid-run just needs
# this script (or the python command below) re-invoked -- it picks up from the next epoch rather
# than restarting from epoch 0.
#
# Same GPU-idle detection as run_stage1_chain.sh: nvidia-smi utilization.gpu (NOT memory.used --
# see that script's header comment for why: a resident-but-idle process, e.g. a locally-served
# LLM, can hold several GB without actually computing), 3 consecutive 0%-utilization checks 20s
# apart before declaring the GPU free.
set -euo pipefail
cd /home/user/AI_Gentle_Viking_RE/.worktrees/ai-model-redesign
set -a; source .env; set +a

CONSECUTIVE_IDLE_CHECKS=3
POLL_INTERVAL_SEC=20

echo "[$(date '+%F %T')] Waiting for GPU to go idle (utilization=0%, x${CONSECUTIVE_IDLE_CHECKS} consecutive checks)..."
idle_streak=0
while [ "$idle_streak" -lt "$CONSECUTIVE_IDLE_CHECKS" ]; do
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')
    if [ "$util" -eq 0 ]; then
        idle_streak=$((idle_streak + 1))
        echo "[$(date '+%F %T')] GPU util=${util}% used=${used}MiB (idle check ${idle_streak}/${CONSECUTIVE_IDLE_CHECKS})"
    else
        if [ "$idle_streak" -gt 0 ]; then
            echo "[$(date '+%F %T')] GPU util=${util}% — busy again, resetting idle streak"
        fi
        idle_streak=0
    fi
    sleep "$POLL_INTERVAL_SEC"
done
echo "[$(date '+%F %T')] GPU confirmed idle (util=0% sustained). memory.used=${used}MiB."
echo "[$(date '+%F %T')] Starting Stage-2 final retrain (training/run_stage2_final_live.py, 15 epochs, full 2019-present)."

PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python -m training.run_stage2_final_live
echo "[$(date '+%F %T')] run_stage2_final_live.py invocation finished (if interrupted mid-run by GPU contention, re-invoke this chain script -- epoch-level checkpoint/resume picks up from the next epoch rather than restarting)."
