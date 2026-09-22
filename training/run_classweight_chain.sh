#!/bin/bash
# classweight-sweep: wait for the GPU to go idle, then launch training/run_classweight_sweep.py
# (balanced-baseline per-class recovery via inference on the existing checkpoint, then 'uniform'
# and 'mild' training runs, 10 epochs each). Direct adaptation of training/run_stage1_chain.sh.
#
# Unlike run_stage1_chain.sh, there is NO pre-flight heavy-hyperparameter check here -- the
# champion architecture is already known-good (it has already trained successfully for both the
# Stage-1 ablation phase and the full Stage-2 live run at the same batch size/hparams), so this
# goes straight from "GPU confirmed idle" to the real sweep.
#
# This chain ENDS after the sweep script completes. It does NOT auto-retrain Stage 2 -- whether a
# new scheme "wins" and Stage 2 should be retrained with it is a judgment call made by whoever
# reviews the sweep's results (see the plan doc's Section 4 decision logic), not something this
# chain decides on its own.
#
# GPU-idle detection: identical technique to run_stage1_chain.sh -- polls nvidia-smi
# utilization.gpu (NOT memory.used; a resident-but-idle process, e.g. a locally-served LLM, can
# hold several GB without actually computing, confirmed in this project 2026-09-17), requires
# CONSECUTIVE_IDLE_CHECKS consecutive 0%-utilization reads POLL_INTERVAL_SEC apart before
# declaring the GPU free.
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
echo "[$(date '+%F %T')] Starting classweight sweep (balanced-baseline recovery + uniform/mild training runs, 10 epochs each)."

PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python training/run_classweight_sweep.py
echo "[$(date '+%F %T')] run_classweight_sweep.py invocation finished (resumable -- if interrupted by GPU contention or a max-minutes budget, re-invoke this chain script; already-recorded schemes are skipped automatically)."
