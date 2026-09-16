#!/bin/bash
# Task 15: wait for the GPU to go idle, run the heavy-hyperparameter pre-flight check, and if it
# passes, launch the real Stage-1 Optuna+ablation search -- unattended, no approval needed at
# each step. Meant to be run inside tmux (see the launch command in the Task 15 report / progress
# ledger) so it survives session disconnects.
#
# GPU-idle detection: polls `nvidia-smi` memory.used and requires it under IDLE_THRESHOLD_MIB for
# CONSECUTIVE_IDLE_CHECKS in a row (not just once) before declaring the GPU free, to avoid a false
# start on a brief gap between successive calls of whatever else is using it.
set -euo pipefail
cd /home/user/AI_Gentle_Viking_RE/.worktrees/ai-model-redesign
set -a; source .env; set +a

IDLE_THRESHOLD_MIB=300
CONSECUTIVE_IDLE_CHECKS=3
POLL_INTERVAL_SEC=20

echo "[$(date '+%F %T')] Waiting for GPU to go idle (<${IDLE_THRESHOLD_MIB}MiB used, x${CONSECUTIVE_IDLE_CHECKS} consecutive checks)..."
idle_streak=0
while [ "$idle_streak" -lt "$CONSECUTIVE_IDLE_CHECKS" ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')
    if [ "$used" -lt "$IDLE_THRESHOLD_MIB" ]; then
        idle_streak=$((idle_streak + 1))
        echo "[$(date '+%F %T')] GPU used=${used}MiB (idle check ${idle_streak}/${CONSECUTIVE_IDLE_CHECKS})"
    else
        if [ "$idle_streak" -gt 0 ]; then
            echo "[$(date '+%F %T')] GPU used=${used}MiB — busy again, resetting idle streak"
        fi
        idle_streak=0
    fi
    sleep "$POLL_INTERVAL_SEC"
done
echo "[$(date '+%F %T')] GPU confirmed idle. Starting pre-flight check."

PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python training/preflight_heavy_hparam_check.py
preflight_status=$(python3 -c "import json; print(json.load(open('training/artifacts/preflight_result.json'))['status'])")

if [ "$preflight_status" != "PASS" ]; then
    echo "[$(date '+%F %T')] Pre-flight check did NOT pass (status=${preflight_status}) — see training/artifacts/preflight_result.json. NOT starting the real search. Needs human attention."
    exit 1
fi

echo "[$(date '+%F %T')] Pre-flight PASSED. Starting the real Stage-1 search (20 trials x 3 epoch, then ablation 7 x 10 epoch)."
PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python training/run_stage1_search.py
echo "[$(date '+%F %T')] run_stage1_search.py invocation finished (may need re-invoking again later if it stopped early due to --max-minutes or was only a partial pass -- rerun this chain, or the search script directly, next time GPU frees up)."
