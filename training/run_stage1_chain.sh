#!/bin/bash
# Task 15: wait for the GPU to go idle, run the heavy-hyperparameter pre-flight check, and if it
# passes, launch the real Stage-1 Optuna+ablation search -- unattended, no approval needed at
# each step. Meant to be run inside tmux (see the launch command in the Task 15 report / progress
# ledger) so it survives session disconnects.
#
# GPU-idle detection: polls `nvidia-smi` utilization.gpu (NOT memory.used -- some GPU workloads,
# e.g. an Ollama-served local LLM, keep several GB resident in VRAM between requests even while
# genuinely idle/not computing, so a memory-based threshold would never fire; 2026-09-17,
# discovered live when the user said the GPU was free but memory.used still showed ~4.9GB from a
# resident-but-idle ollama model). Requires utilization=0% for CONSECUTIVE_IDLE_CHECKS in a row
# (not just once) before declaring the GPU free, to avoid a false start during a brief pause
# between successive calls of whatever else is using it.
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
echo "[$(date '+%F %T')] GPU confirmed idle (util=0% sustained). memory.used=${used}MiB — if another process is holding VRAM resident, the pre-flight check below will surface an OOM rather than silently proceeding into the real search with too little headroom."
echo "[$(date '+%F %T')] Starting pre-flight check."

PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python training/preflight_heavy_hparam_check.py
preflight_status=$(python3 -c "import json; print(json.load(open('training/artifacts/preflight_result.json'))['status'])")

if [ "$preflight_status" != "PASS" ]; then
    echo "[$(date '+%F %T')] Pre-flight check did NOT pass (status=${preflight_status}) — see training/artifacts/preflight_result.json. NOT starting the real search. Needs human attention."
    exit 1
fi

echo "[$(date '+%F %T')] Pre-flight PASSED. Starting the real Stage-1 search (20 trials x 3 epoch, then ablation 7 x 10 epoch)."
PYTHONPATH=. /home/user/miniconda3/envs/dl_env/bin/python training/run_stage1_search.py
echo "[$(date '+%F %T')] run_stage1_search.py invocation finished (may need re-invoking again later if it stopped early due to --max-minutes or was only a partial pass -- rerun this chain, or the search script directly, next time GPU frees up)."
