#!/bin/bash
# classweight-watch: a persistent supervisor around the resumable class-weight sweep.
#
# Keeps watching the GPU and (re)starts training/run_classweight_sweep.py whenever the GPU has
# been idle, until every scheme is recorded -- so the user never has to babysit or re-launch it
# across the many short GPU-availability windows this shared machine gives us.
#
#   * To free the GPU for other work at any time:  pkill -f run_classweight_sweep.py
#     The watcher notices, waits COOLDOWN_AFTER_KILL_SEC (so the user has time to start their own
#     job without us grabbing the GPU in the gap), then goes back to waiting for a sustained idle
#     GPU and resumes -- the sweep resumes at EPOCH granularity (see train_and_score's
#     epoch_checkpoint_path), so an interruption costs at most one epoch of work.
#   * To stop everything for good:  tmux kill-session -t classweight-watch
#
# GPU-idle detection uses utilization.gpu (NOT memory.used): a resident-but-idle process such as a
# locally-served Ollama model can hold ~5GB of VRAM at 0% utilization (confirmed in this project).
#
# The watcher does NOT retrain Stage 2 afterwards -- picking a winning scheme is a judgment call
# (see classweight-sweep-plan.md), made after reading the sweep's results.
set -uo pipefail
cd /home/user/AI_Gentle_Viking_RE/.worktrees/ai-model-redesign
set -a; source .env; set +a

PYTHON=/home/user/miniconda3/envs/dl_env/bin/python
PROGRESS=training/artifacts/classweight_sweep_progress.json
MODEL_VERSIONS=docs/model_versions.md
CONSECUTIVE_IDLE_CHECKS=3
POLL_INTERVAL_SEC=20
COOLDOWN_AFTER_KILL_SEC=300
MAX_QUICK_FAILURES=3      # real crashes (not signal kills) that die fast => stop, needs a human
QUICK_FAILURE_SECONDS=300

log() { echo "[$(date '+%F %T')] $*"; }

sweep_done() {
    # Mirrors run_classweight_sweep.scheme_done(): a scheme counts as done if it is in the progress
    # JSON OR already has a docs/model_versions.md row.
    for scheme in uniform mild; do
        if python3 -c "import json,sys; sys.exit(0 if '$scheme' in json.load(open('$PROGRESS')) else 1)" 2>/dev/null; then
            continue
        fi
        if grep -q "^| classweight-$scheme |" "$MODEL_VERSIONS" 2>/dev/null; then
            continue
        fi
        return 1
    done
    return 0
}

wait_for_idle_gpu() {
    local streak=0 util used
    log "Waiting for GPU idle (utilization=0% x${CONSECUTIVE_IDLE_CHECKS} consecutive checks)..."
    while [ "$streak" -lt "$CONSECUTIVE_IDLE_CHECKS" ]; do
        util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')
        if [ "$util" = "0" ]; then
            streak=$((streak + 1))
            log "GPU util=0% used=${used}MiB (idle check ${streak}/${CONSECUTIVE_IDLE_CHECKS})"
        else
            [ "$streak" -gt 0 ] && log "GPU util=${util}% -- busy again, resetting idle streak"
            streak=0
        fi
        sleep "$POLL_INTERVAL_SEC"
    done
    log "GPU confirmed idle (sustained)."
}

quick_failures=0
while true; do
    if sweep_done; then
        log "classweight sweep COMPLETE (all schemes recorded in $PROGRESS / $MODEL_VERSIONS)."
        exit 0
    fi

    wait_for_idle_gpu

    log "Starting/resuming classweight sweep."
    started=$(date +%s)
    PYTHONPATH=. "$PYTHON" training/run_classweight_sweep.py
    rc=$?
    elapsed=$(( $(date +%s) - started ))
    log "run_classweight_sweep.py exited rc=${rc} after ${elapsed}s."

    if [ "$rc" -eq 0 ]; then
        quick_failures=0
        continue   # loop re-checks sweep_done (a --max-minutes-style early exit would loop again)
    fi

    if [ "$rc" -ge 128 ]; then
        # Killed by a signal (e.g. the user ran `pkill -f run_classweight_sweep.py` to free the GPU).
        quick_failures=0
        log "Killed by signal (rc=${rc}) -- assuming the GPU was reclaimed. Cooling down ${COOLDOWN_AFTER_KILL_SEC}s before watching again."
        sleep "$COOLDOWN_AFTER_KILL_SEC"
        continue
    fi

    # A genuine error exit. A slow one (long run then crash, e.g. CUDA OOM because another job took
    # VRAM) is retried after a cooldown; repeated FAST failures mean a real bug -- stop and ask.
    if [ "$elapsed" -lt "$QUICK_FAILURE_SECONDS" ]; then
        quick_failures=$((quick_failures + 1))
        log "Quick failure ${quick_failures}/${MAX_QUICK_FAILURES}."
        if [ "$quick_failures" -ge "$MAX_QUICK_FAILURES" ]; then
            log "WATCHER ABORTED: ${MAX_QUICK_FAILURES} consecutive quick failures -- needs human attention (see log above)."
            exit 1
        fi
    else
        quick_failures=0
    fi
    sleep "$COOLDOWN_AFTER_KILL_SEC"
done
