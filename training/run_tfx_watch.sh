#!/bin/bash
# tfx-watch: a persistent supervisor around the resumable TFT retraining recipes
# (training/run_tfx_experiments.py: R0 aligned, R1 std, R2 std_vn, R3 std_vn_csr).
#
# Waits until the GPU has been idle, (re)starts the runner, and loops until all four recipes are
# recorded in training/artifacts/tfx_results.json. The runner resumes at EPOCH granularity, so an
# interruption costs at most one epoch.
#
#   * To free the GPU for other work at any time:
#         pkill -f "training/run_tfx_experiments"
#     The watcher notices (signal exit, rc>=128), waits COOLDOWN_AFTER_KILL_SEC (300 s) so you can
#     start your own job, then goes back to waiting for a sustained idle GPU and resumes.
#     WARNING: `pkill -f` matches the FULL command line of every process, including the shell that
#     runs your command when the pattern text appears in it -- pkill can kill its own invoking shell
#     (or tmux send-keys line). Use the bracket trick so the pattern does not match itself:
#         pkill -f "training/[r]un_tfx_experiments"
#   * To stop everything for good:  tmux kill-session -t tfx-watch
#
# GPU-idle detection uses utilization.gpu (NOT memory.used): a resident-but-idle process such as a
# locally-served Ollama model can hold VRAM at 0% utilization. Idle = 0% for 3 consecutive checks,
# 20 s apart.
#
# Launch (NOT launched by whoever wrote this; log goes through tee at the launch site):
#   tmux new-session -d -s tfx-watch \
#     'bash /home/user/AI_Gentle_Viking_RE/.worktrees/ai-model-redesign/training/run_tfx_watch.sh 2>&1 | tee -a /home/user/AI_Gentle_Viking_RE/.worktrees/ai-model-redesign/training/artifacts/tfx_watch.log'
#
# All paths/knobs are overridable through the environment (used by the unit test of tfx_done).
set -uo pipefail

RESULTS="${RESULTS:-training/artifacts/tfx_results.json}"
RECIPES="${RECIPES:-aligned std std_vn std_vn_csr}"
PYTHON="${PYTHON:-/home/user/miniconda3/envs/dl_env/bin/python}"
WORKTREE="${WORKTREE:-/home/user/AI_Gentle_Viking_RE/.worktrees/ai-model-redesign}"
CONSECUTIVE_IDLE_CHECKS="${CONSECUTIVE_IDLE_CHECKS:-3}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-20}"
COOLDOWN_AFTER_KILL_SEC="${COOLDOWN_AFTER_KILL_SEC:-300}"
MAX_QUICK_FAILURES="${MAX_QUICK_FAILURES:-3}"      # real crashes (not signal kills) dying fast => needs a human
QUICK_FAILURE_SECONDS="${QUICK_FAILURE_SECONDS:-300}"

log() { echo "[$(date '+%F %T')] $*"; }

# Exit 0 iff every recipe in $RECIPES is a key of the results JSON (missing/corrupt file => not done).
tfx_done() {
    RESULTS_PATH="$RESULTS" RECIPE_LIST="$RECIPES" python3 - <<'PY' 2>/dev/null
import json, os, sys
try:
    got = json.load(open(os.environ["RESULTS_PATH"]))
except Exception:
    sys.exit(1)
sys.exit(0 if all(r in got for r in os.environ["RECIPE_LIST"].split()) else 1)
PY
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

main() {
    cd "$WORKTREE" || exit 1
    set -a; source .env; set +a
    local quick_failures=0 started rc elapsed
    while true; do
        if tfx_done; then
            log "tfx experiments COMPLETE (all recipes recorded in $RESULTS)."
            exit 0
        fi

        wait_for_idle_gpu

        log "Starting/resuming tfx experiments."
        started=$(date +%s)
        PYTHONPATH=. "$PYTHON" training/run_tfx_experiments.py
        rc=$?
        elapsed=$(( $(date +%s) - started ))
        log "run_tfx_experiments.py exited rc=${rc} after ${elapsed}s."

        if [ "$rc" -eq 0 ]; then
            quick_failures=0
            continue   # loop re-checks tfx_done (a --max-minutes early exit loops again)
        fi

        if [ "$rc" -ge 128 ]; then
            # Killed by a signal (e.g. the user ran pkill to free the GPU).
            quick_failures=0
            log "Killed by signal (rc=${rc}) -- assuming the GPU was reclaimed. Cooling down ${COOLDOWN_AFTER_KILL_SEC}s before watching again."
            sleep "$COOLDOWN_AFTER_KILL_SEC"
            continue
        fi

        # Genuine error exit. A slow one (long run then crash, e.g. CUDA OOM because another job took
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
}

# Only run when executed, not when sourced (the unit test sources this file to call tfx_done).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
