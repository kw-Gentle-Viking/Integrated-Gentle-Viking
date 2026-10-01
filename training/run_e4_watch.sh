#!/bin/bash
# e4-watch: a persistent supervisor around the resumable E4 timing-signal TFT experiments
# (training/run_e4_experiments.py: x1_cslabel, x2_cslabel_nostatic, x3_vn_nostatic, x4_cslabel_vnfeat;
# cross-sectional / volatility-normalised labels, constant static ids, cross-sectional volatility inputs;
# checkpoint selection = val timing IC; plus 6 collapse-diagnosis reruns of x2/x3 with weight_decay=0 or a
# different seed -- x2_nostatic_nowd, x3_vn_nostatic_nowd, x2_nostatic_seed1, x3_vn_nostatic_seed1,
# x2_nostatic_seed2, x3_vn_nostatic_seed2; plus v3_structure_nowd -- V3's own structure (static ids kept,
# fixed label) with just weight_decay=0; plus 8 v3_structure_nowd follow-ups (it is the most balanced E4
# result so far, val AND OOT timing IC both positive): seed1/seed2, label_source=label_vn/cs,
# weight_decay=1e-4/3e-4, patience=8, dropout=0.30 -- each changes exactly one variable off v3_structure_nowd).
#
# Waits until the GPU has been idle, (re)starts the runner, and loops until every recipe in $RECIPES is
# recorded in training/artifacts/e4_results.json. The runner resumes at EPOCH granularity, so an
# interruption costs at most one epoch. Same idle / exit-code / quick-failure / cooldown logic as
# run_e2e3_watch.sh / run_tfx_watch.sh (only the completion check and the runner differ).
#
#   * To free the GPU for other work at any time:
#         pkill -f "training/[r]un_e4_experiments"
#     (bracket trick: the pattern must not match its own invoking shell). The watcher notices (exit code
#     143/130), waits COOLDOWN_AFTER_KILL_SEC (300 s), then goes back to waiting for a sustained idle GPU
#     and resumes.
#   * To stop everything for good:  tmux kill-session -t e4-watch
#
# GPU-idle detection uses utilization.gpu (NOT memory.used): idle = 0% for 10 consecutive checks, 30 s
# apart; our runner's PIDs are excluded from the (only logged) compute-apps list; an unreadable GPU is
# treated as busy and NVSMI_MAX_FAILURES consecutive failures abort the watcher.
# Exit codes: only 143 (SIGTERM) / 130 (SIGINT) count as a user kill (cooldown + resume); 137 (OOM killer),
# 139 (segfault) and every other non-zero code are failures (quick_failure counted if < 300 s).
#
# Launch (NOT launched by whoever wrote this; log goes through tee at the launch site):
#   tmux new-session -d -s e4-watch \
#     'bash /home/user/AI_Gentle_Viking_RE/.worktrees/ai-model-redesign/training/run_e4_watch.sh 2>&1 | tee -a /home/user/AI_Gentle_Viking_RE/.worktrees/ai-model-redesign/training/artifacts/e4_watch.log'
#
# All paths/knobs are overridable through the environment (used by the unit test of e4_done).
set -uo pipefail

RESULTS="${RESULTS:-training/artifacts/e4_results.json}"
RECIPES="${RECIPES:-x1_cslabel x2_cslabel_nostatic x3_vn_nostatic x4_cslabel_vnfeat x2_nostatic_nowd x3_vn_nostatic_nowd x2_nostatic_seed1 x3_vn_nostatic_seed1 x2_nostatic_seed2 x3_vn_nostatic_seed2 v3_structure_nowd v3_structure_nowd_seed1 v3_structure_nowd_seed2 v3_structure_nowd_vn v3_structure_nowd_cs v3_structure_wd1e4 v3_structure_wd3e4 v3_structure_nowd_patience8 v3_structure_nowd_dropout30}"
PYTHON="${PYTHON:-/home/user/miniconda3/envs/dl_env/bin/python}"
WORKTREE="${WORKTREE:-/home/user/AI_Gentle_Viking_RE/.worktrees/ai-model-redesign}"
CONSECUTIVE_IDLE_CHECKS="${CONSECUTIVE_IDLE_CHECKS:-10}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"
COOLDOWN_AFTER_KILL_SEC="${COOLDOWN_AFTER_KILL_SEC:-300}"
MAX_QUICK_FAILURES="${MAX_QUICK_FAILURES:-3}"      # real crashes (not signal kills) dying fast => needs a human
QUICK_FAILURE_SECONDS="${QUICK_FAILURE_SECONDS:-300}"
NVSMI_MAX_FAILURES="${NVSMI_MAX_FAILURES:-20}"     # consecutive nvidia-smi failures before the watcher gives up
USER_KILL_CODES="${USER_KILL_CODES:-143 130}"      # SIGTERM / SIGINT only; 137 (OOM killer) / 139 (segv) are failures

log() { echo "[$(date '+%F %T')] $*"; }

# Exit 0 iff every recipe in $RECIPES is a key of the results JSON (missing/corrupt file => not done).
e4_done() {
    RESULTS_PATH="$RESULTS" RECIPE_LIST="$RECIPES" python3 - <<'PY' 2>/dev/null
import json, os, sys
try:
    got = json.load(open(os.environ["RESULTS_PATH"]))
except Exception:
    sys.exit(1)
sys.exit(0 if all(r in got for r in os.environ["RECIPE_LIST"].split()) else 1)
PY
}

# Sets GPU_UTIL / GPU_USED / GPU_APPS (external compute pids, our runner excluded); returns 1 if any
# nvidia-smi query failed or returned garbage (reason in GPU_ERR).
query_gpu() {
    local out rc mine
    GPU_ERR=""
    out=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>&1); rc=$?
    GPU_UTIL=$(echo "$out" | head -1 | tr -d '[:space:]')
    if [ "$rc" -ne 0 ] || ! [[ "$GPU_UTIL" =~ ^[0-9]+$ ]]; then
        GPU_ERR="utilization query failed (rc=${rc}): $(echo "$out" | head -1)"; return 1
    fi
    GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')
    out=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1); rc=$?
    if [ "$rc" -ne 0 ]; then
        GPU_ERR="compute-apps query failed (rc=${rc}): $(echo "$out" | head -1)"; return 1
    fi
    mine=" $(pgrep -f 'training/[r]un_e4_experiments' 2>/dev/null | tr '\n' ' ') $$ "
    GPU_APPS=""
    local pid
    for pid in $(echo "$out" | tr -d ' ' | grep -E '^[0-9]+$'); do
        case "$mine" in *" $pid "*) ;; *) GPU_APPS="${GPU_APPS}${GPU_APPS:+,}$pid" ;; esac
    done
    return 0
}

wait_for_idle_gpu() {
    local streak=0 failures=0
    log "Waiting for GPU idle (utilization=0% x${CONSECUTIVE_IDLE_CHECKS} consecutive checks, ${POLL_INTERVAL_SEC}s apart)..."
    while [ "$streak" -lt "$CONSECUTIVE_IDLE_CHECKS" ]; do
        if ! query_gpu; then
            failures=$((failures + 1))
            streak=0
            log "nvidia-smi problem (${failures}/${NVSMI_MAX_FAILURES}): ${GPU_ERR} -- treating GPU as busy"
            if [ "$failures" -ge "$NVSMI_MAX_FAILURES" ]; then
                log "WATCHER ABORTED: nvidia-smi failed ${NVSMI_MAX_FAILURES} consecutive checks -- needs human attention."
                exit 1
            fi
        else
            failures=0
            if [ "$GPU_UTIL" = "0" ]; then
                streak=$((streak + 1))
                log "GPU util=0% used=${GPU_USED}MiB external_compute_pids=[${GPU_APPS}] (idle check ${streak}/${CONSECUTIVE_IDLE_CHECKS})"
            else
                [ "$streak" -gt 0 ] && log "GPU util=${GPU_UTIL}% external_compute_pids=[${GPU_APPS}] -- busy again, resetting idle streak"
                streak=0
            fi
        fi
        sleep "$POLL_INTERVAL_SEC"
    done
    log "GPU confirmed idle (sustained)."
}

is_user_kill() {
    local c
    for c in $USER_KILL_CODES; do [ "$1" -eq "$c" ] && return 0; done
    return 1
}

main() {
    cd "$WORKTREE" || exit 1
    set -a; source .env; set +a
    local quick_failures=0 started rc elapsed
    while true; do
        if e4_done; then
            log "e4 experiments COMPLETE (all recipes recorded in $RESULTS)."
            exit 0
        fi

        wait_for_idle_gpu

        log "Starting/resuming e4 experiments."
        started=$(date +%s)
        PYTHONPATH=. "$PYTHON" training/run_e4_experiments.py
        rc=$?
        elapsed=$(( $(date +%s) - started ))
        log "run_e4_experiments.py exited rc=${rc} after ${elapsed}s."

        if [ "$rc" -eq 0 ]; then
            quick_failures=0
            continue   # loop re-checks e4_done (a --max-minutes early exit loops again)
        fi

        if is_user_kill "$rc"; then
            # SIGTERM/SIGINT: the user ran pkill to free the GPU (137 OOM-kill / 139 segv are NOT this).
            quick_failures=0
            log "Killed by user signal (rc=${rc}) -- assuming the GPU was reclaimed. Cooling down ${COOLDOWN_AFTER_KILL_SEC}s before watching again."
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

# Only run when executed, not when sourced (the unit test sources this file to call e4_done).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
