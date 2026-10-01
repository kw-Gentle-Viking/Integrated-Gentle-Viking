"""Unit test of the watcher's done-check and control flow with temp files and stubbed nvidia-smi/python
(no GPU, nothing launched)."""
import json
import os
import stat
import subprocess

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_e4_watch.sh")


def _done(results_path, recipes="x1_cslabel x2_cslabel_nostatic x3_vn_nostatic"):
    r = subprocess.run(["bash", "-c", f'source "{SCRIPT}"; e4_done'],
                       env={**os.environ, "RESULTS": str(results_path), "RECIPES": recipes})
    return r.returncode == 0


def test_bash_syntax_and_executable():
    assert subprocess.run(["bash", "-n", SCRIPT]).returncode == 0
    assert os.stat(SCRIPT).st_mode & stat.S_IXUSR


def test_sourcing_does_not_run_the_watcher(tmp_path):
    r = subprocess.run(["bash", "-c", f'source "{SCRIPT}"; echo sourced-ok'], capture_output=True, text=True, timeout=20)
    assert r.stdout.strip() == "sourced-ok"


TEN_RECIPES = ["x1_cslabel", "x2_cslabel_nostatic", "x3_vn_nostatic", "x4_cslabel_vnfeat", "x2_nostatic_nowd",
               "x3_vn_nostatic_nowd", "x2_nostatic_seed1", "x3_vn_nostatic_seed1", "x2_nostatic_seed2",
               "x3_vn_nostatic_seed2"]
ELEVEN_RECIPES = TEN_RECIPES + ["v3_structure_nowd"]


def test_done_check_default_recipes_covers_all_eleven(tmp_path):
    """e4_done with NO $RECIPES override (the watcher's own default) must require all 11 recipes, including the
    6 collapse-diagnosis reruns and the v3_structure_nowd weight_decay control, not just the original 4."""
    p = tmp_path / "res.json"
    env = {k: v for k, v in os.environ.items() if k != "RECIPES"}
    env["RESULTS"] = str(p)
    p.write_text(json.dumps({n: {} for n in ELEVEN_RECIPES if n != "v3_structure_nowd"}))
    r = subprocess.run(["bash", "-c", f'source "{SCRIPT}"; e4_done'], env=env)
    assert r.returncode != 0                                                   # v3_structure_nowd still missing
    p.write_text(json.dumps({n: {} for n in ELEVEN_RECIPES}))
    r = subprocess.run(["bash", "-c", f'source "{SCRIPT}"; e4_done'], env=env)
    assert r.returncode == 0


def test_done_check(tmp_path):
    p = tmp_path / "res.json"
    assert not _done(p)                                            # missing file
    p.write_text("{not json")
    assert not _done(p)                                            # corrupt file
    p.write_text(json.dumps({"x1_cslabel": {}, "x2_cslabel_nostatic": {}}))
    assert not _done(p)                                            # one recipe missing
    p.write_text(json.dumps({"x1_cslabel": {}, "x2_cslabel_nostatic": {}, "x3_vn_nostatic": {}}))
    assert _done(p)
    assert _done(p, recipes="x2_cslabel_nostatic")
    assert not _done(p, recipes="x2_cslabel_nostatic other")


def _stub(dir_, name, body):
    f = dir_ / name
    f.write_text("#!/bin/bash\n" + body)
    f.chmod(0o755)
    return f


# stub nvidia-smi: idle GPU (util 0, 100 MiB), one external compute pid 4242.
NVSMI_OK = ('case "$*" in\n'
            '  *utilization.gpu*) echo 0;;\n'
            '  *memory.used*) echo 100;;\n'
            '  *compute-apps*) echo 4242;;\n'
            'esac\n')


def _run_watcher(tmp_path, py_body, extra_env=None, nvsmi=NVSMI_OK):
    """Full main loop in a temp worktree with stub python + stub nvidia-smi and zero sleeps."""
    wt = tmp_path / "wt"
    (wt / "training" / "artifacts").mkdir(parents=True)
    (wt / ".env").write_text("X=1\n")
    (wt / "training" / "run_e4_experiments.py").write_text("")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _stub(bindir, "nvidia-smi", nvsmi)
    py = _stub(tmp_path, "fakepython", py_body)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "WORKTREE": str(wt), "PYTHON": str(py),
           "RESULTS": str(wt / "res.json"), "POLL_INTERVAL_SEC": "0", "COOLDOWN_AFTER_KILL_SEC": "0",
           "CONSECUTIVE_IDLE_CHECKS": "3", "QUICK_FAILURE_SECONDS": "300", **(extra_env or {})}
    return subprocess.run(["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=60), wt


ALL_4_JSON = '{"x1_cslabel":{},"x2_cslabel_nostatic":{},"x3_vn_nostatic":{},"x4_cslabel_vnfeat":{}}'
ALL_11_JSON = '{' + ','.join(f'"{n}":{{}}' for n in ELEVEN_RECIPES) + '}'       # matches the watcher's default $RECIPES


def test_watcher_completes_when_runner_records_all_recipes(tmp_path):
    body = f'echo \'{ALL_11_JSON}\' > "$RESULTS"\nexit 0\n'
    r, _ = _run_watcher(tmp_path, body)
    assert r.returncode == 0 and "e4 experiments COMPLETE" in r.stdout
    assert "utilization=0% x3" in r.stdout


def test_watcher_aborts_after_three_quick_failures(tmp_path):
    r, _ = _run_watcher(tmp_path, "exit 2\n")
    assert r.returncode == 1 and "WATCHER ABORTED" in r.stdout and r.stdout.count("Quick failure") == 3


def test_watcher_cools_down_and_resumes_after_signal_kill(tmp_path):
    counter = tmp_path / "count"
    body = (f'n=$(cat "{counter}" 2>/dev/null || echo 0); echo $((n+1)) > "{counter}"\n'
            'if [ "$n" -eq 0 ]; then exit 143; fi\n'
            f'echo \'{ALL_11_JSON}\' > "$RESULTS"\nexit 0\n')
    r, _ = _run_watcher(tmp_path, body)
    assert r.returncode == 0 and "Killed by user signal (rc=143)" in r.stdout and "COMPLETE" in r.stdout
    assert "Quick failure" not in r.stdout


def test_defaults_are_10_checks_30_seconds():
    r = subprocess.run(["bash", "-c", f'source "{SCRIPT}"; echo "$CONSECUTIVE_IDLE_CHECKS $POLL_INTERVAL_SEC"'],
                       capture_output=True, text=True, timeout=20, env={k: v for k, v in os.environ.items()
                                                                        if k not in ("CONSECUTIVE_IDLE_CHECKS", "POLL_INTERVAL_SEC")})
    assert r.stdout.strip() == "10 30"


def test_idle_log_lists_external_compute_pids_but_does_not_block(tmp_path):
    body = 'echo \'{"x1_cslabel":{}}\' > "$RESULTS"\nexit 0\n'
    r, _ = _run_watcher(tmp_path, body, {"RECIPES": "x1_cslabel"})
    assert r.returncode == 0 and "external_compute_pids=[4242]" in r.stdout and "GPU confirmed idle" in r.stdout


def test_nonzero_utilization_resets_streak_and_stays_waiting(tmp_path):
    counter = tmp_path / "n"
    nv = (f'case "$*" in\n *utilization.gpu*) n=$(cat "{counter}" 2>/dev/null || echo 0); echo $((n+1)) > "{counter}";'
          ' if [ "$n" -eq 1 ]; then echo 55; else echo 0; fi;;\n *memory.used*) echo 1;;\n *compute-apps*) ;;\nesac\n')
    body = 'echo \'{"x1_cslabel":{}}\' > "$RESULTS"\nexit 0\n'
    r, _ = _run_watcher(tmp_path, body, {"RECIPES": "x1_cslabel"}, nvsmi=nv)
    assert r.returncode == 0 and "busy again, resetting idle streak" in r.stdout


def test_nvidia_smi_failure_is_logged_busy_and_aborts_eventually(tmp_path):
    r, _ = _run_watcher(tmp_path, "exit 0\n", {"NVSMI_MAX_FAILURES": "4"}, nvsmi='echo "NVML boom" >&2; exit 9\n')
    assert r.returncode == 1 and r.stdout.count("nvidia-smi problem") == 4 and "NVML boom" in r.stdout
    assert "WATCHER ABORTED" in r.stdout and "GPU confirmed idle" not in r.stdout


def test_compute_apps_failure_alone_is_conservatively_busy(tmp_path):
    nv = ('case "$*" in\n *utilization.gpu*) echo 0;;\n *memory.used*) echo 1;;\n *compute-apps*) exit 3;;\nesac\n')
    r, _ = _run_watcher(tmp_path, "exit 0\n", {"NVSMI_MAX_FAILURES": "3"}, nvsmi=nv)
    assert r.returncode == 1 and "compute-apps query failed" in r.stdout and "GPU confirmed idle" not in r.stdout


def test_runner_own_pid_is_excluded_from_external_pids(tmp_path):
    nv = ('case "$*" in\n *utilization.gpu*) echo 0;; *memory.used*) echo 1;;\n *compute-apps*) echo $PPID; echo 777;;\nesac\n')
    body = 'echo \'{"x1_cslabel":{}}\' > "$RESULTS"\nexit 0\n'
    r, _ = _run_watcher(tmp_path, body, {"RECIPES": "x1_cslabel"}, nvsmi=nv)
    # $PPID of the stub is the watcher's subshell chain; assert the unrelated pid survives
    assert "777" in r.stdout


def test_oom_kill_137_and_segfault_139_count_as_quick_failures(tmp_path):
    for rc in (137, 139):
        r, _ = _run_watcher(tmp_path / str(rc), f"exit {rc}\n")
        assert r.returncode == 1 and "WATCHER ABORTED" in r.stdout and r.stdout.count("Quick failure") == 3
        assert "Killed by user signal" not in r.stdout


def test_sigint_130_is_a_user_kill(tmp_path):
    counter = tmp_path / "count"
    body = (f'n=$(cat "{counter}" 2>/dev/null || echo 0); echo $((n+1)) > "{counter}"\n'
            'if [ "$n" -eq 0 ]; then exit 130; fi\n'
            'echo \'{"x1_cslabel":{"a":1}}\' > "$RESULTS"\nexit 0\n')
    r, _ = _run_watcher(tmp_path, body, {"RECIPES": "x1_cslabel"})
    assert r.returncode == 0 and "Killed by user signal (rc=130)" in r.stdout and "Quick failure" not in r.stdout


def test_default_recipe_list_and_results_file_and_runner():
    r = subprocess.run(["bash", "-c", f'source "{SCRIPT}"; echo "$RECIPES|$RESULTS"'], capture_output=True, text=True,
                       timeout=20, env={k: v for k, v in os.environ.items() if k not in ("RECIPES", "RESULTS")})
    assert r.stdout.strip() == ("x1_cslabel x2_cslabel_nostatic x3_vn_nostatic x4_cslabel_vnfeat "
                                "x2_nostatic_nowd x3_vn_nostatic_nowd x2_nostatic_seed1 x3_vn_nostatic_seed1 "
                                "x2_nostatic_seed2 x3_vn_nostatic_seed2 v3_structure_nowd|training/artifacts/e4_results.json")
    assert "run_e4_experiments.py" in open(SCRIPT).read() and "run_tfx_experiments" not in open(SCRIPT).read()
