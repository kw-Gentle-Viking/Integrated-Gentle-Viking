"""Unit test of the watcher's done-check and control flow with temp files and stubbed nvidia-smi/python
(no GPU, nothing launched)."""
import json
import os
import stat
import subprocess

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_tfx_watch.sh")


def _done(results_path, recipes="aligned std std_vn std_vn_csr"):
    r = subprocess.run(["bash", "-c", f'source "{SCRIPT}"; tfx_done'],
                       env={**os.environ, "RESULTS": str(results_path), "RECIPES": recipes})
    return r.returncode == 0


def test_bash_syntax_and_executable():
    assert subprocess.run(["bash", "-n", SCRIPT]).returncode == 0
    assert os.stat(SCRIPT).st_mode & stat.S_IXUSR


def test_sourcing_does_not_run_the_watcher(tmp_path):
    r = subprocess.run(["bash", "-c", f'source "{SCRIPT}"; echo sourced-ok'], capture_output=True, text=True, timeout=20)
    assert r.stdout.strip() == "sourced-ok"


def test_done_check(tmp_path):
    p = tmp_path / "res.json"
    assert not _done(p)                                            # missing file
    p.write_text("{not json")
    assert not _done(p)                                            # corrupt file
    p.write_text(json.dumps({"aligned": {}, "std": {}, "std_vn": {}}))
    assert not _done(p)                                            # one recipe missing
    p.write_text(json.dumps({"aligned": {}, "std": {}, "std_vn": {}, "std_vn_csr": {}}))
    assert _done(p)
    assert _done(p, recipes="std")
    assert not _done(p, recipes="std other")


def _stub(dir_, name, body):
    f = dir_ / name
    f.write_text("#!/bin/bash\n" + body)
    f.chmod(0o755)
    return f


def _run_watcher(tmp_path, py_body, extra_env=None):
    """Full main loop in a temp worktree with stub python + stub nvidia-smi and zero sleeps."""
    wt = tmp_path / "wt"
    (wt / "training" / "artifacts").mkdir(parents=True)
    (wt / ".env").write_text("X=1\n")
    (wt / "training" / "run_tfx_experiments.py").write_text("")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _stub(bindir, "nvidia-smi", "echo 0\n")
    py = _stub(tmp_path, "fakepython", py_body)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "WORKTREE": str(wt), "PYTHON": str(py),
           "RESULTS": str(wt / "res.json"), "POLL_INTERVAL_SEC": "0", "COOLDOWN_AFTER_KILL_SEC": "0",
           "QUICK_FAILURE_SECONDS": "300", **(extra_env or {})}
    return subprocess.run(["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=60), wt


def test_watcher_completes_when_runner_records_all_recipes(tmp_path):
    body = ('echo \'{"aligned":{},"std":{},"std_vn":{},"std_vn_csr":{}}\' > "$RESULTS"\nexit 0\n')
    r, _ = _run_watcher(tmp_path, body)
    assert r.returncode == 0 and "tfx experiments COMPLETE" in r.stdout
    assert "utilization=0% x3" in r.stdout


def test_watcher_aborts_after_three_quick_failures(tmp_path):
    r, _ = _run_watcher(tmp_path, "exit 2\n")
    assert r.returncode == 1 and "WATCHER ABORTED" in r.stdout and r.stdout.count("Quick failure") == 3


def test_watcher_cools_down_and_resumes_after_signal_kill(tmp_path):
    counter = tmp_path / "count"
    body = (f'n=$(cat "{counter}" 2>/dev/null || echo 0); echo $((n+1)) > "{counter}"\n'
            'if [ "$n" -eq 0 ]; then exit 143; fi\n'
            'echo \'{"aligned":{},"std":{},"std_vn":{},"std_vn_csr":{}}\' > "$RESULTS"\nexit 0\n')
    r, _ = _run_watcher(tmp_path, body)
    assert r.returncode == 0 and "Killed by signal (rc=143)" in r.stdout and "COMPLETE" in r.stdout
    assert "Quick failure" not in r.stdout
