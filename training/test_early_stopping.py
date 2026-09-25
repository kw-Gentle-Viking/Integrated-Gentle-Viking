"""CPU tests for run_training's optional early stopping / budget stop (tiny synthetic data, real
torch.save/load resume). Val losses are scripted so plateaus are forced deterministically."""
import pytest
import torch

import training.train as train_mod
from training.test_run_training_resume import _config


@pytest.fixture(autouse=True)
def _wandb_off(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")


def _script(monkeypatch, seq):
    """evaluate_loss returns seq values in order; the string 'boom' raises (simulated kill)."""
    it = iter(seq)

    def fake_eval(*a, **k):
        v = next(it)
        if v == "boom":
            raise RuntimeError("simulated interruption")
        return v

    monkeypatch.setattr(train_mod, "evaluate_loss", fake_eval)


def _count_train_calls(monkeypatch):
    real, calls = train_mod.train_one_epoch, {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(train_mod, "train_one_epoch", counting)
    return calls


def test_default_no_patience_runs_all_epochs_and_checkpoint_has_no_new_keys(tmp_path, monkeypatch):
    _script(monkeypatch, [1.0, 0.9, 0.95, 0.96, 0.97])
    calls = _count_train_calls(monkeypatch)
    res = train_mod.run_training(_config(tmp_path, 5))
    assert calls["n"] == 5
    saved = torch.load(tmp_path / "resume-test_inprogress.pt", map_location="cpu")
    assert set(saved) == {"epoch", "model_state_dict", "optimizer_state_dict", "best_val_loss"}
    assert res["best_val_loss"] == 0.9 and res["stopped_reason"] == "max_epochs"


def test_plateau_stops_early_after_patience_epochs(tmp_path, monkeypatch):
    _script(monkeypatch, [1.0, 0.9, 0.95, 0.96, 0.97, 0.5, 0.4])   # would improve later if not stopped
    calls = _count_train_calls(monkeypatch)
    cfg = _config(tmp_path, 12)
    cfg["early_stopping_patience"] = 3
    res = train_mod.run_training(cfg)
    assert calls["n"] == 5                       # best at epoch 1, then 3 non-improving epochs
    assert res["stopped_reason"] == "early_stopping" and res["best_val_loss"] == 0.9
    assert torch.load(tmp_path / "resume-test.pt", map_location="cpu")   # best-val checkpoint exists


def test_equal_loss_is_not_an_improvement(tmp_path, monkeypatch):
    _script(monkeypatch, [1.0, 1.0, 1.0, 1.0])
    calls = _count_train_calls(monkeypatch)
    cfg = _config(tmp_path, 12)
    cfg["early_stopping_patience"] = 3
    train_mod.run_training(cfg)
    assert calls["n"] == 4


def test_resume_mid_plateau_restores_patience_counter(tmp_path, monkeypatch):
    # epochs 0,1,2 done (counter=1 after epoch 2); kill during epoch 3 validation.
    _script(monkeypatch, [1.0, 0.9, 0.95, "boom"])
    cfg = _config(tmp_path, 12)
    cfg["early_stopping_patience"] = 3
    with pytest.raises(RuntimeError, match="simulated"):
        train_mod.run_training(cfg)
    saved = torch.load(tmp_path / "resume-test_inprogress.pt", map_location="cpu")
    assert saved["epoch"] == 2 and saved["epochs_since_improve"] == 1

    _script(monkeypatch, [0.96, 0.97, 0.1, 0.05])
    calls = _count_train_calls(monkeypatch)
    cfg = _config(tmp_path, 12)
    cfg["early_stopping_patience"] = 3
    res = train_mod.run_training(cfg)
    # counter continues 1 -> 2 (epoch 3) -> 3 (epoch 4) => stop. A reset counter would run 3 more.
    assert calls["n"] == 2 and res["stopped_reason"] == "early_stopping"
    assert res["best_val_loss"] == 0.9


def test_resume_after_early_stop_trains_nothing(tmp_path, monkeypatch):
    _script(monkeypatch, [1.0, 1.1, 1.2, 1.3])
    cfg = _config(tmp_path, 12)
    cfg["early_stopping_patience"] = 3
    train_mod.run_training(cfg)
    calls = _count_train_calls(monkeypatch)
    _script(monkeypatch, [0.0] * 10)
    cfg = _config(tmp_path, 12)
    cfg["early_stopping_patience"] = 3
    res = train_mod.run_training(cfg)
    assert calls["n"] == 0 and res["stopped_reason"] == "early_stopping"


def test_old_checkpoint_without_counter_resumes_with_zero(tmp_path, monkeypatch):
    _script(monkeypatch, [1.0, 0.9, "boom"])
    with pytest.raises(RuntimeError):
        train_mod.run_training(_config(tmp_path, 12))       # no patience => old-format checkpoint
    saved = torch.load(tmp_path / "resume-test_inprogress.pt", map_location="cpu")
    assert "epochs_since_improve" not in saved
    _script(monkeypatch, [0.95, 0.96, 0.97, 0.5])
    calls = _count_train_calls(monkeypatch)
    cfg = _config(tmp_path, 12)
    cfg["early_stopping_patience"] = 3
    train_mod.run_training(cfg)
    assert calls["n"] == 3                                  # counter starts at 0 -> needs 3 bad epochs


def test_should_stop_hook_breaks_after_checkpoint_and_is_resumable(tmp_path, monkeypatch):
    _script(monkeypatch, [1.0, 0.9, 0.8, 0.7])
    cfg = _config(tmp_path, 4)
    flag = {"n": 0}

    def stop():
        flag["n"] += 1
        return flag["n"] >= 2            # asked after epoch 0 (no) and after epoch 1 (yes)

    cfg["should_stop"] = stop
    res = train_mod.run_training(cfg)
    assert res["stopped_reason"] == "budget" and res["last_epoch"] == 1
    saved = torch.load(tmp_path / "resume-test_inprogress.pt", map_location="cpu")
    assert saved["epoch"] == 1                            # state saved BEFORE stopping -> resumable
    _script(monkeypatch, [0.6, 0.5])
    calls = _count_train_calls(monkeypatch)
    res = train_mod.run_training(_config(tmp_path, 4))
    assert calls["n"] == 2 and res["stopped_reason"] == "max_epochs"


def test_should_stop_not_asked_after_final_epoch(tmp_path, monkeypatch):
    _script(monkeypatch, [1.0, 0.9])
    cfg = _config(tmp_path, 2)
    cfg["should_stop"] = lambda: True
    res = train_mod.run_training(cfg)
    assert res["last_epoch"] == 1 or res["stopped_reason"] in ("budget", "max_epochs")


def test_epoch_log_callback(tmp_path, monkeypatch):
    _script(monkeypatch, [1.0, 0.9])
    lines = []
    cfg = _config(tmp_path, 2)
    cfg["epoch_log"] = lines.append
    train_mod.run_training(cfg)
    assert len(lines) == 2 and "epoch 0" in lines[0] and "val_loss" in lines[0]
