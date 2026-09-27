"""CPU tests for run_training's optional per-epoch metric hook (epoch_metric_fn / select_metric_key) and
weight_decay. Defaults must leave the optimizer and checkpoints unchanged (see test_early_stopping.py)."""
import pytest
import torch

import training.train as train_mod
from training.test_run_training_resume import _config


@pytest.fixture(autouse=True)
def _wandb_off(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")


def _script_loss(monkeypatch, seq):
    it = iter(seq)
    monkeypatch.setattr(train_mod, "evaluate_loss", lambda *a, **k: next(it))


def _metric(seq):
    it = iter(seq)

    def fn(model, epoch):
        v = next(it)
        if v == "boom":
            raise RuntimeError("simulated interruption")
        return {"val_ic": v}
    return fn


def _cfg(tmp_path, epochs, metrics, select=True, patience=2):
    cfg = _config(tmp_path, epochs)
    cfg["early_stopping_patience"] = patience
    cfg["epoch_metric_fn"] = _metric(metrics)
    if select:
        cfg["select_metric_key"] = "val_ic"
    return cfg


def test_select_by_metric_ignores_val_loss_and_records_curve(tmp_path, monkeypatch):
    _script_loss(monkeypatch, [1.0, 0.5, 0.4, 0.3, 0.2])          # loss keeps improving
    res = train_mod.run_training(_cfg(tmp_path, 12, [0.01, 0.05, 0.04, 0.03, 0.02]))
    assert res["stopped_reason"] == "early_stopping" and res["last_epoch"] == 3   # best ep 1, 2 bad epochs
    assert res["best_epoch"] == 1 and res["best_metric"] == 0.05
    assert res["best_val_loss"] == 0.3                              # min loss still reported
    assert [h["epoch"] for h in res["epoch_history"]] == [0, 1, 2, 3]
    assert [h["val_ic"] for h in res["epoch_history"]] == [0.01, 0.05, 0.04, 0.03]
    assert [h["val_loss"] for h in res["epoch_history"]] == [1.0, 0.5, 0.4, 0.3]
    assert all("train_loss" in h for h in res["epoch_history"])


def test_best_checkpoint_is_the_best_metric_epoch(tmp_path, monkeypatch):
    _script_loss(monkeypatch, [1.0, 0.9, 0.8])
    cfg = _cfg(tmp_path, 3, [0.01, 0.09, 0.02], patience=5)
    saved = {}
    real_save = torch.save

    def spy(obj, path, *a, **k):
        if str(path).endswith("resume-test.pt"):
            saved["epochs"] = saved.get("epochs", 0) + 1
        return real_save(obj, path, *a, **k)

    monkeypatch.setattr(torch, "save", spy)
    res = train_mod.run_training(cfg)
    assert saved["epochs"] == 2 and res["best_epoch"] == 1      # written at epoch 0 and epoch 1 only


def test_none_or_nan_metric_never_improves(tmp_path, monkeypatch):
    _script_loss(monkeypatch, [1.0] * 5)
    res = train_mod.run_training(_cfg(tmp_path, 12, [0.02, None, float("nan"), 0.01], patience=3))
    assert res["best_epoch"] == 0 and res["last_epoch"] == 3


def test_record_only_mode_selects_by_val_loss(tmp_path, monkeypatch):
    _script_loss(monkeypatch, [1.0, 0.9, 0.95, 0.96, 0.97])
    res = train_mod.run_training(_cfg(tmp_path, 12, [0.1, 0.0, 0.2, 0.3, 0.4], select=False, patience=3))
    assert res["best_epoch"] == 1 and res["stopped_reason"] == "early_stopping" and res["best_val_loss"] == 0.9
    assert len(res["epoch_history"]) == 5


def test_resume_restores_history_best_and_counter(tmp_path, monkeypatch):
    _script_loss(monkeypatch, [1.0, 0.9, 0.8])
    with pytest.raises(RuntimeError, match="simulated"):
        train_mod.run_training(_cfg(tmp_path, 12, [0.05, 0.04, "boom"], patience=3))
    saved = torch.load(tmp_path / "resume-test_inprogress.pt", map_location="cpu")
    assert saved["epoch"] == 1 and saved["epochs_since_improve"] == 1 and saved["best_metric"] == 0.05
    assert saved["best_epoch"] == 0 and len(saved["epoch_history"]) == 2

    _script_loss(monkeypatch, [0.7, 0.6, 0.5, 0.4])
    res = train_mod.run_training(_cfg(tmp_path, 12, [0.03, 0.02, 0.9], patience=3))
    # counter 1 -> 2 (epoch 2) -> 3 (epoch 3) => stop
    assert res["last_epoch"] == 3 and res["best_epoch"] == 0 and res["resumed"]
    assert [h["epoch"] for h in res["epoch_history"]] == [0, 1, 2, 3]


def test_default_config_unchanged_no_hook_keys(tmp_path, monkeypatch):
    _script_loss(monkeypatch, [1.0, 0.9])
    res = train_mod.run_training(_config(tmp_path, 2))
    saved = torch.load(tmp_path / "resume-test_inprogress.pt", map_location="cpu")
    assert set(saved) == {"epoch", "model_state_dict", "optimizer_state_dict", "best_val_loss"}
    assert res["epoch_history"] == []


def test_weight_decay_is_passed_to_adam_only_when_set(tmp_path, monkeypatch):
    seen = []
    real = torch.optim.Adam

    def spy(params, **kw):
        seen.append(kw)
        return real(params, **kw)

    monkeypatch.setattr(torch.optim, "Adam", spy)
    _script_loss(monkeypatch, [1.0])
    train_mod.run_training(_config(tmp_path, 1))
    assert seen[-1] == {"lr": _config(tmp_path, 1)["lr"]}          # default path identical to before
    _script_loss(monkeypatch, [1.0])
    (tmp_path / "wd").mkdir()
    cfg = _config(tmp_path / "wd", 1)
    cfg["weight_decay"] = 1e-3
    train_mod.run_training(cfg)
    assert seen[-1]["weight_decay"] == 1e-3
