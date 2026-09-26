"""CPU-only end-to-end check that run_training's epoch-level resume actually works: interrupt a
real (tiny, synthetic-data) training run mid-way, re-invoke it, and confirm it resumes from the
next epoch instead of restarting from epoch 0.

training/test_train.py already unit-tests the pure resume *decision* (resume_epoch_and_best_loss)
with plain dicts; this test exercises the part that decision can't cover -- the real
torch.save/torch.load round trip of model+optimizer state through run_training, which until now
had only been exercised (never interrupted) by the Stage-2 live run."""
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

import training.train as train_mod
from training.config import build_tft_config
from training.dataset import TickerDayDataset

HIST = ["log_ret", "disparity_20d"]
FUT = ["time_progress", "is_bok"]
STATIC = ["sector_id", "market_id"]


def _tiny_loaders():
    rng = np.random.default_rng(0)
    n = 70  # encoder_len=60 -> 10 usable samples
    df = pd.DataFrame({
        "log_ret": rng.standard_normal(n) * 0.01,
        "disparity_20d": rng.standard_normal(n) * 0.02,
        "time_progress": 1.0, "is_bok": 0.0, "sector_id": 3, "market_id": 1,
        "label": rng.integers(0, 3, n).astype(float),
    })
    ds = TickerDayDataset({"T": df}, HIST, FUT, STATIC, encoder_len=60)
    return DataLoader(ds, batch_size=4), DataLoader(ds, batch_size=4)


def _config(tmp_path, epochs):
    train_loader, val_loader = _tiny_loaders()
    cfg = build_tft_config({"historical": HIST, "future": FUT, "static_cardinalities": [21, 3]},
                            num_classes=3, state_size=8, attention_heads=2, lstm_layers=1)
    return {
        "tft_config": cfg, "class_weights": torch.ones(3),
        "train_loader": train_loader, "val_loader": val_loader,
        "epochs": epochs, "lr": 1e-3, "device": torch.device("cpu"),
        "run_name": "resume-test", "checkpoint_dir": str(tmp_path),
        "epoch_checkpoint_path": str(tmp_path / "resume-test_inprogress.pt"),
    }


def test_interrupted_run_resumes_from_next_epoch_not_from_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    epochs = 3

    real_eval = train_mod.evaluate_loss
    real_train = train_mod.train_one_epoch
    calls = {"eval": 0}

    def flaky_eval(*a, **k):
        calls["eval"] += 1
        if calls["eval"] == 2:  # epoch 0 fully done+checkpointed; die during epoch 1's validation
            raise RuntimeError("simulated interruption")
        return real_eval(*a, **k)

    monkeypatch.setattr(train_mod, "evaluate_loss", flaky_eval)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_mod.run_training(_config(tmp_path, epochs))
    assert (tmp_path / "resume-test_inprogress.pt").exists()
    saved = torch.load(tmp_path / "resume-test_inprogress.pt", map_location="cpu")
    assert saved["epoch"] == 0  # only epoch 0 was fully completed before the interruption

    # Re-invoke with a healthy evaluate_loss and count how many epochs of training actually run.
    monkeypatch.setattr(train_mod, "evaluate_loss", real_eval)
    train_calls = {"n": 0}

    def counting_train(*a, **k):
        train_calls["n"] += 1
        return real_train(*a, **k)

    monkeypatch.setattr(train_mod, "train_one_epoch", counting_train)
    result = train_mod.run_training(_config(tmp_path, epochs))

    # Resumed from epoch 1 -> trains epochs 1 and 2 only (2 calls). A restart-from-scratch bug
    # would train all 3.
    assert train_calls["n"] == epochs - 1
    assert np.isfinite(result["best_val_loss"])
    assert (tmp_path / "resume-test.pt").exists()
    final = torch.load(tmp_path / "resume-test_inprogress.pt", map_location="cpu")
    assert final["epoch"] == epochs - 1


def _interrupt_after_epoch0(tmp_path, monkeypatch, **extra):
    real_eval = train_mod.evaluate_loss
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated interruption")
        return real_eval(*a, **k)

    monkeypatch.setattr(train_mod, "evaluate_loss", flaky)
    with pytest.raises(RuntimeError):
        train_mod.run_training({**_config(tmp_path, 3), **extra})
    monkeypatch.setattr(train_mod, "evaluate_loss", real_eval)


def test_matching_fingerprint_resumes_and_records_resume_meta(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    fp = {"align": "today", "columns_hash": "abc"}
    _interrupt_after_epoch0(tmp_path, monkeypatch, fingerprint=fp, seed=5)
    assert torch.load(tmp_path / "resume-test_inprogress.pt", map_location="cpu")["fingerprint"] == fp
    res = train_mod.run_training({**_config(tmp_path, 3), "fingerprint": fp, "seed": 5})
    assert res["resumed"] and res["start_epoch"] == 1 and res["resume_count"] == 1 and res["stale_checkpoint"] is None


def test_fingerprint_mismatch_marks_inprogress_stale_and_restarts_from_scratch(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    _interrupt_after_epoch0(tmp_path, monkeypatch, fingerprint={"columns_hash": "old"})
    real_train, n = train_mod.train_one_epoch, {"n": 0}
    monkeypatch.setattr(train_mod, "train_one_epoch", lambda *a, **k: (n.__setitem__("n", n["n"] + 1), real_train(*a, **k))[1])
    res = train_mod.run_training({**_config(tmp_path, 3), "fingerprint": {"columns_hash": "new"}})
    assert n["n"] == 3 and not res["resumed"] and res["start_epoch"] == 0
    assert res["stale_checkpoint"].endswith(".stale") and (tmp_path / "resume-test_inprogress.pt.stale").exists()
    assert "MISMATCH" in capsys.readouterr().out
    assert torch.load(tmp_path / "resume-test_inprogress.pt", map_location="cpu")["fingerprint"] == {"columns_hash": "new"}


def test_checkpoint_without_fingerprint_is_stale_when_config_has_one(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    _interrupt_after_epoch0(tmp_path, monkeypatch)          # legacy checkpoint: no fingerprint
    res = train_mod.run_training({**_config(tmp_path, 3), "fingerprint": {"x": 1}})
    assert res["start_epoch"] == 0 and res["stale_checkpoint"]


def test_epoch_seed_makes_shuffle_depend_on_epoch_not_on_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    seen = []
    monkeypatch.setattr(train_mod, "train_one_epoch",
                        lambda *a, **k: (seen.append(torch.rand(1).item()), 0.0)[1])
    monkeypatch.setattr(train_mod, "evaluate_loss", lambda *a, **k: 1.0)
    train_mod.run_training({**_config(tmp_path, 2), "seed": 7})
    full = list(seen)
    assert full[0] != full[1]                       # epoch 1 does not replay epoch 0's RNG stream
    # resume at epoch 1 (state saved by the run above) must draw exactly epoch 1's stream
    st = torch.load(tmp_path / "resume-test_inprogress.pt", map_location="cpu")
    st["epoch"] = 0
    torch.save(st, tmp_path / "resume-test_inprogress.pt")
    seen.clear()
    train_mod.run_training({**_config(tmp_path, 2), "seed": 7})
    assert seen == [full[1]]
