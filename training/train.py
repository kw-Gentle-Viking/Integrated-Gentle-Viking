import os
import sys
import torch
import torch.nn.functional as F
import wandb
sys.path.insert(0, "tft-torch")
from tft_torch.tft import TemporalFusionTransformer

# TickerDayDataset (training/dataset.py, Task 12) emits static_feats_categorical shaped
# [batch, 1, num_static] (a singleton "time" dim left over from the per-sample
# reshape(1, -1)). tft-torch's static input channel is NOT time-distributed
# (InputChannelEmbedding(..., time_distribute=False) in tft_torch/tft.py's
# TemporalFusionTransformer.__init__), so it indexes x_categorical as 2D
# [batch, num_static] and raises IndexError on the 3D tensor as-is. Squeeze that
# singleton dim here in the training loop (glue code) rather than in the
# already-reviewed Task 12 dataset.
_STATIC_KEYS = ("static_feats_categorical", "static_feats_numeric")


def _to_device_batch(batch: dict, device) -> dict:
    batch = dict(batch)
    for key in _STATIC_KEYS:
        if key in batch and batch[key].dim() == 3:
            batch[key] = batch[key].squeeze(1)
    return {k: v.to(device) for k, v in batch.items()}


def train_one_epoch(model, dataloader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = dict(batch)
        labels = batch.pop("label", None)
        batch = _to_device_batch(batch, device)
        out = model(batch)
        logits = out["class_logits"].squeeze(1)  # [B, num_classes]
        loss = criterion(logits, labels.squeeze(-1).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * logits.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate_loss(model, dataloader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = dict(batch)
            labels = batch.pop("label")
            batch = _to_device_batch(batch, device)
            out = model(batch)
            logits = out["class_logits"].squeeze(1)
            loss = criterion(logits, labels.squeeze(-1).to(device))
            total_loss += loss.item() * logits.size(0)
    return total_loss / len(dataloader.dataset)


def predict(model, dataloader, device) -> tuple[list[int], list[int]]:
    """Run inference over a dataloader and return (y_true, y_pred) as plain int lists, suitable
    for evaluation.evaluate.compute_metrics. Added for Task 15 (Stage-1 Optuna/ablation search
    needs val macro-F1, not just val loss, as its objective) and reusable by Task 16's
    evaluate_on_test."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = dict(batch)
            labels = batch.pop("label")
            batch = _to_device_batch(batch, device)
            out = model(batch)
            logits = out["class_logits"].squeeze(1)
            preds = logits.argmax(dim=1)
            y_true.extend(labels.squeeze(-1).tolist())
            y_pred.extend(preds.cpu().tolist())
    return y_true, y_pred


def resume_epoch_and_best_loss(checkpoint_state: dict | None) -> tuple[int, float]:
    """Pure decision logic (no I/O): given a loaded epoch-level checkpoint dict (or None if no
    such checkpoint exists yet), return (start_epoch, best_val_loss) to resume training from.
    Extracted as its own function so the resume decision is unit-testable with plain dicts --
    without a real model/optimizer/torch.load call (see training/test_train.py). The actual
    torch.load(...) / model.load_state_dict(...) / optimizer.load_state_dict(...) calls stay in
    run_training, which is the part that genuinely needs a live model+optimizer and isn't worth
    unit-testing (same split this project already uses elsewhere, e.g. training/ablation.py's
    pure run_ablation vs. training/run_stage1_search.py's live orchestration)."""
    if checkpoint_state is None:
        return 0, float("inf")
    return checkpoint_state["epoch"] + 1, checkpoint_state["best_val_loss"]


def run_training(config: dict) -> dict:
    """config keys: tft_config, class_weights, train_loader, val_loader, epochs, lr,
    device, run_name, checkpoint_dir. Optional: epoch_checkpoint_path -- Task 17 Step 5 addition
    for a single long (~1.4-2.8h) training run that should survive being interrupted by shared-GPU
    contention without restarting from epoch 0 (simpler than Task 15's Optuna/SQLite-style
    resumability, since this is one run, not a multi-trial search). When set, an epoch-level
    checkpoint (model/optimizer state + epoch number + best_val_loss so far) is saved after every
    epoch; if that file already exists when run_training starts, training resumes from the next
    epoch instead of epoch 0. Omitting this key (the default) reproduces the exact prior
    behavior -- always starts at epoch 0, never writes an epoch-level checkpoint -- so existing
    callers (training/run_stage1_search.py's train_and_score) are unaffected."""
    device = config["device"]
    model = TemporalFusionTransformer(config["tft_config"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = torch.nn.CrossEntropyLoss(weight=config["class_weights"].to(device))

    epoch_checkpoint_path = config.get("epoch_checkpoint_path")
    checkpoint_state = None
    if epoch_checkpoint_path and os.path.exists(epoch_checkpoint_path):
        checkpoint_state = torch.load(epoch_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_state["model_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    start_epoch, best_val_loss = resume_epoch_and_best_loss(checkpoint_state)
    if start_epoch > 0:
        print(f"[run_training] resuming {config['run_name']} from epoch {start_epoch} "
              f"(best_val_loss so far={best_val_loss:.4f})")

    wandb.init(project="ai-gentle-viking-re", name=config["run_name"], config=config.get("wandb_config", {}))
    checkpoint_path = f"{config['checkpoint_dir']}/{config['run_name']}.pt"
    try:
        for epoch in range(start_epoch, config["epochs"]):
            train_loss = train_one_epoch(model, config["train_loader"], optimizer, criterion, device)
            val_loss = evaluate_loss(model, config["val_loader"], criterion, device)
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), checkpoint_path)
            if epoch_checkpoint_path:
                torch.save({
                    "epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "best_val_loss": best_val_loss,
                }, epoch_checkpoint_path)
    finally:
        # Without this, an exception mid-loop (e.g. a CUDA OOM a caller catches and recovers
        # from, as training/run_stage1_search.py's Optuna/ablation loops do across a long
        # unattended multi-trial run) would leave this trial's wandb run un-finished, causing
        # wandb state confusion on the next wandb.init() call later in the same process.
        wandb.finish()
    return {"best_val_loss": best_val_loss, "checkpoint_path": checkpoint_path}
