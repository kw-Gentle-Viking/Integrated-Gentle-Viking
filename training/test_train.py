import pytest
from training.train import resume_epoch_and_best_loss


def test_resume_epoch_and_best_loss_no_checkpoint_starts_fresh():
    start_epoch, best_val_loss = resume_epoch_and_best_loss(None)
    assert start_epoch == 0
    assert best_val_loss == float("inf")


def test_resume_epoch_and_best_loss_resumes_from_next_epoch():
    checkpoint_state = {
        "epoch": 4, "best_val_loss": 1.0234,
        "model_state_dict": {}, "optimizer_state_dict": {},
    }
    start_epoch, best_val_loss = resume_epoch_and_best_loss(checkpoint_state)
    assert start_epoch == 5
    assert best_val_loss == 1.0234


def test_resume_epoch_and_best_loss_resumes_from_epoch_zero_checkpoint():
    """A checkpoint saved after epoch 0 (the very first epoch) must resume at epoch 1, not be
    mistaken for 'no checkpoint' -- guards against an off-by-one using falsy epoch=0."""
    checkpoint_state = {
        "epoch": 0, "best_val_loss": 1.5,
        "model_state_dict": {}, "optimizer_state_dict": {},
    }
    start_epoch, best_val_loss = resume_epoch_and_best_loss(checkpoint_state)
    assert start_epoch == 1
    assert best_val_loss == 1.5


# --- predict_proba (plan S3) ---
import torch
from training.train import predict_proba, predict


class _FakeModel(torch.nn.Module):
    """class_logits = the batch's 'x' feature broadcast to [B, 1, 3] (mimics TFT output shape)."""
    def forward(self, batch):
        return {"class_logits": batch["x"].unsqueeze(1)}


def _fake_loader():
    b1 = {"x": torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 3.0]]), "label": torch.tensor([[0], [2]])}
    b2 = {"x": torch.tensor([[0.0, 1.0, 0.0]]), "label": torch.tensor([[2]])}
    return [b1, b2]


def test_predict_proba_returns_labels_and_softmax_rows_summing_to_one():
    y, p = predict_proba(_FakeModel(), _fake_loader(), torch.device("cpu"))
    assert y == [0, 2, 2]
    assert p.shape == (3, 3)
    assert p.sum(axis=1) == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)
    assert p.argmax(axis=1).tolist() == [0, 2, 1]


def test_predict_proba_argmax_matches_predict():
    y1, pred = predict(_FakeModel(), _fake_loader(), torch.device("cpu"))
    y2, p = predict_proba(_FakeModel(), _fake_loader(), torch.device("cpu"))
    assert y1 == y2 and pred == p.argmax(axis=1).tolist()


def test_predict_proba_empty_loader():
    y, p = predict_proba(_FakeModel(), [], torch.device("cpu"))
    assert y == [] and p.shape == (0, 3)
