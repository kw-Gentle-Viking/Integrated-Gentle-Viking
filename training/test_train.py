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
