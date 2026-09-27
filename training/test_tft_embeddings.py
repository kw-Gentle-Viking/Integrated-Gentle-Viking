import numpy as np
import pandas as pd
import torch

from training.config import build_tft_config
from training.dataset import TickerDayDataset
from training.tft_embeddings import extract_embeddings

HIST_COLS = ["f1", "f2"]
FUT_COLS = ["fut1"]
STATIC_COLS = ["s1", "s2"]


def _toy_frames(n_tickers=3, n_days=65, seed=0):
    # training.signal_data.sample_meta (used by extract_embeddings) unconditionally reads
    # row["log_ret"] regardless of which columns the model itself consumes -- every toy frame
    # needs that column even though it is not in HIST_COLS.
    rng = np.random.default_rng(seed)
    frames = {}
    for i in range(n_tickers):
        dates = pd.date_range("2023-01-02", periods=n_days, freq="B")
        df = pd.DataFrame({
            "trade_date": dates,
            "f1": rng.standard_normal(n_days), "f2": rng.standard_normal(n_days),
            "log_ret": rng.standard_normal(n_days) * 0.02,
            "fut1": np.ones(n_days), "s1": rng.integers(0, 2, n_days),
            "s2": rng.integers(0, 2, n_days),
            "label": rng.integers(0, 3, n_days),
        })
        frames[f"T{i}"] = df
    return frames


def _toy_model():
    cfg = build_tft_config(
        {"historical": HIST_COLS, "future": FUT_COLS, "static_cardinalities": [2, 2]},
        num_classes=3, state_size=4, attention_heads=2, lstm_layers=1, dropout=0.0)
    from tft_torch.tft import TemporalFusionTransformer
    return TemporalFusionTransformer(cfg)


def test_extract_embeddings_shape_and_keys_match_dataset_order():
    frames = _toy_frames()
    ds = TickerDayDataset(frames, HIST_COLS, FUT_COLS, STATIC_COLS, encoder_len=60, align="today")
    model = _toy_model()
    keys, emb = extract_embeddings(model, ds, torch.device("cpu"), batch_size=8)
    assert len(keys) == len(ds) == emb.shape[0]
    assert emb.shape[1] == 4  # state_size
    assert np.isfinite(emb).all()
    from training.signal_data import sample_meta
    expected_keys = [(tk, d) for tk, d, _ in sample_meta(ds)]
    assert keys == expected_keys


def test_extract_embeddings_no_grad_and_deterministic_in_eval_mode():
    frames = _toy_frames()
    ds = TickerDayDataset(frames, HIST_COLS, FUT_COLS, STATIC_COLS, encoder_len=60, align="today")
    model = _toy_model()
    model.eval()
    _, emb1 = extract_embeddings(model, ds, torch.device("cpu"))
    _, emb2 = extract_embeddings(model, ds, torch.device("cpu"))
    np.testing.assert_allclose(emb1, emb2)
    assert all(not p.requires_grad or p.grad is None for p in model.parameters())
