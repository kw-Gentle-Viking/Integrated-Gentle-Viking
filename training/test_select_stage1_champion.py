from training.select_stage1_champion import evaluate_on_test, select_champion


def test_selects_highest_val_macro_f1():
    versions = [
        {"version": "v1", "macro_f1_val": 0.41, "columns": ["a", "b"]},
        {"version": "v2", "macro_f1_val": 0.53, "columns": ["a", "b", "c"]},
        {"version": "v3", "macro_f1_val": 0.47, "columns": ["a"]},
    ]
    champion = select_champion(versions)
    assert champion["version"] == "v2"


def test_raises_on_empty_versions():
    import pytest
    with pytest.raises(ValueError, match="no versions"):
        select_champion([])


def _make_tiny_test_loader(n_samples: int, encoder_len: int, n_historical: int, n_future: int,
                            n_static: int):
    """A tiny fake dataset shaped exactly like training.dataset.TickerDayDataset's per-sample
    output (see its __getitem__), so it can stand in for a real TickerDayDataset/DataLoader
    without touching the DB or requiring a GPU."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    class _FakeDataset(Dataset):
        def __len__(self):
            return n_samples

        def __getitem__(self, idx):
            return {
                "historical_ts_numeric": torch.randn(encoder_len, n_historical),
                "future_ts_numeric": torch.randn(1, n_future),
                "static_feats_categorical": torch.zeros(1, n_static, dtype=torch.long),
                "label": torch.tensor([idx % 3]),
            }

    return DataLoader(_FakeDataset(), batch_size=4, shuffle=False)


def test_evaluate_on_test_runs_a_tiny_model_end_to_end(tmp_path):
    """No real DB/GPU: a tiny randomly-initialized TemporalFusionTransformer, saved to a temp
    checkpoint, evaluated over a small synthetic loader (shaped like a real TickerDayDataset
    batch) and a matching synthetic `dates` list. This is a genuine unit test (unlike
    training/run_stage1_champion_selection.py's live DB+GPU orchestration, which is exercised
    manually, not via pytest -- same split this project already uses between test_config.py's
    unit tests and Task 13's live smoke-test script)."""
    import torch

    from training.config import KNOWN_FUTURE_COLS, build_tft_config
    from training.train import TemporalFusionTransformer

    historical_cols = ["a", "b"]
    hparams = {"state_size": 4, "attention_heads": 1, "lstm_layers": 1, "dropout": 0.0}
    feature_columns = {
        "historical": historical_cols, "future": KNOWN_FUTURE_COLS,
        "static_cardinalities": [21, 3],
    }
    tft_config = build_tft_config(feature_columns, num_classes=3, **hparams)
    model = TemporalFusionTransformer(tft_config)

    checkpoint_path = tmp_path / "tiny_champion.pt"
    torch.save(model.state_dict(), checkpoint_path)

    n_samples = 6
    test_loader = _make_tiny_test_loader(
        n_samples, encoder_len=60, n_historical=len(historical_cols),
        n_future=len(KNOWN_FUTURE_COLS), n_static=2)
    dates = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"]

    result = evaluate_on_test(str(checkpoint_path), test_loader, historical_cols, hparams, dates)

    assert set(result.keys()) == {"metrics", "regime_split"}
    assert "macro_f1" in result["metrics"]
    assert result["metrics"]["confusion_matrix"] and len(result["metrics"]["confusion_matrix"]) == 3
    assert set(result["regime_split"].keys()) == {"pre_leverage", "leverage_era"}
    # every synthetic date is well before the 2026-05-27 leverage-ETF launch
    assert result["regime_split"]["leverage_era"]["accuracy"] == 0.0
    assert result["regime_split"]["pre_leverage"]["confusion_matrix"] == result["metrics"]["confusion_matrix"]
