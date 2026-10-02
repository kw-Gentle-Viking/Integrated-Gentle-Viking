import torch

from evaluation.evaluate import compute_metrics, split_by_regime
from training.config import KNOWN_FUTURE_COLS, build_tft_config
from training.run_stage1_search import STATIC_CARDINALITIES
from training.train import TemporalFusionTransformer, predict


def select_champion(versions: list[dict]) -> dict:
    if not versions:
        raise ValueError("no versions to select from")
    return max(versions, key=lambda v: v["macro_f1_val"])


def evaluate_on_test(model_path: str, test_loader, class_columns: list[str],
                      hparams: dict, dates: list[str]) -> dict:
    """Evaluate a trained Stage-1 champion checkpoint on a held-out test set.

    Signature note: the task-16 brief's Interfaces section lists this as
    `evaluate_on_test(model_path, test_loader, class_columns)`. That 3-arg form is not enough to
    actually do the job it describes: reconstructing a `TemporalFusionTransformer` whose
    `load_state_dict(...)` matches `model_path`'s saved shapes requires the exact architecture
    hyperparameters it was trained with (state_size/attention_heads/lstm_layers/dropout -- these
    vary per candidate, see docs/model_versions.md), and computing `split_by_regime` requires each
    sample's trade_date aligned to `test_loader`'s iteration order. Neither is derivable from
    `model_path`/`test_loader`/`class_columns` alone, so this extends the spec with two more
    required params: `hparams` (the champion's architecture dict, same shape as
    training/run_stage1_search.py's `train_and_score(hparams=...)`) and `dates` (per-sample
    trade_date strings, same order as `test_loader`'s -- i.e. built from the same
    TickerDayDataset.index order, shuffle=False). `class_columns` is the champion's historical
    feature column list (what run_stage1_search.py calls `historical_cols`) -- it is what
    `build_tft_config` needs for `num_historical_numeric`.

    Returns a dict with keys: "metrics" (compute_metrics output over the full test set) and
    "regime_split" (split_by_regime output: "pre_leverage" / "leverage_era" sub-dicts).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_columns = {
        "historical": class_columns, "future": KNOWN_FUTURE_COLS,
        "static_cardinalities": STATIC_CARDINALITIES,
    }
    tft_config = build_tft_config(
        feature_columns, num_classes=3, state_size=hparams["state_size"],
        attention_heads=hparams["attention_heads"], lstm_layers=hparams["lstm_layers"],
        dropout=hparams["dropout"],
    )
    model = TemporalFusionTransformer(tft_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    y_true, y_pred = predict(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    regime_split = split_by_regime(dates, y_true, y_pred)
    return {"metrics": metrics, "regime_split": regime_split}
