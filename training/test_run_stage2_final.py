import pandas as pd
import pytest

from training.run_stage2_final import build_stage2_config, split_held_out_by_trading_days


def test_stage2_uses_full_date_range_and_champion_columns():
    champion = {"version": "v2", "columns": ["log_ret", "est_rebalancing_flow"],
                 "state_size": 64, "attention_heads": 4, "lstm_layers": 2, "dropout": 0.15}
    cfg = build_stage2_config(champion, today="2026-09-08")
    assert cfg["train_start"] == "2019-01-02"
    assert cfg["train_end"] == "2026-09-08"
    assert cfg["columns"] == ["log_ret", "est_rebalancing_flow"]
    assert cfg["state_size"] == 64
    assert cfg["run_name"] == "stage2-final-v2"


def _make_ticker_df(dates: list[str], ticker: str = "005930") -> pd.DataFrame:
    return pd.DataFrame({
        "trade_date": dates,
        "log_ret": [0.01 * i for i in range(len(dates))],
        "label": [i % 3 for i in range(len(dates))],
    })


def test_split_held_out_uses_last_n_global_dates_and_excludes_them_from_train():
    dates = [f"2024-01-{d:02d}" for d in range(1, 21)]  # 20 trading days
    ticker_dfs = {"A": _make_ticker_df(dates)}

    train_dfs, held_out_dfs, held_out_dates = split_held_out_by_trading_days(
        ticker_dfs, n_held_out_days=5, encoder_len=3)

    assert held_out_dates == dates[-5:]
    assert list(train_dfs["A"]["trade_date"]) == dates[:15]
    assert max(train_dfs["A"]["trade_date"]) < held_out_dates[0]
    # held-out df = 3-day buffer (from train tail) + 5 held-out target rows = 8 rows
    assert list(held_out_dfs["A"]["trade_date"]) == dates[12:20]


def test_split_held_out_is_global_across_tickers_not_per_ticker():
    """held_out_dates is computed from the union of ALL tickers' dates, so a ticker missing the
    globally-latest date still gets split at the same held_out_start as every other ticker."""
    dates_a = [f"2024-01-{d:02d}" for d in range(1, 11)]  # through Jan 10
    dates_b = [f"2024-01-{d:02d}" for d in range(1, 8)]  # ticker B stops at Jan 7 (e.g. delisted)
    ticker_dfs = {"A": _make_ticker_df(dates_a, "A"), "B": _make_ticker_df(dates_b, "B")}

    train_dfs, held_out_dfs, held_out_dates = split_held_out_by_trading_days(
        ticker_dfs, n_held_out_days=3, encoder_len=2)

    assert held_out_dates == ["2024-01-08", "2024-01-09", "2024-01-10"]
    # B has no rows on/after held_out_start: whatever B contributes to held_out_dfs (at most
    # encoder_len context rows, if any) is pure pre-held-out-window context, never a held-out
    # target row -- so TickerDayDataset windowing would produce zero usable samples for B.
    if "B" in held_out_dfs:
        assert (held_out_dfs["B"]["trade_date"] < held_out_dates[0]).all()
    assert list(train_dfs["B"]["trade_date"]) == dates_b


def test_split_held_out_raises_when_too_few_trading_days():
    ticker_dfs = {"A": _make_ticker_df([f"2024-01-{d:02d}" for d in range(1, 4)])}
    with pytest.raises(ValueError, match="only 3 distinct trade_dates"):
        split_held_out_by_trading_days(ticker_dfs, n_held_out_days=100, encoder_len=60)


# --- classweight-sweep follow-up: scheme-aware run names and gated serving promotion ---

from training.run_stage2_final import (  # noqa: E402
    DEFAULT_WEIGHT_SCHEME,
    promote_to_serving,
    promotion_plan,
    stage2_run_name,
)


def test_run_name_unchanged_for_default_balanced_scheme():
    # Must stay identical to build_stage2_config's run_name so the existing
    # stage2-final-stage1-remove_lev_total_volume row/checkpoint keep working.
    assert DEFAULT_WEIGHT_SCHEME == "balanced"
    assert stage2_run_name("stage2-final-v2", "balanced") == "stage2-final-v2"


def test_run_name_gets_scheme_suffix_for_non_default_schemes():
    assert stage2_run_name("stage2-final-v2", "uniform") == "stage2-final-v2-classweight-uniform"
    assert stage2_run_name("stage2-final-v2", "mild") == "stage2-final-v2-classweight-mild"
    assert stage2_run_name("stage2-final-v2", "mild") != stage2_run_name("stage2-final-v2", "balanced")


def test_promotion_plan_default_scheme_keeps_legacy_auto_copy_without_backup():
    assert promotion_plan("balanced", promote_flag=False) == {"promote": True, "backup": False}


def test_promotion_plan_non_default_scheme_never_promotes_without_flag():
    assert promotion_plan("uniform", promote_flag=False) == {"promote": False, "backup": False}
    assert promotion_plan("mild", promote_flag=False) == {"promote": False, "backup": False}


def test_promotion_plan_non_default_scheme_with_flag_promotes_and_backs_up():
    assert promotion_plan("mild", promote_flag=True) == {"promote": True, "backup": True}


def test_promote_to_serving_backs_up_old_checkpoint_then_overwrites(tmp_path):
    src = tmp_path / "new.pt"
    serving = tmp_path / "best_model_state_dict.pt"
    backup = tmp_path / "best_model_state_dict.pt.stage2-balanced-backup"
    src.write_bytes(b"NEW")
    serving.write_bytes(b"OLD")

    promote_to_serving(str(src), str(serving), str(backup), backup=True)

    assert serving.read_bytes() == b"NEW"
    assert backup.read_bytes() == b"OLD"


def test_promote_to_serving_never_clobbers_an_existing_backup(tmp_path):
    """A second promotion must not overwrite the original balanced backup with the previously
    promoted model -- the backup name promises the balanced model."""
    src = tmp_path / "new.pt"
    serving = tmp_path / "best_model_state_dict.pt"
    backup = tmp_path / "best_model_state_dict.pt.stage2-balanced-backup"
    src.write_bytes(b"NEW2")
    serving.write_bytes(b"PROMOTED1")
    backup.write_bytes(b"ORIGINAL_BALANCED")

    promote_to_serving(str(src), str(serving), str(backup), backup=True)

    assert backup.read_bytes() == b"ORIGINAL_BALANCED"
    assert serving.read_bytes() == b"NEW2"
    extra = [p for p in tmp_path.iterdir() if p.name.startswith("best_model_state_dict.pt.pre-promote-")]
    assert len(extra) == 1 and extra[0].read_bytes() == b"PROMOTED1"


def test_promote_to_serving_without_backup_flag_just_copies(tmp_path):
    src = tmp_path / "new.pt"
    serving = tmp_path / "s.pt"
    backup = tmp_path / "s.pt.bak"
    src.write_bytes(b"NEW")
    serving.write_bytes(b"OLD")

    promote_to_serving(str(src), str(serving), str(backup), backup=False)

    assert serving.read_bytes() == b"NEW"
    assert not backup.exists()
