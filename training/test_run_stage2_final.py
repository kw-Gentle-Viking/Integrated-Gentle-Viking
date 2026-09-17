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
