import pandas as pd


def build_stage2_config(champion: dict, today: str) -> dict:
    return {
        "train_start": "2019-01-02", "train_end": today,
        "columns": champion["columns"],
        "state_size": champion["state_size"], "attention_heads": champion["attention_heads"],
        "lstm_layers": champion["lstm_layers"], "dropout": champion["dropout"],
        "run_name": f"stage2-final-{champion['version']}",
    }


def split_held_out_by_trading_days(ticker_dfs: dict, n_held_out_days: int,
                                    encoder_len: int) -> tuple[dict, dict, list]:
    """Task 17 Step 5: split per-ticker DataFrames (each with a 'trade_date' column, as produced
    by training.stage1_data.build_ticker_dfs) into a training portion and a held-out portion,
    using the last `n_held_out_days` GLOBAL distinct trade_dates (across all tickers, not
    per-ticker) as the held-out window -- e.g. the full 2019-present range's last 100 trading
    days, used as run_training()'s val_loader for checkpoint selection AND for the
    pre_leverage/leverage_era regime-split report. This is NOT a second Task-16-style one-touch
    test set (there is no further held-out future data available -- training extends through the
    present), so it is safe to evaluate against repeatedly.

    Returns (train_ticker_dfs, held_out_ticker_dfs, held_out_dates):
    - train_ticker_dfs: rows strictly before the held-out window's first date -- never sees a
      held-out row, as feature/encoder context or as a label, so there is no leakage.
    - held_out_ticker_dfs: each ticker's held-out target rows (trade_date >= the held-out
      window's first date), PLUS up to `encoder_len` immediately-preceding rows taken from that
      ticker's OWN row order (not re-filtered by global date), so TickerDayDataset can still build
      a full encoder window for the earliest held-out target even for tickers with trading
      gaps/holidays that don't line up with the global date list.
    - held_out_dates: the `n_held_out_days` global dates that define the held-out window
      (ascending) -- used to report the window's actual regime composition (how many fall before
      vs. on/after the leverage-ETF launch date).
    """
    all_dates = sorted({d for df in ticker_dfs.values() for d in df["trade_date"]})
    if len(all_dates) < n_held_out_days:
        raise ValueError(
            f"only {len(all_dates)} distinct trade_dates available, need at least "
            f"{n_held_out_days} to hold out {n_held_out_days} trading days")
    held_out_dates = all_dates[-n_held_out_days:]
    held_out_start = held_out_dates[0]

    train_ticker_dfs: dict = {}
    held_out_ticker_dfs: dict = {}
    for ticker, df in ticker_dfs.items():
        df = df.sort_values("trade_date").reset_index(drop=True)
        train_part = df[df["trade_date"] < held_out_start].reset_index(drop=True)
        held_out_targets = df[df["trade_date"] >= held_out_start]
        buffer = train_part.tail(encoder_len)
        held_out_part = pd.concat([buffer, held_out_targets], ignore_index=True)
        if len(train_part) > 0:
            train_ticker_dfs[ticker] = train_part
        if len(held_out_part) > 0:
            held_out_ticker_dfs[ticker] = held_out_part

    return train_ticker_dfs, held_out_ticker_dfs, held_out_dates
