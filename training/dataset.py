import torch
from torch.utils.data import Dataset
import pandas as pd


def target_offset_of(ds) -> int:
    """Row offset of a sample's target row from its window start; falls back to ds.encoder_len for
    dataset-like objects predating the `align` option (e.g. simple test doubles)."""
    return getattr(ds, "target_offset", ds.encoder_len)


class TickerDayDataset(Dataset):
    def __init__(self, ticker_dfs: dict[str, pd.DataFrame], historical_cols: list[str],
                 future_cols: list[str], static_cols: list[str], encoder_len: int = 60,
                 align: str = "legacy"):
        """align="legacy": encoder = rows [t, t+encoder_len); label / future / static come from row
        t+encoder_len, i.e. the day AFTER the encoder ends (the model never sees the label row's own
        features -- effectively a 2-day-ahead forecast). Kept byte-for-byte for old checkpoints, S3/E0
        rescoring and existing tests.
        align="today": same encoder, but label / future / static come from the LAST ENCODER ROW
        (t+encoder_len-1), matching serving (59 real days + today as the 60th step, output read as
        today's d->d+1 prediction). Windows per ticker = len - encoder_len + 1."""
        if align not in ("legacy", "today"):
            raise ValueError(f"align must be 'legacy' or 'today', got {align!r}")
        self.align = align
        self.historical_cols = historical_cols
        self.future_cols = future_cols
        self.static_cols = static_cols
        self.encoder_len = encoder_len
        self.index: list[tuple[str, int]] = []
        self.ticker_dfs = ticker_dfs
        for ticker, df in ticker_dfs.items():
            df = df.reset_index(drop=True)
            self.ticker_dfs[ticker] = df
            n_usable = len(df) - encoder_len + (1 if align == "today" else 0)
            for t in range(max(n_usable, 0)):
                # window [t : t+encoder_len) 은 history, 예측 시점은 t+target_offset (label 포함)
                if pd.notna(df.loc[t + self.target_offset, "label"]):
                    self.index.append((ticker, t))

    @property
    def target_offset(self) -> int:
        """Row offset (from a window's first row t) of the row supplying label/future/static:
        encoder_len for "legacy", encoder_len-1 (the last encoder row) for "today". Every place that
        maps a sample to its target row must use this, not encoder_len."""
        return self.encoder_len if self.align == "legacy" else self.encoder_len - 1

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        ticker, t = self.index[idx]
        df = self.ticker_dfs[ticker]
        hist = df.loc[t: t + self.encoder_len - 1, self.historical_cols].values.astype("float32")
        target_row = df.loc[t + self.target_offset]
        fut = target_row[self.future_cols].values.astype("float32").reshape(1, -1)
        static = target_row[self.static_cols].values.astype("int64").reshape(1, -1)
        label = int(target_row["label"])
        return {
            "historical_ts_numeric": torch.tensor(hist),
            "future_ts_numeric": torch.tensor(fut),
            "static_feats_categorical": torch.tensor(static, dtype=torch.long),
            "label": torch.tensor([label]),
        }


def build_incomplete_today_bar(intraday_5min_rows: list[dict], open_price: float) -> dict:
    """설계 §3: '오늘'을 5분봉으로 실시간 계산되는 미완성 일봉으로 표현."""
    if not intraday_5min_rows:
        return {"open": open_price, "high": open_price, "low": open_price,
                "close": open_price, "volume": 0}
    highs = [r["high"] for r in intraday_5min_rows]
    lows = [r["low"] for r in intraday_5min_rows]
    latest = max(intraday_5min_rows, key=lambda r: r["datetime"])
    return {
        "open": open_price, "high": max(highs), "low": min(lows),
        "close": latest["close"], "volume": sum(r["volume"] for r in intraday_5min_rows),
    }
