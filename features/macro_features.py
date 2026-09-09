import pandas as pd


def compute_return(series: pd.Series) -> pd.Series:
    return series.pct_change()


def compute_change(series: pd.Series) -> pd.Series:
    return series.diff()


def compute_rate_spread(a: pd.Series, b: pd.Series) -> pd.Series:
    return a - b
