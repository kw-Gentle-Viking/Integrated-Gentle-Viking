"""Pure-logic helpers for the E0 tabular baseline (feature construction, target, preprocessing,
per-ticker time-series diagnostics). No DB / model code here so everything is unit-testable."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def add_lags(df: pd.DataFrame, col: str, lags) -> pd.DataFrame:
    """Return a copy sorted by (ticker, trade_date) with `<col>_lag<k>` = value k rows earlier
    WITHIN the same ticker (strictly past; NaN at each ticker's start, never another ticker's)."""
    out = df.sort_values(["ticker", "trade_date"]).copy()
    g = out.groupby("ticker", sort=False)[col]
    for k in lags:
        out[f"{col}_lag{k}"] = g.shift(k)
    return out


def add_cs_ranks(df: pd.DataFrame, cols, suffix: str = "_csr") -> pd.DataFrame:
    """Copy with per-date cross-sectional percentile rank (0,1] of each col (ties averaged,
    NaN ignored and kept NaN). Uses only same-date information."""
    out = df.copy()
    g = out.groupby("trade_date", sort=False)
    for c in cols:
        out[f"{c}{suffix}"] = g[c].rank(method="average", pct=True)
    return out


def make_z_target(ret, vol, floor: float, clip: float = 5.0) -> np.ndarray:
    """z = ret / max(vol, floor), clipped to [-clip, clip]; NaN if ret or vol is NaN."""
    ret = np.asarray(ret, dtype=float)
    vol = np.asarray(vol, dtype=float)
    z = ret / np.maximum(vol, floor)
    z = np.clip(z, -clip, clip)
    z[~np.isfinite(ret) | ~np.isfinite(vol)] = np.nan
    return z


class Preprocessor:
    """Winsorise at train percentiles, fill NaN with train median, standardise with train
    mean/std. All statistics come from fit() only."""

    def __init__(self, lo: float = 0.5, hi: float = 99.5):
        self.lo, self.hi = lo, hi

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.lo_v_ = np.nanpercentile(X, self.lo, axis=0)
        self.hi_v_ = np.nanpercentile(X, self.hi, axis=0)
        self.med_ = np.nanmedian(X, axis=0)
        Xc = self._clip_fill(X)
        self.mean_ = Xc.mean(axis=0)
        sd = Xc.std(axis=0)
        self.std_ = np.where(sd > 0, sd, 1.0)
        return self

    def _clip_fill(self, X):
        X = np.where(np.isnan(X), self.med_, X)
        return np.clip(X, self.lo_v_, self.hi_v_)

    def transform(self, X):
        return (self._clip_fill(np.asarray(X, dtype=float)) - self.mean_) / self.std_


def per_ticker_ts_metrics(score, ret) -> dict:
    """Time-series diagnostics for one ticker: Spearman(score, ret) over days, sign hit-rate
    (sign(score)==sign(ret), zero-return days excluded) and the base rate of up days."""
    s = np.asarray(score, dtype=float)
    r = np.asarray(ret, dtype=float)
    ok = np.isfinite(s) & np.isfinite(r)
    s, r = s[ok], r[ok]
    n = int(len(s))
    rho = None
    if n >= 3 and s.std() > 0 and r.std() > 0:
        rho = float(spearmanr(s, r)[0])
    nz = r != 0
    hit = float((np.sign(s[nz]) == np.sign(r[nz])).mean()) if nz.any() else None
    up = float((r[nz] > 0).mean()) if nz.any() else None
    return {"n": n, "spearman": rho, "hit_rate": hit, "base_up_rate": up}


def demean_by_date(dates, y) -> np.ndarray:
    """y minus its same-date mean (NaN ignored, kept NaN): a cross-sectional target that removes
    the market-wide component shared by all tickers on a day."""
    s = pd.Series(np.asarray(y, dtype=float))
    return (s - s.groupby(np.asarray(dates, dtype=object)).transform("mean")).values


def argmax_nan_safe(scores: dict):
    """Key with the largest finite value; NaN entries (e.g. undefined IC) are never selected."""
    ok = {k: v for k, v in scores.items() if v == v}
    if not ok:
        raise ValueError("all candidate scores are NaN")
    return max(ok, key=ok.get)
