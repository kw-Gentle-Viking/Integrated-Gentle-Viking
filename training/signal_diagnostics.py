"""Signal diagnostics: split a per-(ticker, day) score's daily rank IC into a ticker FIXED-EFFECT part
and a TIMING part (docs/signal_diagnosis.md, section 3).

    fixed effect  f[ticker] = mean score of that ticker (over `prior_ticker_means` if given, else over the
                  rows of the window being scored)
    timing score  score - f[ticker]

Each of raw score / fixed effect / timing score is scored with the SAME daily rule as
evaluation.evaluate.compute_signal_metrics (per day: rows with finite score and return, >= min_names of
them, Spearman via evaluation.evaluate._spearman, constant days skipped, mean over days; SE = std(ddof=1) /
sqrt(days)). Returns are used ONLY as the IC target: the fixed effect is built from scores alone (never from
returns), so an OOT window scored with the ticker means of the val window uses no OOT information at all.
"""
import numpy as np

from evaluation.evaluate import _spearman

QUANTILE = 0.2


def ticker_score_means(tickers, scores) -> dict:
    """Mean score per ticker over rows with a finite score (returns are not looked at)."""
    tickers = np.asarray(tickers, dtype=object)
    scores = np.asarray(scores, dtype=float)
    ok = np.isfinite(scores)
    acc: dict = {}
    for tk, s in zip(tickers[ok].tolist(), scores[ok].tolist()):
        a = acc.setdefault(tk, [0.0, 0])
        a[0] += s
        a[1] += 1
    return {tk: v[0] / v[1] for tk, v in acc.items()}


def _daily(dates, x, rets, min_names, quantile=QUANTILE):
    """(list of daily Spearman ICs, list of daily top-minus-bottom quantile spreads) of x vs rets."""
    ics, spreads = [], []
    for d in sorted(set(dates.tolist())):
        idx = np.where(dates == d)[0]
        n = len(idx)
        if n < min_names or n < 2:
            continue
        s, r = x[idx], rets[idx]
        ic = _spearman(s, r)
        if ic is None:
            continue
        ics.append(ic)
        k = max(1, int(n * quantile + 1e-9))
        order = np.argsort(s, kind="mergesort")
        spreads.append(float(r[order[-k:]].mean() - r[order[:k]].mean()))
    return ics, spreads


def _summ(ics):
    n = len(ics)
    mean = float(np.mean(ics)) if n else None
    std = float(np.std(ics, ddof=1)) if n >= 2 else None
    se = (std / np.sqrt(n)) if std is not None else None
    ir = (mean / std) if (std not in (None, 0.0) and mean is not None) else None
    return mean, se, ir, n


def timing_decomposition(dates, tickers, scores, returns, prior_ticker_means: dict | None = None,
                         min_names: int = 20) -> dict:
    """Daily rank-IC decomposition. Keys: raw_ic / fixed_effect_ic / timing_ic (+ `_se`, `_ir`, n_days_*),
    raw_quantile_ls / timing_quantile_ls (mean daily top-20% minus bottom-20% return by that score),
    n_days (= raw), ticker_means (the fixed effect actually used; pass a val window's dict as
    `prior_ticker_means` when scoring OOT). Undefined values are None.
    Rows whose ticker is missing from `prior_ticker_means` are dropped from the fixed-effect and timing
    computations only (the raw IC keeps every finite row)."""
    dates_a = np.asarray(list(dates), dtype=object)
    tick_a = np.asarray(list(tickers), dtype=object)
    score_a = np.asarray(scores, dtype=float).reshape(-1)
    ret_a = np.asarray(returns, dtype=float).reshape(-1)
    means = dict(prior_ticker_means) if prior_ticker_means is not None else ticker_score_means(tick_a, score_a)

    valid = np.isfinite(score_a) & np.isfinite(ret_a)
    raw_ics, raw_q = _daily(dates_a[valid], score_a[valid], ret_a[valid], min_names)

    fe = np.array([means.get(tk, np.nan) for tk in tick_a.tolist()], dtype=float)
    v2 = valid & np.isfinite(fe)
    fixed_ics, _ = _daily(dates_a[v2], fe[v2], ret_a[v2], min_names)
    timing = score_a - fe
    # A purely static scorer (a per-ticker constant) leaves float residuals ~1e-17 whose ranks are arbitrary and
    # produce a spurious non-zero timing IC; snap them to 0 so such days are constant -> skipped (undefined).
    scale = max(float(np.nanmax(np.abs(score_a[valid]))) if valid.any() else 0.0, 1e-300)
    timing = np.where(np.abs(timing) < 1e-12 * scale, 0.0, timing)
    timing_ics, timing_q = _daily(dates_a[v2], timing[v2], ret_a[v2], min_names)

    out = {}
    for key, ics in (("raw", raw_ics), ("fixed", fixed_ics), ("timing", timing_ics)):
        mean, se, ir, n = _summ(ics)
        name = {"raw": "raw_ic", "fixed": "fixed_effect_ic", "timing": "timing_ic"}[key]
        out[name], out[f"{name}_se"], out[f"{name}_ir"], out[f"n_days_{key}"] = mean, se, ir, n
    out["n_days"] = out["n_days_raw"]
    out["raw_quantile_ls"] = float(np.mean(raw_q)) if raw_q else None
    out["timing_quantile_ls"] = float(np.mean(timing_q)) if timing_q else None
    out["ticker_means"] = means
    return out


def flat_summary(dec: dict) -> dict:
    """JSON-friendly copy without the per-ticker means (which can be thousands of entries)."""
    return {k: v for k, v in dec.items() if k != "ticker_means"}
