"""Input preprocessing for the TFT retraining recipes (plan "E0 결과 이후 수정", R1..R3).

Pure, unit-tested (training/test_preprocess.py). Statistics are fitted ONLY on the frames the
caller passes (the runner passes train rows dated <= 2023-12-31 and guards this with `max_date`);
val / OOT / serving get apply-only.

Rules (per column, decided from the fit data):
  * dropped_constant : zero variance in train (e.g. the dead leverage columns lev_total_aum,
                       lev_aum_to_mktcap, est_rebalancing_flow) -> removed from the model input list.
  * binary           : values subset of {0, 1} -> passed through untouched (NEVER quantile-clipped:
                       a 0.5/99.5 clip turns a rare event flag into all zeros).
  * continuous       : clip at the train 0.5/99.5 quantiles, then standardise with the mean/std of
                       the CLIPPED train values. If the quantile clip degenerates (clipped std == 0
                       but raw std > 0, e.g. a >99.5%-zero sparse count), the bounds fall back to the
                       raw train min/max so the column is not silently killed.
NaN -> 0.0 AFTER the transform (0 == train mean for continuous, 0 for binary flags). NB: the cached
frames from stage1_data.build_ticker_dfs already carry NaN -> raw 0.0 (the fill semantics of the
pipeline); this NaN handling only matters for columns created later (rank inputs) or fresh frames.

The artifact is a plain JSON-serialisable dict so serving can load it (`load_preprocessor`).
"""
import json
import os

import numpy as np
import pandas as pd

LO_Q, HI_Q = 0.005, 0.995
ARTIFACT_VERSION = 1


def _iter_frames(frames):
    return frames.values() if isinstance(frames, dict) else frames


def _iso(d) -> str:
    return d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)


def fit_preprocessor(train_frames, columns, lo_q: float = LO_Q, hi_q: float = HI_Q,
                     max_date: str | None = None) -> dict:
    """Fit clip bounds + standardisation on `train_frames` (dict or list of per-ticker frames).
    `max_date` (ISO) is a leakage guard: raises if any passed row is dated after it."""
    frames = list(_iter_frames(train_frames))
    if not frames:
        raise ValueError("no frames to fit on")
    dates = pd.concat([f["trade_date"] for f in frames]) if "trade_date" in frames[0] else None
    fit_max = _iso(dates.max()) if dates is not None and len(dates) else None
    if max_date is not None and fit_max is not None and fit_max > max_date:
        raise ValueError(f"fit data contains rows dated {fit_max} after cutoff {max_date}")

    cols_meta, kept, dropped = {}, [], []
    for c in columns:
        x = np.concatenate([f[c].to_numpy(dtype=float) for f in frames])
        x = x[~np.isnan(x)]
        if len(x) == 0 or x.std() == 0:
            cols_meta[c] = {"kind": "dropped_constant",
                            "value": float(x[0]) if len(x) else None}
            dropped.append(c)
            continue
        if set(np.unique(x).tolist()) <= {0.0, 1.0}:
            cols_meta[c] = {"kind": "binary"}
            kept.append(c)
            continue
        lo, hi = float(np.quantile(x, lo_q)), float(np.quantile(x, hi_q))
        xc = np.clip(x, lo, hi)
        fallback = False
        if xc.std() == 0:
            lo, hi = float(x.min()), float(x.max())
            xc = x
            fallback = True
        meta = {"kind": "continuous", "lo": lo, "hi": hi,
                "mean": float(xc.mean()), "std": float(xc.std())}
        if fallback:
            meta["clip_fallback_to_minmax"] = True
        cols_meta[c] = meta
        kept.append(c)
    return {
        "version": ARTIFACT_VERSION, "quantiles": [lo_q, hi_q],
        "columns": cols_meta, "kept_columns": kept, "dropped_columns": dropped,
        "fit_rows": int(sum(len(f) for f in frames)), "fit_max_date": fit_max,
    }


def apply_preprocessor_frame(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Transformed COPY of one frame. Dropped columns are left as-is (they are simply not in
    artifact['kept_columns'], i.e. not model inputs)."""
    out = df.copy()
    for c, meta in artifact["columns"].items():
        if c not in out.columns:
            continue
        x = out[c].to_numpy(dtype=float)
        if meta["kind"] == "continuous":
            x = (np.clip(x, meta["lo"], meta["hi"]) - meta["mean"]) / (meta["std"] if meta["std"] > 0 else 1.0)
            out[c] = np.where(np.isnan(x), 0.0, x)
        elif meta["kind"] == "binary":
            out[c] = np.where(np.isnan(x), 0.0, x)
    return out


def apply_preprocessor(frames, artifact: dict):
    """Apply-only transform of a dict {ticker: frame} (returns a new dict) or a single frame."""
    if isinstance(frames, pd.DataFrame):
        return apply_preprocessor_frame(frames, artifact)
    return {k: apply_preprocessor_frame(v, artifact) for k, v in frames.items()}


def save_preprocessor(artifact: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(artifact, f, indent=2)
    os.replace(tmp, path)


def load_preprocessor(path: str) -> dict:
    """Loader for future serving use."""
    with open(path) as f:
        return json.load(f)


def add_cs_rank_inputs(ticker_dfs: dict, cols, suffix: str = "_csr") -> dict:
    """Per-date cross-sectional percentile rank of each col across the tickers in `ticker_dfs`,
    centred at 0 (rank_pct - (n+1)/(2n), n = non-NaN names that date; range (-0.5, 0.5), mean 0).
    Same-date information only (no cross-date leakage). Uses training.tabular_features.add_cs_ranks
    (the E0 builder). Returns NEW frames with `<col><suffix>` appended; inputs are not mutated.
    Rows whose source value is NaN keep NaN rank. Call on the full ticker dict BEFORE windowing."""
    from training.tabular_features import add_cs_ranks
    cols = list(cols)
    keys = list(ticker_dfs)
    if not keys:
        return {}
    flat = pd.concat([ticker_dfs[k][["trade_date"] + cols] for k in keys], ignore_index=True)
    flat = add_cs_ranks(flat, cols, suffix=suffix)
    g = flat.groupby("trade_date")
    for c in cols:   # centre each date's percentile ranks: (n+1)/(2n) is the mean of pct ranks 1/n..1
        n = g[c].transform("count").to_numpy(dtype=float)
        flat[f"{c}{suffix}"] = flat[f"{c}{suffix}"].to_numpy(dtype=float) - (n + 1.0) / (2.0 * n)
    out, pos = {}, 0
    for k in keys:
        n_rows = len(ticker_dfs[k])
        df = ticker_dfs[k].copy()
        for c in cols:
            df[f"{c}{suffix}"] = flat[f"{c}{suffix}"].to_numpy()[pos:pos + n_rows]
        pos += n_rows
        out[k] = df
    return out
