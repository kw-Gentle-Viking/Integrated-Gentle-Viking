#!/usr/bin/env python
"""Experiment E0: do the champion features contain learnable cross-sectional signal, and is TFT
the thing failing to extract it?  Cheap tabular models (Ridge / HistGradientBoosting), CPU only,
scored on EXACTLY the S3 sample sets / metrics (docs/signal_baseline.md).

Protocol (binding):
  * hyper-parameters chosen ONLY on a time-ordered split inside the train era: fit rows <= 2022-12-31,
    select on 2023 rows (mean daily rank IC), then refit on rows <= 2023-12-31.
  * val_2024 (36,729 samples) and oot_2026 (33,364 samples) each scored once per final config.
  * 2025 rows are never used for anything.

    set -a && source .env && set +a
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python training/run_tabular_baseline.py            # E0v2 (default)

E0v2 (default, --align today): same protocol, but on adjusted-price data (stage1_data.DATA_VERSION) and on
EXACTLY the (ticker, date) sample sets the TFT recipes (training/run_tfx_experiments.py, align="today") score,
so the numbers are like-for-like with R0..R3. Output: tabular_baseline_v2.json + docs/tabular_baseline_v2.md
(the old tabular_baseline.json / docs/tabular_baseline.md are never touched).
--align legacy keeps the old (t+encoder_len) sample enumeration for reference, written to a separate JSON.
"""
import json
import logging
import os
import argparse
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from training.tabular_features import argmax_nan_safe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ART = "training/artifacts"
RESULT_JSON = f"{ART}/tabular_baseline.json"
DOC_PATH = "docs/tabular_baseline.md"            # legacy E0 doc: never written by the v2 flow
V2_JSON = f"{ART}/tabular_baseline_v2.json"
V2_LEGACY_JSON = f"{ART}/tabular_baseline_v2_legacy_align.json"
V2_DOC = "docs/tabular_baseline_v2.md"
TFX_JSON = f"{ART}/tfx_results.json"
EXPECTED_ALIGNED = {"val_2024": 36929, "oot_2026": 33364}   # TFT-runner aligned sample counts (full universe)
RET_CLIP = 0.30
S3_JSON = f"{ART}/signal_baseline.json"
CHAMPION_CONFIG_PATH = "training/champion_config.json"
VN_PATH = "training/threshold_vn.json"
TRAIN_START, TRAIN_END = "2019-01-02", "2023-12-31"
FIT_END, SELECT_START, SELECT_END = "2022-12-31", "2023-01-01", "2023-12-31"
LOAD_END = "2026-09-08"
WARMUP_ROWS = 60           # mirror the TFT encoder: a row needs >=60 prior rows in the loaded frame
EXPECTED = {"val_2024": 36729, "oot_2026": 33364}
WINDOWS = {"val_2024": "2024-01-01..2024-12-31", "oot_2026": "2026-01-01..2026-09-07"}
CACHES = {"val_2024": (f"{ART}/stage1_cache_val_2024-01-01_2024-12-31.pkl", "2024-01-01"),
          "oot_2026": (f"{ART}/stage1_cache_oot_2025-09-01_2026-09-08.pkl", "2026-01-01")}
PROD_TICKERS = ["005930", "000660"]
RANK_BASE = ["log_ret", "disparity_5d", "disparity_20d", "disparity_60d", "rsi_14", "volatility_20d"]
LAGS = [1, 2, 3, 4, 5]
EVENT_COLS = ["is_dividend", "is_bonus_issue", "is_rights_offering", "is_split", "is_vi_triggered"]
RIDGE_ALPHAS = [10.0, 1_000.0, 100_000.0, 1_000_000.0]
HGB_ITERS = [25, 50, 100, 200, 300, 400]
HGB_PARAMS = dict(learning_rate=0.05, max_depth=4, min_samples_leaf=500, l2_regularization=10.0,
                  max_bins=64, early_stopping=False, random_state=0)


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "training/run_tabular_baseline.py",
             "training/tabular_features.py", "training/dataset.py", "training/stage1_data.py"], text=True).strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def load_frame(dsn: str, cols: list[str]) -> pd.DataFrame:
    import psycopg2
    sel = ", ".join(sorted(set(cols + ["next_day_return", "volatility_20d", "label_vn", "label"])))
    q = (f"SELECT ticker, trade_date, {sel} FROM feature_pool "
         f"WHERE trade_date >= %s AND trade_date <= %s ORDER BY ticker, trade_date")
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(q, (TRAIN_START, LOAD_END))
            rows = cur.fetchall()
            names = [d[0] for d in cur.description]
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=names)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for c in df.columns:
        if c not in ("ticker", "trade_date"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in EVENT_COLS:                      # stored NULL == "no event" (see stage1_data.build_ticker_dfs)
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    return df


def build_features(df: pd.DataFrame, champ_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    from training.tabular_features import add_cs_ranks, add_lags
    df = add_lags(df, "log_ret", LAGS)
    lag_cols = [f"log_ret_lag{k}" for k in LAGS]
    df = add_cs_ranks(df, RANK_BASE + lag_cols)
    df["row_no"] = df.groupby("ticker", sort=False).cumcount()
    fs = {"F1": list(champ_cols)}
    fs["F2"] = fs["F1"] + lag_cols
    fs["F3"] = fs["F2"] + [f"{c}_csr" for c in RANK_BASE + lag_cols]
    return df, fs


def enumerate_samples(frames: dict, columns: list[str], min_target: str, align: str) -> list[tuple[str, str]]:
    """(ticker, sample date) list, in dataset order, exactly as the TFT runner's eval dataset
    (run_tfx_experiments.build_window_dataset) enumerates it: TickerDayDataset(align) over the window's
    frames, samples whose target date < min_target dropped. For align="today" the date is that of the
    last encoder row (= the day the prediction is made). Mutates `frames` like the Dataset does."""
    from training.config import KNOWN_FUTURE_COLS, STATIC_COLS
    from training.dataset import TickerDayDataset
    from training.signal_data import filter_index_by_target_date, sample_meta
    ds = TickerDayDataset(frames, columns, KNOWN_FUTURE_COLS, STATIC_COLS, WARMUP_ROWS, align=align)
    filter_index_by_target_date(ds, min_target)
    return [(tk, d) for tk, d, _ in sample_meta(ds)]


def aligned_sample_sets(dsn: str, align: str = "today") -> dict:
    """{window: dict(tickers, dates, n)} on the adj1 caches (built read-only from the DB via `dsn` when a
    cache is missing). For align="today" every window is cross-checked against the TFT runner's own
    build_window_dataset (same list, same order) and the known counts."""
    from training import run_tfx_experiments as tfx
    from training.stage1_data import DATA_VERSION, load_or_build_ticker_dfs
    champ = json.load(open(CHAMPION_CONFIG_PATH))
    out = {}
    for w in tfx.EVAL_WINDOWS:
        sp = tfx.SPLITS[w]
        cache = os.path.join(ART, sp["cache"])
        keys = enumerate_samples(load_or_build_ticker_dfs(dsn, sp["load_start"], sp["load_end"], cache),
                                 champ["columns"], sp["min_target"], align)
        if align == "today":
            frames = load_or_build_ticker_dfs(dsn, sp["load_start"], sp["load_end"], cache)
            _, meta = tfx.build_window_dataset(w, frames, champ["columns"], {"align": "today"})
            assert keys == [(t, d) for t, d, _ in meta], f"{w}: differs from the TFT runner's sample list"
            assert len(keys) == EXPECTED_ALIGNED[w], (w, len(keys), EXPECTED_ALIGNED[w])
        assert len(set(keys)) == len(keys)
        out[w] = dict(tickers=np.array([k[0] for k in keys]), dates=np.array([k[1] for k in keys]), n=len(keys))
        logger.info("[%s] %s samples (%s, data %s): %d, dates %s..%s", w, align, cache, DATA_VERSION, len(keys),
                    min(out[w]["dates"]), max(out[w]["dates"]))
    return out


def sample_sets() -> dict:
    """S3 sample sets: re-enumerated with the same TickerDayDataset + target-date filter S3 used
    and asserted equal to the (ticker, date) list stored by S3 (its npz)."""
    from training.config import KNOWN_FUTURE_COLS, STATIC_COLS
    from training.dataset import TickerDayDataset
    from training.signal_data import filter_index_by_target_date, sample_meta
    from training.stage1_data import load_or_build_ticker_dfs
    champ = json.load(open(CHAMPION_CONFIG_PATH))
    out = {}
    for name, (cache, min_t) in CACHES.items():
        ticker_dfs = load_or_build_ticker_dfs(None, None, None, cache)  # cache hit only
        ds = TickerDayDataset(ticker_dfs, champ["columns"], KNOWN_FUTURE_COLS, STATIC_COLS, 60)
        filter_index_by_target_date(ds, min_t)
        meta = sample_meta(ds)
        z = np.load(f"{ART}/signal_baseline_probs_{name}.npz", allow_pickle=True)
        keys = list(zip([m[0] for m in meta], [m[1] for m in meta]))
        s3_keys = list(zip(z["tickers"].tolist(), z["dates"].tolist()))
        assert keys == s3_keys, f"{name}: re-enumerated samples differ from S3 samples"
        assert len(keys) == EXPECTED[name], (name, len(keys))
        out[name] = dict(tickers=np.array(z["tickers"]), dates=np.array(z["dates"]),
                         ret=z["next_day_return"].astype(float), champ_probs=z["probs"].astype(float),
                         log_ret=z["log_ret"].astype(float))
        logger.info("[%s] %d samples verified against S3 (%s..%s)", name, len(keys),
                    min(out[name]["dates"]), max(out[name]["dates"]))
    return out


def daily_ics(dates, rets, score, min_names=20):
    """Per-day Spearman series with the same skip rules as evaluate.compute_signal_metrics."""
    from evaluation.evaluate import _spearman
    dates = np.asarray(dates, dtype=object)
    score = np.asarray(score, float)
    rets = np.asarray(rets, float)
    ok = np.isfinite(score) & np.isfinite(rets)
    dates, score, rets = dates[ok], score[ok], rets[ok]
    out = {}
    for d in sorted(set(dates.tolist())):
        idx = np.where(dates == d)[0]
        if len(idx) < min_names:
            continue
        ic = _spearman(score[idx], rets[idx])
        if ic is not None:
            out[d] = ic
    return out


def bundle(dates, rets, score, tickers):
    from evaluation.evaluate import compute_signal_metrics
    from training.run_signal_baseline import score_to_probs
    from training.tabular_features import per_ticker_ts_metrics
    probs = score_to_probs(score)
    sig = compute_signal_metrics(list(dates), rets, probs, min_names_per_day=20)
    ics = daily_ics(dates, rets, score)
    n = len(ics)
    se = float(np.std(list(ics.values()), ddof=1) / np.sqrt(n)) if n >= 2 else None
    tickers = np.asarray(tickers)
    per_t = {t: per_ticker_ts_metrics(np.asarray(score)[tickers == t], np.asarray(rets)[tickers == t])
             for t in PROD_TICKERS}
    sig_clip = compute_signal_metrics(list(dates), np.clip(np.asarray(rets, float), -RET_CLIP, RET_CLIP), probs,
                                      min_names_per_day=20)
    return {"signal": sig, "signal_clip30": sig_clip, "ic_se": se, "per_ticker": per_t}, ics


def paired_diff(ics_a: dict, ics_b: dict):
    days = sorted(set(ics_a) & set(ics_b))
    d = np.array([ics_a[k] - ics_b[k] for k in days])
    if len(d) < 2:
        return None
    return {"mean_diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))), "n_days": len(d)}


def mean_ic(dates, rets, score) -> float:
    """Mean daily rank IC; NaN if undefined on every day (e.g. constant-per-day score)."""
    v = list(daily_ics(dates, rets, score).values())
    return float(np.mean(v)) if v else float("nan")


def select_hgb_iters(make, Xtr, ytr, Xsel, sel_dates, sel_ret, score_fn):
    """Fit once with max iters, score staged predictions on the 2023 split, return best n_iter."""
    model = make(max(HGB_ITERS)).fit(Xtr, ytr)
    res = {}
    for i, pred in enumerate(score_fn(model, Xsel)):
        n_it = i + 1
        if n_it in HGB_ITERS:
            res[n_it] = mean_ic(sel_dates, sel_ret, pred)
    best = argmax_nan_safe(res)
    return best, res


CONFIG_DESC = {
    "ridge_F1": "Ridge, F1", "ridge_F2": "Ridge, F2", "ridge_F3": "Ridge, F3",
    "hgb_reg_F1": "HGB reg (z), F1", "hgb_reg_F2": "HGB reg (z), F2", "hgb_reg_F3": "HGB reg (z), F3",
    "hgb_reg_csz_F3": "HGB reg (date-demeaned z), F3", "hgb_clf_F3": "HGB clf (label_vn), F3",
}


def _f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _p(x):
    return "n/a" if x is None else f"{x * 100:.2f}%"


INTERPRETATION = """## Interpretation (SE-aware)

Differences are called real only when they are >= ~2 paired SE.

1. **The features contain a small amount of learnable cross-sectional signal.** Every config has a positive
   mean IC on both windows; the best (HGB regressor, F3) has IC 0.058 (val, SE 0.010) and 0.044 (OOT, SE 0.012), about
   6 SE from zero on val and ~4 SE on OOT. IC IR (HGB) 0.4-0.45 on val and 0.2-0.3 on OOT, i.e. small in absolute terms. Quantile L/S is about 0.1-0.3% per day
   gross, before any transaction costs; this says nothing yet about tradability.
2. **The deployed TFT is well below what a simple tabular model gets from the same 33 columns.** Paired IC difference vs TFT on
   val: HGB F3 +0.043 (SE 0.0135, 3.2 SE), F2 +0.035 (2.7 SE), F1 +0.030 (2.3 SE), clf +0.035 (2.8 SE), demeaned-z +0.039 (3.1 SE);
   ridge F1/F2 (+0.012/+0.013, ~1 SE) are not distinguishable from TFT. On OOT only HGB F3 (+0.049, SE 0.018, 2.7 SE) and
   demeaned-z (+0.049, SE 0.023, 2.2 SE) clear 2 SE; the others are +0.02-0.038 with SE 0.02-0.025 (about 1-1.7 SE), i.e.
   same direction, not individually conclusive. The TFT was selected on val 2024 macro F1, so the val comparison, if anything, favours it.
   Because the tabular F1 models use the same columns as the TFT, the gap is attributable to the model/training setup, not to missing features.
3. **No tabular model beats the trivial 1-day reversal rule; the best ones match it.** IC minus reversal is negative or
   ~0 everywhere (best: HGB F3 -0.005 +/- 0.007 val, +0.002 +/- 0.014 OOT; ridge F1/F2 are significantly below on val, -0.035 +/- 0.008).
   `log_ret` (the reversal input) is in every feature set, so we did not test whether the HGB signal is anything beyond
   reversal plus noise; that decomposition (e.g. drop log_ret and its lags/ranks) is not done here.
4. **Cross-sectional normalisation looks helpful but is not established.** Within HGB and ridge the F3 IC is higher than F1/F2
   on val (ridge +0.014 vs F1, HGB +0.013 vs F1) but we did not compute paired SEs for feature-set differences and on OOT the
   ordering is inconsistent (ridge F1 0.021 vs F3 0.018; HGB F1 0.029 vs F3 0.044). Treat as a hypothesis for the next experiment.
5. **Val to OOT decay** (best HGB 0.058 to 0.044, ridge and clf similar) is within noise (SE ~0.01-0.015 per window) and cannot
   be distinguished from stationarity. argmax L/S (pooled, sign of the score) is negative for most tabular models on OOT because
   the score sign is not calibrated per day; IC and quantile L/S are the meaningful columns.
6. **Production tickers (005930, 000660):** all |rho| <= 0.14 (< 2 SE) and hit-rates are within ~2 SE of 50% (SE ~0.07 / ~0.037); nothing can be
   said per ticker. A cross-sectional IC of ~0.05 says the ranking helps on average across 200 names, not that any single name is predictable.
7. **Selection caveat:** the 8 configs were all scored on val and OOT, and "best" above is picked with hindsight on those windows
   (mildly optimistic, ~8 tries); OOT was scored once per config. HGB hyper-parameters other than n_iter (or ridge alpha) were fixed a priori and untuned.

**What this supports:** the features are not empty and TFT-as-trained is extracting less than gradient boosting does; label/regularisation
experiments on the TFT are worth running with IC (not macro F1) as the criterion, and HGB-F3 / reversal (IC ~0.05-0.06) is the bar to clear.
**What it does not support:** a claim that tabular beats reversal, that the signal survives costs, or that cross-sectional ranks are the cause of the gain.
"""


def render_doc(r: dict) -> str:
    L = ["# Tabular baseline (E0): is there learnable signal in the champion features?", "",
         f"Code commit: `{r['code_commit']}` (`training/run_tabular_baseline.py`, `training/tabular_features.py`). "
         "Raw output: `training/artifacts/tabular_baseline.json` (gitignored, like the other artifacts).", "",
         "## Protocol", "",
         "- Same sample sets and metrics as S3 (`docs/signal_baseline.md`). The samples were re-enumerated with "
         "`TickerDayDataset` + `filter_index_by_target_date` on the S3 caches and asserted identical, in order, to the "
         "(ticker, date) list S3 stored: "
         + "; ".join(f"**{w}** targets {v['targets']}, {v['n_samples']} samples, {v['n_days_scored']} scored days"
                     for w, v in r["windows"].items()) + ". `next_day_return` of every sample was asserted equal to S3's.",
         "- Score = model output (regression prediction, or p_buy - p_sell for the classifier); converted to the S3 "
         "pseudo-probs form via `score_to_probs` and scored with `compute_signal_metrics`. IC SE = IC std / sqrt(n_days).",
         "- Training rows: 2019-01-02.., a row needs >= 60 prior rows (mirrors the TFT encoder), finite `next_day_return` and "
         "`volatility_20d`. Hyper-parameter selection used only a time-ordered split inside the train era: fit on rows "
         "<= 2022-12-31, select on 2023 rows (mean daily rank IC), refit on rows <= 2023-12-31. 2025 rows are never used.",
         f"- Target z = next_day_return / max(volatility_20d, floor={r['floor']:.6f}) clipped to [-5, 5] "
         "(floor from `training/threshold_vn.json`). Classifier target = `label_vn` (classes 0=buy, 1=hold, 2=sell).",
         "- Feature sets. F1: the 33 champion columns at day t. F2: F1 + log_ret lags 1..5 (per-ticker shift). "
         "F3: F2 + per-date cross-sectional percentile ranks of log_ret, disparity_5d/20d/60d, rsi_14, volatility_20d and the 5 lags. "
         "Event flags NULL -> 0; other NaN handled per model.",
         "- Ridge: winsorise at train 0.5/99.5 pct, NaN -> train median, standardise (all train statistics). "
         "HGB (sklearn `HistGradientBoosting`; lightgbm is not installed): lr 0.05, max_depth 4, min_samples_leaf 500, "
         "l2 10, max_bins 64, native NaN, no early stopping.",
         f"- Configs tried: {len(CONFIG_DESC)} (3 Ridge, 3 HGB regressors, 1 HGB regressor on the date-demeaned z, 1 HGB classifier). "
         "The date-demeaned-z config removes the market-wide component from the regression target (the plain regressor's "
         "first trees split on date-level columns, giving a constant score per day, so its IC is undefined at 25 iterations). "
         "All 8 configs are reported.", "",
         "## Selection on 2023 (fit on <= 2022)", "",
         "| config | grid searched (mean daily IC on 2023) | chosen |", "|---|---|---|"]
    for k, v in r["selection"].items():
        grid = v.get("grid_select_ic_2023") or v.get("select_ic_2023_by_iter")
        ch = f"alpha={v['chosen_alpha']:g}" if "chosen_alpha" in v else f"n_iter={v['chosen_n_iter']}"
        L.append(f"| {k} | " + ", ".join(f"{a}: {_f(b, 3)}" for a, b in grid.items()) + f" | {ch} |")
    L += ["", "Caveats: `ridge_F3` picked alpha at 1e5 (interior of the grid, 1e6 is worse); `hgb_reg_csz_F3` picked the "
          "smallest n_iter in the grid (25) with IC still falling as iterations increase, so its optimum may be lower still; "
          "`hgb_clf_F3` picked the largest (400) with IC still rising.", ""]
    for w, v in r["windows"].items():
        L += [f"## {w} (targets {v['targets']}; {v['n_samples']} samples, {v['n_days_scored']} days)", "",
              "| scorer | mean rank IC | SE | IC IR | quantile L/S | argmax L/S | IC minus TFT (SE of diff) | IC minus reversal (SE of diff) |",
              "|---|---|---|---|---|---|---|---|"]
        rows = [(CONFIG_DESC[k], r["configs"][k][w]) for k in CONFIG_DESC]
        rows += [("TFT champion (S3)", r["references"][w]["tft_champion"]), ("reversal (S3)", r["references"][w]["reversal"]),
                 ("random (S3)", r["references"][w]["random"])]
        for name, b in rows:
            s = b["signal"]
            d1, d2 = b.get("vs_tft"), b.get("vs_reversal")
            L.append(f"| {name} | {_f(s['mean_daily_rank_ic'])} | {_f(b['ic_se'])} | {_f(s['ic_ir'], 3)} | "
                     f"{_p(s['quantile_long_short']['mean_spread'])} | {_p(s['argmax_long_short']['spread'])} | "
                     + (f"{d1['mean_diff']:+.4f} ({d1['se']:.4f}) | " if d1 else "- | ")
                     + (f"{d2['mean_diff']:+.4f} ({d2['se']:.4f}) |" if d2 else "- |"))
        L.append("")
    L += ["The TFT / reversal / random rows are recomputed from S3's saved probabilities and asserted equal to "
          "`signal_baseline.json` (mean IC to 1e-9); the paired differences use the per-day IC series over days present in both "
          "(same days for all scorers), so their SE is much tighter than that of two independent means.", "",
          "## Production tickers (noisy)", "",
          "Time-series Spearman(score, next_day_return) over the window's days for one ticker, and sign hit-rate "
          "(sign(score) == sign(return), zero-return days excluded; the base rate of up days is shown for context). "
          "n is about 170-185 days, so SE of a correlation is about 0.07-0.08 and hit-rate SE about 0.037: "
          "**none of these per-ticker numbers is distinguishable from zero / 50%.**", "",
          "| scorer | window | 005930 rho | 005930 hit (up-rate) | 000660 rho | 000660 hit (up-rate) |", "|---|---|---|---|---|---|"]
    for w in r["windows"]:
        rows = [(CONFIG_DESC[k], r["configs"][k][w]) for k in CONFIG_DESC]
        rows += [("TFT champion", r["references"][w]["tft_champion"]), ("reversal", r["references"][w]["reversal"])]
        for name, b in rows:
            cells = []
            for t in PROD_TICKERS:
                m = b["per_ticker"][t]
                cells += [_f(m["spearman"], 3), f"{_f(m['hit_rate'], 3)} ({_f(m['base_up_rate'], 2)}), n={m['n']}"]
            L.append(f"| {name} | {w} | " + " | ".join(cells) + " |")
    L.append("")
    L.append(INTERPRETATION)
    return "\n".join(L)


CONFIG_DESC.update({"hgb_clf_label_F3": "HGB clf (fixed label), F3"})
V2_LABELS = {"hgb_clf_F3": "label_vn", "hgb_clf_label_F3": "label"}   # classifier config -> label column


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def old_e0_reference(path: str = RESULT_JSON) -> dict:
    """Old E0 (unadjusted data, legacy alignment) numbers per config/window, for the change table."""
    e0 = _read_json(path)
    if not e0:
        return {}
    out = {}
    for k, per in e0["configs"].items():
        out[k] = {w: {"ic": per[w]["signal"]["mean_daily_rank_ic"], "ic_se": per[w]["ic_se"],
                      "ic_ir": per[w]["signal"]["ic_ir"], "qls": per[w]["signal"]["quantile_long_short"]["mean_spread"]}
                  for w in per}
    ref = e0.get("references", {})
    out["reversal"] = {w: {"ic": v["reversal"]["signal"]["mean_daily_rank_ic"], "ic_se": v["reversal"]["ic_se"],
                           "ic_ir": v["reversal"]["signal"]["ic_ir"],
                           "qls": v["reversal"]["signal"]["quantile_long_short"]["mean_spread"]}
                       for w, v in ref.items()}
    return out


def tft_reference(path: str = TFX_JSON) -> dict:
    """{recipe: {window: {signal, macro_f1}}} from the TFT runner's results, if it has been run."""
    r = _read_json(path)
    if not r:
        return {}
    out = {}
    for name, rec in r.items():
        if isinstance(rec, dict) and "val_2024" in rec and "signal" in rec.get("val_2024", {}):
            out[name] = {w: {"signal": rec[w]["signal"], "macro_f1": rec[w]["metrics"].get("macro_f1"),
                             "tag": rec.get("tag")} for w in ("val_2024", "oot_2026") if w in rec}
    return out


def render_doc_v2(r: dict) -> str:
    big30 = ", ".join(w + " " + str(v["n_ret_over_30pct"]) for w, v in r["windows"].items())
    L = ["# Tabular baseline v2 (E0v2): like-for-like reference for the TFT recipes R0..R3", "",
         f"Code commit: `{r['code_commit']}` (`training/run_tabular_baseline.py`, `training/tabular_features.py`). "
         "Raw output: `training/artifacts/tabular_baseline_v2.json` (gitignored). "
         "Supersedes the numbers of [`tabular_baseline.md`](tabular_baseline.md) (old E0: unadjusted prices + legacy sample "
         "alignment; see the errata at its top and in [`signal_baseline.md`](signal_baseline.md)); those files are unchanged. "
         "Data rebuild: [`data_adjustment_report.md`](data_adjustment_report.md).", "",
         "## What changed vs old E0", "",
         f"- **Data**: adjusted prices, `DATA_VERSION = {r['data_version']}` (feature_pool/labels rebuilt; caches `__{r['data_version']}`).",
         f"- **Sample set**: exactly the (ticker, date) list the TFT runner scores with `TickerDayDataset(align=\"today\")` "
         "(sample date = the day the prediction is made = last encoder row; features at that day, target = its "
         "`next_day_return`). "
         + "; ".join(f"**{w}**: {v['n_samples']} samples (targets {v['first_date']}..{v['last_date']}, {v['n_days_scored']} scored days)"
                     for w, v in r["windows"].items())
         + ". The lists were asserted identical, in order, to `run_tfx_experiments.build_window_dataset` output and to the "
         "known counts (36,929 / 33,364). Old E0 used the legacy enumeration (36,729 / 33,364, dates shifted one row).",
         "- Everything else is the old E0 protocol: train <= 2023-12-31; hyper-parameters selected on 2023 (fit <= 2022); "
         "val 2024 scored, OOT 2026 scored once per final config; 2025 rows never used; same feature sets F1/F2/F3, "
         "Ridge / HGB (sklearn HistGradientBoosting), fixed HGB params, `random_state=0`, random reference `default_rng(0)`.",
         f"- Added config `hgb_clf_label_F3` (HGB classifier on the fixed-threshold `label`; `hgb_clf_F3` uses `label_vn`) so both "
         "label versions are on record. Macro F1 (auxiliary) exists only for the classifiers: argmax class vs that label on samples "
         "with a non-null label; regressors have no class output.",
         f"- Reproducibility: seed {r['seed']}, floor {r['floor']:.6f} (`threshold_vn.json`), returns are raw `next_day_return` "
         f"(`|ret|>30%` counts: {big30}; "
         "a clipped-return variant is in the JSON as `signal_clip30`).", "",
         "## Selection on 2023 (fit on <= 2022)", "",
         "| config | grid (mean daily IC on 2023) | chosen |", "|---|---|---|"]
    for k, v in r["selection"].items():
        grid = v.get("grid_select_ic_2023") or v.get("select_ic_2023_by_iter")
        ch = f"alpha={v['chosen_alpha']:g}" if "chosen_alpha" in v else f"n_iter={v['chosen_n_iter']}"
        L.append(f"| {k} | " + ", ".join(f"{a}: {_f(b, 3)}" for a, b in grid.items()) + f" | {ch} |")
    L.append("")
    old = r.get("old_e0", {})
    for w, v in r["windows"].items():
        L += [f"## {w} ({v['n_samples']} samples, {v['n_days_scored']} scored days)", "",
              "Signal metrics (not UTIL). IC minus reversal is a paired difference over days (SE of the difference). "
              "Old IC = old E0 (unadjusted, legacy alignment; not like-for-like).", "",
              "| scorer | mean rank IC | SE | IC IR | quantile L/S | argmax L/S | macro F1 | IC minus reversal (SE) | old E0 IC |",
              "|---|---|---|---|---|---|---|---|---|"]
        rows = [(k, CONFIG_DESC[k], r["configs"][k][w]) for k in CONFIG_DESC if k in r["configs"]]
        rows += [("reversal", "1-day reversal (-log_ret)", r["references"][w]["reversal"]),
                 ("random", "random", r["references"][w]["random"])]
        for k, name, b in rows:
            s = b["signal"]
            d2 = b.get("vs_reversal")
            o = old.get(k, {}).get(w)
            mf = b.get("macro_f1")
            L.append(f"| {name} | {_f(s['mean_daily_rank_ic'])} | {_f(b['ic_se'])} | {_f(s['ic_ir'], 3)} | "
                     f"{_p(s['quantile_long_short']['mean_spread'])} | {_p(s['argmax_long_short']['spread'])} | "
                     f"{_f(mf, 3)} | " + (f"{d2['mean_diff']:+.4f} ({d2['se']:.4f}) | " if d2 else "- | ")
                     + (f"{o['ic']:.4f}" if o and o['ic'] is not None else "-") + " |")
        L.append("")
    L += ["## Change vs old E0 (mean rank IC, same config)", "",
          "| config | val old | val v2 | oot old | oot v2 |", "|---|---|---|---|---|"]
    for k in CONFIG_DESC:
        if k not in r["configs"]:
            continue
        c = []
        for w in r["windows"]:
            o = old.get(k, {}).get(w)
            c += [_f(o["ic"]) if o and o["ic"] is not None else "-", _f(r["configs"][k][w]["signal"]["mean_daily_rank_ic"])]
        L.append(f"| {CONFIG_DESC[k]} | " + " | ".join(c) + " |")
    c = []
    for w in r["windows"]:
        o = old.get("reversal", {}).get(w)
        c += [_f(o["ic"]) if o else "-", _f(r["references"][w]["reversal"]["signal"]["mean_daily_rank_ic"])]
    L += ["| reversal | " + " | ".join(c) + " |", ""]
    L += ["## Reference values for the TFT recipes (R0..R3)", "",
          "Compare a TFT recipe's `mean_daily_rank_ic` / `ic_ir` / quantile L/S on the same window against the rows above "
          f"(bar: best tabular = `{r['bar']['config']}`, val IC {_f(r['bar']['val_ic'])}, OOT IC {_f(r['bar']['oot_ic'])}; "
          f"reversal val {_f(r['references']['val_2024']['reversal']['signal']['mean_daily_rank_ic'])}, "
          f"OOT {_f(r['references']['oot_2026']['reversal']['signal']['mean_daily_rank_ic'])}). "
          "Selection of the bar row uses val only (OOT shown for confirmation).", ""]
    tft = r.get("tft_reference") or {}
    if tft:
        L += ["| recipe | window | IC | IC IR | quantile L/S | macro F1 |", "|---|---|---|---|---|---|"]
        for name, per in tft.items():
            for w, v in per.items():
                sg = v["signal"]
                L.append(f"| {name} ({v.get('tag')}) | {w} | {_f(sg['mean_daily_rank_ic'])} | {_f(sg['ic_ir'], 3)} | "
                         f"{_p(sg['quantile_long_short']['mean_spread'])} | {_f(v['macro_f1'], 3)} |")
        L.append("")
    else:
        L += ["`training/artifacts/tfx_results.json` not present at run time: TFT recipes not yet run, so no TFT rows.", ""]
    L += ["## Production tickers (noisy)", "",
          "Time-series Spearman(score, next_day_return) and sign hit-rate per ticker (n about 170-260 days; "
          "none distinguishable from 0 / 50%).", "",
          "| scorer | window | 005930 rho | 005930 hit (up-rate) | 000660 rho | 000660 hit (up-rate) |", "|---|---|---|---|---|---|"]
    for w in r["windows"]:
        rows = [(CONFIG_DESC[k], r["configs"][k][w]) for k in CONFIG_DESC if k in r["configs"]]
        rows += [("reversal", r["references"][w]["reversal"])]
        for name, b in rows:
            cells = []
            for t in PROD_TICKERS:
                m = b["per_ticker"][t]
                cells += [_f(m["spearman"], 3), f"{_f(m['hit_rate'], 3)} ({_f(m['base_up_rate'], 2)}), n={m['n']}"]
            L.append(f"| {name} | {w} | " + " | ".join(cells) + " |")
    L.append("")
    L.append(V2_INTERPRETATION)
    return "\n".join(L)


V2_INTERPRETATION = """## Interpretation (SE-aware; differences are called real only at >= ~2 paired SE)

1. **Bar for the TFT recipes (like-for-like: adjusted data, `align="today"` sample sets, features at day d predicting d->d+1).**
   Best tabular by val IC = HGB classifier on the fixed label, F3: val IC 0.058 (SE 0.009), OOT 0.037 (SE 0.014).
   HGB regressor F3: 0.056 / 0.031. Reversal: 0.0625 / 0.0423. Quantile L/S is about 0.2% per day gross on val for the
   best rows (0.26% for reversal), 0.1-0.35% on OOT; costs not considered. A TFT recipe has to beat these, not the old E0 numbers.
2. **The rebuild changed little.** Old vs v2 IC of the same config moves by at most ~0.015 (HGB reg F3 OOT 0.044 -> 0.031,
   HGB reg F2 OOT 0.017 -> 0.026), i.e. inside the ~0.01-0.015 SE of one window; the sample set and price adjustment do not
   alter the picture. No adjusted-data return exceeds 30% in either window (`n_ret_over_30pct` = 0).
   The unchanged reversal IC (0.0625 val) is expected: rank IC is insensitive to the few adjusted corporate-action days.
3. **No tabular model is distinguishable from the 1-day reversal rule.** IC minus reversal is negative or ~0 everywhere; on val
   Ridge F1/F2/F3 and HGB reg F2, csz and clf(label_vn) are 2-4 paired SE below it, HGB reg F3 -0.006 (1.2 SE), clf(fixed label)
   -0.005 (0.6 SE). On OOT all are within ~2 SE (best clf(label_vn) +0.004 +/- 0.012). `log_ret` and its lags are inputs to every
   feature set, and no decomposition (dropping the reversal inputs) was done, so we do not know whether the HGB signal is more than
   reversal plus noise.
4. **Ranking of configs is unstable across windows** (val best: clf fixed label 0.058; OOT best: clf label_vn 0.046, csz 0.041),
   the per-window SE is 0.009-0.015 and config differences were not tested with paired SEs; treat the ordering as noise. Classifier
   argmax L/S is negative on OOT for most models (uncalibrated sign), so IC / quantile L/S are the meaningful columns.
5. **Label versions.** The fixed-threshold-label classifier is not worse than the label_vn classifier on val (0.058 vs 0.046) and
   slightly worse on OOT (0.037 vs 0.046); both differences are inside one SE. Macro F1 (auxiliary, own label each): fixed label
   0.340 val / 0.356 OOT, label_vn 0.290 / 0.260; not comparable across the two label definitions.
6. **Selection caveat.** 9 configs were all scored on val and OOT; the bar row is chosen on val with hindsight over ~9 tries
   (mildly optimistic). HGB hyper-parameters other than n_iter (and ridge alpha) were fixed a priori. `hgb_reg_csz_F3` again
   picked the smallest n_iter of the grid (25) with IC still falling, so its optimum may be lower.
7. **Provenance.** Numbers were produced by the working-tree version of `run_tabular_baseline.py` on top of commit `9523e24`
   (JSON `code_commit` reads `9523e24+dirty` because the runner was not yet committed); the committed runner differs only by
   this interpretation text. TFT recipes had not been run when this was written, so no TFT row is present;
   `--render-only` re-reads `tfx_results.json` and adds the TFT rows once it exists.
"""


def main(align: str = "today", out_json: str | None = None, doc_path: str | None = None):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from evaluation.evaluate import compute_metrics
    from training.stage1_data import DATA_VERSION
    from training.tabular_features import Preprocessor, demean_by_date, make_z_target

    dsn = os.environ.get("STOCK_DB_V2_DSN")
    if not dsn:
        raise ValueError("STOCK_DB_V2_DSN environment variable not set")
    out_json = out_json or (V2_JSON if align == "today" else V2_LEGACY_JSON)
    doc_path = doc_path if doc_path is not None else (V2_DOC if align == "today" else None)
    champ = json.load(open(CHAMPION_CONFIG_PATH))
    vn = json.load(open(VN_PATH))
    floor = float(vn["floor"])
    commit = _git_commit()
    samples = aligned_sample_sets(dsn, align)

    df = load_frame(dsn, champ["columns"])
    df, fsets = build_features(df, champ["columns"])
    df["z"] = make_z_target(df["next_day_return"].values, df["volatility_20d"].values, floor)
    df["date_s"] = df["trade_date"].dt.strftime("%Y-%m-%d")
    logger.info("frame %s rows, %d tickers, %s..%s", len(df), df.ticker.nunique(), df.date_s.min(), df.date_s.max())

    base_ok = (df["row_no"] >= WARMUP_ROWS) & df["z"].notna()
    tr_fit = df[base_ok & (df.date_s <= FIT_END)]
    tr_sel = df[base_ok & (df.date_s >= SELECT_START) & (df.date_s <= SELECT_END)]
    tr_full = df[base_ok & (df.date_s <= TRAIN_END)]
    assert tr_full.date_s.max() <= TRAIN_END and tr_fit.date_s.max() <= FIT_END
    logger.info("rows: fit<=2022 %d, select-2023 %d, final<=2023 %d", len(tr_fit), len(tr_sel), len(tr_full))

    idx = df.set_index(["ticker", "date_s"])
    assert idx.index.is_unique
    win = {}
    for name, s in samples.items():
        sub = idx.loc[list(zip(s["tickers"].tolist(), s["dates"].tolist()))]
        win[name] = sub.reset_index()
        assert win[name]["label"].notna().all(), f"{name}: sample without a fixed label in feature_pool"
        assert win[name].date_s.min() >= ("2024-01-01" if name == "val_2024" else "2026-01-01")
        assert not ((win[name].date_s >= "2025-01-01") & (win[name].date_s <= "2025-12-31")).any()
        s["ret"] = win[name]["next_day_return"].values.astype(float)
        s["log_ret"] = win[name]["log_ret"].values.astype(float)

    results = {"code_commit": commit, "align": align, "data_version": DATA_VERSION, "seed": 0, "floor": floor,
               "vn_k": vn["k"], "selection": {}, "configs": {}, "references": {}, "windows": {}}
    for name, s in samples.items():
        rr = s["ret"]
        results["windows"][name] = {
            "n_samples": int(len(rr)), "first_date": min(s["dates"].tolist()), "last_date": max(s["dates"].tolist()),
            "n_days_scored": None, "n_ret_nan": int(np.isnan(rr).sum()),
            "n_ret_over_30pct": int((np.abs(rr[np.isfinite(rr)]) > RET_CLIP).sum())}
    ics_store = {n: {} for n in samples}

    def score_windows(cname, score_fn, Xcols, clf=None):
        results["configs"][cname] = {}
        for wn, w in win.items():
            sc = score_fn(w[Xcols].values)
            b, ics = bundle(w["date_s"].values, samples[wn]["ret"], sc, w["ticker"].values)
            if clf is not None:
                model, lab = clf
                pred = model.predict(w[Xcols].values)
                ok = w[lab].notna().values
                b["macro_f1"] = compute_metrics(w.loc[ok, lab].astype(int).tolist(), pred[ok].astype(int).tolist())["macro_f1"]
                b["label_col"] = lab
            results["configs"][cname][wn] = b
            ics_store[wn][cname] = ics
            logger.info("%s %s IC %.4f (SE %.4f)", cname, wn, b["signal"]["mean_daily_rank_ic"], b["ic_se"])

    sel_dates, sel_ret = tr_sel["date_s"].values, tr_sel["next_day_return"].values

    for fname, cols in fsets.items():
        # ---- Ridge ----
        pp = Preprocessor().fit(tr_fit[cols].values)
        Xf, Xs = pp.transform(tr_fit[cols].values), pp.transform(tr_sel[cols].values)
        rs = {}
        for a in RIDGE_ALPHAS:
            m = Ridge(alpha=a).fit(Xf, tr_fit["z"].values)
            rs[a] = mean_ic(sel_dates, sel_ret, m.predict(Xs))
        best_a = argmax_nan_safe(rs)
        results["selection"][f"ridge_{fname}"] = {"grid_select_ic_2023": {str(k): v for k, v in rs.items()},
                                                  "chosen_alpha": best_a}
        pp = Preprocessor().fit(tr_full[cols].values)
        m = Ridge(alpha=best_a).fit(pp.transform(tr_full[cols].values), tr_full["z"].values)
        score_windows(f"ridge_{fname}", lambda X, m=m, pp=pp: m.predict(pp.transform(X)), cols)

        # ---- HGB regressor on z ----
        mk = lambda n: HistGradientBoostingRegressor(max_iter=n, **HGB_PARAMS)  # noqa: E731
        best_n, res = select_hgb_iters(
            mk, tr_fit[cols].values, tr_fit["z"].values, tr_sel[cols].values, sel_dates, sel_ret,
            lambda model, X: model.staged_predict(X))
        results["selection"][f"hgb_reg_{fname}"] = {"select_ic_2023_by_iter": {str(k): v for k, v in res.items()},
                                                    "chosen_n_iter": best_n}
        m = mk(best_n).fit(tr_full[cols].values, tr_full["z"].values)
        score_windows(f"hgb_reg_{fname}", lambda X, m=m: m.predict(X), cols)

    # ---- HGB regressor on the date-demeaned z (cross-sectional target), F3 ----
    cols = fsets["F3"]
    zc_fit = demean_by_date(tr_fit["date_s"].values, tr_fit["z"].values)
    zc_full = demean_by_date(tr_full["date_s"].values, tr_full["z"].values)
    mk = lambda n: HistGradientBoostingRegressor(max_iter=n, **HGB_PARAMS)  # noqa: E731
    best_n, res = select_hgb_iters(mk, tr_fit[cols].values, zc_fit, tr_sel[cols].values, sel_dates, sel_ret,
                                   lambda model, X: model.staged_predict(X))
    results["selection"]["hgb_reg_csz_F3"] = {"select_ic_2023_by_iter": {str(k): v for k, v in res.items()},
                                              "chosen_n_iter": best_n}
    m = mk(best_n).fit(tr_full[cols].values, zc_full)
    score_windows("hgb_reg_csz_F3", lambda X, m=m: m.predict(X), cols)

    # ---- HGB classifiers (F3): score = p_buy - p_sell; label_vn (as old E0) and the fixed-threshold label ----
    cols = fsets["F3"]
    mkc = lambda n: HistGradientBoostingClassifier(max_iter=n, **HGB_PARAMS)  # noqa: E731
    def staged_score(model, X):
        for p in model.staged_predict_proba(X):   # classes_ sorted: 0=buy,1=hold,2=sell
            yield p[:, 0] - p[:, 2]
    for cname, lab in V2_LABELS.items():
        ok_f, ok_full = tr_fit[lab].notna(), tr_full[lab].notna()
        best_n, res = select_hgb_iters(
            mkc, tr_fit.loc[ok_f, cols].values, tr_fit.loc[ok_f, lab].astype(int).values,
            tr_sel[cols].values, sel_dates, sel_ret, staged_score)
        results["selection"][cname] = {"select_ic_2023_by_iter": {str(k): v for k, v in res.items()},
                                       "chosen_n_iter": best_n, "label_col": lab}
        m = mkc(best_n).fit(tr_full.loc[ok_full, cols].values, tr_full.loc[ok_full, lab].astype(int).values)
        assert list(m.classes_) == [0, 1, 2]
        score_windows(cname, lambda X, m=m: (lambda p: p[:, 0] - p[:, 2])(m.predict_proba(X)), cols, clf=(m, lab))

    # ---- references on the SAME sample sets: reversal (-log_ret) and random ----
    for wn, s in samples.items():
        results["references"][wn] = {}
        b, ics = bundle(s["dates"], s["ret"], -s["log_ret"], s["tickers"])
        results["references"][wn]["reversal"] = b
        ics_store[wn]["reversal"] = ics
        b, _ = bundle(s["dates"], s["ret"], np.random.default_rng(0).standard_normal(len(s["ret"])), s["tickers"])
        results["references"][wn]["random"] = b
        results["windows"][wn]["n_days_scored"] = results["references"][wn]["reversal"]["signal"]["n_days_used"]
        for cname in results["configs"]:
            results["configs"][cname][wn]["vs_reversal"] = paired_diff(ics_store[wn][cname], ics_store[wn]["reversal"])

    # bar for TFT: best tabular config chosen by VAL IC only
    best = max(results["configs"], key=lambda k: results["configs"][k]["val_2024"]["signal"]["mean_daily_rank_ic"] or -9)
    results["bar"] = {"config": best, "val_ic": results["configs"][best]["val_2024"]["signal"]["mean_daily_rank_ic"],
                      "oot_ic": results["configs"][best]["oot_2026"]["signal"]["mean_daily_rank_ic"]}
    results["old_e0"] = old_e0_reference()
    results["tft_reference"] = tft_reference()

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("wrote %s", out_json)
    if doc_path:
        with open(doc_path, "w") as f:
            f.write(render_doc_v2(results))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--align", choices=["today", "legacy"], default="today",
                   help="sample enumeration: 'today' (E0v2, matches the TFT runner) or 'legacy' (old t+encoder_len rows)")
    p.add_argument("--out-json", default=None)
    p.add_argument("--doc", default=None, help="markdown output (default docs/tabular_baseline_v2.md for --align today)")
    p.add_argument("--render-only", action="store_true", help="re-render the v2 doc from an existing v2 JSON")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.render_only:
        src = args.out_json or V2_JSON
        res = json.load(open(src))
        res["tft_reference"] = tft_reference()          # pick up TFT results produced after the tabular run
        with open(args.doc or V2_DOC, "w") as f:
            f.write(render_doc_v2(res))
    else:
        main(args.align, args.out_json, args.doc)
