#!/usr/bin/env python
"""Timing-signal probe (CPU): how much of each simple scorer's IC is 'timing' (score minus its own ticker mean)
rather than a static per-ticker ranking?  docs/signal_diagnosis.md showed V3's IC is all static ranking with a
NEGATIVE timing IC; this gives the reference values (reversal, HGB variants, a low-vol static ranker) on exactly
the same aligned val/OOT sample sets, so E4's 'timing IC' target has a meaningful bar.

    set -a && source .env && set +a
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python training/probe_timing_baselines.py

Protocol: HGB models are refit on rows <= 2023-12-31 with the n_iter chosen by E0v2 (tabular_baseline_v2.json), so
val/OOT are never used for selection. Fixed effect = ticker mean score in the val window (OOT uses the val means as
a frozen prior); scores only, never returns (training.signal_diagnostics.timing_decomposition).
"""
import json
import logging
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT_JSON = "training/artifacts/timing_probe.json"
OUT_DOC = "docs/timing_probe.md"
E0V2_JSON = "training/artifacts/tabular_baseline_v2.json"


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def summarize(scores: dict, samples: dict) -> dict:
    """{scorer: {window: timing_decomposition dict}}. `scores[scorer][window]` is aligned to `samples[window]`."""
    from training.signal_diagnostics import timing_decomposition
    out = {}
    for name, per_window in scores.items():
        out[name] = {}
        prior = None
        for w in ("val_2024", "oot_2026"):
            s = samples[w]
            r = timing_decomposition(s["dates"], s["tickers"], per_window[w], s["ret"], prior_ticker_means=prior)
            out[name][w] = r
            if w == "val_2024":
                prior = r["ticker_means"]        # frozen val means -> OOT prior
    return out


def render(summary: dict, meta: dict) -> str:
    def f(x):
        return "n/a" if x is None else f"{x:+.4f}"
    L = ["# Timing probe (CPU)", "",
         f"code_commit={meta['code_commit']}, data_version={meta['data_version']}. Fixed effect = ticker mean score in val; "
         "OOT uses the val means. SE ~0.014 (val) / ~0.018 (OOT): differences under 2 SE are noise.", "",
         "| scorer | window | raw IC | fixed-effect IC | timing IC | timing SE | timing top-20% minus bottom-20% /day |",
         "|---|---|---|---|---|---|---|"]
    for name, per_window in summary.items():
        for w, r in per_window.items():
            L.append(f"| {name} | {w} | {f(r['raw_ic'])} | {f(r['fixed_effect_ic'])} | {f(r['timing_ic'])} | "
                     f"{f(r.get('timing_ic_se'))} | {f(r.get('timing_quantile_ls'))} |")
    L += ["", "Reference (docs/signal_diagnosis.md): V3 TFT raw 0.0412/0.0392, fixed 0.0409/0.0392, timing -0.0253/-0.0393.", ""]
    return "\n".join(L)


def main() -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    from training.run_tabular_baseline import (CHAMPION_CONFIG_PATH, FIT_END, HGB_PARAMS, TRAIN_END, VN_PATH,
                                               WARMUP_ROWS, aligned_sample_sets, build_features, load_frame)
    from training.stage1_data import DATA_VERSION
    from training.tabular_features import demean_by_date, make_z_target

    dsn = os.environ.get("STOCK_DB_V2_DSN")
    if not dsn:
        raise SystemExit("STOCK_DB_V2_DSN environment variable not set")
    champ = json.load(open(CHAMPION_CONFIG_PATH))
    floor = float(json.load(open(VN_PATH))["floor"])
    n_iter = {k: v["chosen_n_iter"] for k, v in json.load(open(E0V2_JSON))["selection"].items() if "chosen_n_iter" in v}

    df = load_frame(dsn, champ["columns"])
    df, fsets = build_features(df, champ["columns"])
    df["z"] = make_z_target(df["next_day_return"].values, df["volatility_20d"].values, floor)
    df["date_s"] = df["trade_date"].dt.strftime("%Y-%m-%d")
    tr = df[(df["row_no"] >= WARMUP_ROWS) & df["z"].notna() & (df["date_s"] <= TRAIN_END)]
    f3 = fsets["F3"]
    logger.info("train rows %d, F3 cols %d", len(tr), len(f3))

    samples = aligned_sample_sets(dsn, "today")
    idx = df.set_index(["ticker", "date_s"])
    win = {}
    for w, s in samples.items():
        win[w] = idx.loc[list(zip(s["tickers"].tolist(), s["dates"].tolist()))].reset_index()
        s["ret"] = win[w]["next_day_return"].values.astype(float)

    # train-era mean volatility per ticker (static low-vol ranker; no val/OOT information)
    tk_vol = tr.groupby("ticker")["volatility_20d"].mean()

    scores = {"reversal (-log_ret)": {w: -win[w]["log_ret"].values for w in win},
              "static low-vol (-train mean vol)": {w: -win[w]["ticker"].map(tk_vol).values.astype(float) for w in win}}

    reg = HistGradientBoostingRegressor(max_iter=n_iter["hgb_reg_F3"], **HGB_PARAMS).fit(tr[f3].values, tr["z"].values)
    scores["HGB reg (z) F3"] = {w: reg.predict(win[w][f3].values) for w in win}

    zc = demean_by_date(tr["date_s"].values, tr["z"].values)
    regc = HistGradientBoostingRegressor(max_iter=n_iter["hgb_reg_csz_F3"], **HGB_PARAMS).fit(tr[f3].values, zc)
    scores["HGB reg (date-demeaned z) F3"] = {w: regc.predict(win[w][f3].values) for w in win}

    ok = tr["label"].notna()
    clf = HistGradientBoostingClassifier(max_iter=n_iter["hgb_clf_label_F3"], **HGB_PARAMS).fit(
        tr.loc[ok, f3].values, tr.loc[ok, "label"].astype(int).values)
    assert list(clf.classes_) == [0, 1, 2]
    scores["HGB clf (fixed label) F3"] = {w: (lambda p: p[:, 0] - p[:, 2])(clf.predict_proba(win[w][f3].values)) for w in win}

    summary = summarize(scores, samples)
    meta = {"code_commit": _commit(), "data_version": DATA_VERSION, "n_iter": n_iter, "fit_end": FIT_END}
    with open(OUT_JSON, "w") as f:
        json.dump({"meta": meta, "summary": {k: {w: {kk: vv for kk, vv in r.items() if kk != "ticker_means"}
                                                 for w, r in v.items()} for k, v in summary.items()}},
                  f, indent=2, ensure_ascii=False, default=float)
    with open(OUT_DOC, "w") as f:
        f.write(render(summary, meta))
    logger.info("wrote %s, %s", OUT_JSON, OUT_DOC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
