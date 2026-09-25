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
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python training/run_tabular_baseline.py
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

ART = "training/artifacts"
RESULT_JSON = f"{ART}/tabular_baseline.json"
DOC_PATH = "docs/tabular_baseline.md"
S3_JSON = f"{ART}/signal_baseline.json"
CHAMPION_CONFIG_PATH = "training/champion_config.json"
VN_PATH = "training/threshold_vn.json"
TRAIN_START, TRAIN_END = "2019-01-02", "2023-12-31"
FIT_END, SELECT_START, SELECT_END = "2022-12-31", "2023-01-01", "2023-12-31"
LOAD_END = "2026-09-08"
WARMUP_ROWS = 60           # mirror the TFT encoder: a row needs >=60 prior rows in the loaded frame
EXPECTED = {"val_2024": 36729, "oot_2026": 33364}
WINDOWS = {"val_2024": "2024-01-01..2024-12-31", "oot_2026": "2026-01-01..2026-09-07"}
CACHES = {"val_2024": (f"{ART}/stage1_cache_val_2024-01-01_2024-12-31.pkl", "2026-01-01"),
          "oot_2026": (f"{ART}/stage1_cache_oot_2025-09-01_2026-09-08.pkl", "2026-01-01")}
PROD_TICKERS = ["005930", "000660"]
RANK_BASE = ["log_ret", "disparity_5d", "disparity_20d", "disparity_60d", "rsi_14", "volatility_20d"]
LAGS = [1, 2, 3, 4, 5]
EVENT_COLS = ["is_dividend", "is_bonus_issue", "is_rights_offering", "is_split", "is_vi_triggered"]
RIDGE_ALPHAS = [10.0, 1_000.0, 100_000.0]
HGB_ITERS = [25, 50, 100, 200, 300, 400]
HGB_PARAMS = dict(learning_rate=0.05, max_depth=4, min_samples_leaf=500, l2_regularization=10.0,
                  max_bins=64, early_stopping=False, random_state=0)


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "training/run_tabular_baseline.py",
             "training/tabular_features.py"], text=True).strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def load_frame(dsn: str, cols: list[str]) -> pd.DataFrame:
    import psycopg2
    sel = ", ".join(sorted(set(cols + ["next_day_return", "volatility_20d", "label_vn"])))
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
    return {"signal": sig, "ic_se": se, "per_ticker": per_t}, ics


def paired_diff(ics_a: dict, ics_b: dict):
    days = sorted(set(ics_a) & set(ics_b))
    d = np.array([ics_a[k] - ics_b[k] for k in days])
    if len(d) < 2:
        return None
    return {"mean_diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))), "n_days": len(d)}


def select_hgb_iters(make, Xtr, ytr, Xsel, sel_dates, sel_ret, score_fn):
    """Fit once with max iters, score staged predictions on the 2023 split, return best n_iter."""
    model = make(max(HGB_ITERS)).fit(Xtr, ytr)
    res = {}
    for i, pred in enumerate(score_fn(model, Xsel)):
        n_it = i + 1
        if n_it in HGB_ITERS:
            ics = daily_ics(sel_dates, sel_ret, pred)
            res[n_it] = float(np.mean(list(ics.values())))
    best = max(res, key=res.get)
    return best, res


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from training.tabular_features import Preprocessor, make_z_target

    dsn = os.environ.get("STOCK_DB_V2_DSN")
    if not dsn:
        raise ValueError("STOCK_DB_V2_DSN environment variable not set")
    champ = json.load(open(CHAMPION_CONFIG_PATH))
    vn = json.load(open(VN_PATH))
    floor = float(vn["floor"])
    commit = _git_commit()
    samples = sample_sets()

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
    win = {}
    for name, s in samples.items():
        sub = idx.loc[list(zip(s["tickers"].tolist(), s["dates"].tolist()))]
        assert np.allclose(sub["next_day_return"].values, s["ret"], equal_nan=True), "next_day_return mismatch vs S3"
        win[name] = sub.reset_index()
        assert win[name].date_s.min() >= ("2024-01-01" if name == "val_2024" else "2026-01-01")
        assert not ((win[name].date_s >= "2025-01-01") & (win[name].date_s <= "2025-12-31")).any()

    results = {"code_commit": commit, "floor": floor, "vn_k": vn["k"], "selection": {}, "configs": {},
               "references": {}, "windows": {}}
    for name, s in samples.items():
        results["windows"][name] = {"targets": WINDOWS[name], "n_samples": int(len(s["ret"])),
                                    "n_days_scored": None}
    ics_store = {n: {} for n in samples}

    def score_windows(cname, score_fn, Xcols):
        results["configs"][cname] = {}
        for wn, w in win.items():
            sc = score_fn(w[Xcols].values)
            b, ics = bundle(w["date_s"].values, samples[wn]["ret"], sc, w["ticker"].values)
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
            rs[a] = float(np.mean(list(daily_ics(sel_dates, sel_ret, m.predict(Xs)).values())))
        best_a = max(rs, key=rs.get)
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

    # ---- HGB classifier on label_vn (F3): score = p_buy - p_sell ----
    cols = fsets["F3"]
    ok_f, ok_full = tr_fit["label_vn"].notna(), tr_full["label_vn"].notna()
    mkc = lambda n: HistGradientBoostingClassifier(max_iter=n, **HGB_PARAMS)  # noqa: E731
    def staged_score(model, X):
        for p in model.staged_predict_proba(X):   # classes_ sorted: 0=buy,1=hold,2=sell
            yield p[:, 0] - p[:, 2]
    best_n, res = select_hgb_iters(
        mkc, tr_fit.loc[ok_f, cols].values, tr_fit.loc[ok_f, "label_vn"].astype(int).values,
        tr_sel[cols].values, sel_dates, sel_ret, staged_score)
    results["selection"]["hgb_clf_F3"] = {"select_ic_2023_by_iter": {str(k): v for k, v in res.items()},
                                          "chosen_n_iter": best_n}
    m = mkc(best_n).fit(tr_full.loc[ok_full, cols].values, tr_full.loc[ok_full, "label_vn"].astype(int).values)
    assert list(m.classes_) == [0, 1, 2]
    score_windows("hgb_clf_F3", lambda X, m=m: (lambda p: p[:, 0] - p[:, 2])(m.predict_proba(X)), cols)

    # ---- references: S3 rows copied from JSON; daily IC series rebuilt from S3 npz for paired tests ----
    s3 = json.load(open(S3_JSON))
    for wn, s in samples.items():
        results["references"][wn] = {}
        for rname, sc in (("tft_champion", s["champ_probs"][:, 0] - s["champ_probs"][:, 2]),
                          ("reversal", -s["log_ret"])):
            b, ics = bundle(samples[wn]["dates"], s["ret"], sc, s["tickers"])
            src = s3[wn]["model" if rname == "tft_champion" else "references"]
            src = src if rname == "tft_champion" else src["reversal"]
            assert abs(b["signal"]["mean_daily_rank_ic"] - src["signal"]["mean_daily_rank_ic"]) < 1e-9, rname
            b["s3_copy"] = {"signal": src["signal"]}       # verbatim S3 numbers
            results["references"][wn][rname] = b
            ics_store[wn][rname] = ics
        b, _ = bundle(samples[wn]["dates"], s["ret"], np.random.default_rng(0).standard_normal(len(s["ret"])),
                      s["tickers"])
        assert abs(b["signal"]["mean_daily_rank_ic"] - s3[wn]["references"]["random"]["signal"]["mean_daily_rank_ic"]) < 1e-9
        results["references"][wn]["random"] = b
        results["windows"][wn]["n_days_scored"] = results["references"][wn]["reversal"]["signal"]["n_days_used"]
        for cname in results["configs"]:
            results["configs"][cname][wn]["vs_tft"] = paired_diff(ics_store[wn][cname], ics_store[wn]["tft_champion"])
            results["configs"][cname][wn]["vs_reversal"] = paired_diff(ics_store[wn][cname], ics_store[wn]["reversal"])

    with open(RESULT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("wrote %s", RESULT_JSON)


if __name__ == "__main__":
    main()
