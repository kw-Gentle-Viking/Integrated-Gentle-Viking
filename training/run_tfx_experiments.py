#!/usr/bin/env python
"""TFX: TFT retraining recipes R1..R3 (plan 2026-09-26-signal-improvement-plan.md, section
"E0 결과 이후 수정"). Scored on the SAME windows/metrics as S3/E0 so rows are directly comparable.

  R0 aligned     raw inputs (no preprocessing), fixed label, align="today" -- isolates the
                 train/serving alignment fix
  R1 std         R0 + input standardisation (fixed label)
  R2 std_vn      R1 + label_vn (volatility-normalised label)
  R3 std_vn_csr  R2 + per-date cross-sectional rank inputs (log_ret, disparity_5d/20d/60d, rsi_14,
                 volatility_20d)

ALL recipes use TickerDayDataset(align="today"): label / future / static come from the LAST encoder
row (the day whose features the model has seen), as serving does. The legacy dataset built the label
from the row AFTER the encoder (an effective 2-day-ahead forecast, out of step with serving). Legacy
alignment is kept only to re-derive the S3/E0 sample sets (`--align legacy --check-windows`).

Common: champion architecture/hparams (training/champion_config.json), batch 128, max 12 epochs,
val-loss early stopping (patience 3), balanced class weights from each recipe's TRAIN label
distribution (after glitch masking), epoch-level resume, best-val-loss checkpoint.

Discipline (binding): every preprocessing statistic is fitted on train rows dated <= 2023-12-31 only
(guarded in training.preprocess.fit_preprocessor); val/OOT are apply-only. Training-label masking of
glitch rows (|next_day_return| > 0.31 -> label NaN) touches TRAIN frames only; the evaluation sample
set is gated by the fixed-label NULL set for every recipe and is asserted BEFORE training starts:
legacy alignment reproduces S3/E0 exactly (counts 36729 / 33364 and the exact (ticker, date) list);
for align="today" the sample DATES are the last-encoder-row dates (a few hundred more samples than S3:
first window per ticker) and are asserted to lie inside the window (no 2025-dated OOT sample). Selection is by val
2024 rank IC; the 2026 OOT window is confirmation only. The 2025 test window is never scored.

Resumable: recipes already in tfx_results.json are skipped; an interrupted recipe resumes from its
epoch-level checkpoint. `--max-minutes` is honoured between recipes AND between epochs (a budget stop
leaves a resumable state and exits 0 without recording the recipe).

    set -a && source .env && set +a
    PYTHONPATH=. python training/run_tfx_experiments.py [--recipes std std_vn] [--max-minutes 600]

Output paths are overridable (--artifacts-dir / --results-path / --doc-path / --model-versions-path)
so dry runs never touch the real ones.
"""
import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "training/artifacts"
CHAMPION_CONFIG_PATH = "training/champion_config.json"
DOC_PATH = "docs/tfx_experiments.md"
MODEL_VERSIONS_PATH = "docs/model_versions.md"
ENCODER_LEN = 60
DEFAULT_ALIGN = "today"
BATCH_SIZE = 128
MAX_EPOCHS = 12
PATIENCE = 3
MIN_NAMES_PER_DAY = 20
GLITCH_MASK_THRESHOLD = 0.31     # |next_day_return| above this -> TRAIN label NaN
RET_CLIP = 0.30                  # spread variant with returns clipped to +-30% (KRX daily limit)
FIT_CUTOFF = "2023-12-31"
CS_RANK_COLS = ["log_ret", "disparity_5d", "disparity_20d", "disparity_60d", "rsi_14", "volatility_20d"]
OOT_IC_TARGET = 0.044

TRAIN_SPLIT = dict(start="2019-01-02", end="2023-12-31")
SPLITS = {   # name -> load window, target-date window, cache file (default label; others get __<label>)
    "train": dict(load_start="2019-01-02", load_end="2023-12-31",
                  cache="stage1_cache_train_2019-01-02_2023-12-31.pkl"),
    "val_2024": dict(load_start="2024-01-01", load_end="2024-12-31", min_target="2024-01-01",
                     max_target="2024-12-31", cache="stage1_cache_val_2024-01-01_2024-12-31.pkl",
                     expected_samples=36729, s3_npz="signal_baseline_probs_val_2024.npz"),
    "oot_2026": dict(load_start="2025-09-01", load_end="2026-09-08", min_target="2026-01-01",
                     max_target="2026-09-08", cache="stage1_cache_oot_2025-09-01_2026-09-08.pkl",
                     expected_samples=33364, s3_npz="signal_baseline_probs_oot_2026.npz"),
}
EVAL_WINDOWS = ("val_2024", "oot_2026")

RECIPES = {
    "aligned": dict(label_col="label", rank_inputs=False, preprocess=False, tag="R0",
                    desc="raw inputs (no preprocessing, all 33 champion columns), fixed label; only the "
                         "alignment fix vs the legacy champion recipe"),
    "std": dict(label_col="label", rank_inputs=False, preprocess=True, tag="R1",
                desc="R0 + input standardisation (train<=2023 clip 0.5/99.5 + z-score, binary untouched, "
                     "constant cols dropped)"),
    "std_vn": dict(label_col="label_vn", rank_inputs=False, preprocess=True, tag="R2",
                   desc="R1 + volatility-normalised label (label_vn)"),
    "std_vn_csr": dict(label_col="label_vn", rank_inputs=True, preprocess=True, tag="R3",
                       desc="R2 + per-date cross-sectional rank inputs "
                            "(log_ret, disparity_5d/20d/60d, rsi_14, volatility_20d)"),
}


# --------------------------------------------------------------------------------------------
# pure helpers (unit-tested in training/test_run_tfx_experiments.py)
# --------------------------------------------------------------------------------------------
def _date_str(d) -> str:
    return d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)


def attach_next_day_return(frames: dict, lookup: dict) -> dict:
    """NEW frames with a `next_day_return` column joined on (ticker, 'YYYY-MM-DD'); NaN if the
    lookup has no / NULL value. The default training path never reads this column."""
    out = {}
    for tk, df in frames.items():
        df = df.copy()
        df["next_day_return"] = [
            np.nan if lookup.get((tk, _date_str(d))) is None else lookup[(tk, _date_str(d))]
            for d in df["trade_date"]]
        out[tk] = df
    return out


def mask_glitch_labels(frames: dict, threshold: float = GLITCH_MASK_THRESHOLD) -> tuple[dict, int]:
    """TRAIN frames only: label -> NaN where |next_day_return| > threshold (unadjusted corporate
    actions). NaN return is not masked. Returns (new frames, number of rows newly masked)."""
    out, n = {}, 0
    for tk, df in frames.items():
        df = df.copy()
        bad = (df["next_day_return"].abs() > threshold) & df["label"].notna()
        n += int(bad.sum())
        df.loc[bad, "label"] = np.nan
        out[tk] = df
    return out, n


def build_eval_frames(default_frames: dict, recipe_frames: dict) -> dict:
    """Evaluation frames whose SAMPLE SET equals the fixed-label (S3/E0) one regardless of recipe.
    `label` = recipe label where both exist; rows the fixed label keeps but the recipe label lacks
    (label_vn NULL from missing volatility) get a placeholder 1.0 so they stay in the sample set;
    `label_ok` marks samples with a genuine recipe label (only those enter F1/per-class metrics).
    Rows the fixed label lacks stay NaN (dropped by the Dataset), exactly as in S3."""
    if list(default_frames) != list(recipe_frames):
        raise ValueError("default-label and recipe-label frames cover different tickers")
    out = {}
    for tk, base in default_frames.items():
        rec = recipe_frames[tk]
        if len(base) != len(rec) or not (base["trade_date"].values == rec["trade_date"].values).all():
            raise ValueError(f"{tk}: default-label and recipe-label frames have different rows")
        df = rec.copy()
        gate = base["label"].notna().to_numpy()
        have = rec["label"].notna().to_numpy()
        lab = np.where(gate, np.where(have, rec["label"].to_numpy(dtype=float), 1.0), np.nan)
        df["label"] = lab
        df["label_ok"] = gate & have
        out[tk] = df
    return out


def clip_returns(rets, limit: float = RET_CLIP) -> np.ndarray:
    return np.clip(np.asarray(rets, dtype=float), -limit, limit)


def subsample_tickers(frames: dict, max_tickers: int | None) -> dict:
    if not max_tickers:
        return frames
    return {k: frames[k] for k in sorted(frames)[:max_tickers]}


def ic_se(signal: dict):
    if signal.get("ic_std") is None or not signal.get("n_days_used"):
        return None
    return signal["ic_std"] / math.sqrt(signal["n_days_used"])


def load_results(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_results_atomic(path: str, results: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def pending_recipes(results: dict, wanted: list[str]) -> list[str]:
    return [r for r in wanted if r not in results]


def model_versions_row_exists(path: str, version: str) -> bool:
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return any(line.startswith(f"| {version} |") for line in f)


def read_reference_rows(ref_dir: str) -> dict:
    """{window: [(name, signal_dict, macro_f1_or_None), ...]}: TFT champion + reversal from S3's
    signal_baseline.json, HGB F3 (regressor on z) from E0's tabular_baseline.json. Missing files or
    keys are skipped (the doc simply has fewer comparison rows)."""
    out = {w: [] for w in EVAL_WINDOWS}
    s3, e0 = {}, {}
    try:
        with open(os.path.join(ref_dir, "signal_baseline.json")) as f:
            s3 = json.load(f)
    except (OSError, ValueError):
        pass
    try:
        with open(os.path.join(ref_dir, "tabular_baseline.json")) as f:
            e0 = json.load(f)
    except (OSError, ValueError):
        pass
    for w in EVAL_WINDOWS:
        r = s3.get(w)
        if r:
            out[w].append(("TFT champion (S3, legacy-aligned)", r["model"]["signal"], r["model"]["metrics"].get("macro_f1")))
            if "reversal" in r.get("references", {}):
                out[w].append(("1-day reversal (S3, legacy sample set)", r["references"]["reversal"]["signal"], None))
        h = e0.get("configs", {}).get("hgb_reg_F3", {}).get(w)
        if h:
            out[w].append(("HGB reg (z), F3 (E0, old data)", h["signal"], None))
    return out


def _fmt(x, nd=4, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 100:.{max(nd - 2, 1)}f}%" if pct else f"{x:.{nd}f}"


def _sig_cells(sig: dict, sig_clip: dict | None) -> list[str]:
    ls, q = sig["argmax_long_short"], sig["quantile_long_short"]
    cl = sig_clip or {}
    cls_, cq = cl.get("argmax_long_short", {}), cl.get("quantile_long_short", {})
    return [_fmt(sig["mean_daily_rank_ic"]), _fmt(ic_se(sig)), _fmt(sig["ic_ir"], 3), str(sig["n_days_used"]),
            _fmt(q["mean_spread"], 4, True), _fmt(cq.get("mean_spread"), 4, True),
            _fmt(ls["spread"], 4, True), _fmt(cls_.get("spread"), 4, True)]


def render_doc(results: dict, refs: dict, code_commit: str, recipes=None) -> str:
    recipes = recipes or list(RECIPES)
    L = [
        "# TFX: TFT retraining recipes (input standardisation / label_vn / cross-sectional rank inputs)", "",
        "Plan: `docs/superpowers/plans/2026-09-26-signal-improvement-plan.md`, section \"E0 결과 이후 수정\". "
        f"Code commit: `{code_commit}` (`training/run_tfx_experiments.py`, `training/preprocess.py`). "
        "Raw output: `training/artifacts/tfx_results.json` (gitignored); preprocessing artifacts "
        "`training/artifacts/preproc_<recipe>.json`.", "",
        "**Selection by val IC; OOT is confirmation only.** Model/recipe choice uses the 2024 val window "
        "(mean daily rank IC). The 2026 OOT window was scored once per recipe (3 recipes) and is NOT a "
        "selection criterion. The 2025 test window is never scored. "
        f"Bar to clear: OOT IC >= ~{OOT_IC_TARGET} (HGB F3 level).", "",
        "## Protocol", "",
        "- Architecture/hparams from `training/champion_config.json` (state 32, heads 8, layers 2, dropout 0.1658, "
        f"lr 0.000291), batch {BATCH_SIZE}, max {MAX_EPOCHS} epochs, val-loss early stopping (patience {PATIENCE}), "
        "best-val-loss checkpoint, balanced class weights from each recipe's train label distribution.",
        "- **Alignment**: all recipes use `TickerDayDataset(align=\"today\")`: label / future / static features come from the last "
        "encoder row (the model sees the labelled day's own features, as serving does). The legacy dataset took them from the row "
        "AFTER the encoder (effective 2-day-ahead forecast). Sample date = date of the last encoder row, so the aligned sample sets "
        "contain a few hundred more samples than S3/E0 (first window per ticker); they are NOT asserted equal to S3's 36729 / 33364 "
        "(that legacy reproduction is checked separately with `--align legacy --check-windows`).",
        f"- R1-R3 preprocessing fitted on train rows dated <= {FIT_CUTOFF} only (clip at train 0.5/99.5 quantiles, z-score of the "
        "clipped values; binary 0/1 untouched; zero-variance columns dropped); val/OOT apply-only. R0 feeds raw values.",
        f"- Train labels of glitch rows (|next_day_return| > {GLITCH_MASK_THRESHOLD}) masked to NaN; evaluation sample sets "
        "gated by the fixed-label NULL set for every recipe (same across recipes). Spreads shown raw and with returns clipped to +-30%; IC is rank-based (primary). "
        "Comparison rows (TFT champion, reversal from S3; HGB F3 from E0) carry raw spreads only (clipped = n/a) and are "
        "**legacy-aligned TFT / old-data tabular; not like-for-like** (different sample sets and price data version; clean-data "
        "aligned tabular baselines are re-run separately).",
        "- Score = p_buy - p_sell; rank IC = per-day cross-sectional Spearman with next_day_return "
        f"(days with < {MIN_NAMES_PER_DAY} names skipped); SE = IC std / sqrt(days). Macro F1 / per-class P/R are against each "
        "recipe's own label, so R1 (fixed label) vs R2/R3 (label_vn) F1 values are NOT comparable with each other.", "",
        "## Recipes", "",
        "| recipe | tag | alignment | preprocessing | label | model inputs | dropped (constant in train) | epochs run / stop | best val loss | class weights (buy/hold/sell) | train samples (glitch-masked rows) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in recipes:
        spec, r = RECIPES[name], results.get(name)
        if r is None:
            L.append(f"| {name} | {spec['tag']} | today | | {spec['label_col']} | _pending_ | | | | | |")
            continue
        t = r["train"]
        L.append(f"| {name} | {spec['tag']} | {t['align']} | {'standardised' if t['preprocessed'] else 'raw'} | "
                 f"{spec['label_col']} | {t['n_inputs']} ({t['n_rank_inputs']} rank) | "
                 f"{', '.join(t['dropped_columns']) or 'none'} | {t['last_epoch'] + 1} / {t['stopped_reason']} | "
                 f"{_fmt(t['best_val_loss'])} | {' / '.join(f'{w:.3f}' for w in t['class_weights'])} | "
                 f"{t['n_train_samples']} ({t['n_glitch_masked']}) |")
    L += ["", "Recipe descriptions: " + "; ".join(f"**{RECIPES[n]['tag']} {n}**: {RECIPES[n]['desc']}" for n in recipes) + ".", ""]

    hdr = ("| scorer | mean rank IC | SE | IC IR | days | quantile L/S | quantile L/S (clip30) | argmax L/S | "
           "argmax L/S (clip30) | macro F1 | pred share buy/hold/sell |")
    sep = "|" + "---|" * 11
    for w in EVAL_WINDOWS:
        spec = SPLITS[w]
        title = "val_2024 (selection window)" if w == "val_2024" else "oot_2026 (confirmation only, not a selection criterion)"
        L += [f"## {title}: targets {spec['min_target']}..{spec['max_target']}", ""]
        first = next((results[n][w] for n in recipes if n in results), None)
        if first:
            L += [f"Samples {first['n_samples']} (with returns {first['signal']['n_samples']}), actual target dates "
                  f"{first['target_date_min']}..{first['target_date_max']}, {first['signal']['n_days_used']} scored days, "
                  f"mean names/day {_fmt(first['mean_names_per_day'], 1)}; feature rows loaded "
                  f"{spec['load_start']}..{spec['load_end']}. Legacy-aligned S3/E0 counts were {spec['expected_samples']} (aligned sets are larger). "
                  f"Samples with |next_day_return| > 30%: {first['n_ret_over_30pct']}.", ""]
        L += [hdr, sep]
        for name in recipes:
            r = results.get(name)
            if r is None:
                L.append(f"| {RECIPES[name]['tag']} {name} | _pending_ |" + " |" * 9)
                continue
            x = r[w]
            sh = x["pred_share"]
            L.append(f"| **{RECIPES[name]['tag']} {name}** | " + " | ".join(_sig_cells(x["signal"], x["signal_clip30"])) +
                     f" | {_fmt(x['metrics']['macro_f1'])} ({x['n_label_ok']} lbl) | "
                     f"{_fmt(sh['buy'], 3)} / {_fmt(sh['hold'], 3)} / {_fmt(sh['sell'], 3)} |")
        for rname, sig, f1 in refs.get(w, []):
            L.append(f"| {rname} | " + " | ".join(_sig_cells(sig, None)) + f" | {_fmt(f1)} | n/a |")
        L += [""]
        pc_rows = [(n, results[n][w]) for n in recipes if n in results]
        if pc_rows:
            L += ["Per-class precision / recall (0=buy, 1=hold, 2=sell; against each recipe's own label):", "",
                  "| recipe | buy P / R | hold P / R | sell P / R | accuracy | MCC |", "|---|---|---|---|---|---|"]
            for n, x in pc_rows:
                pc = x["metrics"]["per_class"]
                cells = [f"{_fmt(pc[str(c)]['precision'], 3)} / {_fmt(pc[str(c)]['recall'], 3)}" for c in range(3)]
                L.append(f"| {n} | " + " | ".join(cells) + f" | {_fmt(x['metrics']['accuracy'])} | {_fmt(x['metrics']['mcc'])} |")
            L.append("")
    dropped = sorted({c for r in results.values() for c in r["train"]["dropped_columns"]})
    L += ["## Columns", "",
          f"Dropped as constant in train (<= {FIT_CUTOFF}): {', '.join(dropped) if dropped else 'none recorded yet'}. "
          "Kept/dropped column lists per recipe and the fitted clip bounds / mean / std are in "
          "`training/artifacts/preproc_<recipe>.json`.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------------------------
# data access (injectable so tests / dry runs never need the DB)
# --------------------------------------------------------------------------------------------
class RealData:
    """Frames via training.stage1_data.load_or_build_ticker_dfs (never hard-code cache names -- the
    loader owns them), returns via training.signal_data.fetch_next_day_returns (read-only)."""

    def __init__(self, dsn: str):
        self.dsn, self._ret = dsn, {}

    def frames(self, split: str, label_col: str) -> dict:
        from training.stage1_data import load_or_build_ticker_dfs
        sp = SPLITS[split]
        return load_or_build_ticker_dfs(self.dsn, sp["load_start"], sp["load_end"],
                                        os.path.join(ARTIFACTS_DIR, sp["cache"]),   # shared read-mostly caches
                                        label_col=label_col)

    def next_day_returns(self, split: str) -> dict:
        if split not in self._ret:
            from training.signal_data import fetch_next_day_returns
            sp = SPLITS[split]
            self._ret[split] = fetch_next_day_returns(self.dsn, sp["load_start"], sp["load_end"])
        return self._ret[split]


# --------------------------------------------------------------------------------------------
# recipe execution
# --------------------------------------------------------------------------------------------
THRESHOLD_FILES = {"label": "threshold.json", "label_vn": "threshold_vn.json"}
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
# fingerprint keys computable without loading data (used for the cheap skip-check of recorded results)
CHEAP_FP_KEYS = ("align", "data_version", "seed", "hparams", "threshold", "label_col", "batch_size", "preprocessed")


def columns_hash(cols) -> str:
    return hashlib.sha256(json.dumps(list(cols)).encode()).hexdigest()[:16]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_threshold(label_col: str, training_dir: str = TRAINING_DIR) -> dict:
    """The label-threshold parameters this recipe's label was derived with (timestamps excluded)."""
    path = os.path.join(training_dir, THRESHOLD_FILES[label_col])
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {"file": THRESHOLD_FILES[label_col], "error": "unreadable"}
    return {"file": THRESHOLD_FILES[label_col],
            **{k: v for k, v in d.items() if k not in ("timestamp", "computed_on")}}


def cheap_fingerprint(name: str, opts: dict) -> dict:
    from training.stage1_data import DATA_VERSION
    spec = RECIPES[name]
    return json.loads(json.dumps({
        "align": opts.get("align", DEFAULT_ALIGN), "data_version": DATA_VERSION, "seed": opts.get("seed", 0),
        "hparams": opts["hparams"], "batch_size": opts["batch_size"], "label_col": spec["label_col"],
        "preprocessed": spec["preprocess"],
        "threshold": read_threshold(spec["label_col"], opts.get("threshold_dir", TRAINING_DIR))}, sort_keys=True))


def full_fingerprint(name: str, opts: dict, cols, class_weights) -> dict:
    fp = cheap_fingerprint(name, opts)
    fp["columns_hash"] = columns_hash(cols)
    fp["class_weights"] = [round(float(w), 6) for w in class_weights]
    return fp


def fingerprint_mismatch(recorded: dict | None, current: dict) -> list[str]:
    """Keys (of `current`) whose recorded value differs; a missing fingerprint mismatches on everything."""
    if not recorded:
        return sorted(current)
    return sorted(k for k in current if recorded.get(k) != current[k])


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "training/run_tfx_experiments.py", "training/preprocess.py",
             "training/train.py", "training/tabular_features.py", "training/stage1_data.py",
             "training/signal_data.py", "training/dataset.py", "training/config.py", "training/label.py",
             "evaluation/evaluate.py"], text=True).strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def prepare_recipe(name: str, data, opts: dict) -> dict:
    """Load + join + (rank inputs) + fit preprocessing on train + apply everywhere. Returns
    everything training/scoring needs; no model work here."""
    from training import preprocess as pp
    spec = RECIPES[name]
    label_col = spec["label_col"]
    champion = opts["champion"]
    base_cols = list(champion["columns"])
    max_t = opts.get("max_tickers")

    def load(split, lab):
        fr = subsample_tickers(data.frames(split, lab), max_t)
        return attach_next_day_return(fr, data.next_day_returns(split))

    raw = {"train": load("train", label_col)}
    raw_default, raw_recipe = {}, {}
    for w in EVAL_WINDOWS:
        raw_default[w] = load(w, "label")
        raw_recipe[w] = raw_default[w] if label_col == "label" else load(w, label_col)

    columns = list(base_cols)
    if spec["rank_inputs"]:
        raw["train"] = pp.add_cs_rank_inputs(raw["train"], CS_RANK_COLS)
        for w in EVAL_WINDOWS:
            raw_recipe[w] = pp.add_cs_rank_inputs(raw_recipe[w], CS_RANK_COLS)   # gate frames need no ranks
        columns += [f"{c}_csr" for c in CS_RANK_COLS]

    train_masked, n_glitch = mask_glitch_labels(raw["train"])
    if spec["preprocess"]:
        # Fit sees ONLY the train frames (rows <= FIT_CUTOFF; enforced by the guard).
        art = pp.fit_preprocessor(train_masked, columns, max_date=FIT_CUTOFF)
        kept = art["kept_columns"]
        logger.info("[%s] preprocessing fitted on %d train rows (max date %s): %d kept, dropped constant: %s",
                    name, art["fit_rows"], art["fit_max_date"], len(kept), art["dropped_columns"])
        apply = lambda fr: pp.apply_preprocessor(fr, art)          # noqa: E731
    else:
        art, kept = None, list(columns)
        logger.info("[%s] raw inputs (no preprocessing): %d columns", name, len(kept))
        apply = lambda fr: fr                                       # noqa: E731
    train_frames = apply(train_masked)
    val_loss_frames = apply(raw_recipe["val_2024"])   # natural recipe label (NaN rows skipped)
    eval_frames = {}
    for w in EVAL_WINDOWS:
        eval_frames[w] = apply(build_eval_frames(raw_default[w], raw_recipe[w]))
    return {"artifact": art, "kept_columns": kept, "n_rank_inputs": len(columns) - len(base_cols),
            "train_frames": train_frames, "val_loss_frames": val_loss_frames, "eval_frames": eval_frames,
            "n_glitch_masked": n_glitch}


def build_window_dataset(w: str, frames: dict, columns: list[str], opts: dict):
    """Eval dataset for window `w` + structural assertions (run BEFORE any training)."""
    from training.config import KNOWN_FUTURE_COLS, STATIC_COLS
    from training.dataset import TickerDayDataset
    from training.signal_data import filter_index_by_target_date, sample_meta
    sp, align = SPLITS[w], opts.get("align", DEFAULT_ALIGN)
    ds = TickerDayDataset(frames, columns, KNOWN_FUTURE_COLS, STATIC_COLS, ENCODER_LEN, align=align)
    n_all = len(ds)
    dropped = filter_index_by_target_date(ds, sp["min_target"])
    meta = sample_meta(ds)
    tdates = [d for _, d, _ in meta]
    assert tdates, f"{w}: no samples"
    # sample date == date of the LAST ENCODER ROW (today) / row after it (legacy), inside the window
    assert min(tdates) >= sp["min_target"] and max(tdates) <= sp["max_target"], (
        f"{w}: target dates {min(tdates)}..{max(tdates)} outside {sp['min_target']}..{sp['max_target']}")
    assert not any("2025-01-01" <= d <= "2025-12-31" for d in tdates), "2025 test-window target leaked in"
    logger.info("[%s] align=%s samples before target filter=%d dropped=%d kept=%d; target dates %s..%s",
                w, align, n_all, dropped, len(ds), min(tdates), max(tdates))
    if align == "legacy" and not opts.get("max_tickers"):
        # the legacy path must still reproduce the S3/E0 sample sets exactly.
        exp = opts.get("expected_samples", {}).get(w, sp["expected_samples"])
        assert len(ds) == exp, f"{w}: {len(ds)} samples, expected {exp} (S3/E0 window)"
        npz = os.path.join(opts.get("ref_dir", ARTIFACTS_DIR), sp["s3_npz"])
        if os.path.exists(npz):
            z = np.load(npz, allow_pickle=True)
            same = (list(z["tickers"]) == [t for t, _, _ in meta]) and (list(z["dates"]) == tdates)
            assert same, f"{w}: (ticker, date) sample list differs from S3's {npz}"
            logger.info("[%s] sample list identical to S3's (%d samples)", w, len(ds))
    return ds, meta


def sample_arrays(ds) -> tuple[np.ndarray, np.ndarray]:
    """(next_day_return, label_ok) per sample in ds.index order, read from the frames."""
    rets, ok = [], []
    for tk, t in ds.index:
        row = ds.ticker_dfs[tk].loc[t + ds.target_offset]
        rets.append(row["next_day_return"])
        ok.append(bool(row["label_ok"]))
    return np.asarray(rets, dtype=float), np.asarray(ok, dtype=bool)


def score_window(w: str, ds, meta, model, device) -> dict:
    from torch.utils.data import DataLoader
    from evaluation.evaluate import compute_metrics, compute_signal_metrics
    from training.run_signal_baseline import class_shares
    from training.train import predict_proba
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    y_true, probs = predict_proba(model, loader, device, progress_every=50, log=logger.info)
    tdates = [d for _, d, _ in meta]
    rets, ok = sample_arrays(ds)
    y_true = np.asarray(y_true)
    pred = probs.argmax(axis=1)
    sig = compute_signal_metrics(tdates, rets, probs, min_names_per_day=MIN_NAMES_PER_DAY)
    sig_clip = compute_signal_metrics(tdates, clip_returns(rets), probs, min_names_per_day=MIN_NAMES_PER_DAY)
    per_day = {}
    for d in tdates:
        per_day[d] = per_day.get(d, 0) + 1
    return {
        "n_samples": len(y_true), "n_label_ok": int(ok.sum()),
        "target_date_min": min(tdates), "target_date_max": max(tdates),
        "mean_names_per_day": float(np.mean(list(per_day.values()))),
        "n_ret_over_30pct": int((np.abs(rets[np.isfinite(rets)]) > RET_CLIP).sum()),
        "signal": sig, "signal_clip30": sig_clip,
        "metrics": compute_metrics(y_true[ok].tolist(), pred[ok].tolist()),
        "pred_share": class_shares(pred.tolist()),
    }


def run_recipe(name: str, data, opts: dict, budget) -> dict | None:
    """Returns the recipe's result record, or None if the budget stopped training mid-way (state is
    left resumable in the epoch checkpoint)."""
    import torch
    from torch.utils.data import DataLoader
    from training import preprocess as pp
    from training.config import KNOWN_FUTURE_COLS, STATIC_COLS, build_tft_config
    from training.dataset import TickerDayDataset
    from training.run_stage1_search import STATIC_CARDINALITIES
    from training.stage1_data import compute_class_weights, compute_label_distribution
    from training.train import TemporalFusionTransformer, run_training

    champion, hp = opts["champion"], opts["hparams"]
    device = opts["device"]
    torch.manual_seed(opts.get("seed", 0))
    np.random.seed(opts.get("seed", 0))
    logger.info("=== recipe %s (%s) ===", name, RECIPES[name]["tag"])
    prep = prepare_recipe(name, data, opts)
    cols = prep["kept_columns"]
    art_dir = opts["artifacts_dir"]
    preproc_path = None
    if prep["artifact"] is not None:
        preproc_path = os.path.join(art_dir, f"preproc_{name}.json")
        pp.save_preprocessor(prep["artifact"], preproc_path)

    # Structural assertions first (fail fast, before hours of training).
    eval_ds = {w: build_window_dataset(w, prep["eval_frames"][w], cols, opts) for w in EVAL_WINDOWS}

    align = opts.get("align", DEFAULT_ALIGN)
    train_ds = TickerDayDataset(prep["train_frames"], cols, KNOWN_FUTURE_COLS, STATIC_COLS, ENCODER_LEN, align=align)
    val_ds = TickerDayDataset(prep["val_loss_frames"], cols, KNOWN_FUTURE_COLS, STATIC_COLS, ENCODER_LEN, align=align)
    counts = compute_label_distribution(train_ds)
    if min(counts.values()) == 0:
        raise ValueError(f"[{name}] a train class is empty: {counts}")
    class_weights = compute_class_weights(counts)
    logger.info("[%s] inputs=%d train samples=%d (glitch-masked rows %d) val-loss samples=%d label counts=%s weights=%s",
                name, len(cols), len(train_ds), prep["n_glitch_masked"], len(val_ds), counts, class_weights.tolist())

    tft_config = build_tft_config(
        {"historical": cols, "future": KNOWN_FUTURE_COLS, "static_cardinalities": STATIC_CARDINALITIES},
        num_classes=3, state_size=hp["state_size"], attention_heads=hp["attention_heads"],
        lstm_layers=hp["lstm_layers"], dropout=hp["dropout"])
    fingerprint = full_fingerprint(name, opts, cols, class_weights.tolist())
    ckpt_dir = os.path.join(art_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    run_name = f"tfx-{name}"
    result = run_training({
        "tft_config": tft_config, "class_weights": class_weights, "device": device,
        "train_loader": DataLoader(train_ds, batch_size=opts["batch_size"], shuffle=True),
        "val_loader": DataLoader(val_ds, batch_size=opts["batch_size"], shuffle=False),
        "epochs": opts["epochs"], "lr": hp["lr"], "run_name": run_name, "checkpoint_dir": ckpt_dir,
        "epoch_checkpoint_path": os.path.join(ckpt_dir, f"{run_name}_inprogress.pt"),
        "early_stopping_patience": opts["patience"],
        "should_stop": budget.exceeded, "epoch_log": logger.info,
        "fingerprint": fingerprint, "seed": opts.get("seed", 0),
        "wandb_config": {**hp, "recipe": name, "label_col": RECIPES[name]["label_col"], "n_inputs": len(cols)},
    })
    if result["stopped_reason"] == "budget":
        logger.info("[%s] time budget reached after epoch %d; state saved, resumable.", name, result["last_epoch"])
        return None
    logger.info("[%s] training finished (%s, last epoch %d, best val loss %.4f); scoring.",
                name, result["stopped_reason"], result["last_epoch"], result["best_val_loss"])

    model = TemporalFusionTransformer(tft_config).to(device)
    model.load_state_dict(torch.load(result["checkpoint_path"], map_location=device))
    meta = {
        "seed": opts.get("seed", 0), "data_version": fingerprint["data_version"], "threshold": fingerprint["threshold"],
        "columns_hash": fingerprint["columns_hash"],
        "preproc": ({"path": preproc_path, "sha256": file_sha256(preproc_path)} if preproc_path else None),
        "resumed": result["resumed"], "resume_count": result["resume_count"], "resumed_from_epoch": result["start_epoch"],
        "stale_inprogress_ignored": result["stale_checkpoint"],
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
    }
    rec = {
        "fingerprint": fingerprint, "meta": meta,
        "recipe": name, "tag": RECIPES[name]["tag"], "label_col": RECIPES[name]["label_col"], "align": align,
        "hparams": hp, "batch_size": opts["batch_size"], "max_epochs": opts["epochs"], "patience": opts["patience"],
        "train": {
            "n_inputs": len(cols), "n_rank_inputs": prep["n_rank_inputs"], "kept_columns": cols,
            "dropped_columns": prep["artifact"]["dropped_columns"] if prep["artifact"] else [],
            "preprocessed": prep["artifact"] is not None, "align": align,
            "n_train_samples": len(train_ds), "n_glitch_masked": prep["n_glitch_masked"],
            "label_counts": {str(k): v for k, v in counts.items()}, "class_weights": class_weights.tolist(),
            "best_val_loss": result["best_val_loss"], "last_epoch": result["last_epoch"],
            "stopped_reason": result["stopped_reason"], "checkpoint": result["checkpoint_path"],
        },
        "code_commit": opts["commit"], "finished_at": time.strftime("%F %T"),
    }
    for w in EVAL_WINDOWS:
        ds, meta = eval_ds[w]
        rec[w] = score_window(w, ds, meta, model, device)
        logger.info("[%s] %s: IC=%s SE=%s F1=%s", name, w, rec[w]["signal"]["mean_daily_rank_ic"],
                    ic_se(rec[w]["signal"]), rec[w]["metrics"]["macro_f1"])
    del model
    return json.loads(json.dumps(rec))   # plain JSON types (e.g. per_class keys become str, like a reloaded file)


def record_model_version(name: str, rec: dict, path: str) -> None:
    """One docs/model_versions.md row per finished recipe (existing helper, idempotent)."""
    if model_versions_row_exists(path, f"tfx-{name}"):
        return
    from training import run_stage1_search as rss
    t, v, o = rec["train"], rec["val_2024"], rec["oot_2026"]
    dropped_txt = f"죽은 상수 컬럼 제거: {', '.join(t['dropped_columns'])}" if t["dropped_columns"] else "원본 그대로(전처리 없음)"
    rank_txt = f", 횡단면 랭크 입력 {t['n_rank_inputs']}개 추가" if t["n_rank_inputs"] else ""
    desc = (f"챔피언 33컬럼 기준 {t['n_inputs']}개 입력({dropped_txt}{rank_txt}), "
            f"{'train(≤2023) fit 클립+표준화, ' if t['preprocessed'] else ''}label={rec['label_col']}, align={rec['align']}")
    note = (f"TFX {rec['tag']}, epochs_run={t['last_epoch'] + 1}({t['stopped_reason']}), val_loss={t['best_val_loss']:.4f}, "
            f"val IC={_fmt(v['signal']['mean_daily_rank_ic'])}(선택 기준), OOT IC={_fmt(o['signal']['mean_daily_rank_ic'])}"
            f"(확인용, 선택 근거 아님), val F1는 자체 라벨 기준. 상세: docs/tfx_experiments.md")
    old = rss.MODEL_VERSIONS_PATH
    rss.MODEL_VERSIONS_PATH = path
    try:
        rss.append_model_version_row(version=f"tfx-{name}", stage="1단계-TFX", feature_desc=desc,
                                     hparams=rec["hparams"], macro_f1_val=v["metrics"]["macro_f1"], note=note)
    finally:
        rss.MODEL_VERSIONS_PATH = old


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipes", nargs="+", default=list(RECIPES), choices=list(RECIPES))
    p.add_argument("--max-minutes", type=float, default=600.0)
    p.add_argument("--artifacts-dir", default=None,
                   help=f"checkpoints + preproc artifacts + results JSON (default {ARTIFACTS_DIR}; REQUIRED with --max-tickers)")
    p.add_argument("--results-path", default=None, help="default: <artifacts-dir>/tfx_results.json")
    p.add_argument("--doc-path", default=DOC_PATH)
    p.add_argument("--model-versions-path", default=MODEL_VERSIONS_PATH)
    p.add_argument("--ref-dir", default=ARTIFACTS_DIR, help="where S3/E0 JSON (comparison rows) and S3 npz live")
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="default: cuda if available else cpu")
    p.add_argument("--no-wandb", action="store_true", help="WANDB_MODE=disabled (dry runs)")
    p.add_argument("--max-tickers", type=int, default=None, help="DRY RUN: subsample tickers (skips S3 count asserts)")
    p.add_argument("--state-size", type=int, default=None, help="DRY RUN override of champion hparam")
    p.add_argument("--align", choices=["today", "legacy"], default=DEFAULT_ALIGN,
                   help="dataset alignment. Recipes always use 'today'; 'legacy' exists to re-verify the "
                        "S3/E0 sample sets (36729 / 33364) via --check-windows")
    p.add_argument("--check-windows", action="store_true",
                   help="only build the val/OOT sample sets for the given recipes and assert counts (no training)")
    args = p.parse_args(argv)
    if args.max_tickers and args.artifacts_dir is None:
        p.error("--max-tickers (dry run) requires an explicit --artifacts-dir so the real preproc/checkpoint "
                "artifacts are not overwritten with subsampled ones")
    if args.artifacts_dir is None:
        args.artifacts_dir = ARTIFACTS_DIR
    return args


def main(argv=None, data=None) -> int:
    args = parse_args(argv)
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    import torch
    from training.run_stage1_search import Budget
    budget = Budget(args.max_minutes)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with open(CHAMPION_CONFIG_PATH) as f:
        champion = json.load(f)
    hp = {k: champion[k] for k in ("state_size", "attention_heads", "lstm_layers", "dropout", "lr")}
    if args.state_size:
        hp["state_size"] = args.state_size
    if data is None:
        dsn = os.environ.get("STOCK_DB_V2_DSN")
        if not dsn:
            raise ValueError("STOCK_DB_V2_DSN environment variable not set")
        data = RealData(dsn)
    os.makedirs(args.artifacts_dir, exist_ok=True)
    results_path = args.results_path or os.path.join(args.artifacts_dir, "tfx_results.json")
    opts = {"champion": champion, "hparams": hp, "device": device, "epochs": args.epochs, "patience": args.patience,
            "batch_size": args.batch_size, "artifacts_dir": args.artifacts_dir, "seed": args.seed,
            "max_tickers": args.max_tickers, "align": args.align, "ref_dir": args.ref_dir, "commit": _git_commit(),
            "expected_samples": getattr(data, "expected_samples", {})}   # test hook; real run uses SPLITS
    logger.info("device=%s recipes=%s max_minutes=%s results=%s hparams=%s", device, args.recipes,
                args.max_minutes, results_path, hp)

    if args.check_windows:
        for name in args.recipes:
            prep = prepare_recipe(name, data, opts)
            for w in EVAL_WINDOWS:
                ds, _ = build_window_dataset(w, prep["eval_frames"][w], prep["kept_columns"], opts)
                logger.info("[check-windows] %s %s: %d samples OK", name, w, len(ds))
        return 0

    results = load_results(results_path)
    refs_dir = args.ref_dir

    def write_outputs():
        with open(args.doc_path, "w") as f:
            f.write(render_doc(results, read_reference_rows(refs_dir), opts["commit"]))

    for name in args.recipes:
        if name in results:
            bad = fingerprint_mismatch(results[name].get("fingerprint"), cheap_fingerprint(name, opts))
            if not bad:
                logger.info("[%s] already recorded in %s -- skipping", name, results_path)
                record_model_version(name, results[name], args.model_versions_path)   # heal a missed row
                continue
            logger.warning("[%s] recorded result's fingerprint differs from the current setup (%s) -- "
                           "moving it to %s.stale.%s.json and re-running", name, ", ".join(bad), results_path, name)
            save_results_atomic(f"{results_path}.stale.{name}.json", {name: results.pop(name)})
            save_results_atomic(results_path, results)
        if budget.exceeded():
            logger.info("time budget exhausted; not starting recipe %s this invocation.", name)
            break
        rec = run_recipe(name, data, opts, budget)
        if rec is None:      # budget stop mid-training
            break
        results[name] = rec
        save_results_atomic(results_path, results)
        record_model_version(name, rec, args.model_versions_path)
        os.makedirs(os.path.dirname(args.doc_path) or ".", exist_ok=True)
        write_outputs()
        logger.info("[%s] recorded.", name)
    os.makedirs(os.path.dirname(args.doc_path) or ".", exist_ok=True)
    write_outputs()
    done = [r for r in RECIPES if r in results]
    logger.info("run_tfx_experiments invocation complete: %d/%d recipes recorded (%s).", len(done), len(RECIPES), done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
