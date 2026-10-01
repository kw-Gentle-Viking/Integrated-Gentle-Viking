#!/usr/bin/env python
"""E4: make the R0 TFT learn a TIMING signal instead of a per-ticker fixed ranking (docs/signal_diagnosis.md).

Diagnosis (V3 = R0 33 raw columns, align=today, weight_decay 1e-3, seed 0): the whole IC of ~0.04 comes from a
fixed per-ticker score (== a low-volatility factor); the timing part (score minus the ticker's mean score) has
a NEGATIVE IC (val -0.025, OOT -0.039). The fixed label threshold (+-1.238%) gives volatile tickers more
BUY/SELL labels, so "how volatile is this ticker" is the easiest thing to learn. E4 keeps the TFT and changes
what it is asked to learn / allowed to see. Every recipe is R0-based (raw 33 champion columns, no
preprocessing), align="today", Adam weight_decay 1e-3 (V3), champion hparams, seed 0, batch 128, max 12
epochs, patience 4, balanced class weights from the recipe's own train label distribution:

  x1_cslabel           label = per-DATE cross-sectional quantile of next_day_return (top 30% BUY=0, bottom 30%
                       SELL=2, rest HOLD=1; days with < MIN_NAMES_PER_DAY names -> NaN). Replaces the `label`
                       column of the train / val-loss / eval frames.
  x2_cslabel_nostatic  x1 + sector_id / market_id set to the constant 0 (model structure and embedding
                       cardinalities unchanged; only the inputs are constant).
  x3_vn_nostatic       the DB volatility-normalised label (label_vn) + constant static ids.
  x4_cslabel_vnfeat    x1 + the volatility-type input columns (VOL_RANK_COLS) replaced, in place, by their
                       per-date cross-sectional percentile rank in (0, 1] (same-date information only).

Selection (binding): the checkpoint / early stopping criterion is the val-2024 TIMING IC (mean daily rank IC of
score - ticker mean; the ticker mean is the val window's own mean score, i.e. no return information), computed
every epoch from val predictions (epoch_metric_fn / select_metric_key hooks of training.train.run_training).
The 2026 OOT window is scored exactly once per recipe, on the final selected checkpoint (its fixed effect =
the val-window ticker means of that same checkpoint), is confirmation only and is never a selection criterion.
The 2025 test window is never scored.

Data preparation, windows, fingerprint, meta, per-epoch shuffle seed, glitch masking, scoring and results-JSON
idempotence come from training/run_tfx_experiments.py (unchanged); epoch resume and the time budget from
training/train.py / run_stage1_search.Budget. Resumable at epoch granularity; a recipe is recorded in
e4_results.json only when finished; `--max-minutes` stops between epochs (exit 0, state resumable).

Collapse diagnosis (x2/x3 finished first): both x2_cslabel_nostatic and x3_vn_nostatic show train_loss frozen
at ln(3)~=1.0986 (3-class random-guess loss) after epoch 1, with val predictions identical to the decimal in
every later epoch -- a degenerate "predict the class prior" fit, not real convergence. x1/x4 (static ids kept)
train_loss keeps falling every epoch under the same hparams. weight_decay=1e-3 (Adam L2, inherited from V3) is
one candidate cause once the static path is removed (a weak-signal model pulled to 0 fast); seed is the other.
Six reruns test this, reusing x2/x3's label/static/vol_rank spec unchanged (only seed or weight_decay differ):
  x2_nostatic_nowd / x3_vn_nostatic_nowd    x2/x3, weight_decay=0 (seed 0)
  x2_nostatic_seed1 / x3_vn_nostatic_seed1  x2/x3, seed=1 (weight_decay 1e-3)
  x2_nostatic_seed2 / x3_vn_nostatic_seed2  x2/x3, seed=2 (weight_decay 1e-3)
Run order (weight_decay hypothesis checked first): x2_nostatic_nowd -> x3_vn_nostatic_nowd -> x2_nostatic_seed1
-> x3_vn_nostatic_seed1 -> x2_nostatic_seed2 -> x3_vn_nostatic_seed2 (see DEFAULT_ORDER). Each is a fresh
fingerprint (different seed/weight_decay in the recorded fingerprint's "seed"/"hparams" keys), so it cannot be
skip-matched against x2/x3's already-recorded result, and vice versa.

V3-structure weight_decay control (v3_structure_nowd, run last): the six collapse-diagnosis reruns only ever
tested weight_decay=0 together with static ids REMOVED (x2_nostatic_nowd / x3_vn_nostatic_nowd). V3's own
structure (static ids kept, fixed label) was never retried with just weight_decay=0. v3_structure_nowd =
V3 (e2e3 `v3_wd`: label_source="label" i.e. the same fixed-threshold `label` column, static_const=False,
vol_rank=False, same champion hparams) with weight_decay=0 instead of 1e-3 -- the single-variable control that
isolates the weight_decay fix from the static-id change.

v3_structure_nowd follow-ups (run last, after v3_structure_nowd): it is the most balanced E4 result so far -- val
timing IC +0.048, OOT timing IC +0.018, BOTH positive and same-signed (every earlier recipe either had val/OOT
disagree or was never retried with static ids kept + weight_decay=0). best_epoch=3, last_epoch=7 (patience=4
early-stopped it). Eight recipes each change exactly ONE variable off v3_structure_nowd, everything else
(static_const=False, champion hparams otherwise) held fixed:
  v3_structure_nowd_seed1/_seed2      seed=1 / seed=2 -- is the result seed-independent?
  v3_structure_nowd_vn                label_source="label_vn" -- label_vn combined with static-ids-kept + wd=0,
                                       for the first time.
  v3_structure_nowd_cs                label_source="cs" -- the cross-sectional quantile label combined with
                                       static-ids-kept + wd=0, for the first time.
  v3_structure_wd1e4 / _wd3e4         weight_decay=1e-4 / 3e-4 instead of 0 -- a small Adam L2 between 0 and V3's
                                       1e-3.
  v3_structure_nowd_patience8         patience=8 instead of 4 -- v3_structure_nowd stopped at last_epoch=7 under
                                       patience=4; does it keep improving with more room?
  v3_structure_nowd_dropout30         dropout=0.30 instead of the champion default (~0.1658) -- more regularization.
Run order (seed check first -- cheapest confirmation it is not a fluke -- then the never-tried label
combinations, then fine-tuning): seed1 -> seed2 -> vn -> cs -> wd1e4 -> wd3e4 -> patience8 -> dropout30 (see
DEFAULT_ORDER). `patience` and `dropout` are new optional per-recipe overrides read by `recipe_opts`, the same
pattern as the existing `weight_decay`/`seed` overrides; a recipe that does not set them gets the module defaults
(PATIENCE=4, champion dropout) unchanged, so the original 11 recipes' fingerprints and behavior are unaffected.

    set -a && source .env && set +a
    PYTHONPATH=. python training/run_e4_experiments.py [--recipes x1_cslabel x2_cslabel_nostatic] [--max-minutes 600]

Output paths are overridable (--artifacts-dir / --results-path / --doc-path / --model-versions-path) so dry
runs never touch the real ones.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import training.run_tfx_experiments as tfx
from training.signal_diagnostics import flat_summary, timing_decomposition

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = tfx.ARTIFACTS_DIR
DOC_PATH = "docs/e4_experiments.md"
RESULTS_NAME = "e4_results.json"
BASE_RECIPE = "aligned"                     # R0: raw 33 columns, fixed-label gate for the evaluation sample sets
MAX_EPOCHS = 12
PATIENCE = 4
SEED = 0
WEIGHT_DECAY = 1e-3                         # V3 (Adam L2, not AdamW)
SELECT_KEY = "val_timing_ic"
CS_TOP, CS_BOTTOM = 0.30, 0.30              # cross-sectional label quantiles (BUY / SELL share per day)
E4_VERSION = 1                              # bump when label / transform semantics change (part of the fingerprint)
# Volatility-type inputs among the 33 champion columns, replaced by their per-date cross-sectional percentile
# rank in x4. volatility_20d = 20-day std of the ticker's log return (per-ticker volatility level);
# sector_volatility = (high-low)/close of the ticker's sector index (identifies the sector's daily range).
# NOT included on purpose: vix_chg / kospi_ret ... (market-wide, identical for all tickers on a date, a
# cross-sectional rank would be constant and would remove the market-timing feature), is_vi_triggered /
# vi_count_recent5d (sparse event flags/counts, not a volatility level), log_ret / disparity_* / rsi_14
# (returns/momentum, not volatility).
VOL_RANK_COLS = ["volatility_20d", "sector_volatility"]

# V3 seed-0 diagnosis numbers (docs/signal_diagnosis.md section 3; fixed / timing IC are not in any results
# JSON, the V3 raw IC / IR / quantile L/S are read from the results JSON).
V3_DIAGNOSIS = {
    "val_2024": {"raw": 0.0412, "fixed": 0.0409, "timing": -0.0253},
    "oot_2026": {"raw": 0.0392, "fixed": 0.0392, "timing": -0.0393, "timing_se": 0.018},
}

RECIPES = {
    "x1_cslabel": dict(tag="X1", label_source="cs", static_const=False, vol_rank=False,
                       desc="per-date cross-sectional quantile label (top 30% BUY / bottom 30% SELL / rest HOLD)"),
    "x2_cslabel_nostatic": dict(tag="X2", label_source="cs", static_const=True, vol_rank=False,
                                desc="X1 + sector_id / market_id constant 0"),
    "x3_vn_nostatic": dict(tag="X3", label_source="label_vn", static_const=True, vol_rank=False,
                           desc="volatility-normalised label (label_vn) + constant static ids"),
    "x4_cslabel_vnfeat": dict(tag="X4", label_source="cs", static_const=False, vol_rank=True,
                              desc="X1 + volatility_20d / sector_volatility as per-date cross-sectional percentile ranks"),
    # Collapse diagnosis (x2/x3's train_loss freezes at ln(3) after epoch 1; see the module docstring). Same
    # label/static/vol_rank spec as x2/x3 -- only `weight_decay` or `seed` differ from the WEIGHT_DECAY/SEED
    # module defaults, via the optional per-recipe overrides `recipe_opts` reads.
    "x2_nostatic_nowd": dict(tag="X2wd0", label_source="cs", static_const=True, vol_rank=False, weight_decay=0.0,
                             desc="X2 (cs label, static ids const 0) + weight_decay=0 -- does removing Adam L2 "
                                  "stop the 1-epoch train_loss freeze at ln(3)?"),
    "x3_vn_nostatic_nowd": dict(tag="X3wd0", label_source="label_vn", static_const=True, vol_rank=False, weight_decay=0.0,
                                desc="X3 (label_vn, static ids const 0) + weight_decay=0 -- does removing Adam L2 "
                                     "stop the 1-epoch train_loss freeze at ln(3)?"),
    "x2_nostatic_seed1": dict(tag="X2s1", label_source="cs", static_const=True, vol_rank=False, seed=1,
                              desc="X2 (cs label, static ids const 0), seed=1 -- is the 1-epoch freeze seed-independent?"),
    "x3_vn_nostatic_seed1": dict(tag="X3s1", label_source="label_vn", static_const=True, vol_rank=False, seed=1,
                                 desc="X3 (label_vn, static ids const 0), seed=1 -- is the 1-epoch freeze seed-independent?"),
    "x2_nostatic_seed2": dict(tag="X2s2", label_source="cs", static_const=True, vol_rank=False, seed=2,
                              desc="X2 (cs label, static ids const 0), seed=2 -- is the 1-epoch freeze seed-independent?"),
    "x3_vn_nostatic_seed2": dict(tag="X3s2", label_source="label_vn", static_const=True, vol_rank=False, seed=2,
                                 desc="X3 (label_vn, static ids const 0), seed=2 -- is the 1-epoch freeze seed-independent?"),
    # V3-structure control: the collapse diagnosis only ever tested weight_decay=0 together with static ids
    # removed (x2_nostatic_nowd / x3_vn_nostatic_nowd). This recipe isolates the weight_decay fix alone: same
    # label, same static ids, same every other hparam as V3 (e2e3 `v3_wd`: BASE_RECIPE "aligned" = R0, label_col
    # "label", static ids untouched) -- only weight_decay is 0 instead of 1e-3. label_source="label" reuses
    # prepare_e4's "not cs, not label_vn" branch, which loads data.frames(split, "label") unchanged and skips
    # cs_label_frames entirely, i.e. the exact same fixed-threshold label V3 trains/evaluates on (see
    # training/test_run_e4_experiments.py::test_prepare_v3_structure_nowd_label_matches_v3_exactly, which asserts
    # byte-identical frames against tfx.prepare_recipe("aligned", ...)).
    "v3_structure_nowd": dict(tag="V3wd0", label_source="label", static_const=False, vol_rank=False, weight_decay=0.0,
                              desc="V3(e2e3 v3_wd)와 라벨/정적변수/하이퍼파라미터 전부 동일, weight_decay만 0 -- "
                                   "does the wd=0 fix help timing IC with static ids (and the V3 label) left alone?"),
    # v3_structure_nowd follow-ups: it is the most balanced E4 result so far (val timing IC +0.048, OOT +0.018,
    # both positive and same-signed -- every other recipe either had val/OOT disagree or was never retried with
    # static ids kept + weight_decay=0). These 8 each change exactly ONE variable off v3_structure_nowd (seed,
    # label_source, weight_decay, patience, dropout); every other setting (static_const=False, champion hparams)
    # is identical to v3_structure_nowd. patience/dropout use the recipe_opts `patience`/`dropout` overrides
    # (same optional-override pattern as weight_decay/seed; unset -> the module defaults PATIENCE / champion
    # dropout, so the original 11 recipes are byte-identical in fingerprint and behavior).
    "v3_structure_nowd_seed1": dict(tag="V3wd0s1", label_source="label", static_const=False, vol_rank=False,
                                    weight_decay=0.0, seed=1,
                                    desc="v3_structure_nowd, seed=1 -- is the balanced val/OOT timing-IC result seed-independent?"),
    "v3_structure_nowd_seed2": dict(tag="V3wd0s2", label_source="label", static_const=False, vol_rank=False,
                                    weight_decay=0.0, seed=2,
                                    desc="v3_structure_nowd, seed=2 -- is the balanced val/OOT timing-IC result seed-independent?"),
    "v3_structure_nowd_vn": dict(tag="V3wd0vn", label_source="label_vn", static_const=False, vol_rank=False,
                                 weight_decay=0.0,
                                 desc="v3_structure_nowd + label_source=label_vn -- first test of label_vn together with "
                                      "static ids kept and weight_decay=0"),
    "v3_structure_nowd_cs": dict(tag="V3wd0cs", label_source="cs", static_const=False, vol_rank=False,
                                 weight_decay=0.0,
                                 desc="v3_structure_nowd + label_source=cs (cross-sectional quantile) -- first test of the "
                                      "cs label together with static ids kept and weight_decay=0"),
    "v3_structure_wd1e4": dict(tag="V3wd1e4", label_source="label", static_const=False, vol_rank=False,
                               weight_decay=0.0001,
                               desc="v3_structure_nowd with weight_decay=1e-4 instead of 0 -- a small Adam L2 between 0 "
                                    "and V3's 1e-3"),
    "v3_structure_wd3e4": dict(tag="V3wd3e4", label_source="label", static_const=False, vol_rank=False,
                               weight_decay=0.0003,
                               desc="v3_structure_nowd with weight_decay=3e-4 instead of 0"),
    "v3_structure_nowd_patience8": dict(tag="V3wd0p8", label_source="label", static_const=False, vol_rank=False,
                                        weight_decay=0.0, patience=8,
                                        desc="v3_structure_nowd with patience=8 instead of 4 -- best_epoch=3/last_epoch=7 "
                                             "stopped early under patience=4; does it still improve given more room?"),
    "v3_structure_nowd_dropout30": dict(tag="V3wd0d30", label_source="label", static_const=False, vol_rank=False,
                                        weight_decay=0.0, dropout=0.30,
                                        desc="v3_structure_nowd with dropout=0.30 instead of the champion default "
                                             "(~0.1658) -- more regularization"),
}
DEFAULT_ORDER = ["x1_cslabel", "x2_cslabel_nostatic", "x3_vn_nostatic", "x4_cslabel_vnfeat",
                  # weight_decay hypothesis checked first (cheaper to falsify), then the seed reruns.
                  "x2_nostatic_nowd", "x3_vn_nostatic_nowd",
                  "x2_nostatic_seed1", "x3_vn_nostatic_seed1", "x2_nostatic_seed2", "x3_vn_nostatic_seed2",
                  "v3_structure_nowd",
                  # v3_structure_nowd follow-ups: seeds first (cheapest confirmation it's not a fluke), then the
                  # label combinations never tried with this structure, then fine-tuning (weight_decay, patience,
                  # dropout).
                  "v3_structure_nowd_seed1", "v3_structure_nowd_seed2",
                  "v3_structure_nowd_vn", "v3_structure_nowd_cs",
                  "v3_structure_wd1e4", "v3_structure_wd3e4",
                  "v3_structure_nowd_patience8", "v3_structure_nowd_dropout30"]


# --------------------------------------------------------------------------------------------
# pure helpers (unit-tested in training/test_run_e4_experiments.py)
# --------------------------------------------------------------------------------------------
def recipe_opts(name: str, base: dict) -> dict:
    """Per-recipe opts: champion hparams + weight_decay 1e-3, seed 0, patience 4, val-timing-IC selection --
    unless the recipe's own RECIPES entry overrides `weight_decay`, `seed`, `patience` and/or `dropout` (the
    collapse-diagnosis reruns of x2/x3 override weight_decay/seed; the v3_structure_nowd follow-ups also use the
    patience/dropout overrides). A recipe with none of these keys gets the exact module defaults, unchanged.
    `--state-size` (dry run) always wins over the champion state size (and over any `dropout` override, a
    different hp key)."""
    spec = RECIPES[name]
    hp = {**base["hparams"], "weight_decay": spec.get("weight_decay", WEIGHT_DECAY)}
    if "dropout" in spec:
        hp["dropout"] = spec["dropout"]
    if base.get("state_size_override"):
        hp["state_size"] = base["state_size_override"]
    seed = spec.get("seed", SEED)
    patience = spec.get("patience", PATIENCE)
    return {**base, "hparams": hp, "seed": seed, "patience": patience, "select": SELECT_KEY, "align": "today"}


def label_col_of(name: str) -> str:
    src = RECIPES[name]["label_source"]
    return "cs_quantile" if src == "cs" else src


def cheap_fingerprint(name: str, opts: dict) -> dict:
    """tfx cheap fingerprint of R0 with this recipe's hparams, plus the E4-specific switches and the selection
    rule. The threshold entry covers the fixed label (evaluation sample-set gate) and, for label_vn recipes, the
    label_vn threshold file as well."""
    spec = RECIPES[name]
    fp = tfx.cheap_fingerprint(BASE_RECIPE, opts)
    fp["label_col"] = label_col_of(name)
    if spec["label_source"] == "label_vn":
        fp["threshold"] = {"gate": fp["threshold"], "label_vn": tfx.read_threshold("label_vn", opts.get("threshold_dir", tfx.TRAINING_DIR))}
    fp["selection"] = {"mode": opts["select"], "patience": opts["patience"]}
    fp["e4"] = {"version": E4_VERSION, "label_source": spec["label_source"], "static_const": spec["static_const"],
                "vol_rank_cols": list(VOL_RANK_COLS) if spec["vol_rank"] else [],
                "cs_top": CS_TOP, "cs_bottom": CS_BOTTOM, "min_names": tfx.MIN_NAMES_PER_DAY,
                "glitch_mask": tfx.GLITCH_MASK_THRESHOLD}
    return json.loads(json.dumps(fp, sort_keys=True))


def full_fingerprint(name: str, opts: dict, cols, class_weights) -> dict:
    fp = cheap_fingerprint(name, opts)
    fp["columns_hash"] = tfx.columns_hash(cols)
    fp["class_weights"] = [round(float(w), 6) for w in class_weights]
    return fp


def _flatten(frames: dict, cols):
    """(keys, per-ticker lengths, flat DataFrame of `cols`) of the ticker dict, in dict order."""
    keys = list(frames)
    flat = pd.concat([frames[k][cols] for k in keys], ignore_index=True) if keys else pd.DataFrame(columns=cols)
    return keys, [len(frames[k]) for k in keys], flat


def cs_label_frames(frames: dict, top: float = CS_TOP, bottom: float = CS_BOTTOM, min_names: int | None = None) -> dict:
    """NEW frames whose `label` column is replaced by the per-date cross-sectional quantile label of
    `next_day_return`: BUY(0) = top `top` share of that day's names, SELL(2) = bottom `bottom` share, HOLD(1) the
    rest. Eligible rows = base `label` notna AND finite `next_day_return` (so glitch rows masked to NaN by
    tfx.mask_glitch_labels, or rows without a fixed label, are excluded BEFORE the ranks are formed and get NaN);
    days with fewer than `min_names` eligible names (default tfx.MIN_NAMES_PER_DAY) get NaN for every row.
    Ranks are average ranks (ties share one label; the result never depends on ticker order); with midrank
    percentile p = (rank - 0.5) / n: BUY if p >= 1 - top, SELL if p <= bottom. Uses only same-date returns."""
    min_names = tfx.MIN_NAMES_PER_DAY if min_names is None else min_names
    keys, lens, flat = _flatten(frames, ["trade_date", "next_day_return", "label"])
    ret = flat["next_day_return"].to_numpy(dtype=float)
    elig = flat["label"].notna().to_numpy() & np.isfinite(ret)
    tmp = pd.DataFrame({"d": flat["trade_date"].to_numpy(), "r": np.where(elig, ret, np.nan)})
    g = tmp.groupby("d", sort=False)["r"]
    n = g.transform("count").to_numpy(dtype=float)
    rank = g.rank(method="average").to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = (rank - 0.5) / n
    lab = np.where(pct >= 1.0 - top - 1e-9, 0.0, np.where(pct <= bottom + 1e-9, 2.0, 1.0))
    lab[~elig | (n < min_names) | ~np.isfinite(pct)] = np.nan
    out, pos = {}, 0
    for k, m in zip(keys, lens):
        df = frames[k].copy()
        df["label"] = lab[pos:pos + m]
        pos += m
        out[k] = df
    return out


def constant_static_frames(frames: dict, static_cols=None) -> dict:
    """NEW frames with every static categorical column set to the constant 0."""
    from training.config import STATIC_COLS
    cols = list(static_cols or STATIC_COLS)
    out = {}
    for k, df in frames.items():
        df = df.copy()
        for c in cols:
            df[c] = 0
        out[k] = df
    return out


def vol_rank_frames(frames: dict, cols) -> dict:
    """NEW frames where each of `cols` is replaced (same column name) by its per-date cross-sectional percentile
    rank in (0, 1] (average ranks, NaN kept NaN) over the tickers of `frames`. Same-date information only."""
    from training.tabular_features import add_cs_ranks
    cols = list(cols)
    keys, lens, flat = _flatten(frames, ["trade_date"] + cols)
    flat = add_cs_ranks(flat, cols, suffix="_csr")
    out, pos = {}, 0
    for k, m in zip(keys, lens):
        df = frames[k].copy()
        for c in cols:
            df[c] = flat[f"{c}_csr"].to_numpy()[pos:pos + m]
        pos += m
        out[k] = df
    return out


def _transform_inputs(frames: dict, spec: dict) -> dict:
    if spec["static_const"]:
        frames = constant_static_frames(frames)
    if spec["vol_rank"]:
        frames = vol_rank_frames(frames, VOL_RANK_COLS)
    return frames


def prepare_e4(name: str, data, opts: dict) -> dict:
    """Mirror of tfx.prepare_recipe for an unpreprocessed R0 base, with the E4 label / input changes. Returns
    train_frames / val_loss_frames / eval_frames (val, OOT; sample sets = the fixed-label gate, `label_ok` marks
    samples with a genuine recipe label), kept_columns and n_glitch_masked. No model work here."""
    spec = RECIPES[name]
    src = spec["label_source"]
    base_cols = list(opts["champion"]["columns"])
    max_t = opts.get("max_tickers")
    min_names = tfx.MIN_NAMES_PER_DAY

    def load(split, lab):
        fr = tfx.subsample_tickers(data.frames(split, lab), max_t)
        return tfx.attach_next_day_return(fr, data.next_day_returns(split))

    raw_train = load("train", "label_vn" if src == "label_vn" else "label")
    raw_default, raw_recipe = {}, {}
    for w in tfx.EVAL_WINDOWS:
        raw_default[w] = load(w, "label")
        raw_recipe[w] = load(w, "label_vn") if src == "label_vn" else raw_default[w]

    train_masked, n_glitch = tfx.mask_glitch_labels(raw_train)          # glitch rows: label NaN BEFORE any ranking
    if src == "cs":
        train_masked = cs_label_frames(train_masked, min_names=min_names)
        raw_recipe = {w: cs_label_frames(raw_default[w], min_names=min_names) for w in tfx.EVAL_WINDOWS}
    train_frames = _transform_inputs(train_masked, spec)
    val_loss_frames = _transform_inputs(raw_recipe["val_2024"], spec)
    eval_frames = {w: _transform_inputs(tfx.build_eval_frames(raw_default[w], raw_recipe[w]), spec)
                   for w in tfx.EVAL_WINDOWS}
    logger.info("[%s] label=%s static_const=%s vol_rank=%s: %d input columns, glitch-masked train rows %d",
                name, src, spec["static_const"], spec["vol_rank"], len(base_cols), n_glitch)
    return {"kept_columns": base_cols, "train_frames": train_frames, "val_loss_frames": val_loss_frames,
            "eval_frames": eval_frames, "n_glitch_masked": n_glitch}


def _default_score_fn(model, ds, device) -> np.ndarray:
    """score = p_buy - p_sell for every sample of `ds`, in ds.index order."""
    from torch.utils.data import DataLoader
    from training.train import predict_proba
    _, probs = predict_proba(model, DataLoader(ds, batch_size=256, shuffle=False), device)
    return probs[:, 0].astype(float) - probs[:, 2].astype(float)


def make_timing_fn(ds, meta, rets, device, score_fn=None):
    """epoch_metric_fn for run_training: val-window predictions -> timing_decomposition with the window's own
    ticker means. Returns plain python floats / None only (epoch checkpoints are torch.load(weights_only) safe).
    RNG-free. Only ever built for the val window."""
    score_fn = score_fn or _default_score_fn
    tdates = [d for _, d, _ in meta]
    tickers = [t for t, _, _ in meta]
    f = lambda x: None if x is None else float(x)          # noqa: E731

    def fn(model, epoch):
        dec = timing_decomposition(tdates, tickers, score_fn(model, ds, device), rets, None, tfx.MIN_NAMES_PER_DAY)
        return {"val_timing_ic": f(dec["timing_ic"]), "val_ic": f(dec["raw_ic"]), "val_fixed_ic": f(dec["fixed_effect_ic"]),
                "val_timing_ic_se": f(dec["timing_ic_se"]), "val_ic_se": f(dec["raw_ic_se"]),
                "val_timing_ic_ir": f(dec["timing_ic_ir"])}
    return fn


def _fmt(x, nd=4, pct=False):
    return tfx._fmt(x, nd, pct)


def _signed(x, nd=4):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def _curve_line(rec: dict) -> str:
    be = rec.get("best_epoch")
    parts = []
    for h in rec.get("epoch_curve", []):
        mark = "*" if h["epoch"] == be else ""
        parts.append(f"{h['epoch']}{mark}: timing {_signed(h.get('val_timing_ic'))} / raw {_signed(h.get('val_ic'))}"
                     f" / vloss {_fmt(h['val_loss'])} / tloss {_fmt(h.get('train_loss'))}")
    return "; ".join(parts) if parts else "n/a"


LN3 = 1.0986  # 3-class random-guess cross-entropy loss; a train_loss stuck here past epoch 1 is a degenerate
              # "predict the class prior" fit, not real convergence (see module docstring / x2 & x3 in E4).


def _min_train_loss(rec: dict):
    """min(train_loss) over `rec`'s epoch_curve, or None (curve is train.run_training's epoch_history, which
    always carries train_loss -- None only for a pending/empty curve)."""
    vals = [h["train_loss"] for h in rec.get("epoch_curve", []) if h.get("train_loss") is not None]
    return min(vals) if vals else None


def _v3_record(refs: dict):
    return (refs or {}).get("v3_wd")


def render_doc(results: dict, refs: dict, code_commit: str, recipes=None) -> str:
    """docs/e4_experiments.md. `refs` = {"v3_wd": V3 seed-0 record} read from e2e3_results.json (or tfx)."""
    recipes = recipes or DEFAULT_ORDER
    v3 = _v3_record(refs)
    L = [
        "# E4: timing-signal experiments (TFT, R0 base)", "",
        "Background: `docs/signal_diagnosis.md`. Code commit: "
        f"`{code_commit}` (`training/run_e4_experiments.py`, `training/signal_diagnostics.py`, reuses "
        "`training/run_tfx_experiments.py`). Raw output: `training/artifacts/e4_results.json` (gitignored). "
        "Baseline row = V3 seed 0 (`v3_wd` of `training/artifacts/e2e3_results.json`), not retrained.", "",
        "**Goal.** V3's IC of ~0.04 is entirely a fixed per-ticker ranking (identical to a low-volatility factor); "
        "its timing IC (score minus the ticker's mean score) is negative: val -0.0253, OOT -0.0393. The fixed label "
        "threshold (+-1.238%) makes volatile tickers BUY/SELL-heavy, so ticker volatility is the easiest thing to "
        "learn. E4 keeps the TFT and changes the label / inputs so that the only thing left to learn is timing. "
        "Success criterion: timing IC > 0 (val AND OOT), not raw IC.", "",
        "**Selection discipline.** Epoch selection and early stopping use the val 2024 TIMING IC only. The 2026 OOT "
        "window is scored exactly ONCE per recipe, on its final selected checkpoint, for confirmation; it is never "
        "used to select an epoch, a recipe or a hyper-parameter (never scored per epoch). The 2025 test window is "
        "never scored.", "",
        "## Protocol", "",
        "- All recipes: R0 = raw 33 champion columns (no preprocessing), `align=\"today\"`, adjusted prices, champion "
        f"architecture/hparams (`training/champion_config.json`), Adam `weight_decay` {WEIGHT_DECAY:g} (L2, as V3), seed {SEED}, "
        f"batch {tfx.BATCH_SIZE}, max {MAX_EPOCHS} epochs, patience {PATIENCE}, class weights = balanced from each "
        "recipe's own train label distribution (after glitch masking). Same val/OOT sample sets as V3 (the fixed-label "
        "gate); samples the recipe label cannot label are kept for the IC and marked out of the F1.",
        "- **Timing IC.** score = p_buy - p_sell. fixed effect = the ticker's mean score (val window: its own mean over the "
        "window; OOT: the val-window means of the SAME final checkpoint - scores only, never returns). timing = score - "
        "fixed effect. All three ICs (raw, fixed-effect, timing) use the `evaluation.evaluate` daily Spearman rule "
        f"(>= {tfx.MIN_NAMES_PER_DAY} names per day, constant days skipped; SE = std / sqrt(days)). Per-epoch val curve: "
        "timing / raw IC / val loss.",
        f"- **Cross-sectional label (X1, X2, X4).** Per trading day, over the names with a valid next_day_return: top "
        f"{CS_TOP:.0%} = BUY(0), bottom {CS_BOTTOM:.0%} = SELL(2), rest HOLD(1). Average-rank percentile p = (rank - 0.5)/n; "
        "BUY if p >= 0.7, SELL if p <= 0.3 (tied returns share one label; no dependence on ticker order). Days with fewer "
        f"than {tfx.MIN_NAMES_PER_DAY} eligible names -> NaN. Glitch rows (|next_day_return| > {tfx.GLITCH_MASK_THRESHOLD}) are "
        "masked out of the TRAIN frames BEFORE the ranks are formed. The label replaces the `label` column of the train / val-loss / "
        "eval frames, so `TickerDayDataset` is unchanged. Macro F1 is against each recipe's own label: X1/X2/X4 (cs label) "
        "and X3 (label_vn) F1 values, and V3's fixed-label F1, are NOT comparable with each other.",
        "- **Constant static ids (X2, X3).** `sector_id` and `market_id` are set to 0 in every frame; the model still has its "
        "two static embeddings (cardinalities unchanged), they just always receive id 0.",
        "- **V3wd0 (`v3_structure_nowd`).** `label_source=\"label\"`: the SAME fixed-threshold label as V3 (`label_col_of` "
        "falls through to the raw column name; `prepare_e4` skips `cs_label_frames` and loads `data.frames(split, \"label\")` "
        "unchanged, same as `tfx.prepare_recipe(\"aligned\", ...)`). Static ids and vol inputs are left alone (`static_const=False`, "
        "`vol_rank=False`, matching X1/X4/V3), every other hparam matches V3 (e2e3 `v3_wd`) -- only `weight_decay` is 0 instead "
        "of 1e-3. Isolates the weight_decay fix from the static-id removal that the collapse-diagnosis reruns (X2wd0/X3wd0) "
        "always combined it with; its F1 IS comparable to V3's (same label).",
        "- **Volatility inputs as cross-sectional ranks (X4).** In every frame (train / val / OOT) these columns of the 33 are "
        "replaced, under the same column name, by their per-date percentile rank in (0, 1] over the tickers of that date "
        "(average ranks; same-date information only): " + ", ".join(f"`{c}`" for c in VOL_RANK_COLS) + ". `volatility_20d` is "
        "the ticker's 20-day std of log returns; `sector_volatility` = (high - low) / close of the ticker's sector index "
        "(it identifies the sector's range). Deliberately NOT ranked: `vix_chg` and the other market-wide columns "
        "(identical for all tickers on a date, a rank would be constant), `is_vi_triggered` / `vi_count_recent5d` (sparse "
        "event flags), `log_ret` / `disparity_*` / `rsi_14` (returns / momentum).", "",
        "## Recipes", "",
        f"`min train_loss` = the lowest train_loss over the run's epoch_curve; stuck at ln(3)~={LN3:.4f} (3-class "
        "random-guess cross-entropy) past epoch 1 flags a degenerate class-prior fit rather than real learning "
        "(see the module docstring; this is what x2_cslabel_nostatic / x3_vn_nostatic showed).", "",
        "| recipe | tag | label | static ids | vol inputs | description | epochs run / stop | best epoch | best val timing IC | min train_loss | class weights (buy/hold/sell) | train samples (glitch-masked rows) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in recipes:
        sp, r = RECIPES[n], results.get(n)
        lab = {"cs": "cs quantile 30/30", "label_vn": "label_vn", "label": "fixed label (V3)"}[sp["label_source"]]
        st = "const 0" if sp["static_const"] else "as is"
        vi = "cs rank" if sp["vol_rank"] else "raw"
        if r is None:
            L.append(f"| {n} | {sp['tag']} | {lab} | {st} | {vi} | {sp['desc']} | _pending_ | | | | | |")
            continue
        t = r["train"]
        L.append(f"| {n} | {sp['tag']} | {lab} | {st} | {vi} | {sp['desc']} | {t['last_epoch'] + 1} / {t['stopped_reason']} | "
                 f"{_fmt(r.get('best_epoch'), 0)} | {_signed(r.get('best_val_timing_ic'))} | {_fmt(_min_train_loss(r))} | "
                 f"{' / '.join(f'{w:.3f}' for w in t['class_weights'])} | {t['n_train_samples']} ({t['n_glitch_masked']}) |")
    L += [""]

    hdr = ("| scorer | raw IC | raw SE | raw IR | fixed-effect IC | timing IC | timing SE | timing IR | days | "
           "quantile L/S (raw score) | quantile L/S (timing score) | macro F1 (own label) | best epoch | epochs run |")
    sep = "|" + "---|" * 14
    for w in tfx.EVAL_WINDOWS:
        title = ("val_2024 (selection window; timing IC of the selected epoch)" if w == "val_2024" else
                 "oot_2026 (final checkpoint scored once; confirmation only, not a selection criterion)")
        L += [f"## {title}", "", hdr, sep]
        if v3:
            x = v3[w]["signal"]
            diag = V3_DIAGNOSIS[w]
            tse = diag.get("timing_se")
            L.append(f"| **V3 seed 0 (`v3_wd`, fixed label)** | {_fmt(x['mean_daily_rank_ic'])} | {_fmt(tfx.ic_se(x))} | "
                     f"{_fmt(x['ic_ir'], 3)} | {_signed(diag['fixed'])}† | {_signed(diag['timing'])}† | "
                     f"{_fmt(tse) if tse else 'n/a'} | n/a | {x['n_days_used']} | {_fmt(x['quantile_long_short']['mean_spread'], 4, True)} | "
                     f"n/a | {_fmt(v3[w]['metrics']['macro_f1'])} (fixed label) | {_fmt(v3.get('best_epoch'), 0)} | "
                     f"{v3['train']['last_epoch'] + 1} |")
        else:
            L.append("| V3 seed 0 (`v3_wd`) | _not found in e2e3_results.json / tfx_results.json_ |" + " |" * 12)
        for n in recipes:
            r = results.get(n)
            if r is None:
                L.append(f"| {RECIPES[n]['tag']} {n} | _pending_ |" + " |" * 12)
                continue
            d, s = r[w]["decomposition"], r[w]["signal"]
            L.append(f"| **{RECIPES[n]['tag']} {n}** | {_signed(d['raw_ic'])} | {_fmt(d['raw_ic_se'])} | {_fmt(d['raw_ic_ir'], 3)} | "
                     f"{_signed(d['fixed_effect_ic'])} | {_signed(d['timing_ic'])} | {_fmt(d['timing_ic_se'])} | "
                     f"{_fmt(d['timing_ic_ir'], 3)} | {d['n_days_raw']} | {_fmt(s['quantile_long_short']['mean_spread'], 4, True)} | "
                     f"{_fmt(d['timing_quantile_ls'], 4, True)} | {_fmt(r[w]['metrics']['macro_f1'])} ({r[w]['n_label_ok']} lbl) | "
                     f"{_fmt(r.get('best_epoch'), 0)} | {r['train']['last_epoch'] + 1} |")
        L += ["", "† V3 fixed-effect / timing IC are quoted from `docs/signal_diagnosis.md` (computed from the V3 score dump); "
                  "raw IC / IR / quantile L/S / F1 are read from the results JSON. " if v3 else "", ""]
    L += ["## Per-epoch val curves (`*` = selected epoch; timing IC is the selection metric)", ""]
    for n in recipes:
        if n in results:
            L.append(f"- **{RECIPES[n]['tag']} {n}**: {_curve_line(results[n])}")
    L += ["", "Reading guide: a recipe helps only if its timing IC is clearly positive on val AND OOT (SE ~0.014 val / ~0.018 OOT; "
              "differences below ~2 SE are noise). A high raw IC with fixed-effect IC ~ raw IC and timing IC <= 0 means the model "
              "still learned only a ticker ranking.", ""]
    return "\n".join(L)


def _git_commit() -> str:
    base = tfx._git_commit()
    try:
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "training/run_e4_experiments.py",
                                         "training/signal_diagnostics.py"], text=True).strip()
    except Exception:  # noqa: BLE001
        return base
    return base if (base.endswith("+dirty") or not dirty) else base + "+dirty"


# --------------------------------------------------------------------------------------------
# recipe execution
# --------------------------------------------------------------------------------------------
def score_with_decomposition(ds, meta, model, device, prior=None):
    """(decomposition dict, ticker means) of the final checkpoint on one window."""
    from training.run_tfx_experiments import sample_arrays
    rets, _ = sample_arrays(ds)
    dec = timing_decomposition([d for _, d, _ in meta], [t for t, _, _ in meta], _default_score_fn(model, ds, device),
                               rets, prior, tfx.MIN_NAMES_PER_DAY)
    means = dec["ticker_means"]
    out = flat_summary(dec)
    out["fixed_effect_source"] = "window mean" if prior is None else "val_2024 ticker means of the final checkpoint"
    return out, means


def run_recipe(name: str, data, opts: dict, budget) -> dict | None:
    """Result record, or None if the budget stopped training mid-way (resumable epoch checkpoint)."""
    import torch
    from torch.utils.data import DataLoader
    from training.config import KNOWN_FUTURE_COLS, STATIC_COLS, build_tft_config
    from training.dataset import TickerDayDataset
    from training.run_stage1_search import STATIC_CARDINALITIES
    from training.stage1_data import compute_class_weights, compute_label_distribution
    from training.train import TemporalFusionTransformer, run_training

    spec = RECIPES[name]
    ropts = recipe_opts(name, opts)
    hp, seed, device = ropts["hparams"], ropts["seed"], ropts["device"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info("=== e4 recipe %s (%s) seed=%d hparams=%s select=%s ===", name, spec["tag"], seed, hp, ropts["select"])
    prep = prepare_e4(name, data, ropts)
    cols = prep["kept_columns"]
    art_dir = ropts["artifacts_dir"]

    eval_ds = {w: tfx.build_window_dataset(w, prep["eval_frames"][w], cols, ropts) for w in tfx.EVAL_WINDOWS}
    align = ropts.get("align", tfx.DEFAULT_ALIGN)
    train_ds = TickerDayDataset(prep["train_frames"], cols, KNOWN_FUTURE_COLS, STATIC_COLS, tfx.ENCODER_LEN, align=align)
    val_ds = TickerDayDataset(prep["val_loss_frames"], cols, KNOWN_FUTURE_COLS, STATIC_COLS, tfx.ENCODER_LEN, align=align)
    counts = compute_label_distribution(train_ds)
    if min(counts.values()) == 0:
        raise ValueError(f"[{name}] a train class is empty: {counts}")
    class_weights = compute_class_weights(counts)
    logger.info("[%s] inputs=%d train samples=%d val-loss samples=%d label counts=%s weights=%s",
                name, len(cols), len(train_ds), len(val_ds), counts, class_weights.tolist())

    tft_config = build_tft_config(
        {"historical": cols, "future": KNOWN_FUTURE_COLS, "static_cardinalities": STATIC_CARDINALITIES},
        num_classes=3, state_size=hp["state_size"], attention_heads=hp["attention_heads"],
        lstm_layers=hp["lstm_layers"], dropout=hp["dropout"])
    fingerprint = full_fingerprint(name, ropts, cols, class_weights.tolist())
    ckpt_dir = os.path.join(art_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    run_name = f"e4-{name}"
    val_ds_eval, val_meta = eval_ds["val_2024"]           # val 2024 ONLY feeds the per-epoch hook
    val_rets, _ = tfx.sample_arrays(val_ds_eval)
    result = run_training({
        "tft_config": tft_config, "class_weights": class_weights, "device": device,
        "train_loader": DataLoader(train_ds, batch_size=ropts["batch_size"], shuffle=True),
        "val_loader": DataLoader(val_ds, batch_size=ropts["batch_size"], shuffle=False),
        "epochs": ropts["epochs"], "lr": hp["lr"], "weight_decay": hp.get("weight_decay", 0.0),
        "run_name": run_name, "checkpoint_dir": ckpt_dir,
        "epoch_checkpoint_path": os.path.join(ckpt_dir, f"{run_name}_inprogress.pt"),
        "early_stopping_patience": ropts["patience"],
        "epoch_metric_fn": make_timing_fn(val_ds_eval, val_meta, val_rets, device),
        "select_metric_key": SELECT_KEY,
        "should_stop": budget.exceeded, "epoch_log": logger.info,
        "fingerprint": fingerprint, "seed": seed,
        "wandb_config": {**hp, "recipe": name, "seed": seed, "select": ropts["select"], "n_inputs": len(cols),
                         "label_source": spec["label_source"], "static_const": spec["static_const"],
                         "vol_rank": spec["vol_rank"]},
    })
    if result["stopped_reason"] == "budget":
        logger.info("[%s] time budget reached after epoch %d; state saved, resumable.", name, result["last_epoch"])
        return None
    logger.info("[%s] training finished (%s, last epoch %d, best epoch %s); scoring.",
                name, result["stopped_reason"], result["last_epoch"], result["best_epoch"])

    model = TemporalFusionTransformer(tft_config).to(device)
    model.load_state_dict(torch.load(result["checkpoint_path"], map_location=device))
    meta = {
        "seed": seed, "data_version": fingerprint["data_version"], "threshold": fingerprint["threshold"],
        "columns_hash": fingerprint["columns_hash"], "preproc": None,
        "resumed": result["resumed"], "resume_count": result["resume_count"], "resumed_from_epoch": result["start_epoch"],
        "stale_inprogress_ignored": result["stale_checkpoint"],
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
    }
    curve = result["epoch_history"]
    best_ic = next((h[SELECT_KEY] for h in curve if h["epoch"] == result["best_epoch"]), None)
    rec = {
        "fingerprint": fingerprint, "meta": meta, "recipe": name, "tag": spec["tag"], "group": "E4",
        "base_recipe": BASE_RECIPE, "label_source": spec["label_source"], "static_const": spec["static_const"],
        "vol_rank_cols": list(VOL_RANK_COLS) if spec["vol_rank"] else [], "align": align, "hparams": hp,
        "selection": ropts["select"], "seed": seed, "batch_size": ropts["batch_size"], "max_epochs": ropts["epochs"],
        "patience": ropts["patience"], "best_epoch": result["best_epoch"], "best_val_timing_ic": best_ic,
        "epoch_curve": curve, "oot_scored": "final checkpoint only, once; not a selection criterion",
        "train": {
            "n_inputs": len(cols), "kept_columns": cols, "preprocessed": False, "align": align,
            "n_train_samples": len(train_ds), "n_glitch_masked": prep["n_glitch_masked"],
            "label_counts": {str(k): v for k, v in counts.items()}, "class_weights": class_weights.tolist(),
            "best_val_loss": result["best_val_loss"], "last_epoch": result["last_epoch"],
            "stopped_reason": result["stopped_reason"], "checkpoint": result["checkpoint_path"],
        },
        "code_commit": ropts["commit"], "finished_at": time.strftime("%F %T"),
    }
    prior = None
    for w in tfx.EVAL_WINDOWS:                              # val first: its ticker means are the OOT fixed effect
        ds, wmeta = eval_ds[w]
        rec[w] = tfx.score_window(w, ds, wmeta, model, device)
        rec[w]["decomposition"], means = score_with_decomposition(ds, wmeta, model, device, prior)
        if w == "val_2024":
            prior = means
        logger.info("[%s] %s: raw IC=%s fixed IC=%s timing IC=%s F1=%s", name, w, rec[w]["decomposition"]["raw_ic"],
                    rec[w]["decomposition"]["fixed_effect_ic"], rec[w]["decomposition"]["timing_ic"],
                    rec[w]["metrics"]["macro_f1"])
    del model
    return json.loads(json.dumps(rec))


def record_model_version(name: str, rec: dict, path: str) -> None:
    """One docs/model_versions.md row per finished recipe (idempotent)."""
    version = f"e4-{name}"
    if tfx.model_versions_row_exists(path, version):
        return
    from training import run_stage1_search as rss
    t, v, o = rec["train"], rec["val_2024"], rec["oot_2026"]
    lab = {"cs": "날짜별 횡단면 분위 라벨(상위30% 매수/하위30% 매도)", "label_vn": "label_vn",
           "label": "고정 임계값 라벨(V3와 동일, label)"}[rec["label_source"]]
    extra = (", 정적 범주 sector_id/market_id 상수 0" if rec["static_const"] else "") + \
            (f", 변동성 열 횡단면 순위 치환({', '.join(rec['vol_rank_cols'])})" if rec["vol_rank_cols"] else "")
    wd = rec["hparams"].get("weight_decay", 0)
    desc = f"R0 기반(33컬럼 원본, align={rec['align']}), 라벨={lab}{extra}, weight_decay={wd:g}, seed={rec['seed']}, 체크포인트 선택=val 타이밍 IC"
    note = (f"E4 {rec['tag']}, best_epoch={rec['best_epoch']}, epochs_run={t['last_epoch'] + 1}({t['stopped_reason']}), "
            f"val 타이밍 IC={_signed(v['decomposition']['timing_ic'])}(선택 기준), val 원점수 IC={_signed(v['decomposition']['raw_ic'])}, "
            f"OOT 타이밍 IC={_signed(o['decomposition']['timing_ic'])}/원점수 IC={_signed(o['decomposition']['raw_ic'])}"
            f"(최종 체크포인트 1회 채점, 확인용, 선택 근거 아님), F1는 자체 라벨 기준. 상세: docs/e4_experiments.md")
    old = rss.MODEL_VERSIONS_PATH
    rss.MODEL_VERSIONS_PATH = path
    try:
        rss.append_model_version_row(version=version, stage="1단계-E4", feature_desc=desc, hparams=rec["hparams"],
                                     macro_f1_val=v["metrics"]["macro_f1"], note=note)
    finally:
        rss.MODEL_VERSIONS_PATH = old


def read_v3_reference(ref_dir: str, e2e3_path: str | None = None, tfx_path: str | None = None) -> dict:
    """{"v3_wd": V3 seed-0 record}: e2e3_results.json `v3_wd`; empty if that file / key is missing."""
    e2 = tfx.load_results(e2e3_path or os.path.join(ref_dir, "e2e3_results.json"))
    return {"v3_wd": e2["v3_wd"]} if "v3_wd" in e2 else {}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipes", nargs="+", default=list(DEFAULT_ORDER), choices=list(RECIPES),
                   help="run order = the order given (default: x1 -> x2 -> x3 -> x4 -> "
                        "x2_nostatic_nowd -> x3_vn_nostatic_nowd -> x2/x3_..._seed1 -> x2/x3_..._seed2 -> "
                        "v3_structure_nowd -> seed1 -> seed2 -> vn -> cs -> wd1e4 -> wd3e4 -> patience8 -> dropout30)")
    p.add_argument("--max-minutes", type=float, default=600.0)
    p.add_argument("--artifacts-dir", default=None,
                   help=f"checkpoints + results JSON (default {ARTIFACTS_DIR}; REQUIRED with --max-tickers)")
    p.add_argument("--results-path", default=None, help=f"default: <artifacts-dir>/{RESULTS_NAME}")
    p.add_argument("--e2e3-results-path", default=None, help="V3 seed-0 record source; default: <ref-dir>/e2e3_results.json")
    p.add_argument("--doc-path", default=DOC_PATH)
    p.add_argument("--model-versions-path", default=tfx.MODEL_VERSIONS_PATH)
    p.add_argument("--ref-dir", default=ARTIFACTS_DIR, help="where e2e3_results.json lives")
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--batch-size", type=int, default=tfx.BATCH_SIZE)
    p.add_argument("--device", default=None, help="default: cuda if available else cpu")
    p.add_argument("--no-wandb", action="store_true", help="WANDB_MODE=disabled (dry runs)")
    p.add_argument("--max-tickers", type=int, default=None, help="DRY RUN: subsample tickers")
    p.add_argument("--state-size", type=int, default=None, help="DRY RUN override of state_size for every recipe")
    args = p.parse_args(argv)
    if args.max_tickers and args.artifacts_dir is None:
        p.error("--max-tickers (dry run) requires an explicit --artifacts-dir so real checkpoints are not overwritten")
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
    with open(tfx.CHAMPION_CONFIG_PATH) as f:
        champion = json.load(f)
    hp = {k: champion[k] for k in ("state_size", "attention_heads", "lstm_layers", "dropout", "lr")}
    if data is None:
        dsn = os.environ.get("STOCK_DB_V2_DSN")
        if not dsn:
            raise ValueError("STOCK_DB_V2_DSN environment variable not set")
        data = tfx.RealData(dsn)
    os.makedirs(args.artifacts_dir, exist_ok=True)
    results_path = args.results_path or os.path.join(args.artifacts_dir, RESULTS_NAME)
    opts = {"champion": champion, "hparams": hp, "device": device, "epochs": args.epochs, "batch_size": args.batch_size,
            "artifacts_dir": args.artifacts_dir, "max_tickers": args.max_tickers, "align": tfx.DEFAULT_ALIGN,
            "ref_dir": args.ref_dir, "commit": _git_commit(), "state_size_override": args.state_size,
            "expected_samples": getattr(data, "expected_samples", {})}
    logger.info("device=%s recipes=%s max_minutes=%s results=%s", device, args.recipes, args.max_minutes, results_path)

    results = tfx.load_results(results_path)

    def write_outputs():
        os.makedirs(os.path.dirname(args.doc_path) or ".", exist_ok=True)
        with open(args.doc_path, "w") as f:
            f.write(render_doc(results, read_v3_reference(args.ref_dir, args.e2e3_results_path), opts["commit"]))

    for name in args.recipes:
        ropts = recipe_opts(name, opts)
        if name in results:
            bad = tfx.fingerprint_mismatch(results[name].get("fingerprint"), cheap_fingerprint(name, ropts))
            if not bad:
                logger.info("[%s] already recorded in %s -- skipping", name, results_path)
                record_model_version(name, results[name], args.model_versions_path)   # heal a missed row
                continue
            logger.warning("[%s] recorded result's fingerprint differs from the current setup (%s) -- moving it to "
                           "%s.stale.%s.json and re-running", name, ", ".join(bad), results_path, name)
            tfx.save_results_atomic(f"{results_path}.stale.{name}.json", {name: results.pop(name)})
            tfx.save_results_atomic(results_path, results)
        if budget.exceeded():
            logger.info("time budget exhausted; not starting recipe %s this invocation.", name)
            break
        rec = run_recipe(name, data, opts, budget)
        if rec is None:
            break
        results[name] = rec
        tfx.save_results_atomic(results_path, results)
        record_model_version(name, rec, args.model_versions_path)
        write_outputs()
        logger.info("[%s] recorded.", name)
    write_outputs()
    done = [r for r in RECIPES if r in results]
    logger.info("run_e4_experiments invocation complete: %d recorded (%s).", len(done), done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
