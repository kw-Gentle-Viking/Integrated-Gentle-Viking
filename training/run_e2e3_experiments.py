#!/usr/bin/env python
"""E2/E3: seed variance and overfitting suppression for the R0 TFT recipe (plan
2026-09-26-signal-improvement-plan.md). Everything is R0-based: raw 33 champion columns (no
preprocessing), fixed label, TickerDayDataset(align="today"), adjusted prices (adj1). Data preparation,
windows, fingerprint, meta, per-epoch shuffle seed, glitch masking, scoring and the results-JSON
idempotence are imported from training/run_tfx_experiments.py (unchanged).

  E3 seed variance     R0 recipe (val-loss selection, patience 3) with seed 1 / seed 2. Seed 0 is NOT
                       retrained: it is the `aligned` record of tfx_results.json (reused at render time).
                       A per-epoch val IC curve is recorded (record-only hook, selection stays val loss).
  E2 overfit control   seed 0, varied hyper-parameters, val-IC epoch selection:
                         v1_lr 1e-4 | v2_do dropout 0.3 | v3_wd Adam weight_decay 1e-3 (L2, not AdamW)
                         v4_lr_do lr 1e-4 + dropout 0.3 | v5_lr_do_wd v4 + wd 1e-3 | v6_state16 state 16
                       Each epoch: val 2024 predictions -> mean daily rank IC; the checkpoint of the epoch
                       with the highest val IC is kept (patience 4, max 12 epochs); the val IC / val loss
                       curve is stored per recipe.

Discipline (binding): variants are selected on the val 2024 rank IC ONLY. The 2026 OOT window is scored
exactly once per recipe, on the final selected checkpoint, is recorded for confirmation and is NEVER a
selection criterion (never scored per epoch). The 2025 test window is never scored.

Priority order = e3_seed1, e3_seed2, v4_lr_do, v1_lr, v2_do, v3_wd, v5_lr_do_wd (v6_state16 only on
request via --recipes). Resumable at epoch granularity; a recipe is recorded in e2e3_results.json only
when finished; `--max-minutes` stops between epochs (exit 0, state resumable).

    set -a && source .env && set +a
    PYTHONPATH=. python training/run_e2e3_experiments.py [--recipes v4_lr_do v1_lr] [--max-minutes 600]

Output paths are overridable (--artifacts-dir / --results-path / --doc-path / --model-versions-path) so
dry runs never touch the real ones.
"""
import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import training.run_tfx_experiments as tfx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = tfx.ARTIFACTS_DIR
DOC_PATH = "docs/e2e3_experiments.md"
RESULTS_NAME = "e2e3_results.json"
BASE_RECIPE = "aligned"                     # R0
MAX_EPOCHS = 12
E3_PATIENCE = 3                             # identical to R0 (val-loss selection)
E2_PATIENCE = 4                             # val-IC selection

RECIPES = {
    "e3_seed1": dict(tag="E3-s1", group="E3", seed=1, overrides={}, select="val_loss", patience=E3_PATIENCE,
                     desc="R0 recipe, seed 1"),
    "e3_seed2": dict(tag="E3-s2", group="E3", seed=2, overrides={}, select="val_loss", patience=E3_PATIENCE,
                     desc="R0 recipe, seed 2"),
    "v4_lr_do": dict(tag="V4", group="E2", seed=0, overrides={"lr": 1e-4, "dropout": 0.3}, select="val_ic",
                     patience=E2_PATIENCE, desc="lr 1e-4 + dropout 0.3"),
    "v1_lr": dict(tag="V1", group="E2", seed=0, overrides={"lr": 1e-4}, select="val_ic", patience=E2_PATIENCE,
                  desc="lr 1e-4"),
    "v2_do": dict(tag="V2", group="E2", seed=0, overrides={"dropout": 0.3}, select="val_ic",
                  patience=E2_PATIENCE, desc="dropout 0.3"),
    "v3_wd": dict(tag="V3", group="E2", seed=0, overrides={"weight_decay": 1e-3}, select="val_ic",
                  patience=E2_PATIENCE, desc="Adam weight_decay 1e-3 (L2, not AdamW)"),
    "v5_lr_do_wd": dict(tag="V5", group="E2", seed=0, overrides={"lr": 1e-4, "dropout": 0.3, "weight_decay": 1e-3},
                        select="val_ic", patience=E2_PATIENCE, desc="V4 + weight_decay 1e-3"),
    "v6_state16": dict(tag="V6", group="E2", seed=0, overrides={"state_size": 16}, select="val_ic",
                       patience=E2_PATIENCE, desc="state_size 16 (optional)"),
}
DEFAULT_ORDER = ["e3_seed1", "e3_seed2", "v4_lr_do", "v1_lr", "v2_do", "v3_wd", "v5_lr_do_wd"]


# --------------------------------------------------------------------------------------------
# pure helpers (unit-tested in training/test_run_e2e3_experiments.py)
# --------------------------------------------------------------------------------------------
def recipe_opts(name: str, base: dict) -> dict:
    """Per-recipe opts: R0 hparams + the recipe's overrides, its seed, patience and selection mode.
    `--state-size` (dry-run) wins over everything."""
    spec = RECIPES[name]
    hp = {**base["hparams"], **spec["overrides"]}
    if base.get("state_size_override"):
        hp["state_size"] = base["state_size_override"]
    return {**base, "hparams": hp, "seed": spec["seed"], "patience": spec["patience"], "select": spec["select"]}


def cheap_fingerprint(name: str, opts: dict) -> dict:
    """tfx cheap fingerprint of the R0 recipe with this recipe's hparams/seed, plus the selection rule."""
    fp = tfx.cheap_fingerprint(BASE_RECIPE, opts)
    fp["selection"] = {"mode": opts["select"], "patience": opts["patience"]}
    return json.loads(json.dumps(fp, sort_keys=True))


def full_fingerprint(name: str, opts: dict, cols, class_weights) -> dict:
    fp = cheap_fingerprint(name, opts)
    fp["columns_hash"] = tfx.columns_hash(cols)
    fp["class_weights"] = [round(float(w), 6) for w in class_weights]
    return fp


def best_epoch_of(rec: dict):
    """Epoch of the selected checkpoint. Seed-0 R0 (tfx record, no curve): with early stopping the best
    val-loss epoch is last_epoch - patience; otherwise unknown (None)."""
    if rec.get("best_epoch") is not None:
        return rec["best_epoch"]
    t = rec["train"]
    if t.get("stopped_reason") == "early_stopping" and rec.get("patience"):
        return t["last_epoch"] - rec["patience"]
    return None


def mean_std(xs: list[float]):
    """(mean, sample std ddof=1 or None when n < 2) of the finite values; (None, None) if empty."""
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def e3_seed_rows(tfx_results: dict, results: dict) -> list[tuple[int, dict]]:
    """[(seed, record)] for seed 0 (tfx `aligned`) and every finished E3 seed recipe."""
    rows = []
    if BASE_RECIPE in tfx_results:
        rows.append((0, tfx_results[BASE_RECIPE]))
    for n, spec in RECIPES.items():
        if spec["group"] == "E3" and n in results:
            rows.append((spec["seed"], results[n]))
    return rows


def make_val_ic_fn(ds, meta, device):
    """epoch_metric_fn for run_training: mean daily rank IC of the val-2024 eval window (RNG-free)."""
    from torch.utils.data import DataLoader
    from evaluation.evaluate import compute_signal_metrics
    from training.train import predict_proba
    tdates = [d for _, d, _ in meta]
    rets, _ = tfx.sample_arrays(ds)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    def fn(model, epoch):
        _, probs = predict_proba(model, loader, device)
        sig = compute_signal_metrics(tdates, rets, probs, min_names_per_day=tfx.MIN_NAMES_PER_DAY)
        return {"val_ic": sig["mean_daily_rank_ic"], "val_ic_ir": sig["ic_ir"], "val_ic_se": tfx.ic_se(sig)}
    return fn


def _fmt(x, nd=4, pct=False):
    return tfx._fmt(x, nd, pct)


def _curve_line(rec: dict) -> str:
    be = rec.get("best_epoch")
    parts = []
    for h in rec.get("epoch_curve", []):
        mark = "*" if h["epoch"] == be else ""
        parts.append(f"{h['epoch']}{mark}: IC {_fmt(h.get('val_ic'))} / loss {_fmt(h['val_loss'])}")
    return "; ".join(parts) if parts else "n/a"


def render_doc(results: dict, tfx_results: dict, refs: dict, code_commit: str, recipes=None) -> str:
    recipes = recipes or DEFAULT_ORDER
    r0 = tfx_results.get(BASE_RECIPE)
    L = [
        "# E2/E3: seed variance and overfitting suppression (R0 TFT recipe)", "",
        "Plan: `docs/superpowers/plans/2026-09-26-signal-improvement-plan.md`. "
        f"Code commit: `{code_commit}` (`training/run_e2e3_experiments.py`, reuses `training/run_tfx_experiments.py`). "
        "Raw output: `training/artifacts/e2e3_results.json` (gitignored). Baseline (R0, seed 0) = `aligned` "
        "record of `training/artifacts/tfx_results.json`, not retrained.", "",
        "**Selection discipline.** Variants are chosen on the val 2024 mean daily rank IC ONLY. The 2026 OOT "
        "window is scored exactly ONCE per recipe, on its final selected checkpoint, and is recorded for "
        "confirmation; it is NOT used to select an epoch, a variant or a hyper-parameter (it is never scored "
        "per epoch). The 2025 test window is never scored.", "",
        "## Protocol", "",
        "- All recipes: R0 = raw 33 champion columns (no preprocessing), fixed label, `align=\"today\"`, adjusted "
        f"prices, batch {tfx.BATCH_SIZE}, max {MAX_EPOCHS} epochs, class weights from the train label distribution, "
        "same val/OOT sample sets and metrics as TFX (SE = IC std / sqrt(days)).",
        f"- **E3** (seeds 1, 2): identical to R0 - val-loss checkpoint selection, patience {E3_PATIENCE}. The val IC "
        "per epoch is recorded only (no effect on training). Seed 0 is the recorded R0 run.",
        f"- **E2** (V1..V6, seed 0): Adam with the stated lr / dropout / `weight_decay` (plain Adam L2, not AdamW), "
        f"checkpoint = epoch with the highest val IC (patience {E2_PATIENCE}); val loss recorded alongside.",
        "- **Caveat**: R0/E3 select by val loss, E2 variants by val IC, so E2-vs-R0 differences mix the hyper-parameter "
        "change with the selection rule. Single-seed differences below ~2 SE (about 0.02 for val IC, "
        "see E3 seed spread) are within noise.", "",
    ]
    # ---- E3
    rows = e3_seed_rows(tfx_results, results)
    L += ["## E3: seed variance (R0 recipe)", "",
          "| seed | source | val IC | val SE | OOT IC | OOT SE | best epoch (val loss) | epochs run |",
          "|---|---|---|---|---|---|---|---|"]
    for seed, rec in rows:
        v, o, t = rec["val_2024"]["signal"], rec["oot_2026"]["signal"], rec["train"]
        src = "tfx_results.json `aligned` (reused)" if seed == 0 else f"e2e3 `e3_seed{seed}`"
        L.append(f"| {seed} | {src} | {_fmt(v['mean_daily_rank_ic'])} | {_fmt(tfx.ic_se(v))} | "
                 f"{_fmt(o['mean_daily_rank_ic'])} | {_fmt(tfx.ic_se(o))} | {_fmt(best_epoch_of(rec), 0)} | "
                 f"{t['last_epoch'] + 1} |")
    vm, vs = mean_std([r["val_2024"]["signal"]["mean_daily_rank_ic"] for _, r in rows])
    om, os_ = mean_std([r["oot_2026"]["signal"]["mean_daily_rank_ic"] for _, r in rows])
    L += ["", f"Across {len(rows)} seed(s): val IC mean {_fmt(vm)} +- {_fmt(vs)} (std, ddof=1), "
              f"OOT IC mean {_fmt(om)} +- {_fmt(os_)}." +
          ("" if len(rows) >= 3 else " (fewer than 3 seeds recorded so far)"), ""]
    # ---- main comparison per window
    hdr = ("| scorer | mean rank IC | SE | IC IR | days | quantile L/S | quantile L/S (clip30) | argmax L/S | "
           "argmax L/S (clip30) | macro F1 | best epoch | epochs run |")
    sep = "|" + "---|" * 12
    order = [(f"R0 seed 0 (tfx)", r0, None)] if r0 else []
    for n in recipes:
        order.append((f"{RECIPES[n]['tag']} {n}", results.get(n), n))
    for w in tfx.EVAL_WINDOWS:
        title = ("val_2024 (selection window)" if w == "val_2024"
                 else "oot_2026 (final checkpoint scored once; confirmation only, not a selection criterion)")
        L += [f"## {title}", "", hdr, sep]
        base_ic = r0[w]["signal"]["mean_daily_rank_ic"] if r0 else None
        for label, rec, n in order:
            if rec is None:
                L.append(f"| {label} | _pending_ |" + " |" * 10)
                continue
            x = rec[w]
            L.append(f"| **{label}** | " + " | ".join(tfx._sig_cells(x["signal"], x["signal_clip30"])) +
                     f" | {_fmt(x['metrics']['macro_f1'])} | {_fmt(best_epoch_of(rec), 0)} | {rec['train']['last_epoch'] + 1} |")
        for rname, sig, f1 in refs.get(w, []):
            L.append(f"| {rname} | " + " | ".join(tfx._sig_cells(sig, None)) + f" | {_fmt(f1)} | n/a | n/a |")
        L += [""]
        if base_ic is not None:
            d = [f"{RECIPES[n]['tag']} {_fmt(results[n][w]['signal']['mean_daily_rank_ic'] - base_ic)}"
                 for n in recipes if n in results]
            L += ["Delta IC vs R0 seed 0: " + (", ".join(d) if d else "none finished yet") + ".", ""]
    # ---- hparams + curves
    L += ["## Hyper-parameters, selected epochs and val curves (all recipes)", "",
          "| recipe | lr | dropout | weight_decay | state | selection | best epoch | best val IC | min val loss | stop |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n in recipes:
        rec = results.get(n)
        if rec is None:
            L.append(f"| {RECIPES[n]['tag']} {n} | _pending_ |" + " |" * 8)
            continue
        hp, t = rec["hparams"], rec["train"]
        L.append(f"| {RECIPES[n]['tag']} {n} | {hp['lr']:g} | {hp['dropout']:.4g} | {hp.get('weight_decay', 0):g} | "
                 f"{hp['state_size']} | {rec['selection']} | {_fmt(rec.get('best_epoch'), 0)} | "
                 f"{_fmt(rec.get('best_val_ic'))} | {_fmt(t['best_val_loss'])} | {t['stopped_reason']} |")
    L += ["", "Per-epoch val curves (`*` = selected epoch):", ""]
    for n in recipes:
        if n in results:
            L.append(f"- **{RECIPES[n]['tag']} {n}**: {_curve_line(results[n])}")
    L += [""]
    L += ["Comparison rows (reversal, HGB) are from the S3 / E0v2 JSON files; legacy-aligned or old-data, "
          "raw spreads only, **not like-for-like** with the aligned TFT rows.", ""]
    return "\n".join(L)


def _git_commit() -> str:
    base = tfx._git_commit()
    try:
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "training/run_e2e3_experiments.py"],
                                        text=True).strip()
    except Exception:  # noqa: BLE001
        return base
    return base if (base.endswith("+dirty") or not dirty) else base + "+dirty"


# --------------------------------------------------------------------------------------------
# recipe execution
# --------------------------------------------------------------------------------------------
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
    logger.info("=== e2e3 recipe %s (%s) seed=%d hparams=%s select=%s ===", name, spec["tag"], seed, hp, ropts["select"])
    prep = tfx.prepare_recipe(BASE_RECIPE, data, ropts)
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
    logger.info("[%s] inputs=%d train samples=%d val-loss samples=%d weights=%s",
                name, len(cols), len(train_ds), len(val_ds), class_weights.tolist())

    tft_config = build_tft_config(
        {"historical": cols, "future": KNOWN_FUTURE_COLS, "static_cardinalities": STATIC_CARDINALITIES},
        num_classes=3, state_size=hp["state_size"], attention_heads=hp["attention_heads"],
        lstm_layers=hp["lstm_layers"], dropout=hp["dropout"])
    fingerprint = full_fingerprint(name, ropts, cols, class_weights.tolist())
    ckpt_dir = os.path.join(art_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    run_name = f"e2e3-{name}"
    val_ds_eval, val_meta = eval_ds["val_2024"]           # val 2024 ONLY feeds the per-epoch hook
    result = run_training({
        "tft_config": tft_config, "class_weights": class_weights, "device": device,
        "train_loader": DataLoader(train_ds, batch_size=ropts["batch_size"], shuffle=True),
        "val_loader": DataLoader(val_ds, batch_size=ropts["batch_size"], shuffle=False),
        "epochs": ropts["epochs"], "lr": hp["lr"], "weight_decay": hp.get("weight_decay", 0.0),
        "run_name": run_name, "checkpoint_dir": ckpt_dir,
        "epoch_checkpoint_path": os.path.join(ckpt_dir, f"{run_name}_inprogress.pt"),
        "early_stopping_patience": ropts["patience"],
        "epoch_metric_fn": make_val_ic_fn(val_ds_eval, val_meta, device),
        "select_metric_key": "val_ic" if ropts["select"] == "val_ic" else None,
        "should_stop": budget.exceeded, "epoch_log": logger.info,
        "fingerprint": fingerprint, "seed": seed,
        "wandb_config": {**hp, "recipe": name, "seed": seed, "select": ropts["select"], "n_inputs": len(cols)},
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
    best_ic = next((h["val_ic"] for h in curve if h["epoch"] == result["best_epoch"]), None)
    rec = {
        "fingerprint": fingerprint, "meta": meta, "recipe": name, "tag": spec["tag"], "group": spec["group"],
        "base_recipe": BASE_RECIPE, "label_col": "label", "align": align, "hparams": hp,
        "selection": ropts["select"], "seed": seed, "overrides": spec["overrides"],
        "batch_size": ropts["batch_size"], "max_epochs": ropts["epochs"], "patience": ropts["patience"],
        "best_epoch": result["best_epoch"], "best_val_ic": best_ic, "epoch_curve": curve,
        "oot_scored": "final checkpoint only, once; not a selection criterion",
        "train": {
            "n_inputs": len(cols), "n_rank_inputs": 0, "kept_columns": cols, "dropped_columns": [],
            "preprocessed": False, "align": align, "n_train_samples": len(train_ds),
            "n_glitch_masked": prep["n_glitch_masked"],
            "label_counts": {str(k): v for k, v in counts.items()}, "class_weights": class_weights.tolist(),
            "best_val_loss": result["best_val_loss"], "last_epoch": result["last_epoch"],
            "stopped_reason": result["stopped_reason"], "checkpoint": result["checkpoint_path"],
        },
        "code_commit": ropts["commit"], "finished_at": time.strftime("%F %T"),
    }
    for w in tfx.EVAL_WINDOWS:
        ds, wmeta = eval_ds[w]
        rec[w] = tfx.score_window(w, ds, wmeta, model, device)
        logger.info("[%s] %s: IC=%s SE=%s F1=%s", name, w, rec[w]["signal"]["mean_daily_rank_ic"],
                    tfx.ic_se(rec[w]["signal"]), rec[w]["metrics"]["macro_f1"])
    del model
    return json.loads(json.dumps(rec))


def record_model_version(name: str, rec: dict, path: str) -> None:
    """One docs/model_versions.md row per finished recipe (idempotent)."""
    version = f"e2e3-{name}"
    if tfx.model_versions_row_exists(path, version):
        return
    from training import run_stage1_search as rss
    t, v, o = rec["train"], rec["val_2024"], rec["oot_2026"]
    desc = (f"R0 기반(33컬럼 원본, label=label, align={rec['align']}), {RECIPES[name]['desc']}, seed={rec['seed']}, "
            f"체크포인트 선택={'val IC' if rec['selection'] == 'val_ic' else 'val loss'}")
    note = (f"E2/E3 {rec['tag']}, best_epoch={rec['best_epoch']}, epochs_run={t['last_epoch'] + 1}({t['stopped_reason']}), "
            f"val IC={_fmt(v['signal']['mean_daily_rank_ic'])}(선택 기준), OOT IC={_fmt(o['signal']['mean_daily_rank_ic'])}"
            f"(최종 체크포인트 1회 채점, 확인용, 선택 근거 아님). 상세: docs/e2e3_experiments.md")
    old = rss.MODEL_VERSIONS_PATH
    rss.MODEL_VERSIONS_PATH = path
    try:
        rss.append_model_version_row(version=version, stage="1단계-E2E3", feature_desc=desc, hparams=rec["hparams"],
                                     macro_f1_val=v["metrics"]["macro_f1"], note=note)
    finally:
        rss.MODEL_VERSIONS_PATH = old


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipes", nargs="+", default=list(DEFAULT_ORDER), choices=list(RECIPES),
                   help="run order = the order given (default: priority order)")
    p.add_argument("--max-minutes", type=float, default=600.0)
    p.add_argument("--artifacts-dir", default=None,
                   help=f"checkpoints + results JSON (default {ARTIFACTS_DIR}; REQUIRED with --max-tickers)")
    p.add_argument("--results-path", default=None, help=f"default: <artifacts-dir>/{RESULTS_NAME}")
    p.add_argument("--tfx-results-path", default=None, help="R0 seed-0 record; default: <ref-dir>/tfx_results.json")
    p.add_argument("--doc-path", default=DOC_PATH)
    p.add_argument("--model-versions-path", default=tfx.MODEL_VERSIONS_PATH)
    p.add_argument("--ref-dir", default=ARTIFACTS_DIR, help="where S3/E0 JSON (comparison rows) and tfx_results.json live")
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
    tfx_path = args.tfx_results_path or os.path.join(args.ref_dir, "tfx_results.json")
    opts = {"champion": champion, "hparams": hp, "device": device, "epochs": args.epochs, "batch_size": args.batch_size,
            "artifacts_dir": args.artifacts_dir, "max_tickers": args.max_tickers, "align": tfx.DEFAULT_ALIGN,
            "ref_dir": args.ref_dir, "commit": _git_commit(), "state_size_override": args.state_size,
            "expected_samples": getattr(data, "expected_samples", {})}
    logger.info("device=%s recipes=%s max_minutes=%s results=%s", device, args.recipes, args.max_minutes, results_path)

    results = tfx.load_results(results_path)

    def write_outputs():
        os.makedirs(os.path.dirname(args.doc_path) or ".", exist_ok=True)
        with open(args.doc_path, "w") as f:
            f.write(render_doc(results, tfx.load_results(tfx_path), tfx.read_reference_rows(args.ref_dir),
                               opts["commit"]))

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
    logger.info("run_e2e3_experiments invocation complete: %d recorded (%s).", len(done), done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
