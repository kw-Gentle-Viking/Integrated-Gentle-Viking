#!/usr/bin/env python
"""Plan S3: reference measurement of the deployed Stage-1 champion's REAL signal.

Scores the frozen champion (training/champion_config.json + stage1-ablation-<version>.pt) on
  * val_2024 : targets 2024-01-01..2024-12-31 (the model-selection window)
  * oot_2026 : targets 2026-01-01..2026-09-08 (fresh out-of-time; features loaded from 2025-09-01
               only as 60-day lookback, samples with target date < 2026-01-01 are dropped)
The 2025 test window is never scored. Inference only, CPU by default. Resumable per window: a
window whose result is already in the JSON is skipped. Writes
  training/artifacts/signal_baseline.json, training/artifacts/signal_baseline_probs_<window>.npz,
  docs/signal_baseline.md

    set -a && source .env && set +a
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python training/run_signal_baseline.py
"""
import argparse
import json
import logging
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "training/artifacts"
CHAMPION_CONFIG_PATH = "training/champion_config.json"
RESULT_JSON = f"{ARTIFACTS_DIR}/signal_baseline.json"
DOC_PATH = "docs/signal_baseline.md"
ENCODER_LEN = 60
BATCH_SIZE = 128
MIN_NAMES_PER_DAY = 20

# name -> (load_start, load_end, min_target_date, max_target_date, cache file)
WINDOWS = {
    "val_2024": dict(load_start="2024-01-01", load_end="2024-12-31",
                     min_target="2024-01-01", max_target="2024-12-31",
                     cache=f"{ARTIFACTS_DIR}/stage1_cache_val_2024-01-01_2024-12-31.pkl"),
    "oot_2026": dict(load_start="2025-09-01", load_end="2026-09-08",
                     min_target="2026-01-01", max_target="2026-09-08",
                     cache=f"{ARTIFACTS_DIR}/stage1_cache_oot_2025-09-01_2026-09-08.pkl"),
}
CLASS_KEYS = ("buy", "hold", "sell")


def score_to_probs(score: np.ndarray) -> np.ndarray:
    """Wrap a real-valued score into [N,3] pseudo-probs so heuristics go through the same
    compute_signal_metrics path: p_buy - p_sell is a strictly increasing function of score;
    argmax = buy if score>0, sell if score<0, hold if score==0 (no opinion)."""
    s = np.asarray(score, dtype=float).reshape(-1)
    if len(s) == 0:
        return np.zeros((0, 3))
    sd = s.std()
    z = np.clip(s / sd if sd > 0 else s, -30, 30)
    p_buy = 1.0 / (1.0 + np.exp(-z))
    nz = (z != 0).astype(float)
    return np.stack([p_buy * nz, 1.0 - nz, (1.0 - p_buy) * nz], axis=1)


def class_shares(pred) -> dict:
    n = len(pred)
    if n == 0:
        return {k: 0.0 for k in CLASS_KEYS}
    pred = np.asarray(pred)
    return {name: float((pred == c).sum() / n) for c, name in enumerate(CLASS_KEYS)}


def _fmt(x, nd=4, pct=False):
    if x is None:
        return "n/a"
    return f"{x * 100:.{nd - 2}f}%" if pct else f"{x:.{nd}f}"


def _row(name, r):
    s, m, sh = r["signal"], r["metrics"], r["pred_share"]
    ls, q = s["argmax_long_short"], s["quantile_long_short"]
    return (f"| {name} | {_fmt(s['mean_daily_rank_ic'])} | {_fmt(s['ic_std'])} | {_fmt(s['ic_ir'])} | "
            f"{s['n_days_used']} | {_fmt(ls['spread'], 4, True)} | {ls['n_buy']} / {ls['n_sell']} | "
            f"{_fmt(q['mean_spread'], 4, True)} | {_fmt(m['macro_f1'])} | "
            f"{_fmt(sh['buy'], 3)} / {_fmt(sh['hold'], 3)} / {_fmt(sh['sell'], 3)} |")


def render_markdown(results: dict, code_commit: str) -> str:
    lines = [
        "# Signal baseline (deployed Stage-1 champion, fixed label)", "",
        "Plan S3 reference measurement: how much real, label-agnostic signal the current champion has.",
        f"Code commit: `{code_commit}`. Champion: see `training/champion_config.json` "
        "(`stage1-remove_lev_total_volume`, checkpoint `stage1-ablation-remove_lev_total_volume.pt`).", "",
        "Score = p_buy - p_sell. Rank IC = per-day cross-sectional Spearman(score, next_day_return), "
        f"days with < {MIN_NAMES_PER_DAY} names skipped; IC IR = mean/std of daily IC (ddof=1). "
        "argmax L/S = mean return of predicted-buy minus predicted-sell (pooled). Quantile L/S = per-day "
        "top-20% minus bottom-20% by score, averaged over days. Returns are raw next_day_return (t to t+1).", "",
        "Windows: the 2025 test window is never scored.", "",
    ]
    for name, spec in WINDOWS.items():
        lines.append(f"## {name}: targets {spec['min_target']}..{spec['max_target']}")
        r = results.get(name)
        if r is None:
            lines += ["", "_pending (not yet scored)_", ""]
            continue
        lines += [
            "", f"Samples: {r['n_samples']} (scored with returns: {r['model']['signal']['n_samples']}); "
                f"feature rows loaded {spec['load_start']}..{spec['load_end']}; "
                f"actual target-date range {r.get('target_date_min')}..{r.get('target_date_max')}; "
                f"mean names/day {_fmt(r.get('mean_names_per_day'), 1)}.", "",
            "| Scorer | mean rank IC | IC std | IC IR | days | argmax L/S | n buy / n sell | quantile L/S | macro F1 | pred share buy/hold/sell |",
            "|---|---|---|---|---|---|---|---|---|---|",
            _row("champion", r["model"]),
        ]
        for rn, rr in r["references"].items():
            lines.append(_row(rn, rr))
        m, ls = r["model"]["metrics"], r["model"]["signal"]["argmax_long_short"]
        lines += [
            "", "Champion per-class (0=buy, 1=hold, 2=sell):", "",
            "| class | precision | recall | F1 |", "|---|---|---|---|",
        ]
        for c in ("0", "1", "2"):
            pc = m["per_class"][c] if c in m["per_class"] else m["per_class"][int(c)]  # str after JSON round-trip
            lines.append(f"| {c} | {_fmt(pc['precision'])} | {_fmt(pc['recall'])} | {_fmt(pc['f1'])} |")
        lines += [
            "", f"accuracy {_fmt(m['accuracy'])}, MCC {_fmt(m['mcc'])}. Mean next-day return by argmax group: "
                f"buy {_fmt(ls['mean_ret_buy'], 4, True)}, hold {_fmt(ls['mean_ret_hold'], 4, True)}, "
                f"sell {_fmt(ls['mean_ret_sell'], 4, True)} "
                f"(n {ls['n_buy']} / {ls['n_hold']} / {ls['n_sell']}).", "",
        ]
    lines += [
        "References (macro F1 is meaningless for them since they never predict hold; they only give "
        "context for the IC/spread scale): `random` = seeded Gaussian score; `reversal` = -log_ret of the "
        "target day; `momentum` = +log_ret of the target day.", "",
    ]
    return "\n".join(lines)


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "training/run_signal_baseline.py",
                                         "training/signal_data.py", "training/train.py",
                                         "evaluation/evaluate.py"], text=True).strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _load_results() -> dict:
    if os.path.exists(RESULT_JSON):
        with open(RESULT_JSON) as f:
            return json.load(f)
    return {}


def _score_bundle(dates, rets, probs, y_true):
    from evaluation.evaluate import compute_metrics
    pred = probs.argmax(axis=1).tolist() if len(probs) else []
    return {
        "signal": __import__("evaluation.evaluate", fromlist=["x"]).compute_signal_metrics(
            dates, rets, probs, min_names_per_day=MIN_NAMES_PER_DAY),
        "metrics": compute_metrics(y_true, pred),
        "pred_share": class_shares(pred),
    }


def score_window(name: str, spec: dict, dsn: str, champion: dict, device) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from training.config import KNOWN_FUTURE_COLS, STATIC_COLS, build_tft_config
    from training.dataset import TickerDayDataset
    from training.run_stage1_search import STATIC_CARDINALITIES
    from training.signal_data import (fetch_next_day_returns, filter_index_by_target_date,
                                      next_day_returns_for_samples, sample_meta)
    from training.stage1_data import load_or_build_ticker_dfs
    from training.train import TemporalFusionTransformer, predict_proba

    ticker_dfs = load_or_build_ticker_dfs(dsn, spec["load_start"], spec["load_end"], spec["cache"])
    ds = TickerDayDataset(ticker_dfs, champion["columns"], KNOWN_FUTURE_COLS, STATIC_COLS, ENCODER_LEN)
    n_all = len(ds)
    dropped = filter_index_by_target_date(ds, spec["min_target"])
    logger.info("[%s] samples before target-date filter=%d, dropped=%d, kept=%d", name, n_all, dropped, len(ds))
    meta = sample_meta(ds)
    tdates = [d for _, d, _ in meta]
    assert tdates, f"{name}: no samples"
    assert min(tdates) >= spec["min_target"] and max(tdates) <= spec["max_target"], (
        f"{name}: target dates {min(tdates)}..{max(tdates)} outside {spec['min_target']}..{spec['max_target']}")
    assert not any("2025-01-01" <= d <= "2025-12-31" for d in tdates), "2025 test-window target leaked in"
    logger.info("[%s] target dates %s..%s", name, min(tdates), max(tdates))

    lookup = fetch_next_day_returns(dsn, spec["min_target"], spec["max_target"])
    rets = next_day_returns_for_samples(meta, lookup)
    logger.info("[%s] next_day_return available for %d/%d samples", name, int(np.isfinite(rets).sum()), len(rets))

    tft_config = build_tft_config(
        {"historical": champion["columns"], "future": KNOWN_FUTURE_COLS,
         "static_cardinalities": STATIC_CARDINALITIES},
        num_classes=3, state_size=champion["state_size"], attention_heads=champion["attention_heads"],
        lstm_layers=champion["lstm_layers"], dropout=champion["dropout"])
    model = TemporalFusionTransformer(tft_config).to(device)
    ckpt = f"{ARTIFACTS_DIR}/checkpoints/stage1-ablation-{champion['version'][len('stage1-'):]}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    logger.info("[%s] scoring %d samples on %s (checkpoint %s)", name, len(ds), device, ckpt)
    y_true, probs = predict_proba(model, loader, device, progress_every=10, log=logger.info)

    tickers = np.array([t for t, _, _ in meta])
    logret = np.array([lr for _, _, lr in meta])
    np.savez_compressed(f"{ARTIFACTS_DIR}/signal_baseline_probs_{name}.npz", y_true=np.array(y_true),
                        probs=probs, dates=np.array(tdates), tickers=tickers, next_day_return=rets, log_ret=logret)

    rng = np.random.default_rng(0)
    refs = {
        "random": rng.standard_normal(len(rets)),
        "reversal": -logret,
        "momentum": logret,
    }
    per_day = {}
    for d in tdates:
        per_day[d] = per_day.get(d, 0) + 1
    return {
        "window": f"targets {spec['min_target']}..{spec['max_target']}",
        "n_samples": len(y_true),
        "n_dropped_before_min_target": dropped,
        "target_date_min": min(tdates), "target_date_max": max(tdates),
        "mean_names_per_day": float(np.mean(list(per_day.values()))),
        "model": _score_bundle(tdates, rets, probs, y_true),
        "references": {k: _score_bundle(tdates, rets, score_to_probs(v), y_true) for k, v in refs.items()},
    }


def main(argv=None) -> None:
    import torch
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    dsn = os.environ.get("STOCK_DB_V2_DSN")
    if not dsn:
        raise ValueError("STOCK_DB_V2_DSN environment variable not set")
    with open(CHAMPION_CONFIG_PATH) as f:
        champion = json.load(f)
    ckpt = f"{ARTIFACTS_DIR}/checkpoints/stage1-ablation-{champion['version'][len('stage1-'):]}.pt"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)

    commit = _git_commit()
    results = _load_results()
    for name, spec in WINDOWS.items():
        if name in results:
            logger.info("[%s] already scored in %s -- skipping", name, RESULT_JSON)
            continue
        results[name] = score_window(name, spec, dsn, champion, device)
        results[name]["code_commit"] = commit
        tmp = RESULT_JSON + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        os.replace(tmp, RESULT_JSON)
        with open(DOC_PATH, "w") as f:
            f.write(render_markdown(results, commit))
        logger.info("[%s] done: %s", name, json.dumps(results[name]["model"]["signal"]))
    with open(DOC_PATH, "w") as f:
        f.write(render_markdown(results, commit))
    logger.info("run_signal_baseline complete.")


if __name__ == "__main__":
    main()
