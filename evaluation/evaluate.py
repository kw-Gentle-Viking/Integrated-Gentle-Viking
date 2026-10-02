import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    matthews_corrcoef, confusion_matrix,
)

CLASS_NAMES = {0: "buy", 1: "hold", 2: "sell"}


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    labels = [0, 1, 2]
    if len(y_true) == 0:
        empty_per_class = {label: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for label in labels}
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "mcc": 0.0,
            "per_class": empty_per_class,
            "confusion_matrix": [[0] * len(labels) for _ in labels],
        }
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    per_class = {
        labels[i]: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
        for i in range(len(labels))
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def split_by_regime(dates: list[str], y_true: list[int], y_pred: list[int],
                     leverage_start: str = "2026-05-27") -> dict:
    pre_idx = [i for i, d in enumerate(dates) if d < leverage_start]
    post_idx = [i for i, d in enumerate(dates) if d >= leverage_start]
    return {
        "pre_leverage": compute_metrics([y_true[i] for i in pre_idx], [y_pred[i] for i in pre_idx]),
        "leverage_era": compute_metrics([y_true[i] for i in post_idx], [y_pred[i] for i in post_idx]),
    }


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given their average rank (what Spearman requires)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray):
    """Spearman rank correlation via average ranks; None if either side is constant."""
    ra, rb = _average_ranks(a), _average_ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom == 0:
        return None
    return float((ra * rb).sum() / denom)


def _mean_or_none(x):
    return float(np.mean(x)) if len(x) else None


def compute_signal_metrics(dates: list[str], next_day_returns, probs, min_names_per_day: int = 20,
                            quantile: float = 0.2) -> dict:
    """Label-agnostic, trading-relevant signal metrics (plan S3).

    score = p_buy - p_sell (probs columns: 0=buy, 1=hold, 2=sell). Per date (>= min_names_per_day
    names with finite score and return): Spearman(score, next_day_return) across names -> mean
    daily rank IC / std (ddof=1) / IR; top-`quantile` minus bottom-`quantile` mean return by
    score. Days whose scores are constant are skipped (IC undefined). argmax_long_short is pooled
    over all samples: mean return of predicted-buy minus predicted-sell. Undefined values are None
    (JSON-safe). Never raises on empty / NaN / constant input.
    """
    dates_arr = np.asarray(list(dates), dtype=object)
    rets = np.asarray(next_day_returns, dtype=float).reshape(-1)
    probs = np.asarray(probs, dtype=float).reshape(-1, 3) if len(rets) else np.zeros((0, 3))
    score = probs[:, 0] - probs[:, 2]
    valid = np.isfinite(rets) & np.isfinite(score) & np.isfinite(probs).all(axis=1)
    dates_arr, rets, probs, score = dates_arr[valid], rets[valid], probs[valid], score[valid]

    ics, q_spreads = [], []
    for d in sorted(set(dates_arr.tolist())):
        idx = np.where(dates_arr == d)[0]
        n = len(idx)
        if n < min_names_per_day or n < 2:
            continue
        s, r = score[idx], rets[idx]
        ic = _spearman(s, r)
        if ic is None:
            continue
        ics.append(ic)
        k = max(1, int(n * quantile + 1e-9))
        order = np.argsort(s, kind="mergesort")
        q_spreads.append(float(r[order[-k:]].mean() - r[order[:k]].mean()))

    n_days = len(ics)
    ic_std = float(np.std(ics, ddof=1)) if n_days >= 2 else None
    mean_ic = _mean_or_none(ics)
    ic_ir = (mean_ic / ic_std) if (ic_std not in (None, 0.0)) else None

    pred = probs.argmax(axis=1) if len(rets) else np.zeros(0, dtype=int)
    groups = {}
    for name, cls in (("buy", 0), ("hold", 1), ("sell", 2)):
        g = rets[pred == cls]
        groups[f"n_{name}"] = int(len(g))
        groups[f"mean_ret_{name}"] = _mean_or_none(g)
    mb, ms = groups["mean_ret_buy"], groups["mean_ret_sell"]
    groups["spread"] = (mb - ms) if (mb is not None and ms is not None) else None

    return {
        "n_samples": int(len(rets)),
        "n_days_used": n_days,
        "mean_daily_rank_ic": mean_ic,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "argmax_long_short": groups,
        "quantile_long_short": {
            "quantile": quantile,
            "n_days": len(q_spreads),
            "mean_spread": _mean_or_none(q_spreads),
        },
    }
