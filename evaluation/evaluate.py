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
