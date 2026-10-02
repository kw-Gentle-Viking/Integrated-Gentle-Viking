from typing import Callable


def run_ablation(base_columns: list[str], candidate_removals: list[str],
                  train_fn: Callable[[list[str]], float]) -> list[dict]:
    results = [{"removed": None, "macro_f1": train_fn(base_columns)}]
    for col in candidate_removals:
        reduced = [c for c in base_columns if c != col]
        results.append({"removed": col, "macro_f1": train_fn(reduced)})
    return results
