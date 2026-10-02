from training.ablation import run_ablation


def test_ablation_tries_removing_each_candidate_once():
    calls = []

    def fake_train_fn(columns: list[str]) -> float:
        calls.append(list(columns))
        return len(columns) * 0.01  # 컬럼 많을수록 점수 높은 가짜 함수

    base = ["a", "b", "c"]
    results = run_ablation(base, candidate_removals=["a", "b"], train_fn=fake_train_fn)

    assert len(calls) == 3  # baseline(전체) + 2개 제거 실험
    assert {"removed": None, "macro_f1": 0.03} in results
    assert {"removed": "a", "macro_f1": 0.02} in results
    assert {"removed": "b", "macro_f1": 0.02} in results


def test_ablation_removed_column_actually_excluded_from_train_call():
    seen_columns = []

    def fake_train_fn(columns: list[str]) -> float:
        seen_columns.append(columns)
        return 0.5

    run_ablation(["x", "y"], candidate_removals=["x"], train_fn=fake_train_fn)
    removal_call = [c for c in seen_columns if "x" not in c]
    assert removal_call == [["y"]]
