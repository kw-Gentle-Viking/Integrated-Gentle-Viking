# TFT 임베딩 + GBM 메타 모델 (Phase 1, 오프라인)

code_commit=2e10bed, chosen_n_iter=400, n_features=82

| window | IC | IC IR | macro F1 |
|---|---|---|---|
| val_2024 | 0.0467 | 0.345 | 0.3216 |
| oot_2026 | 0.0204 | 0.105 | 0.3283 |

선택: HGB n_iter는 train 내부 2023 분할(fit<=2022, select 2023)로만 골랐다 -- val 2024/OOT 2026은 이 선택에 전혀 쓰이지 않았다.
