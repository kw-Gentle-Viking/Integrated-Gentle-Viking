# 신호 품질 개선 계획 (2026-09-26, 18개 태스크 완료 후 후속 사이클)

## 배경과 근거 (전부 실측)

- 18개 태스크 완료 후 성능: 2024 val macro F1 0.382 / 2025 test 0.330 / 2단계 held-out 0.319. 랜덤 기준선(3클래스, 실제 클래스 비율)이 ≈0.326이라 **test에서 랜덤과 거의 차이가 없음** → 병목은 클래스 가중치가 아니라 신호/일반화.
- class-weight 스윕은 접음: 기존 `balanced` 모델은 2024 val에서 이미 예측 비율 27/38/35% (정답 25/47/28%)로 균형 → 2단계 held-out(2026)의 매수 쏠림(45%)/관망 recall 16%는 가중치 문제가 아님. (`training/run_classweight_*`는 코드로만 남기고 실행하지 않음)
- **진짜 원인 후보 = 라벨 정의**: 고정 임계값(±1.238%)에서 연도별 관망 비율은 2019 55% / 2020 43% / … / **2026 31.6%**. `next_day_return / volatility_20d`(임계값 k=0.578, train |z| 중앙값)로 라벨링하면 연도별 관망 비율이 48~52%로 안정(2026도 48.4%). 두 라벨 일치율 85.9%. (분석: 2026-09-26, feature_pool 전체 376,682행)
- 학습 과적합이 빠름: val loss가 2~3에폭 이후 상승 (1단계 ablation, 2단계 모두), 2단계 15에폭 중 대부분 낭비.
- ablation 차이(0.349~0.382)가 시드 노이즈 범위인지 미검증.

## 평가 규율 (test set 오염 방지)

- 모델/설정 **선택은 2024 val만** 사용. 2025 test는 Task 16에서 이미 1회 소진 → 재사용 금지.
- 1단계 계열 모델(학습 2019~2023)에게 **2026-01-01~2026-09-08은 완전한 미사용 구간**(레버리지 국면 포함) → "신선한 OOT 창"으로 **최종 후보 레시피당 1회만** 확인(윈도우용 룩백은 2025-09부터 로드, 타깃일이 2026-01-01 이후인 샘플만 채점).
- 라벨 정의가 다른 모델끼리는 macro F1을 직접 비교할 수 없음 → **라벨 무관 지표**를 1급 평가 지표로 추가: 일별 횡단면 rank IC(score = p_buy − p_sell vs 실제 next_day_return, 날짜별 Spearman 평균), 예측 매수−매도 그룹 평균 수익률 스프레드(argmax 기준 + 일별 상/하위 분위 기준). macro F1/클래스별 P/R/예측비율은 보조로 유지.

## 태스크

### S1. 변동성 정규화 라벨 (순수 로직, TDD)
- `training/label.py`에 추가: `derive_volnorm_params(returns, vols, target_hold_ratio=0.5, floor_quantile=0.01) -> {"k","floor"}` (train 구간에서만 산출: floor = vols의 1% 분위, k = |ret / max(vol, floor)|의 target_hold_ratio 분위), `assign_label_volnorm(returns, vols, k, floor) -> list[Optional[int]]` (z = ret / max(vol, floor); z≥k 매수(0), z≤−k 매도(2), 그 외 관망(1); ret 또는 vol이 None/NaN이면 None). 기존 `assign_label`/`derive_threshold`는 수정 금지(기존 결과 재현성).
- 파라미터는 코드에 하드코딩 금지: `training/threshold_vn.json`(gitignored, threshold.json과 동일 관례)에 `k, floor, target_hold_ratio, computed_on(2019-01-02~2023-12-31), timestamp` 저장. 산출은 **train 구간(2019-01-02~2023-12-31)만** 사용.
- 변동성 입력은 `feature_pool.volatility_20d` (t 시점까지의 정보만 사용 → 룩어헤드 없음. next_day_return은 t→t+1).

### S2. 라벨 적재 + 데이터 로더 스위치
- `labels` 및 `feature_pool`에 `label_vn SMALLINT` 컬럼 추가(ALTER … IF NOT EXISTS) 후 전체 기간 라벨 산출·적재. **기존 `label`/`next_day_return`은 건드리지 않음.**
- **함정 주의**: `features/build_features.py`는 자기 df에 없는 컬럼을 stale로 drop함(과거 매크로 컬럼 사고, 커밋 1eaf049) → `label_vn`이 재빌드 시 사라지지 않도록 build_features의 labels 조인/기대 컬럼에 `label_vn` 포함 + 회귀 테스트.
- `training/stage1_data.py`: `query_feature_pool`/`build_ticker_dfs`/`load_or_build_ticker_dfs`에 `label_col: str = "label"` 인자 추가(선택 컬럼을 `label`로 alias). 캐시 파일명에 label_col 반영(고정 라벨 캐시와 충돌 금지).
- 검증(실DB): 연도별 클래스 비율이 분석 결과(2019~2026 모두 관망 ≈48~52%)와 일치, train 구간 관망 비율 ≈50%, NULL 행 수 = 기존 label의 NULL + vol 결측분.

### S3. 라벨 무관 신호 지표 + 확률 추론
- `training/train.py`: `predict_proba(model, loader, device) -> (y_true, probs[N,3])` 추가 (기존 `predict` 수정 금지, 내부 재사용 가능).
- `evaluation/evaluate.py`: `compute_signal_metrics(dates, next_day_returns, probs) -> dict` — `mean_daily_rank_ic`(날짜별 Spearman(p_buy−p_sell, ret) 평균, 최소 종목수 미만 날짜 제외), `ic_ir`(일별 IC 평균/표준편차), `argmax_long_short`(예측 매수 평균수익 − 예측 매도 평균수익, 각 그룹 표본수 포함), `quantile_long_short`(일별 score 상위/하위 20% 평균수익 차), 모두 순수 함수 TDD(빈 입력·상수 score 등 엣지 포함, 크래시 금지).
- 샘플→(티커, 타깃일, next_day_return) 매핑 헬퍼(기존 `dates_for_dataset` 재사용/확장).
- 기준선 재채점 스크립트: 기존 챔피언 체크포인트(고정 라벨 학습)를 **2024 val**과 **2026 OOT 창**에서 신호 지표+기존 지표로 채점(추론만, GPU가 사용 중이면 CPU 허용) → `docs/model_versions.md`가 아닌 별도 `docs/signal_baseline.md`에 기록. 이게 "지금 모델에 실제 신호가 얼마나 있는가"의 기준값.

### E1. 라벨 비교 (GPU)
챔피언 구조/33피처 그대로, {고정 라벨(기존 체크포인트 재사용), volnorm 라벨(신규 학습)} — 신규는 2019~2023 학습, 조기종료(patience 3, 최대 10에폭), 최적 val loss 체크포인트. 2024 val에서 신호 지표+F1/클래스별 P/R/예측비율 비교. 승자 라벨로 이후 진행(동률이면 라벨 안정성이 좋은 volnorm 우선 — 근거: 국면 안정성).

### E2. 과적합 억제 탐색 (GPU, Optuna, 재개 가능)
승자 라벨 위에서 소규모 Optuna(≈12 trial, ≤5에폭, 조기종료): state_size {8,16,32}, dropout ≤0.5, lr, **weight_decay(1e-6~1e-2 log, 신규)**, lstm_layers {1,2}. 목적함수 = 2024 val mean daily rank IC. 재개/유휴감시 패턴은 Task 15의 `run_stage1_search.py`/`run_stage1_chain.sh` 방식 재사용.

### E3. 시드 분산 + 앙상블 (GPU)
E2 최적 설정을 시드 3개로 학습 → IC/F1 평균±표준편차, 확률 평균 앙상블 성능. 같은 시드 수로 "33피처(lev_total_volume 제거) vs 34피처" 비교해 ablation 결론이 노이즈인지 확인.

### F. 최종
승자 레시피로 2단계 재학습(전체 기간, held-out 100일 보고 슬라이스 유지) → 서빙 교체(앙상블 채택 시 `serving/model.py` 다중 체크포인트 로딩 필요 → 별도 태스크). 최종 신뢰 구간 확인용 OOT(2026) 1회 사용 규율은 위와 동일.

## 실행 원칙
- GPU는 사용자와 공유: 유휴 감시 체인(utilization 0% 연속) + 에폭/트라이얼 단위 재개. `pkill -f <sweep script>`로 언제든 양보 가능. `kis_collector` env는 torch 관련 금지(프로덕션 공유), `dl_env` 사용.
- 각 태스크: TDD → 실DB/실데이터 검증 → 리뷰 → 원장 기록(SDD).

---

## E0 결과 이후 수정 (2026-09-26 밤) — TFT 재학습 레시피 R1~R3로 E1~E3를 대체/재정렬

### 새로 확인된 사실 (전부 실측)
1. **학습·서빙 어디에서도 입력 스케일링이 적용된 적이 없음.** Task 11이 만든 clip/scale 아티팩트(`features/artifacts/stage1.*`)는 파이프라인에 연결되지 않았음(`load_artifacts`/`apply_clip`은 `features/` 밖에서 호출 0회). TFT는 원본 값을 그대로 먹음: rsi_14 0~100(std 17), usd_krw_chg 최대 50, vix_chg 최대 25, sector_volume_ratio 최대 14, log_ret 최대 2.4 등. 태스크 간 경계에 생긴 실행 갭(각 태스크는 통과, 연결 태스크가 없었음).
2. **레버리지 피처 3개가 죽어 있음**: `lev_total_aum`, `lev_aum_to_mktcap`, `est_rebalancing_flow`는 train뿐 아니라 2026 OOT에서도 전부 0(AUM 이력을 KIS에서 못 받아 계산이 0). → 이전 ablation의 "est_rebalancing_flow 제거 시 하락" 등 해석은 노이즈. (별도 후속: AUM 프록시로 살릴지 결정 — 이번 사이클 범위 밖, 원장에 기록)
3. **코퍼레이트 액션 미수정 가격 글리치**: |next_day_return|>30%(KRX 상하한 30%)인 행 42개(40종목), |log_ret|>30% 79개 (예: 300720 2021-09-07 다음날 −90.9%). 소수지만 극단 라벨/이상치.
4. E0: 같은 피처로 HGB(F3)는 IC 0.058(val)/0.044(OOT), TFT 챔피언은 0.014/−0.005 → TFT가 신호를 못 뽑음. 목표선 = **OOT IC ≥ 0.044 (HGB F3 수준)**.

### 레시피 (모두 챔피언 구조 state 32/heads 8/layers 2/dropout .1658/lr .000291, batch 128, 최대 12에폭, val-loss 조기종료 patience 3, 에폭 단위 재개)
- **R1 std**: 입력 정규화만(고정 라벨). 연속형은 train(≤2023-12-31)에서 0.5/99.5 분위 클립 → 평균/표준편차 표준화, 이진 플래그(0/1)는 그대로(분위 클립하면 희소 플래그가 0으로 죽음 — Task 11이 제외했던 이유), train에서 분산 0인 컬럼(위 죽은 레버리지 3개)은 입력에서 제거 → 30컬럼. 아티팩트는 train으로만 fit, val/OOT/서빙은 적용만.
- **R2 std_vn**: R1 + `label_vn`.
- **R3 std_vn_csr**: R2 + 날짜별 횡단면 백분위 랭크 입력(E0의 F3와 동일 정신; 대상 log_ret, disparity_5d/20d/60d, rsi_14, volatility_20d; E0의 `training/tabular_features.py` 재사용).
- (R1~R3 결과에 따라) R4: 회귀 헤드(z_next) 또는 순위 타깃 — 그때 결정.
- 학습 라벨 마스킹: |next_day_return|>0.31 행은 **학습 라벨만** NaN 처리(Dataset이 NaN 라벨 샘플 스킵). 평가 표본 집합은 S3/E0와 동일 유지(직접 비교 위해); 스프레드는 ±0.30 클립 버전도 함께 보고(IC는 랭크 기반이라 영향 미미).
- 선택은 2024 val IC(1차)로만, 2026 OOT는 확인용(레시피 3개 한정, 문서에 "선택 근거 아님" 명시). 비교행: TFT 챔피언, 1일 반전, HGB F3(E0 JSON에서 복사).

### 서빙 함의 (레시피 채택 시 후속)
- R1~ 채택 → `serving/`에 스케일러 아티팩트 탑재 + `feature_builder`가 60행 모두에 동일 변환 적용 필요.
- R3(랭크 입력) 채택 → 오늘 값의 횡단면 랭크를 200종목 라이브 데이터 없이 계산할 방법 필요(예: 전일 EOD 횡단면 분포를 기준분포로 사용). 채택 시 별도 설계.

### 실행
GPU 유휴 감시 체인(utilization 0% 연속) 자동 실행 + 레시피/에폭 단위 재개, `pkill`로 양보. 사용자 승인 없이 유휴 시 자동 진행(2026-09-26 사용자 지시).
