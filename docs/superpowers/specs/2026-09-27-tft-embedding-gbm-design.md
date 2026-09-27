# TFT 임베딩 + GBM 최종 예측기 ("2안") — 설계

## 배경

원래 스펙(`docs/superpowers/specs/2026-09-08-ai-model-redesign-design.md` §13)에서 "2안"은
TFT를 임베딩 추출기로 쓰고 트리 모델이 최종 예측을 담당하는 후속 페이즈로 정의됐다. 이번
신호 개선 사이클(`docs/superpowers/plans/2026-09-26-signal-improvement-plan.md`)에서
데이터 정렬 결함(§"E0 결과 이후 수정")을 고치고 R0~R3 레시피와 E2(과적합 억제)/E3(시드
분산)를 실측한 결과:

- 정렬 수정만으로 TFT 자체 신호는 크게 개선됐다(OOT IC −0.005 → 시드 평균 약 0.027).
- 하지만 시드 분산이 매우 크다(R0 seed 0/1/2 val IC: 0.046 / 0.024 / 0.003). 단일 시드로
  레시피를 비교할 수 없다.
- 표 모델(HGB, E0v2)도 TFT와 비슷하거나 약간 나은 수준이다(HGB clf F3: val 0.058 / OOT
  0.037).
- **TFT도 HGB도 1일 반전(reversal, val 0.063 / OOT 0.042)을 확실히 넘지 못한다.**
- `log_ret`(반전 신호의 원천)은 이미 챔피언 33컬럼(F1)에 포함되어 있고, F2/F3 피처셋에도
  그대로 들어간다(`training/run_tabular_baseline.py:104-111`). 즉 반전 신호를 "추가"할
  필요 없이, F3 표 피처를 쓰는 것만으로 이미 GBM 입력에 반전 정보가 포함된다.

사용자 결정(브레인스토밍 세션, 2026-09-27):
1. 범위는 원래 2안 그대로(TFT 임베딩 + 트리모델), 단순 GBM-only가 아니다.
2. GBM이 반전을 못 넘으면 "그만둔다"가 아니라, 반전 정보를 GBM 피처로 흡수해 앙상블 효과를
   노린다 — 위 분석대로 F3를 쓰면 이미 충족된다.
3. GBM 표 피처셋은 F3(횡단면 랭크 포함, 오프라인 IC 최고)로 하고, 이를 위해 필요한 서빙
   측 200종목 라이브 횡단면 스냅샷 파이프라인도 이번 스펙 범위에 포함한다.
4. 임베딩 추출은 접근 A(단일 TFT 체크포인트, 1회 추출)로 한다. K-fold OOF(접근 B)는
   GPU 비용 대비 이득이 불확실해 범위 밖. 확률만 쓰는 접근 C는 원래 2안 취지(내부 표현
   활용)에서 벗어나 채택하지 않는다.

## 목표

- TFT의 학습된 표현(임베딩)과 표 피처(F3, 반전 정보 포함)를 함께 쓰는 GBM 메타 모델을
  오프라인으로 학습하고, val 2024 IC로 선택, OOT 2026 IC를 1회 확인한다.
- 이 GBM을 서빙에서 최종 예측기로 쓸 수 있도록 필요한 인프라(라이브 임베딩 추출, 라이브
  200종목 횡단면 스냅샷, GBM 추론)를 만들되, 기존 TFT-only 서빙 경로는 그대로 유지하고
  env 스위치로 전환 가능하게 한다.

## 비목표 (Out of scope)

- K-fold OOF 임베딩(접근 B).
- 새로운 라벨/전처리 방식 실험(R0~R3 결론 재검토는 별도 트랙).
- 프론트/백엔드 UI, 알림, 결제 등 — 담당 아님.
- 2025 test 사용 — 어떤 이유로도 금지.
- 거래비용/슬리피지 모델링 — 신호 품질만 다룬다.

## 아키텍처

### 오프라인 (학습)

```
[R0 체크포인트 (tfx-aligned.pt, 고정)]
        │ forward hook (pos_wise_ff_gating 출력, 32-dim)
        ▼
[training/tft_embeddings.py] ── extract_embeddings(dataset) → (ticker, date) 순서 정렬된 [N,32] 배열
        │
        │  join on (ticker, date)
        ▼
[F3 표 피처 (training/run_tabular_baseline.py의 build_features 재사용)]
        │
        ▼
[training/run_gbm_meta.py] ── sklearn HistGradientBoosting (E0v2와 동일 계열)
        │  select: val 2024 mean_daily_rank_ic 최대
        ▼
[training/artifacts/gbm_meta.joblib + gbm_meta_results.json + docs/gbm_meta.md]
        (OOT 2026은 최종 config 1회만 채점, 2025 미사용)
```

### 서빙

```
[일별 배치: 200종목 유니버스 전체 원시 피처 스냅샷]   (신규, serving/feature_builder.py 확장)
        │ 횡단면 랭크 계산 (log_ret, disparity_5d/20d/60d, rsi_14, volatility_20d + lag1..5)
        ▼
[종목별 추론 요청]
        │
        ├─ TFT forward (기존 경로 재사용) → 32-dim 임베딩 (hook)
        ├─ 해당 종목의 F3 랭크값 (스냅샷에서 조회)
        ▼
[GBM predict_proba] → {prob_buy, prob_hold, prob_sell}   (env FINAL_PREDICTOR=tft_gbm일 때)
        기존 TFT-only 소프트맥스 경로는 FINAL_PREDICTOR=tft(기본값)로 유지
```

## 컴포넌트

1. **`training/tft_embeddings.py`** (신규)
   - `extract_embeddings(model, dataloader, device) -> (list[(ticker,date)], np.ndarray[N,32])`.
   - `torch.nn.Module.register_forward_hook`으로 `model.pos_wise_ff_gating`의 출력을 캡처.
     tft-torch 벤더 코드는 수정하지 않는다.
   - 그래디언트 불필요(`torch.no_grad()`), 배치 순서가 곧 (ticker, date) 순서임을
     `run_tfx_experiments.build_window_dataset`과 동일한 방식으로 보장.

2. **`training/run_gbm_meta.py`** (신규)
   - 데이터 준비: `run_tfx_experiments`의 캐시 로딩/윈도 구성 로직과
     `run_tabular_baseline`의 `build_features`(F3)를 그대로 import해서 쓴다(재구현 금지).
   - TFT 임베딩과 F3 피처를 (ticker, date)로 내부 조인. **조인 후 행 수가 TFT 샘플 수 및
     기존 표 샘플 수와 정확히 같은지 assert** — 다르면 즉시 예외(조용한 드롭 금지).
   - GBM 학습(HistGradientBoostingClassifier, E0v2와 같은 하이퍼파라미터 그리드),
     val 2024 IC로 모델 선택, OOT 2026 1회 채점.
   - 결과: `training/artifacts/gbm_meta_results.json`, `docs/gbm_meta_results.md`,
     `model_versions.md`에 `gbm-meta` 행 기록(중복 방지, 기존 러너 패턴 재사용).
   - 재현성 메타데이터(시드, TFT 체크포인트 해시, feature set, 코드 커밋)를 기존 지문
     패턴과 동일하게 기록.

3. **`serving/feature_builder.py` 확장** (기존 파일 수정)
   - 신규 함수: 그날 200종목 유니버스 전체의 원시 피처를 한 번에 계산해 횡단면 랭크를
     만드는 배치 함수(예: `build_universe_snapshot(trade_date) -> DataFrame`). 일별 배치
     job(크론 후보, 단 **기존 운영 crontab에는 추가하지 않고** 이 브랜치 범위 안에서만
     구현·테스트)로 돌릴 것을 전제로 설계.
   - 스냅샷이 없거나 불완전(일부 종목 결측)한 경우의 처리는 두 옵션 중 하나로 구현 단계에서
     확정: (a) 그 종목은 `FINAL_PREDICTOR=tft`로 자동 폴백, (b) 에러 응답. 백엔드 계약을
     아직 모르므로 이 스펙은 두 옵션 모두를 인터페이스로 열어두고, writing-plans 단계에서
     기본값을 정한다. **조용한 0-채움은 어느 경우에도 금지.**

4. **`serving/model.py` / `serving/inference.py` 확장**
   - GBM 아티팩트 로딩(joblib), env `FINAL_PREDICTOR`(`tft` 기본값 / `tft_gbm`)로 분기.
   - `tft_gbm` 경로: TFT forward(hook으로 임베딩 캡처) → 스냅샷에서 해당 종목 F3 값 조회 →
     GBM `predict_proba` → 기존과 같은 `{pred_label, pred_str, prob_buy, prob_hold,
     prob_sell}` 응답 포맷 유지(백엔드 계약 불변).

## 모델 선택 규율 (기존 사이클과 동일)

- 학습: ≤2023. 선택: val 2024 `mean_daily_rank_ic`. 확인: OOT 2026(2026-01-01~09-07) 1회.
- 2025 test는 어떤 이유로도 사용하지 않는다.
- GBM 자체 성능이 반전/HGB-단독을 못 넘어도 "실패"로 간주해 되돌리지 않는다 — 이 스펙의
  목적은 F3(반전 정보 포함)와 TFT 임베딩을 합쳤을 때의 결과를 실측하는 것이며, 결과가
  나쁘면 그 자체가 유효한 산출물(문서화하고 다음 결정에 반영)이다.

## 에러 처리

- 임베딩·표 피처 조인 불일치 → 즉시 예외.
- 서빙 스냅샷 결측 → 폴백 또는 에러(위 3번 참고), 조용한 기본값 대입 금지.
- GBM 아티팩트/TFT 체크포인트 버전 불일치(지문 mismatch) → 기존 TFX 러너의 지문 검증
  패턴을 재사용해 로드 시점에 예외.

## 테스트 계획

- `tft_embeddings.py`: 합성 모델/배치로 hook이 올바른 shape([N,32])과 값(그래디언트 없이
  순전파만)을 잡는지 단위 테스트.
- `run_gbm_meta.py`: 조인 정합성(불일치 시 예외), 시드 고정 재현성, val IC 선택 로직,
  결과 JSON/문서/model_versions 멱등성 — 기존 러너들의 테스트 패턴 재사용.
- `feature_builder.py`의 유니버스 스냅샷: 합성 200종목 데이터로 횡단면 랭크 계산 검증,
  결측 종목 처리(폴백/에러) 테스트.
- 서빙 통합: 합성 스냅샷 + 합성 TFT/GBM으로 종목별 추론 전체 경로 테스트, 기존
  `FINAL_PREDICTOR=tft` 경로가 완전히 불변인지 회귀 테스트.

## 리스크 / 알려진 한계

- 임베딩은 R0 seed 0 체크포인트 1개에서만 추출한다. TFT 자체의 큰 시드 분산이 임베딩
  품질에도 영향을 줄 수 있다 — 필요하면 추후 시드 앙상블 임베딩(여러 체크포인트의 임베딩을
  concat)으로 확장 가능하나 이번 스펙 범위 밖.
- GBM 학습에 쓰는 TFT 임베딩은 TFT 자신이 학습 때 본 train 구간 라벨로 만들어진 것이라
  약한 스태킹 누수 위험이 있다(업계 표준 관행, val/OOT 선택에는 영향 없음) — 문서에 명시.
- 라이브 200종목 스냅샷 배치는 새로운 운영 부담(주기적 전체 유니버스 피처 계산)이다.
  크론 등록은 사용자 승인 없이 하지 않는다.

## 참고 파일

- `docs/superpowers/specs/2026-09-08-ai-model-redesign-design.md` §13 (원래 2안 정의)
- `docs/superpowers/plans/2026-09-26-signal-improvement-plan.md` (이번 사이클 배경)
- `training/run_tfx_experiments.py`, `training/run_e2e3_experiments.py` (재사용 대상 로직)
- `training/run_tabular_baseline.py` (F3 피처 정의, `build_features`)
- `serving/feature_builder.py`, `serving/model.py`, `serving/inference.py` (서빙 확장 대상)
- `tft-torch/tft_torch/tft.py:990-1011` (임베딩 추출 지점: `pos_wise_ff_gating` 출력)
