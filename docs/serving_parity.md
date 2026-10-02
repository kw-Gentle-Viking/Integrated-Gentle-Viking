# 학습-서빙 패리티 (serving parity)

기준일 2026-09-28. 모델 무관(체크포인트와 상관없이 입력 텐서 정합성만 다룬다). 자동 검증은
`serving/test_train_serving_parity.py` (`STOCK_DB_V2_DSN` + adj1 val 캐시가 있을 때만 실행, 없으면 skip).

## 검증 방식

- 임의 3종목 x 3날짜(val 2024 창, 시드 고정)에 대해
  - 학습 쪽: `training.dataset.TickerDayDataset(align="today")` 샘플 (`historical` 60x33 / `future` 1x5 / `static` 1x2)
  - 서빙 쪽: feature_pool의 그 날짜 직전 59행 + 그 날 완성 일봉(OHLCV를 5분봉 1개로 감싸 "오늘" 행으로 사용)
    -> `serving.feature_builder.build_encoder_df` -> `serving.inference.assemble_model_inputs`
    (run_inference가 쓰는 바로 그 텐서 조립 함수)
- 비교 허용오차: float32, `rtol=atol=1e-5`.
- 아래 표는 별도로 20종목 x 3날짜 = 60샘플(시드 7)로 잰 값이다(일회성 진단, 테스트는 9샘플).

## 결과 요약

| 구분 | 결과 |
|---|---|
| 과거 59행 전체(33열) | 60샘플 모두 학습과 일치 (불일치 0) |
| LIVE 열 (log_ret, disparity_5d/20d/60d, rsi_14, volatility_20d, day_of_week) | 완성 일봉 기준 오늘 행이 학습값과 일치 |
| time_progress, static(sector_id, market_id) | 일치 |
| FFILL 열, ZERO 열, 이벤트 플래그 | 아래 "의도된 어긋남"(정책상 다름) |

## 발견해서 고친 버그 (2건) + 가드 추가

1. **time_progress** — 학습은 모든 행에 상수 1.0(`training.stage1_data.TIME_PROGRESS_CONSTANT`)인데 서빙은
   오늘 행에 `compute_time_progress(now)`(장중 0~1, 예: 정오 0.5)를 넣고 있었다. 모델이 본 적 없는 값. ->
   오늘 행도 1.0. `compute_time_progress`는 정보용으로만 남김(모델 입력에 사용 안 함).
2. **is_bok / is_fomc / is_witching_kr / is_witching_us 를 오늘 행에서 전부 0으로 넣던 문제** —
   feature_pool의 이 4열은 "당일 이벤트 발생" 플래그가 아니라 `features/build_features.py`가 ffill한
   "가장 최근 이벤트 종류" 상태 플래그다. 실측: feature_pool 376,682행 중 375,686행(99.7%)이 4열 중 정확히
   하나가 1(나머지 996행은 전부 NULL). 예: 종목 014680의 2024-06-10~12는 is_witching_us=1, 06-13~17은
   is_bok=1, 06-18~20은 is_fomc=1. 서빙이 오늘 행에 (0,0,0,0)을 넣으면 학습 2023+ 어느 행에서도 없던
   패턴이다. -> 직전 행의 값을 이월(FFILL). 새 이벤트가 오늘 시작되는 날만 어긋난다(아래 표).
   근거: 9개 샘플 날짜 전부에서 옛 정책은 학습값(1)과 서빙값(0)이 어긋났다(예 183300 2024-12-17 is_fomc 1 vs 0).
3. **히스토리 가드**(`serving.feature_builder.validate_history_rows`) — 날짜 오름차순/중복 검증,
   오늘 날짜 행 제외(SQL `trade_date < today` + 순수함수 이중 방어), 오늘 이후 행은 look-ahead 예외,
   마지막 히스토리와 오늘 사이가 벌어지면 `HistoryIntegrityError`.
   - `stock_db_v2.calendar.is_market_open`은 한국 공휴일을 반영하지 않아(2024-09-16~18 추석이 open=true 등,
     feature_pool 실제 거래일과 119일 불일치) **영업일 판정에 쓰지 않는다.** 대신 feature_pool 자체의
     시장 거래일(전 종목 DISTINCT trade_date)을 사용: 종목 히스토리 이후 시장은 거래했는데 그 종목 행이 없으면
     "missing trading days"로 예외.
   - 그 외에는 달력일 5일 초과 예외(기본). 추석/설 연휴 직후 첫 거래일(간격 6~9일)에는 전 종목이 예외가
     되므로 그날만 `HISTORY_MAX_GAP_DAYS=10` 환경변수(또는 `max_gap_days` 인자)로 완화해야 한다.

## 의도된 어긋남 (오늘 행의 정책열)

라이브에 섹터/매크로/레버리지/이벤트 피드가 없어 재현할 수 없는 열들. 학습은 그날 값을 보고 서빙은
정책값을 쓴다. 60샘플 실측 (target 행에서 허용오차를 넘은 샘플 수 / 평균 |학습-서빙|):

| 정책 | 열 | 어긋난 샘플 | 평균 abs 차이 |
|---|---|---|---|
| FFILL (직전 거래일 값 이월) | sector_ret_1d | 47/60 | 0.0141 |
| | sector_ret_5d | 47/60 | 0.0118 |
| | sector_ret_20d | 47/60 | 0.0175 |
| | sector_ma_ratio_20d | 46/60 | 0.0077 |
| | sector_volatility | 47/60 | 0.0061 |
| | sector_volume_ratio | 47/60 | 0.3664 |
| | kospi_ret | 60/60 | 0.0112 |
| | kosdaq_ret | 60/60 | 0.0167 |
| | snp500_ret | 60/60 | 0.0097 |
| | nasdaq_ret | 60/60 | 0.0142 |
| | phlx_semi_ret | 60/60 | 0.0265 |
| | vix_chg | 60/60 | 1.552 |
| | usd_krw_chg | 60/60 | 8.602 |
| | us_10y_yield_chg | 59/60 | 0.0771 |
| | rate_spread_us_kr | 56/60 | 0.0557 |
| | wti_ret | 60/60 | 0.0230 |
| | gold_ret | 60/60 | 0.0137 |
| | lev_total_aum, lev_aum_to_mktcap, est_rebalancing_flow | 0/60 | - (표본 종목에서 일 변동 없음) |
| | vi_count_recent5d | 1/60 | 1.0 |
| ZERO (이벤트 당일 플래그, 0 고정) | is_dividend | 1/60 | 1.0 |
| | is_bonus_issue, is_rights_offering, is_split, is_vi_triggered | 0/60 | - |
| FUTURE_DB (직전 행 이월, 새 이벤트 시작일만 어긋남) | is_bok | 10/60 | 1.0 |
| | is_fomc | 5/60 | 1.0 |
| | is_witching_kr | 1/60 | 1.0 |
| | is_witching_us | 14/60 | 1.0 |

이벤트 플래그는 상태가 바뀌는 날(새 이벤트 시작일, 60샘플 중 15건 = 25%)마다 두 열이 동시에 어긋난다.
`market_events`에 2025-12-19 이후 행이 없어 앞날 캘린더가 없다(미래 일정 테이블이 채워지면 오늘의 플래그를
미리 알 수 있으므로 해소 가능).

참고(정렬 근거): feature_pool의 미국/원자재 열은 KRX 거래일 D 행에 "D 직전 미국 세션" 값이 들어 있다
(2024-08-05 행 snp500_ret=-1.84%는 8/2 세션, 8/6 행 -3.00%가 8/5 세션). 즉 US 계열은 개장 전에 이미
알 수 있는 값인데 현재는 전일값을 이월하고 있다 -> 매크로 피드를 붙이면 FFILL 열 중 US 계열은 손실 없이
정합시킬 수 있다. kospi/kosdaq_ret와 섹터 계열은 KRX 당일 값이라 장중엔 미완성이다.

장중 추론(5분마다)의 "미완성 봉" 재계산은 LIVE 열(위 표에 없음)에서 의도된 동작이다: 완성 일봉과 일치함을
위 테스트가 보장하고, 장중에는 그 시점 close로 계산되므로 종가 기준 값과 다르다(당연한 차이).

## 남은 한계

- 위 FFILL/ZERO/이벤트 열의 어긋남은 데이터 소스가 없어 못 없앤다. 모델이 이 열에 얼마나 민감한지는
  체크포인트별로 별도 평가가 필요하다(이 문서/테스트는 모델 무관).
- 패리티 테스트는 val 2024 캐시 기준이며 종목 3 x 날짜 3 표본이다(전수 아님).
- 학습 쪽 이슈(서빙 버그 아님): 이벤트 4열이 ffill 상태 플래그라 "이벤트 예정일 정보"로서의 의미가 약하다.
