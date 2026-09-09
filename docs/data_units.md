# 데이터 단위 기록

> 새 필드를 다룰 때마다 이 문서에 단위를 기록한다(Global Constraint). 확인 전엔 배율을 가정하지 않는다 — Task 8 Step 6, Task 10의 시총 이중검증 참고.

## 원칙

- 모든 원화 금액 필드는 **raw 원(₩)** 단위로 통일한다.
- WTI 유가·금값·S&P500 등 외화/지수 표시 매크로 지표는 환산 대상 아님(수익률/변동폭으로만 피처화).
- "미확인"은 실제 API 응답 샘플로 확인 전까지 배율을 가정하지 않았다는 뜻 — 절대 추측치로 채우지 말 것.

## 확인된 단위

| 테이블.컬럼 | 소스 API 필드 | 원본 단위 | 배율 | 확인 근거 |
|---|---|---|---|---|
| `daily_valuation.market_cap` | KIS `hts_avls` (FHKST01010100) | 백만원 | × 1,000,000 | 원본 캡스톤 `pipeline_overview.md`에 문서화됨(Task 5 코드에 이미 반영) |
| `investor_flow_daily.individual_net_amt` | KIS `prsn_ntby_tr_pbmn` (FHKST01010900) | 백만원 | × 1,000,000 | 위와 동일 |
| `investor_flow_daily.foreign_net_amt` | KIS `frgn_ntby_tr_pbmn` (FHKST01010900) | 백만원 | × 1,000,000 | 위와 동일 |
| `investor_flow_daily.inst_net_amt` | KIS `orgn_ntby_tr_pbmn` (FHKST01010900) | 백만원 | × 1,000,000 | 위와 동일 |
| `price_daily.turnover` | KIS `acml_tr_pbmn` (FHKST03010100) | raw 원 | × 1 | 2026-09-08 실제 백필(200종목, 376,682행) 데이터로 검증 — 005930 최근 3거래일 `turnover`/`(close_price×volume)` 비율이 0.99~1.02 (VWAP 기반이라 완전히 1은 아니지만 자릿수 일치 확인됨) |
| `price_daily.shares_outstanding` | KIS `lstn_stcn` — **`output2`가 아니라 `output1`**(현재가, 같은 응답에 포함) | 주 | × 1 | 2026-09-08 라이브 실행 중 발견: `output2`(일별 시세)엔 이 필드가 없어서 전량 NULL이 되는 버그가 있었음(수정 완료, 커밋 af1057a). `output1`은 "조회 시점 현재" 기준이라 과거 시점 근사치(원본 캡스톤과 동일한 한계) |
| `ticker_universe.market_cap`, `price_daily` 기반 계산 시총 | 없음 (직접 계산) | raw 원 | × 1 | `close_price(원) × shares_outstanding(주)` 직접 곱셈이라 API 배율 이슈 자체가 없음 |
| `market_global.*` (snp500, gold, wti 등) | yfinance/FRED | 각 지표 원래 통화·단위(USD, index point 등) | 환산 안 함 | Global Constraint — 원화 환산 대상 아님 |

## 미확인 (사용 전 반드시 채울 것)

| 테이블.컬럼 | 소스 API 필드 | 상태 |
|---|---|---|
| `leverage_daily.volume` | KIS `acml_vol` (FHKST03010100) | Task 8 Step 6에서 확인됨 → 아래로 이동 |
| `leverage_daily.aum` | KIS 일일 NAV/AUM 조회 불가 | 역사 데이터 미제공, NULL로 유지 |
| `leverage_daily.nav` | KIS 일일 NAV/AUM 조회 불가 | 역사 데이터 미제공, NULL로 유지 |

---

## 확인된 단위 추가 (Task 8, 2026-09-08)

| 테이블.컬럼 | 소스 API 필드 | 원본 단위 | 배율 | 확인 근거 |
|---|---|---|---|---|
| `leverage_daily.close_price` | KIS `stck_clpr` (FHKST03010100) | raw 원 | × 1 | 2026-09-08 1종목(0193W0) 실제 데이터: 가격 20,975원으로 합리적인 ETF 가격대 확인됨 |
| `leverage_daily.volume` | KIS `acml_vol` (FHKST03010100) | 주 | × 1 | 위와 동일, 누적 거래량 65M으로 수량 단위 확인됨 |
| `leverage_daily.turnover` | KIS `acml_tr_pbmn` (FHKST03010100) | raw 원 | × 1 | 위와 동일, 가격×거래량 비율이 0.93~1.08 범위(VWAP 기반이라 완전 1은 아니지만 원 단위 확인) |

---

## `feature_pool` 매크로 파생 컬럼 11개 (Task 10.5, 2026-09-09)

Task 13 리뷰에서 발견된 스코프 갭 — `feature_pool`엔 매크로 원본 레벨값만 있고 그 파생(수익률/변화량/스프레드)이 없었음. `features/add_macro_features.py`로 라이브 백필 완료(376,682행 전부 UPDATE).

원칙: `_ret` 접미사 = `pct_change()`(무차원 비율), `_chg` 접미사 = `diff()`(원본 단위의 절대 변화량), `rate_spread_us_kr`만 예외로 전일 대비가 아니라 **당일 레벨의 차분**(`us_10y_yield - kr_base_rate`).

| 신규 컬럼 | 원본 컬럼 | 산식 | 단위 |
|---|---|---|---|
| `kospi_ret` | `index_0001` | `pct_change()` | 무차원 비율 (예: 0.01 = +1%) |
| `kosdaq_ret` | `index_1001` | `pct_change()` | 무차원 비율 |
| `snp500_ret` | `snp500_close` | `pct_change()` | 무차원 비율 |
| `nasdaq_ret` | `nasdaq_close` | `pct_change()` | 무차원 비율 |
| `phlx_semi_ret` | `phlx_semi_close` | `pct_change()` | 무차원 비율 |
| `wti_ret` | `wti_crude_oil` | `pct_change()` | 무차원 비율 |
| `gold_ret` | `gold_price` | `pct_change()` | 무차원 비율 |
| `vix_chg` | `vix` | `diff()` | VIX 포인트 절대 변화량(VIX 자체가 이미 %값이라 diff 사용, pct_change 아님) |
| `usd_krw_chg` | `usd_krw` | `diff()` | 원화 절대 변화량(원/달러), pct 아님 |
| `us_10y_yield_chg` | `us_10y_yield` | `diff()` | bp(퍼센트포인트) 절대 변화량, pct 아님 |
| `rate_spread_us_kr` | `us_10y_yield`, `kr_base_rate` | `us_10y_yield - kr_base_rate` (당일 레벨 차분, 전일 대비 아님) | 퍼센트포인트(%p) |

**계산 grain**: 거래일별 값(전종목 동일값 broadcast) — `feature_pool`에서 `trade_date`로 DISTINCT ON dedup 후 날짜순 정렬해서 계산, `trade_date` 기준으로 200종목 전체에 join-back. 종목별로 나눠 계산하면 같은 날짜가 200번 반복돼 diff가 0이 되는 버그가 생기므로 반드시 날짜 grain에서만 계산.

**첫 거래일(2019-01-02) 결측**: `pct_change()`/`diff()` 기반 10개 컬럼(`rate_spread_us_kr` 제외 전부)은 전일 데이터가 구조적으로 없어 NULL — ffill 대상 아님, Task 3의 상장일 결측과 같은 성격의 진짜 최초 시점 구조적 결측. `rate_spread_us_kr`은 당일 레벨 차분이라 전일 의존이 없어 첫 거래일에도 NULL 없음. 라이브 검증(2026-09-09): 200개 티커 × 10개 컬럼 = 정확히 2,000개 셀이 2019-01-02에만 NULL, 그 외 날짜/컬럼 조합엔 NULL 0건.

**브리프와의 편차**: 브리프 Step 5.5는 "9개 컬럼 NULL"이라 적었으나, 브리프 자신의 공식표(11개 컬럼 중 `rate_spread_us_kr` 1개만 당일 레벨 차분이고 나머지 10개가 `pct_change`/`diff` 기반)를 그대로 따르면 11-1=10개가 맞음 — `pct_change`/`diff`는 정의상 첫 원소가 항상 NaN이므로 10이 수학적으로 옳은 값. 실측도 10을 확인함(브리프의 "9"는 단순 계산 오류로 판단, 공식표 자체는 그대로 따름).
