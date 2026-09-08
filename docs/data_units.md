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
| `price_daily.turnover` | KIS `acml_tr_pbmn` (FHKST03010100) | raw 원으로 추정 | × 1 | 원본 문서에 배율 언급 없음 — **확정 아님**, Task 4 실행 시 실제 응답값을 종가×거래량과 대조해서 자릿수 재확인 권장 |
| `ticker_universe.market_cap`, `price_daily` 기반 계산 시총 | 없음 (직접 계산) | raw 원 | × 1 | `close_price(원) × shares_outstanding(주)` 직접 곱셈이라 API 배율 이슈 자체가 없음 |
| `market_global.*` (snp500, gold, wti 등) | yfinance/FRED | 각 지표 원래 통화·단위(USD, index point 등) | 환산 안 함 | Global Constraint — 원화 환산 대상 아님 |

## 미확인 (사용 전 반드시 채울 것)

| 테이블.컬럼 | 소스 API 필드 | 상태 |
|---|---|---|
| `leverage_daily.close_price` | KIS ETF/ETN 현재가 API | Task 8 Step 6에서 1종목 1일치로 확인 예정 |
| `leverage_daily.aum` | KIS ETF/ETN NAV 비교추이 API | 위와 동일 |
| `leverage_daily.nav` | KIS ETF/ETN NAV 비교추이 API | 위와 동일 |
