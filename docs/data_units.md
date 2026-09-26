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

## 백로그: ETN 2종목 코드 해결 + 백필 (2026-09-09)

Task 8에서 `code: None`으로 남겨뒀던 ETN 2종목(TIGER 삼성전자레버리지/TIGER SK하이닉스레버리지, 이름은 당시 추정치)을
KRX_API_KEY 확보 후 재조사해 해결. 상세 경위·검증 근거는
`.superpowers/sdd/2026-09-08-ai-model-redesign-plan/backlog-etn-codes-report.md` 참고.

**KRX Open API(`etp/etn_bydd_trd`)는 이 키로 이용 신청이 안 돼 있어(전체 `etp` 카테고리 401, `sto`/`idx`는 200 — 키 자체는
유효함을 대조 확인) 사용 불가.** 대신 KIS 공식 종목마스터파일(`kospi_code.mst`, koreainvestment/open-trading-api 레포의
`kis_kospi_code_mst.py` 파싱 로직 그대로 사용)에서 발행사 코드 "Q520"(미래에셋) 계열로 후보를 찾고, KIS 일별시세 API
(FHKST03010100)로 실거래 데이터를 받아 교차검증했다.

| 테이블.컬럼 | 값 | 확인 근거 |
|---|---|---|
| `leverage_products.code` = `Q520100` | 미래에셋 레버리지 삼성전자 단일종목 ETN | `kospi_code.mst` 상장일자 필드 `20260527`(다른 16종과 동일 상장일 일치), KIS 응답 `stck_shrn_iscd`/`hts_kor_isnm`이 코드·명 그대로 일치 |
| `leverage_products.code` = `Q520101` | 미래에셋 레버리지 SK하이닉스 단일종목ETN | 위와 동일 패턴으로 검증 |
| `leverage_daily.close_price`(Q520100/Q520101) | raw 원 | 2026-05-27~09-09 73거래일, 가격대 6,940~42,155원 — 기존 16종 범위(5,785~44,000원)와 정합 |
| `leverage_daily.turnover`(Q520100/Q520101) | raw 원 | 전체 146행(73거래일×2종목) turnover/(price×volume) 비율 — Q520100 0.922~1.192(평균 1.008), Q520101 0.828~1.192(평균 1.015). 기존 16종의 전체 분포(평균 1.001~1.019, 범위 최대 0.632~1.246)와 통계적으로 구분 안 됨 — 리뷰(2026-09-09)에서 표본 6행 기준이던 걸 전수 재확인 |
| `leverage_products.code`/`leverage_daily.code` 컬럼 길이 | VARCHAR(7) | ETN 코드가 "Q"+6자리(7자)라 기존 VARCHAR(6)로는 저장 불가 — stock_db_v2에 `ALTER TABLE ... ALTER COLUMN code TYPE VARCHAR(7)` 실행, `db/schema.sql`도 동기화 |

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

---

## 백로그: `vi_events` 데이터 소스 재조사 + 백필 (2026-09-09)

Task 8에서 시도했던 KIS TR `FHKST02900200`(404 확인됨)와 별개로, `vi_events`(0행, 전체 기간)를 채울 실제 작동하는 소스가 있는지 재조사. 상세 경위·검증 근거는 `.superpowers/sdd/2026-09-08-ai-model-redesign-plan/backlog-vi-events-report.md` 참고. **결론: DONE — 실제 데이터로 백필 완료(4,926행, 200/200 티커).**

조사한 3개 경로:
- KRX Open API(`openapi.krx.co.kr`, `KRX_API_KEY` 대상): 라이브 확인 — 전체 ~35개 서비스 카탈로그(지수/주식/증권상품/채권/파생상품/일반상품/ESG)에 VI 관련 서비스 없음. **불가.**
- KRX 정보데이터시스템(`data.krx.co.kr`) 공개 MDC 페이지: "변동성완화장치 발동종목현황"(bld `dbms/MDC/STAT/issue/MDCSTAT22401`) 페이지 자체는 정확한 컬럼 구조로 실존하나, 2026-09-09 기준 `getJsonData.cmd` 호출이 전부 HTTP 400 `LOGOUT` — KRX가 익명 접근을 최근 차단함(대조군으로 과거 공개였던 일별시세 페이지도 동일하게 차단 확인, 외부 블로그 보고와도 일치). **불가.**
- KIS 신규 TR `FHPST01390000`(`/uapi/domestic-stock/v1/quotations/inquire-vi-status`, "변동성완화장치(VI) 현황"): KIS 공식 GitHub 예제 레포에 문서화된, Task 8이 시도한 것과 다른 TR. 라이브 검증(2026-05-27, 2026-09-09, 티커별 조회) 후 실제 사용. **가능.**

| 테이블.컬럼 | 소스 API 필드 | 확인 근거 |
|---|---|---|
| `vi_events.ticker` | KIS `mksc_shrn_iscd` (FHPST01390000) | 200/200 유니버스 티커 전부 최소 1건 이상 확인 |
| `vi_events.triggered_at`/`released_at` | KIS `bsop_date`+`cntg_vi_hour`/`vi_cncl_hour` | HHMMSS, KST 현지시각 그대로(naive timestamp, 타임존 변환 없음). `cntg_vi_hour`가 16:00~18:00(시간외 단일가) 구간도 포함함을 라이브로 확인 |
| `vi_events.vi_type` | KIS `vi_kind_code` | `1`=STATIC(정적), `2`=DYNAMIC(동적) — 같은 TR 계열의 요청 파라미터 문서("0:전체 1:정적 2:동적 3:정적&동적")로 매핑. `3`=BOTH는 라이브로 별도 검증: `vi_kind_code=3`인 표본 행(140410, 2019-06-28)이 `vi_stnd_prc`(정적 기준가)와 `vi_dmc_stnd_prc`(동적 기준가) 둘 다 non-zero로 채워짐을 확인(코드 1/2는 둘 중 하나만 채워짐) — "정적+동적 동시발동" 확정 |
| `vi_events.released_at` = `000000` 처리 | KIS `vi_cncl_hour="000000"` | 1/4,926행에서 관측된 일시적 이상값(같은 (ticker,date) 재조회시 정상값 복원됨) — VI가 물리적으로 자정에 해제될 수 없으므로(거래시간 09:00~18:00) 빈 문자열과 동일하게 `None` 처리하도록 파서에 방어 코드 추가, 기존 1건은 재조회한 실값(09:03:42)으로 직접 수정 |

**완전성 관련 중요 발견**: 시장 전체 조회(`FID_INPUT_ISCD=""`)는 호출당 최대 30행으로 캡핑됨(라이브로 확인). 전체 스윕(2005거래일×2시장=4,010콜)에서 **87%(3,488/4,010)가 이 캡에 걸림** — 애초 예상("드문 극단적 변동일에만 걸림")과 크게 다름. 티커 지정 조회(`FID_INPUT_ISCD=<ticker>`)는 캡 없음(검증: 007120 단일종목 조회 시 그 종목의 실제 4건 전부 정상 반환). 보완용 티커별 재조회를 캡된 3,488개 (시장,날짜) 쌍에 대해 실행했으나 예산(4,000콜)이 2019년 데이터부터 소진되어 2026-05-27 인근은 커버 못함 — 이후 **2026-05-27 ±10거래일(21일×200종목=4,200콜) 전용 타겟 재검증을 별도 실행**해 이 구간만 캡 없이 완전 커버. 나머지 기간(2019~2026년, 05-27 구간 제외)은 시장전체스윕 표본(캡의 영향으로 실제보다 과소집계 가능성 있음) — 결과표의 "가장 바쁜 날" 랭킹이 05-27 구간에 쏠린 것은 이 커버리지 차이 때문이지 반드시 실제 변동성 차이만은 아님(단, 05-27 구간 자체가 레버리지 ETF 출시로 실제로도 변동성이 컸다는 것은 기초 사실과 일치).

**⚠️Task 11 후속조치 필요**: `features/artifacts/stage1.bounds.json`의 `vi_count_recent5d: [0.0, 0.0]`은 `vi_events`가 항상 비어있던 시절에 적합(fit)된 퇴화(degenerate) 경계값. 이제 실 데이터가 존재하므로 **재적합 필요** — 이 백로그 태스크 스코프 밖이라 직접 재적합하지 않음, `features/build_features.py`(`load_vi_events`)+`features/run_fit_clip_scale.py` 재실행 필요.

### 후속: `vi_count_recent5d` 롤링 윈도우 버그 수정 + 재계산 (2026-09-09)

위 백필(4,926행) 이후 `feature_pool.is_vi_triggered`/`vi_count_recent5d`가 여전히 전체 376,682행 100% 0으로 확인됨 — `vi_events`가 항상 비어있던 시절에 짜인 `load_vi_events()`가 그대로 남아있었기 때문. 재계산 과정에서 버그를 하나 발견해 함께 수정했다.

**버그**: 기존 `load_vi_events()`는 `vi_daily = df.groupby(['ticker','trade_date']).size()`로 만든, **VI 이벤트가 실제로 발생한 날짜만 있는 sparse 프레임**에 대고 바로 `.rolling(window=5)`를 걸었다. 이 프레임엔 이벤트가 없는 날의 행 자체가 없으므로, `window=5`는 "최근 5거래일"이 아니라 "최근 5번의 VI 이벤트 발생일"을 의미하게 된다. VI 이벤트가 드문 종목은 수년 전 이벤트가 이번 이벤트의 "5일 윈도우"에 섞여 들어갈 수 있음 — 이 코드베이스의 다른 모든 `*_5d`/`*_20d` 피처(`disparity_5d`, `sector_ret_5d` 등)가 실제 거래일 인덱스 기반 trailing 윈도우인 것과 어긋남.

**수정**: `features/build_features.py`에 순수 함수 `compute_vi_features(vi_raw_df, trading_days_df)`를 신설(`load_vi_events()`는 이제 DB 조회 후 이 함수를 호출하는 얇은 래퍼). `trading_days_df`(=`price_df`)로 종목별 실제 거래일 캘린더 전체를 만들고, 이벤트 없는 날은 `vi_count=0`으로 채워 넣은 뒤 그 날짜 인덱스 위에서 `.rolling(window=5, min_periods=1)`을 적용 — 다른 trailing-N-거래일 피처와 동일한 패턴. `is_vi_triggered = (vi_count > 0)` 로직 자체는 변경 없음. TDD 근거: `features/test_build_features.py`에 회귀 테스트 4개 추가(210거래일 떨어진 두 이벤트가 서로의 5일 윈도우에 섞이지 않는지, 3거래일 이내 이벤트는 합산되는지, 비이벤트일도 0으로 커버되는지, 빈 입력 처리).

**부수 효과**: `compute_vi_features()`가 이제 (ticker, trade_date) 쌍마다 정확히 한 행을 반환(price_df와 동일한 키 집합)하므로, `merge_features()`의 VI 조인이 완전한 1:1 매치가 됨 — 예전엔 `vi_events_df`가 항상 비어있어 else 분기(0 채움)만 타서 드러나지 않았지만, sparse 프레임을 그대로 좌측 조인했다면 비이벤트 행에 `NaN`이 남는 잠재 버그도 함께 해소됨.

**라이브 재계산 결과**(`python -m features.build_features --start 2019-01-02 --end 2026-09-08`): `is_vi_triggered=1` 376,682행 중 4,426행(= `vi_events`의 고유 (ticker, date) 발동일 수와 일치), `vi_count_recent5d` 분포는 0~14 사이(0: 356,795 / 1: 16,729 / 2: 2,250 / ... / 14: 1) — 더 이상 100% 0이 아님. 스팟체크: `000660`(SK하이닉스) `2026-05-27`은 KIS 원본에도 그 날짜에 STATIC VI 1건만 있고 근방(±10거래일)에 다른 이벤트가 없음 — `feature_pool`에서 `is_vi_triggered=1`(05-27), `vi_count_recent5d`가 05-27부터 1로 올라가 이벤트가 trailing 5거래일 윈도우 안에 있는 동안(06-02까지) 유지되다 06-04에 0으로 빠짐 — 정확히 trailing-5-거래일 의미론과 일치.

**⚠️재확인**: `build_features.py`는 자기 출력에 없는 기존 컬럼을 "stale"로 간주해 DROP한다(Task 10 fix round 2 설계) — Task 10.5(`features/add_macro_features.py`)가 별도로 추가한 매크로 파생 11개 컬럼(`kospi_ret` 등)이 이 재실행으로 한 차례 삭제되었다가 `add_macro_features.py`를 곧바로 재실행해 즉시 복구함(73개 컬럼, 376,682행 모두 정상 확인). **`build_features.py`를 다시 돌릴 때마다 `add_macro_features.py`도 함께 재실행해야 한다** — 두 스크립트가 서로 독립적인 후처리 단계라 이 순서가 문서화되어 있지 않았던 것 자체가 향후 재발 가능한 위험.

---

## `price_daily` 수정주가 (2026-09-26, DATAADJ)

**`price_daily`의 open/high/low/close/volume/turnover는 KIS `FID_ORG_ADJ_PRC="0"` = 수정주가(액면분할·무상증자·유상증자 반영)로 백필한다.**
2026-09-08 최초 백필은 `FID_ORG_ADJ_PRC="1"`로 받았는데, 이 값은 **원주가(미수정)** 다 — 분할/증자 시점마다 가격이 점프해
수익률·이동평균 피처·RSI·변동성·라벨이 그 시점 전후로 오염됐다(수정 전 |log_ret|>0.3 79행, |next_day_return|>0.3 42행 / 40종목;
KRX 일일 가격제한폭은 ±30%). 2026-09-26에 전 종목을 `"0"`으로 재백필해 덮어썼다(`data_collection/run_adjusted_price_backfill.py`).

**확인 근거** — 종목 300720, KIS `inquire-daily-price`(TR `FHKST03010100`), 2021-08-25~2021-09-17:

| `FID_ORG_ADJ_PRC` | 종가 흐름 | 거래량 |
|---|---|---|
| `"0"` (수정) | 20,450, 21,600, 21,800, 23,550 … 23,800, 그다음 21,700, 20,100 … (연속) | ×10 스케일 (297,949) |
| `"1"` (원, 기존 저장분) | 204,500 … 238,000, 그다음 21,700 (**−90.9%**) | 원 거래량 (29,795) |

따라서 `"0"` = 수정, `"1"` = 원주가.

- **거래량도 소급 조정된다**(분할 비율만큼 과거 거래량이 곱해짐). 반면 `turnover`(거래대금, 원)는 금액이라 조정 대상이 아니다 —
  수정 후 과거 구간에서 `turnover / (close × volume)`은 여전히 ≈1(둘 다 같은 비율로 조정돼 상쇄)이어야 정상이다.
- **`shares_outstanding`은 여전히 조회 시점 현재값**(`output1.lstn_stcn`)이라 과거 행에 그대로 적용되는 근사치다. 수정주가 × 현재 주식수이므로
  분할 종목의 과거 시가총액(close × shares_outstanding)은 이제 일관된다(이전엔 원주가 × 현재주식수라 분할 종목 과거 시총이 과소/과대).
- **서빙 시 주의(caveat)**: 수정주가 이력은 "백필 실행일 기준"이다. 앞으로 새 액면분할/증자가 생기면 KIS가 과거 이력을 다시 소급 조정하므로,
  오늘 저장한 과거 행은 그 시점부터 다시 stale해진다. 신규 일봉(수집기가 저장하는 당일 값)과 과거 백필 사이의 연속성은
  이벤트 발생 시 해당 종목을 재백필해야 유지된다. 학습 데이터 갱신·서빙 전 이벤트 종목 재백필 절차가 필요하다.
