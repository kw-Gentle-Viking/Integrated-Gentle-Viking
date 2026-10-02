# 백엔드 연동 메모 (AI ↔ Back-Gentle-Viking, 2026-09-28)

기준: 백엔드 `demo/integrated-backend`(main은 2026-05-10에서 정체). 로컬 수정 브랜치 `ai-fit` (push/PR 안 함).

## 계약 (AI → 백엔드)
- `POST /ai/realtime` (헤더 `X-API-Key`), 본문 `{inference_at, results:[{ticker, trade_datetime, pred_label, pred_str, prob_buy, prob_hold, prob_sell, model_version}]}`.
- 백엔드는 `signal=pred_str`, `confidence=max(prob_*)`로 저장하고 `min_confidence`(기본 0.60) 미만이면 매매를 건너뛴다.
- `ai-fit`에서 `model_version`을 선택(기본 "unknown")으로 완화했고, 서빙은 `TFT_MODEL_VERSION`(없으면 모델 파일명)을 채워 보낸다.

## 현재 모델과의 정합성 (docs/signal_diagnosis.md)
- V3 출력은 확률이 0.33~0.50에 몰려 confidence 0.60 이상이 0%, argmax BUY가 0% → 지금 연결하면 자동매매는 전부 건너뜀(안전하지만 신호 없음).
- 비용 반영 순초과 수익이 통계적으로 0과 구분되지 않는다 → 모델이 개선되기 전에는 게이트를 낮추지 말 것.

## ai-fit 브랜치 변경
1. 백테스트 엔진: 종가 10만원 이상 조용한 `break`, 100만원 초과 예외, 100주 초과 예외 제거(가격 유효성=유한·양수, 포지션 금액 sanity로 대체).
2. `AI_PRED_MAX_AGE_MIN`(기본 30분)보다 오래된 예측은 HOLD/확신도 0으로 낮춤.
3. `/ai/realtime` 수신 예측을 `ai_prediction_history`에 저장(실패해도 웹훅 정상).
4. `ai_signal` 백테스트 전략: 저장된 BUY/HOLD/SELL 이력 재생(롱온리, 정수 주, `lag` 옵션).

## 백엔드 담당자와 논의할 것 (수정하지 않음)
- 엔진: 종가 동시 체결, `np.random` 지정가 체결(시드 없음), 소수점 주식.
- `/backtest/strategies`에 나오는 aggressive/balanced/conservative/ultra_safe 는 `BacktestService._load_strategy`에 없음("Unknown strategy").
- 신호 정의: 절대 확률 임계값은 기간 간 분포가 달라 불안정(OOT 점수가 val 분포 밖). 매매에 쓰려면 종목 간 일별 순위(200종목 배치 추론)가 필요.
- 멀티프로세스/레플리카 배포 시 `active_tasks`(프로세스 메모리 dict)가 중복 `/trade/start` 실행을 막지 못함 — 로컬 단일 프로세스 드라이런에는 영향 없음, GCP 배포 때 재점검 필요.
- KIS 에러코드별 분기 없음(레이트리밋 `EGW00133`과 잔고부족을 구분 안 하고 전부 고정 10초 재시도).

~~자동매매 루프가 체결 후 Portfolio를 갱신하지 않음 / RiskManager가 실시간 루프에 연결 안 됨~~ →
**2026-10-02에 수정됨** (아래 섹션 참고).

## 2026-09-28 추가 수정 (Back `ai-fit` d24279e~, Front `ai-fit` 4fd1017)
Back: `/trade/history` status 포함, 전략 대체 사실을 `/trade/start` 응답 `strategies`로 노출, 에러 detail 문자열화, Google 콜백 리다이렉트(`FRONTEND_GOOGLE_CALLBACK_URL`),
인증 공백 수정(`/users/{id}` 본인만, `/users` 목록 `ENABLE_DEBUG_ENDPOINTS`, `/account/assets`·`/ai/agreement` 로그인, `/ai/commands/*`·`POST /prices` 서비스 키),
`APP_ENV=production` 이면 dev 시크릿/`LOCAL_DEMO_MODE`로 기동 거부. AI 폴러(`poll_commands.py`)는 `GCP_BACKEND_API_KEY`를 Bearer로 보내므로 백엔드 `AI_COMMAND_API_KEY`에 같은 값을 설정.
Front(로직만, UI 무변경): 새로고침이 `job_id`(GET /ai/once/{job_id})로 새 결과 도착을 확인한 뒤에만 채택, `total_capital`을 '1주 가격 합'으로 보내지 않음(서버가 예수금 조회), FAILED 주문은 체결 내역에서 제외, 토큰 갱신 single-flight.

## 2026-10-02 전체 정합성 감사 + 수정 (`docs/integration_audit_2026-10-02.md` 전체 감사 결과)

치명(10/6 전 필수) 3건:
- **ONCE 콜백(`/ai/callback`) 500 고정 실패** — AI는 ONCE 결과에 `interpretability` 키를 안 보내는데 백엔드가 이를
  `None`으로 저장, `dict.get(key, {})`는 키가 존재하면(값이 None이어도) default를 안 쓰는 함정 → `None.get(...)`에서
  매번 `AttributeError`. `result.get("interpretability") or {}`로 수정 + 빈 리스트 인덱싱(`events[0]`) 2차 크래시도 수정.
  (Back `3c61738`)
- **자동매매 포지션 추적 버그** — `trading_loop`가 매 사이클 빈 `Portfolio()`를 새로 만들고 체결 후에도 갱신하지 않아
  "이미 보유 중인지" 판단(전략의 `current_qty`)이 영원히 거짓 → 같은 종목 반복 매수가 무한정 가능, 매도(청산)는
  거의 발화 안 함. `app/kis_positions.py` 신설: 매 사이클 KIS 실제 잔고(`fetch_balance().output1`)를 그대로 읽어
  `portfolio.positions`를 채움(메모리 추적 아님 — 재시작/수동매매/부분체결에도 자동 정합). 조회 실패 시 "보유 없음"으로
  가정하지 않고 그 사이클을 통째로 스킵. 이미 구현돼 있던 `RiskManager.check_pretrade()`(종목당/1회 주문/전체 노출 상한)를
  주문 전에 호출하도록 연결. 체결 직후 `apply_fill()`로 즉시 반영. (Back `f68437a`)
- **Basket 종목 검증 없음** — 코스닥 등 AI가 전혀 모르는 종목을 바구니에 담으면 `ai_client.predict()`가 confidence-0
  HOLD 또는(`AI_ALLOW_DUMMY_PREDICTIONS=true`) 완전 무작위 BUY/SELL로 치환하는데 구분 플래그가 없었음.
  `app/ai_universe.py` 신설: `stock_db_v2.ticker_universe`(AI의 실제 200종목)를 조회해 `POST /basket`과
  `/trade/start|once`(Basket 동기화) 양쪽에서 검증. (Back `7e98ac8`)

높음 7건:
- `trading_loop`의 `allocate_portfolio` 0-나누기(미지정 confidence=0 BUY) → task 전체가 조용히 죽던 버그. 루트 수정
  (`total_conf<=0`면 `[]` 반환) + 티커별 `try/except/finally: db.commit()`로 한 종목 예외가 나머지/이전 기록에 영향 안 줌. (Back `9f22950`)
- `broker.create_order`/`fetch_balance`/`get_balance` 동기 호출이 이벤트 루프를 막던 문제 → `asyncio.to_thread`로 이동. (Back `0e4ad1a`)
- AI가 실제로 보낼 수 없는 `warmup`(5분봉 다일치 과거 데이터, KIS/AI 양쪽 다 소스 없음) 대기 90초 고정 → `WARMUP_TIMEOUT_SEC`(기본 5초)로 축소. (Back `c692e45`)
- 프론트가 백엔드에 없는 `strategy_id="rsi_reversal"`을 보내 매번 조용히 `conservative`로 대체 → 전략 선택 UI가 없으므로 빈 배열 전송, 백엔드 기본값 사용. (Front `b468bab`)
- `GET /ai/predictions*`의 `stale` 플래그를 프론트가 전혀 안 읽어 30분 지난 신호를 최신처럼 표시 → "오래된 신호" 배지 추가. (Front `b468bab`)
- 포트폴리오 페이지 "1회 분석" 폴링에 `job_id` 게이팅이 없어 콜백 도착 전 이전 예측을 새 결과로 오인할 레이스 → `lib/ai/predictionJobStore.ts`의 `isOnceJobDone` 재사용. (Front `b468bab`)
- AI 유니버스(코스피 200) 밖 종목의 추론 결과를 무조건 "실제 응답"으로 표시 → `modelVersion` 존재 여부로 조건화. (Front `c0872d8`)

중간 7건:
- `pred_label`이 0/1/2 밖이면 로그 없이 HOLD로 폴백(모델 버그 은폐 가능) → WARNING 로그 추가. (Back `18c8563`)
- ONCE 결과가 `AIPredictionHistory`에 전혀 안 남음(실시간 push만 기록) → `record_predictions()` 호출 추가. (Back `18c8563`)
- `job_id`/`user_id`를 AI가 매번 보내는데 `RealtimePayload`에 필드가 없어 조용히 드롭 → 필드 추가(선택값). (Back `18c8563`)
- `/trade/history`의 `status`(FAILED)를 한 화면(`AITradeHistory`)은 거르고 다른 화면(`portfolio`)은 그대로 노출 → 실패 배지 추가. (Front `b468bab`)
- `/trade/status` 등 조회 실패(401/500)해도 조용히 이전 상태 유지 → 에러 메시지 노출. (Front `b468bab`)
- `/ai/agreement`가 AI 커버리지 밖 종목도 "TFT는 관망 의견"처럼 허위 해석 생성 → `is_ai_covered_ticker` 선확인 후 "비교 불가" 응답. (Back `f2f481a`)
- `FEATURE_DESC`(interpretability 리포트용 피처 설명)가 지금 모델에 없는 옛 피처 8개(`bb_position`,
  `vol_ratio`, `macd_ratio`, `per`, `pbr`, `prop_*`)에 매핑돼 있고, 실제 33개 피처 중 25개 가량
  (`disparity_5d/20d/60d`, `sector_*`, `lev_*` 등)은 매핑이 없어 원본 컬럼명이 그대로 노출됨 →
  실제 champion 컬럼 33개와 1:1로 재작성 + 드리프트 방지 테스트 추가. (Back `f361ffe`)

기타: `run_once()`/`get_market_close()` 죽은 코드 제거(호출부 없음, `broker.create_order` 없이 `TradeLog`에 `FILLED`를
가짜로 기록하도록 drift돼 있었음, Back `f68437a` 포함) · AI/백엔드 `AI_SERVER_API_KEY` 기본값 3군데 불일치 통일(AI `e506b21`) ·
프론트 미사용 목데이터(`mockPopularStocks`, 코스닥 종목+ETF 혼재) 제거(Front `679f7a2`).

전체 감사 원본: `docs/integration_audit_2026-10-02.md`.
