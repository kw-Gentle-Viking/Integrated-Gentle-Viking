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

## 2026-10-03 실제 end-to-end 테스트로 발견/수정

서버를 실제로 띄워서 curl로 직접 호출해봄 — 위 수정들이 실제로 동작하는지 확인하는 과정에서 추가로
3건을 발견/수정했다(정적 분석/단위테스트만으로는 못 잡는 종류).

1. **`generate_report`가 `GEMINI_API_KEY` 비어있으면 그대로 크래시** — `genai.Client(...)` 생성이
   try 밖에 있어서, interpretability 수정을 통과한 뒤에도 ONCE 콜백이 여전히 500. try 안으로 이동.
   (Back `c819ed7`)
2. **`personas` 테이블이 비어있었음**(seed 안 됨, 환경설정 누락) — ONCE 콜백의 `save_report`가 FK
   위반으로 500. `app.seed_rec.seed_personas()` 실행해서 5개 페르소나 채움.
3. **`stock_db_v2.feature_pool`이 25일간 업데이트 안 됨**(2026-09-08에서 멈춤, 2026-10-03 기준) —
   AI 모델이 실제로는 어떤 종목도 추론 못 하는 상태였음(정상적으로 "stale history" 거부는 했지만).
   `price_daily`/`daily_valuation`/`investor_flow_daily`/`stock_events`/`market_global`/
   `market_index_daily`/`sector_daily_ohlcv`/`market_events`/`calendar`는 운영 `stock_db`(같은 KIS
   데이터, 350종목 전체를 매일 수집 중이던 별개 파이프라인)에서 AI 200종목분만 캐치업 복사,
   `leverage_daily`는 `data_collection/run_leverage_backfill.py` 재실행으로 캐치업(`vi_events`는
   KIS API에 과거 조회 엔드포인트가 없어 원래부터 비어있음, 알려진 제약). `features/build_features.py
   --start 2026-06-01 --end 2026-10-02`로 feature_pool 재빌드 완료, 60일 이동평균 등 롤링 피처가
   충분한 lookback을 갖도록 범위를 넉넉히 잡음.
4. **`/trade/once`(및 start/stop)가 큐에 넣은 커맨드를 아무도 가져가지 않음** — 운영 원본
   `/home/user/poll_commands.py`는 운영 `/home/user/api_server.py`를 직접 import해서 호출하는
   방식이라 이 프로젝트의 서빙/모델과는 무관했음. `serving/poll_commands.py` 신설(단순 HTTP
   릴레이: 백엔드 큐 폴링 -> 이 프로젝트 `serving/api_server.py`의 `/command`로 전달). 백엔드
   `/trade/once` -> 큐 -> 이 폴러 -> AI 서버 -> 실제 TFT 추론(`v3_structure_nowd_cs-seed0`) ->
   콜백 -> `GET /ai/once/{job_id}` 조회까지 전체 경로 실제로 확인함. (AI `247e061`)

**확인됨(실제 요청으로 검증, 추가 수정 불필요)**: Basket AI-커버리지 검증(400 거부), `/ai/agreement`
no_tft_coverage 응답, `/ai/predictions` stale 필드, `/ai/realtime` pred_label 이상값 WARNING 로그,
`/trade/once` 혼합 바구니의 `excluded_unsupported_tickers` 분리, `AIPredictionHistory`에 ONCE
결과 기록, 프론트 dev 서버.

**프로세스 상태**: `backend-api`(8000)/`frontend-dev`(3000)는 tmux로 수동 실행 중(재부팅 시 다시
띄워야 함). `ai-serving`(8001, `serving.api_server:app`, dl_env에 fastapi/uvicorn 설치함)과
`serving.poll_commands`(커맨드 큐 중계)는 **crontab에 등록 완료**(`@reboot`+5분마다 헬스체크/
재시작 + 1분마다 poll) — 재부팅에도 살아남음. 기존 `serving.inference_pipeline`(5분 push) 크론도
전역 GCP 설정 대신 로컬 `.env`를 쓰도록 같이 고쳤음(자세한 내용은 `docs/serving_architecture.md`의
"배포(crontab)" 절).
