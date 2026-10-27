# AI/백엔드/프론트엔드 통합 정합성 감사 (2026-10-02)

범위: `AI_Gentle_Viking_RE`(serving), `Back-Gentle-Viking`(`ai-fit`), `Front-Gentle-Viking`(`ai-fit`). 읽기 전용 감사 — 아래 항목은 전부 미수정 상태.

## 치명 (지금 당장 깨져 있음 / 10/6 전 필수 수정)

1. **`/ai/callback`(ONCE 추론 콜백)이 항상 500으로 실패한다.** AI는 ONCE 결과에 `interpretability` 키를 보내지 않는데, 백엔드가 이를 `None`으로 명시 저장 후 `services_report.py`에서 `interp.get("top_features", {})`를 호출 — `dict.get(key, default)`는 키가 존재하면 그 값(`None`)을 그대로 반환하므로 `default={}`가 적용되지 않고 `AttributeError`.
   - `Back-Gentle-Viking/app/services_report.py:120-121`, `app/routes_ai_webhook.py:130`
2. **자동매매의 포지션 추적이 체결 후 전혀 갱신되지 않는다.** `trading_loop`는 매 사이클 `portfolio.positions = {}`로 새로 초기화하고, 주문이 실제로 나간 뒤에도 갱신하는 코드가 단 한 줄도 없음. 4개 전략 클래스 전부가 `portfolio.positions.get(ticker)`로 "이미 보유 중인지"를 판단하므로 이 값이 영원히 비어있다는 뜻은:
   - 매수 조건이 몇 사이클 연속 유지되면 **같은 종목에 반복 매수**가 무한정 가능(`max_weight` 캡이 "계좌의 30%까지"가 아니라 "호출당 30%씩 계속"이 됨)
   - 매도/청산 조건(`current_qty > 0`)이 절대 참이 안 되므로 **자동 청산이 이 경로로는 거의 발생하지 않음**
   - `Back-Gentle-Viking/app/routes_trade.py:179-180` (초기화만 있고 갱신 없음), `backtest/strategies/{ultra_safe,conservative,balanced,aggressive}.py`의 `current_qty` 판단부
   - 이미 구현된 `backtest/engine/risk.py`의 `RiskManager.check_pretrade()`(종목당/1회 주문당 노출 상한, 드로다운 정지)가 실거래 경로(`routes_trade.py`)에는 전혀 연결돼 있지 않음 — 가장 빠른 완화책
3. **종목 바구니(Basket)에 티커 검증이 전혀 없다.** 사용자가 코스닥 종목이나 AI가 전혀 모르는 KOSPI 종목을 자유롭게 등록할 수 있고, 이게 `ai_client.predict()`에 들어가면 AI 커버리지 부재가 "confidence 0 HOLD" 또는(`AI_ALLOW_DUMMY_PREDICTIONS=true`일 때) **완전 무작위 BUY/SELL**로 치환됨. 두 경우 모두 반환 dict에 `is_mock`/`source` 같은 구분 플래그가 없어 실제 AI 신호와 구조적으로 동일한 shape — 호출부(`allocate_portfolio`)가 구분할 방법이 없음.
   - `Back-Gentle-Viking/app/routes_basket.py:13-40`, `app/ai_client.py:65-116`, `app/routes_trade.py:273,498`

## 높음 (조건이 맞으면 바로 터지는 리스크)

4. **프론트가 보내는 `strategy_id: "rsi_reversal"`이 백엔드에 존재하지 않는 값** → 매번 조용히 `conservative`로 대체됨(백엔드 주석이 직접 이 값을 지적). 사용자가 고른 전략이 한 번도 실제로 적용된 적이 없음. `/start` 응답의 `strategies`(fallback 여부) 필드를 프론트가 안 읽어서 사용자는 이 사실을 알 수 없음.
   - `Front-Gentle-Viking/app/portfolio/page.tsx:350-353` ↔ `Back-Gentle-Viking/app/strategy_factory.py:7-39`
5. **`trading_loop` 안에서 보호되지 않은 예외가 나면 asyncio task가 조용히 죽는다.** 예: `min_confidence=0`이고 전부 confidence 0인 BUY 신호가 들어오면 `allocate_portfolio`의 `total_conf == 0`에서 `ZeroDivisionError`. 이 task는 fire-and-forget(`asyncio.create_task`, 아무도 await 안 함)이라 예외가 나면 DB 기록도, 알림도, 재시작도 없이 "자동매매가 멈췄는데 아무도 모르는" 상태가 됨.
   - `Back-Gentle-Viking/app/routes_trade.py:234-464`(보호 범위는 410-439 주문호출뿐), `app/services_allocation.py:47`
6. **같은 사이클 내 크래시 시 그 전에 성공한 주문의 DB 기록이 통째로 사라질 수 있다.** `db.commit()`이 `for ticker in tickers:` 루프 전체가 끝난 뒤 1회만 호출되므로, 티커 A 주문이 KIS에 실제로 들어간 뒤 티커 B 처리 중 예외가 나면 A의 `TradeLog`/`AutoTradeDecision`이 커밋 없이 버려짐 — 실제 체결된 주문이 감사 기록에는 전혀 안 남는 시나리오.
   - `Back-Gentle-Viking/app/routes_trade.py:462-464`
7. **멀티프로세스/레플리카 배포 시 중복 실행 가드가 없다.** `active_tasks`가 프로세스 메모리 내 dict일 뿐이라 워커가 2개 이상이면 같은 유저의 `trading_loop`가 중복으로 뜰 수 있음(단일 프로세스에서는 안전 확인됨). 로컬 드라이런(단일 uvicorn)엔 영향 없지만 추후 GCP 배포 시 반드시 재점검 필요.
   - `Back-Gentle-Viking/app/routes_trade.py:35,636-637`
8. **백엔드가 보내는 `warmup` 필드를 AI가 스키마에 선언하지 않아 매번 무시된다.** AI 쪽에 `/ai/warmup`을 호출하는 코드 자체가 없음 → 자동매매 시작마다 백엔드가 **90초를 그냥 타임아웃으로 날림**(죽은 기능, 매매 자체는 계속 진행).
   - `Back-Gentle-Viking/app/routes_trade.py:195-224` ↔ `AI_Gentle_Viking_RE/serving/api_server.py:140-144`
9. **프론트 "종목 레포트"가 AI가 전혀 모르는 350종목(코스피200+코스닥150) 추천을 검증 없이 받아, 예측 요청에 흘리고 결과를 무조건 "실제 응답"으로 표시.** 백엔드의 이 추천 경로(`rec_report.py`) 자체가 TFT 모델을 호출하지 않는 별도의 Gemini 기반 로직임.
   - `Front-Gentle-Viking/app/ai-report/page.tsx:379-392`, `app/ai-report/predict/page.tsx:236,441` ↔ `Back-Gentle-Viking/app/rec_report.py:350-358`
10. **`stale`(오래된 신호 경고) 플래그가 백엔드 응답엔 있는데 프론트 어디서도 읽지 않는다.** AI 서버가 30분 넘게 멈춰도 사용자는 낡은 BUY/SELL 신호를 최신처럼 계속 봄(매매 판단 로직 자체는 안전 — `AIClient.predict()`가 stale이면 강제 HOLD).
    - `Back-Gentle-Viking/app/ai_client.py:56-59` ↔ `Front-Gentle-Viking` 전체에 `stale` 참조 0건
11. **포트폴리오 페이지의 "1회 분석" 폴링에 `job_id` 게이팅이 처음부터 적용되지 않았다.** `lib/ai/predictionJobStore.ts`에는 있지만 `app/portfolio/page.tsx`의 자체 ONCE 흐름에는 없어서, 콜백 도착 전 이전 예측을 새 결과로 오인할 수 있는 레이스 컨디션이 남아 있음(회귀가 아니라 애초에 미적용).
    - `Front-Gentle-Viking/app/portfolio/page.tsx:362-377`

## 중간

12. `SIGNAL_MAP.get(..., "HOLD")`이 `pred_label`이 0/1/2 범위를 벗어나는 비정상값을 로그 없이 HOLD로 조용히 폴백 — 모델/서빙 버그를 은폐할 수 있음 (`Back-Gentle-Viking/app/routes_ai_webhook.py:77-80`).
13. ONCE 추론 결과는 `model_version`이 `AIPredictionHistory`에 전혀 기록 안 됨(실시간 push만 기록) — 모델 교체 이력 추적에 구멍 (`app/routes_ai_webhook.py:119-162`).
14. AI가 보내는 `job_id`/`user_id`가 `RealtimePayload` 스키마에 없어 조용히 드롭 — push를 job 단위로 추적 불가 (`app/schemas.py:175-177`).
15. `/trade/history`의 `status`(FAILED) 처리가 화면마다 다름 — `AITradeHistory.tsx`는 필터링하지만 `portfolio/page.tsx`는 타입에도 없어 실패 주문이 체결처럼 보일 수 있음.
16. KIS 에러코드별 분기가 없음(레이트리밋 `EGW00133`과 잔고부족을 구분하지 않고 전부 고정 10초 재시도) — 치명적이진 않으나 튜닝 여지 (`app/routes_trade.py:410-439`).
17. `broker.create_order`/`fetch_balance`가 동기 호출인데 `async def trading_loop` 안에서 `await` 없이 실행 — 호출 중 전체 이벤트 루프가 블로킹됨.
18. `run_once`(`app/routes_trade.py:475`) 함수는 현재 아무도 호출하지 않는 죽은 코드인데, `trading_loop`와 전혀 다른(위험한) 방향으로 로직이 drift됨 — 실제 `broker.create_order` 호출 없이 DB에 `status="FILLED"`로 가짜 기록을 남김. 지금은 안전하지만 누군가 나중에 연결하면 바로 사고로 이어질 수 있어 제거 또는 `NotImplementedError` 처리 권장.
19. 여러 유저가 단일 공유 KIS 계정의 전체 잔고를 각자 독립적으로 `total_capital`로 사용 — 동시 사용 시 포지션 사이징이 서로 간섭 가능 (3-D 항목, 1·7과 결합 시 악화).
20. 배분(allocation) 결정의 세부치(weight, persona multiplier)와 실제 KIS 체결가가 저장되지 않음 — "왜 그 비중/가격으로 체결됐는지" 사후 재구성 불가.
21. `_extract_stock_map`(`Back-Gentle-Viking/app/rec_report.py:280-313`)이 로컬 CSV의 `market`(KOSPI200/KOSDAQ150) 컬럼을 읽고도 버림 — KOSDAQ 종목을 걸러낼 수 있는 유일한 소스가 무시되고 있음.
22. `/ai/agreement`가 AI 커버리지 밖 종목에도 `tft_signal`을 `"HOLD"`로 조용히 정규화해 "TFT는 관망 의견"이라는 허위 해석 문장을 생성 (`app/services_report.py:194-243`).
23. DB 스키마(`Basket`, `ValueWatchlist`, `RecommendationReport` 등) 어디에도 "AI가 실제로 커버하는 200종목"과 "리포트 전용 종목"을 구분하는 컬럼/제약이 없음.

## 낮음

24. `AI_SERVER_API_KEY` 기본값이 AI/백엔드 세 곳에서 전부 다름(`"changeme"` / `""` / `"dev-ai-key"`) — 배포 환경 간 설정 누락 시 불일치 위험.
25. 백테스트 `BacktestResponse.total_trades` 필드는 프론트에 백테스트 화면 자체가 없어 현재는 영향 없음(향후 연동 시 전체 스키마 재매핑 필요).
26. 프론트 랜딩 페이지 미리보기에 KOSDAQ 지수가 정적 장식으로 노출돼 "코스닥까지 다루는 서비스"라는 톤 불일치(`app/page.tsx:276-277`, 실제 데이터 호출과 무관).
27. `lib/dashboard/mock.ts`의 미사용 목데이터에 코스닥 종목·ETF가 "인기 종목 TOP30"으로 혼재 — 현재는 죽은 코드지만 재사용 시 바로 위험 재현.

## 확인됨 (안전, 걱정 안 해도 됨)

- confidence 게이트·allocation_map 게이트 자체의 로직은 우회 분기 없이 안전하게 구현됨(단, 치명 #2의 포지션 추적 결함으로 실질적으로 무력화되는 게 핵심 문제).
- `is_stale()`/`AIClient.predict()`의 매매 판단 경로는 stale/누락 신호를 안전하게 강제 HOLD 처리 — 견고함.
- `stop_trading`은 진짜 asyncio task cancellation + cleanup. 가짜 정지가 아님.
- 단일 프로세스 내에서는 `start_trading` 중복 호출 가드가 안전(레이스 컨디션 없음).
- `SIGNAL_MAP`의 한글 3종 + 정수 라벨 3종 커버리지는 AI의 `LABEL_MAP`과 정확히 일치.
- `X-API-Key` 인증은 실제로 호출되는 검증 함수로 구현됨(단순 파라미터 선언이 아님).
- `model_version`은 실시간 조회 API(`GET /ai/predictions*`)까지는 정상적으로 전달됨.
- 프론트의 보호된 라우트는 예외 없이 Bearer 토큰을 부착(`apiFetch`), 공개 라우트도 문제없음.
- 토큰 리프레시 single-flight 처리는 정상 유지(리그레션 없음).
- `/trade/order`(KIS 프록시)의 프론트-백엔드 타입 일치 확인됨.
- 포트폴리오 상태 폴링(15초, RUNNING일 때만)은 정상 게이팅.

## 10/6 로컬 드라이런 기준 권장 우선순위

마감(10/6) 전 반드시: **#1 (ONCE 콜백 500), #2 (포지션 추적 결함 — RiskManager 연결이 가장 빠른 완화책), #3 (바구니 티커 검증)**. 이 셋은 "추론+자동매매+수익률 기록"이라는 핵심 목표 자체를 직접 깨뜨린다. 나머지 높음/중간 항목은 로컬 단일 유저·단일 프로세스 드라이런에서는 발생 확률이 낮은 것(#7 멀티프로세스, #19 멀티유저 등)과, 발생해도 치명적이지 않은 것(#4 전략 fallback, #10 stale 미표시)을 구분해서 여유 되는 대로 처리.
