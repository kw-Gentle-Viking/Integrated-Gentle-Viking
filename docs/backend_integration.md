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
- 자동매매 루프가 체결 후 `Portfolio`를 갱신하지 않음 → SELL이 항상 "보유 없음"으로 걸러지고 5분마다 같은 종목 재매수 가능.
- `RiskManager`(낙폭 정지·주문 한도)는 백테스트에서만 사용, 실시간 루프에는 없음.
- 엔진: 종가 동시 체결, `np.random` 지정가 체결(시드 없음), 소수점 주식.
- `/backtest/strategies`에 나오는 aggressive/balanced/conservative/ultra_safe 는 `BacktestService._load_strategy`에 없음("Unknown strategy").
- 신호 정의: 절대 확률 임계값은 기간 간 분포가 달라 불안정(OOT 점수가 val 분포 밖). 매매에 쓰려면 종목 간 일별 순위(200종목 배치 추론)가 필요.

## 2026-09-28 추가 수정 (Back `ai-fit` d24279e~, Front `ai-fit` 4fd1017)
Back: `/trade/history` status 포함, 전략 대체 사실을 `/trade/start` 응답 `strategies`로 노출, 에러 detail 문자열화, Google 콜백 리다이렉트(`FRONTEND_GOOGLE_CALLBACK_URL`),
인증 공백 수정(`/users/{id}` 본인만, `/users` 목록 `ENABLE_DEBUG_ENDPOINTS`, `/account/assets`·`/ai/agreement` 로그인, `/ai/commands/*`·`POST /prices` 서비스 키),
`APP_ENV=production` 이면 dev 시크릿/`LOCAL_DEMO_MODE`로 기동 거부. AI 폴러(`poll_commands.py`)는 `GCP_BACKEND_API_KEY`를 Bearer로 보내므로 백엔드 `AI_COMMAND_API_KEY`에 같은 값을 설정.
Front(로직만, UI 무변경): 새로고침이 `job_id`(GET /ai/once/{job_id})로 새 결과 도착을 확인한 뒤에만 채택, `total_capital`을 '1주 가격 합'으로 보내지 않음(서버가 예수금 조회), FAILED 주문은 체결 내역에서 제외, 토큰 갱신 single-flight.
