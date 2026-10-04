# 실행 계획 (2026-10-04 정리)

범위: AI 추론 · 백엔드 · 데이터 수집. 로그인·리포트는 제외(서비스 단계에서 확인).
대상 종목: 삼성전자(005930), SK하이닉스(000660). 추론은 5분 간격.

---

## 1. 10월 6일부터 장 중 추론 결과와 체결 수익률 데이터를 확보하기 위해 할 것

### 이미 완료된 것
- 추론 대상 등록: `serving/active_tickers.json` = 005930, 000660 (상한 5종목 코드 적용됨)
- 실시간 5분 추론 크론: `inference_pipeline`, 평일 09:00~15:30 (로컬 `.env` 사용)
- 명령 중계 크론: `poll_commands`, 매분 (자동매매 START/STOP/ONCE용)
- 백엔드 서버(로컬 8000), AI 서버(8001) 기동 상태
- 운영 수집 대상: `/home/user/active_tickers.json` = 005930, 000660
- AI 데이터 일별 갱신 크론: 10/6부터 평일 16:45 (`scripts/daily_v2_refresh.sh`)
- 워밍업: 운영 5분봉 이력으로 전략 240봉 채움 (백엔드 `f71db83`, 로컬 커밋)

### 아직 필요한 것 (우선순위 순)
1. **KIS 계좌 불일치 해결 (필수)** — 체결 수익률을 만들려면 실제 주문이 나가야 하고, 주문과 잔고 조회는 지금 `INVALID_CHECK_ACNO`로 막혀 있음. 일반 모의투자 키와 계좌번호를 맞춰 주거나, 상시대회 키로 통일해야 함.
2. **10/6 장 시작 전 점검**
   - 백엔드(8000), AI 서버(8001), 프론트(3000) 기동 확인 (tmux 프로세스는 재부팅 시 사라짐)
   - `crontab -l`에 inference_pipeline, poll_commands, 헬스체크, daily_v2_refresh 항목 확인
   - `serving/active_tickers.json`과 `/home/user/active_tickers.json` 종목 일치 확인
3. **장 중 확인**
   - 09:05 이후 `serving/pipeline.log`에 종목별 추론 기록 확인
   - 백엔드 `ai_prediction_history`에 5분마다 행이 쌓이는지 확인
   - `intraday_5min`에 005930, 000660 5분봉이 쌓이는지 확인 (수집기 `realtime.log`)
   - 자동매매 시작 후 `trade_logs` / `auto_trade_decisions`에 판단·주문 기록이 남는지 확인
4. **체결 가격 기록 보완 (수익률 정확도)** — 현재 `TradeLog.price`는 결정 시점 현재가이고, KIS가 실제로 체결한 가격은 저장하지 않음. 수익률 정확도를 위해 주문 응답의 체결가/주문번호를 저장하도록 수정 필요.
5. **수익률 계산 스크립트** — 쌓인 추론(BUY/HOLD/SELL)과 체결, 익일 종가를 맞춰 한 달 성과 산출. 데이터가 어느 정도 쌓인 뒤 작성해도 됨 (기록은 지금부터 누적됨).

### 10/6 이후 한 달간 확인할 것
- 수집 실패 건수, 추론 누락 건수 (`realtime.log`, `pipeline.log`)
- `daily_v2_refresh.log`에서 feature_pool 갱신 실패 여부
- 워밍업 상태에서 전략 신호 발생 여부

---

## 2. 백엔드를 GCP에 올리고 전체 시스템이 물 흐르듯 작동하기 위해 할 것

### 결정이 필요한 것
- **AI 추론 서버와 수집·DB를 로컬에 둘지, GCP로 옮길지**. 현재 데이터(`stock_db`, `stock_db_v2`)와 수집 크론은 모두 로컬에 있음. 백엔드만 GCP로 옮기면 AI→백엔드 통신이 외부 네트워크를 타게 됨.
- **백엔드 배포 브랜치**: `ai-fit`(로컬에만 있음, 여러 커밋 미푸시)을 배포할지, 팀 레포 쪽 브랜치(`feature/gcp-deploy` 등)에 합칠지.

### 백엔드 배포
1. GCP VM 준비 (Python 환경, Postgres 또는 DB 접속 경로)
2. `.env` 값을 GCP용으로 분리하고 시크릿은 Secret Manager 또는 VM 환경변수로 관리
   - `APP_ENV=production`이면 개발용 기본 키/데모 모드로는 기동 거부됨 (이미 구현됨)
   - `ENABLE_DEBUG_ENDPOINTS=false` 유지
3. 프로세스 관리: systemd 등으로 uvicorn 상시 기동, 재부팅 후 자동 시작
4. 외부 노출: HTTPS(또는 최소한 IP 제한), CORS에 프론트 도메인만 허용

### AI ↔ 백엔드 연결 (AI를 로컬에 둘 경우)
- AI의 `BACKEND_WEBHOOK_URL`을 GCP 백엔드 주소로 변경 (로컬 크론의 `.env` 값 교체)
- 양쪽 `AI_SERVER_API_KEY` 일치 (GCP 백엔드 env와 AI `.env`)
- `poll_commands`가 GCP 큐를 폴링하도록 `BACKEND_WEBHOOK_URL` 변경
- 워밍업용 `PROD_STOCK_DB_DSN`: GCP 백엔드가 운영 `stock_db`에 접근해야 함 → 접근 경로(DB 공개 vs 워밍업 데이터를 AI가 push) 결정 필요

### KIS 접속
- GCP VM의 외부 IP가 KIS에서 허용되는지 확인 (IP 등록 필요 여부)
- 계좌 불일치 해결(1번 항목)이 선행되어야 함

### 프론트
- `NEXT_PUBLIC_API_URL`을 GCP 백엔드 주소로 변경 후 빌드·배포
- Google 로그인을 쓸 경우 `FRONTEND_GOOGLE_CALLBACK_URL`과 Google OAuth 리디렉션 설정 필요 (서비스 단계)

### 전체 흐름 검증 (GCP 배포 후)
1. 로그인 → 바구니 추가(상한 5, AI 커버리지 확인) → `/trade/start`
2. START가 AI 서버 등록까지 전달되는지 (poller 로그)
3. 5분 추론 push가 GCP 백엔드 `ai_prediction_history`에 쌓이는지
4. 자동매매 한 사이클이 돌고 주문/판단 기록이 남는지
5. `/trade/stop` 후 AI 등록이 해제되는지
6. ONCE 경로: 프론트 "1회 분석" → 콜백 → 결과 조회

### 알려진 제약 (배포 전 정리)
- 멀티프로세스 실행 시 자동매매 중복 실행 가드 없음 → 단일 워커로 운영
- KIS 에러코드별 재시도 세분화 없음
- 5분봉 워밍업은 운영 DB 접근이 되어야 동작

---

## 체크리스트 요약
| 구분 | 항목 | 상태 |
|---|---|---|
| 10/6 | KIS 계좌/키 매칭 | 미해결 (필수) |
| 10/6 | 추론·수집·일별 갱신 크론 | 완료 |
| 10/6 | 체결 가격 저장 | 미구현 |
| GCP | 백엔드 배포 방식·브랜치 결정 | 미결정 |
| GCP | AI↔백엔드 URL/키 전환 | 미실행 |
| GCP | 운영 DB 접근 경로 | 미결정 |
| GCP | KIS IP 등록 | 확인 필요 |
