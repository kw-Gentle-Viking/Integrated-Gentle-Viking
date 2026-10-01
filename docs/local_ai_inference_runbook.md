# Local AI Inference Runbook

마지막 갱신: 2026-10-01 (10/6 로컬 드라이런 준비 작업 중 전면 재작성 — 이전 버전은 `docker-compose.yml`로 AI DB(`stock_db` dump)를 백엔드가 직접 읽는 낡은 구조를 전제로 했고, 팀원 개인 경로(`/home/jeongeun/...`)가 섞여 있었다. 지금 구조는 아래 "현재 아키텍처"가 유일하게 맞는 설명이다.)

## 10/6(화) 장 시작 전 체크리스트

1. **백엔드 기동** — `trading` DB가 떠 있는 로컬 postgres에 붙는지 확인 후 uvicorn 기동 (아래 "백엔드 기동" 참고). `GET /health` 200 확인.
2. **모델 체크포인트 확정 여부 확인** — `serving/model.py`가 가리키는 `TFT_MODEL_PATH`/`TFT_MODEL_VERSION`이 최종 모델인지 확인 (2026-10-01 기준 아직 미확정 — 바뀌지 않았으면 장중 추론을 돌리지 말 것).
3. **`active_tickers.json` 등록** — 모델이 확정된 뒤에만 `serving/active_tickers.json`에 운용할 티커(005930/000660 등)를 등록. 미등록 상태면 crontab의 `serving.inference_pipeline`이 "활성 종목 없음"으로 스킵만 하고 아무 일도 하지 않는다(안전한 기본값).
4. **AI → 백엔드 push 배선 재확인** — `serving/inference_pipeline.py`의 `push_results()`에 알려진 버그가 있음 (아래 "알려진 문제" 참고). 고치지 않았다면 crontab이 돌아도 백엔드에 결과가 쌓이지 않는다.
5. **프론트 기동** — `npm run dev`, `http://localhost:3000`.
6. **로그인** — 아래 "로컬 테스트 계정"으로 로그인하거나 신규 가입.
7. **자동매매 시작은 사용자가 프론트/API로 명시적으로 `/trade/start` 호출할 때만** — 이 러너가 미리 실행해두지 않았음 (범위 밖, KIS 실호출 위험 때문).

## 현재 아키텍처 (2026-10-01 기준 실제 구조)

- **앱 DB**: 네이티브 postgres(5432, docker 아님) 안의 `trading` 데이터베이스. `stock_user` 소유, 운영 `stock_db`/AI `stock_db_v2`와 완전히 분리됨. `DATABASE_URL`로 지정.
- **market DB(TimescaleDB, 시세/백테스트용)**: 이번 범위에서 만들지 않음. `MARKET_DB_URL`은 `.env.example` 기본값(가리키는 postgres:5433 인스턴스가 실제로 없음)을 그대로 둬도 백엔드 기동은 막히지 않는다 — `app/db.py`의 `create_engine(MARKET_DB_URL, ...)`은 SQLAlchemy 특성상 실제 커넥션을 맺지 않고 지연 생성되고, `market_engine`을 실제로 쓰는 곳(`app/routes_market.py`, `backtest/db.py`)은 전부 요청이 들어왔을 때만 접속을 시도하는 지연 경로다. 즉 `/market/*`을 호출하기 전까지는 영향 없음. 차트/백테스트 UI를 쓸 때 가서 별도로 구성.
- **ClickHouse(백테스트 결과 저장)**: 마찬가지로 이번 범위 밖, 미설치. 관련 함수(`save_to_clickhouse` 등)도 호출 시에만 실패한다.
- **redis**: 코드에서 실제로 쓰는 곳 없음(`docker-compose.yml`에만 선언). 설치/실행 불필요.
- **AI ↔ 백엔드 연동은 "AI가 push" 구조다. 백엔드가 AI DB를 직접 읽지 않는다.**
  - AI 쪽(`AI_Gentle_Viking_RE` worktree `serving/inference_pipeline.py`, crontab에 이미 등록됨, 장중 5분 주기)이 자체 DB(`stock_db_v2`, 운영 `stock_db`)에서 피처를 만들고 추론한 뒤, 결과를 `POST {BACKEND_WEBHOOK_URL}/ai/realtime`로 백엔드에 밀어넣는다(X-API-Key 헤더로 인증).
  - 백엔드는 그 결과를 메모리(`realtime_predictions`)에 보관하면서 동시에 postgres `trading.ai_prediction_history` 테이블에 영구 저장한다(`app/ai_history.py`).
  - 사용자가 보는 추천 리포트("ONCE")는 반대로 백엔드가 AI에 커맨드를 큐에 쌓아두고(`POST /ai/commands/push` 또는 `/trade/once`), AI 쪽 `poll_commands.py`가 주기적으로 `GET /ai/commands/pending`을 폴링해서 가져간 뒤 `POST /ai/callback`으로 결과를 돌려준다.
  - 자동매매 루프(`/trade/start` → `app/routes_trade.py`의 `trading_loop`)는 시세를 DB가 아니라 KIS websocket 실시간가(`KISWebSocket.get_price`)로 받는다. 따라서 market-db가 없어도 자동매매 자체는 동작 가능한 구조다(단, KIS 앱키가 있어야 websocket이 붙는다).

## 포트 / DB / 환경변수 (이름만 — 실제 값은 각 `.env`에만 있음)

| 구성요소 | 위치 | 비고 |
|---|---|---|
| 백엔드 API | `http://localhost:8000` | uvicorn |
| 프론트 | `http://localhost:3000` | `next dev` |
| 앱 DB | postgres `localhost:5432/trading` | `DATABASE_URL` |
| 백엔드 `.env` | `Back-Gentle-Viking/.env` (gitignored) | `.env.example` 기반 |
| AI worktree `.env` | `AI_Gentle_Viking_RE/.worktrees/ai-model-redesign/.env` (gitignored) | 기존 DB/KIS/DART 등 키에 `BACKEND_WEBHOOK_URL`, `AI_SERVER_API_KEY` 추가함 |
| 프론트 `.env.local` | `Front-Gentle-Viking/.env.local` (gitignored) | `NEXT_PUBLIC_API_URL=http://localhost:8000`만 |

공유 키(값은 각 `.env`에만 있고 여기 적지 않음 — 양쪽 `.env`에 **동일한 값**을 넣는다는 뜻으로만 읽을 것):

- `AI_SERVER_API_KEY` — 백엔드 `.env`와 AI worktree `.env` 양쪽에 동일한 값. AI → 백엔드 push(`/ai/realtime`, `/ai/callback`, `/ai/warmup`) 인증.
- `AI_COMMAND_API_KEY` — 백엔드 `.env`에만 존재. `poll_commands.py`가 읽는 `GCP_BACKEND_API_KEY` 환경변수와 같은 값을 넣어야 `/ai/commands/pending` Bearer 인증이 통과한다. (이번 러너에서는 `AI_SERVER_API_KEY`와 동일한 값으로 통일해서 넣었다 — 둘 다 백엔드 쪽 `service_key_auth`가 허용하는 값이라 실제로는 하나만 맞아도 되지만, `poll_commands.py` 실행 환경에는 `GCP_BACKEND_API_KEY`라는 이름으로 넣어야 한다.)

`poll_commands.py`를 실제로 실행할 때 필요한 환경변수(이번 작업에서는 **실행하지 않았음** — 인증 경로만 curl로 검증):

```
BACKEND_WEBHOOK_URL=http://localhost:8000
GCP_BACKEND_API_KEY=<백엔드 .env의 AI_COMMAND_API_KEY와 동일한 값>
AI_SERVER_API_KEY=<백엔드 .env의 AI_SERVER_API_KEY와 동일한 값>
```

## 사용자가 직접 채워야 하는 값 (절대 대신 추측/생성하지 않음)

| 변수 | 위치 | 없으면 안 되는 기능 |
|---|---|---|
| `KIS_APP_KEY` / `KIS_APP_SECRET` / `KIS_ACC_NO` | 백엔드 `.env` | KIS 실시간 시세(websocket), 잔고 조회, 모의/실계좌 주문. 비어 있으면 `get_broker()`가 조용히 `None`을 반환하고 잔고는 로컬 기본값(1000만원)으로 대체됨 — 기동은 막히지 않지만 실제 매매 루프는 "시세 없음"으로 전부 SKIP 처리된다. |
| `GEMINI_API_KEY` | 백엔드 `.env` | 추천 리포트(ONCE 콜백 후 `generate_report`)의 Gemini 기반 해설 생성. |
| `NAVER_CLIENT_ID` / `NAVER_CLIENT_SECRET` | 백엔드 `.env` | 네이버 관련 리포트/뉴스 조회 기능(해당 라우트 호출 시에만 실패). |
| `DART_API_KEY` | 백엔드 `.env` | 공시 기반 리포트 생성(`app/rec_report.py`, `OpenDartReader` 경유). AI worktree `.env`에는 이미 값이 들어있지만(운영 공유 키) **백엔드 쪽에는 의도적으로 넣지 않았다** — 사용자가 백엔드용으로 별도 결정/입력할 것. |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | 백엔드 `.env` | Google 로그인. 이번 범위에서는 이메일/비번 로그인으로 충분해서 비워둠. |

## 알려진 문제 (고치지 않고 서술만 — 사용자 승인 후 조치)

**`AI_Gentle_Viking_RE/.worktrees/ai-model-redesign/serving/inference_pipeline.py`의 `push_results()`가 실제로는 백엔드에 꽂히지 않는다.**

- crontab에는 이미 `*/5 * * * 1-5 ... -m serving.inference_pipeline`이 등록되어 있어 장중이면 돈다(모델 체크포인트가 확정되지 않았어도, `active_tickers.json`이 비어 있는 한 "활성 종목 없음"으로 스킵만 하니 당장 위험하진 않다).
- 문제는 `push_results()`가 `requests.post(webhook_url, json=payload, ...)`로 **`BACKEND_WEBHOOK_URL`을 그대로** POST 대상으로 쓰고, **X-API-Key 헤더를 전혀 보내지 않는다**는 점이다.
  - 백엔드의 `/ai/realtime`은 `BACKEND_WEBHOOK_URL` + `/ai/realtime` 경로에 있고 `X-API-Key` 헤더로 인증한다(`app/routes_ai_webhook.py`의 `verify_api_key`).
  - 즉 `BACKEND_WEBHOOK_URL=http://localhost:8000`으로만 설정하면 이 함수는 `http://localhost:8000`(루트)에 키 없이 POST를 보내게 되어 404(또는 인증 누락)로 실패한다.
  - 같은 프로젝트의 원본 스크립트(`/home/user/push_realtime_results.py`)는 이 두 가지(`f"{BACKEND_WEBHOOK_URL}/ai/realtime"` 경로 + `X-API-Key` 헤더)를 올바르게 하고 있어서 비교하면 바로 보인다.
- 이번 작업에서 이 파일은 지시대로 건드리지 않았다. **모델이 확정돼서 crontab 추론이 실제로 쓸모 있어지기 전에, `push_results()`에 경로 접미사(`/ai/realtime`)와 `X-API-Key` 헤더 추가가 필요하다.**
- 대신 이번 검증(아래 "AI → 백엔드 push 경로 검증")은 `push_results()`를 거치지 않고, 올바른 경로+헤더로 직접 합성 payload를 쏴서 백엔드 수신/저장 쪽만 증명했다. 백엔드 쪽은 정상.

## 로컬 테스트 계정 (만들어두고 삭제하지 않음 — `trading` DB, 운영 아님)

- 이메일: `dryrun-1006@example.com`
- 비밀번호: `DryRun1006!`
- 바구니: `005930`(삼성전자), `000660`(SK하이닉스) 등록됨.
- 10/6에 이 계정으로 로그인해서 이어서 쓸 수 있다.

## 백엔드 기동

```bash
cd /home/user/team_repos/Back-Gentle-Viking
<venv>/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

파이썬 의존성: `requirements.txt` 그대로 설치하되 **`OpenDartReader==0.2.3`은 PyPI 휠 자체가 깨져 있어(패키지 파일들이 `OpenDartReader/` 하위가 아니라 site-packages 루트에 바로 풀리고, `python_requires>=3.13`으로 3.11을 막음) 설치/임포트가 안 된다.** `OpenDartReader==0.2.2`로 대체 설치하면 정상 임포트된다(API는 동일). 또한 `passlib==1.7.4` + 최신 `bcrypt`(5.x) 조합은 72바이트 이상 자체 셀프테스트 과정에서 깨지므로(`ValueError: password cannot be longer than 72 bytes`) `bcrypt==4.0.1`로 고정 설치할 것 — 이 두 가지는 `requirements.txt`에 핀이 없어서 매번 최신이 깔리니 재현 시 주의.

기동 후 확인:

```bash
curl http://localhost:8000/health   # {"ok":true}
```

## 프론트 기동

```bash
cd /home/user/team_repos/Front-Gentle-Viking
npm run dev
```

`.env.local`의 `NEXT_PUBLIC_API_URL=http://localhost:8000`만 있으면 됨. UI 변경 없음.

## AI → 백엔드 push 경로 검증 (이번에 한 방식)

실제 모델 추론 없이, `PredictionResult` 스키마에 맞는 합성 데이터를 올바른 경로+헤더로 직접 POST:

```bash
curl -X POST http://localhost:8000/ai/realtime \
  -H "Content-Type: application/json" -H "X-API-Key: <AI_SERVER_API_KEY>" \
  -d '{"inference_at":"...","results":[{"ticker":"005930","trade_datetime":"...","pred_label":0,"pred_str":"BUY","prob_buy":0.7,"prob_hold":0.2,"prob_sell":0.1,"model_version":"..."}]}'
```

`trading.ai_prediction_history` 테이블에 실제로 쌓이는지 직접 확인:

```bash
psql -U stock_user -d trading -c "SELECT ticker, trade_datetime, signal, model_version FROM ai_prediction_history ORDER BY id DESC LIMIT 5;"
```

## poll_commands.py 인증 경로 검증 (실제 실행은 하지 않음)

```bash
curl http://localhost:8000/ai/commands/pending -H "Authorization: Bearer <AI_COMMAND_API_KEY>"
```

200과 `{"commands": [...]}`가 와야 한다(비어 있어도 OK). 키 없음/오답이면 401.

## trade 조회 (10/6에 실제 기록이 쌓일 창구)

```bash
curl http://localhost:8000/trade/history -H "Authorization: Bearer <access_token>"
curl http://localhost:8000/trade/decisions -H "Authorization: Bearer <access_token>"
```

로그인 안 된 상태(`LOCAL_DEMO_MODE=0`)에서는 access_token이 반드시 필요하다 — `/users` 가입 후 `/auth/login`으로 발급.
