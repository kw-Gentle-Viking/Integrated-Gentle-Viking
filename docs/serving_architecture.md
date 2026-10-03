# AI 추론/서빙 아키텍처 (2026-10-02 기준)

TFT(Temporal Fusion Transformer) 모델을 백엔드(Back-Gentle-Viking)에 공급하는 `serving/` 모듈의
전체 구조. 확정 모델 자체(레시피/라벨/하이퍼파라미터/성능)는 `/home/user/AI_Gentle_Viking_RE/README.md` 참고 —
이 문서는 "그 모델을 어떻게 실제로 돌려서 백엔드에 전달하는가"만 다룬다.

## 구성 파일

| 파일 | 역할 |
|---|---|
| `serving/model.py` | 모델/피처컬럼 로딩 (공유 싱글턴) |
| `serving/feature_builder.py` | 종목 1개의 60행 인코더 입력 조립 (59일 과거 + "오늘" 1행) |
| `serving/inference.py` | 텐서 조립 + forward pass + softmax -> `pred_label/pred_str/prob_*` |
| `serving/inference_pipeline.py` | 5분 주기 크론 루프: 활성 종목 전체 추론 -> 백엔드 push |
| `serving/api_server.py` | 백엔드가 호출하는 FastAPI 서버 (`/command`, `/health`) |
| `serving/universe_batch.py` | 하루 1회, 전체 200종목 배치 랭킹(실시간 매매 루프와 무관) |

## 모델 로딩 (`model.py`)

- 아키텍처/하이퍼파라미터: `training/champion_config.json` (state_size=32, heads=8, layers=2, dropout≈0.1658 등).
- 가중치: `serving/best_model_state_dict.pt` — 확정된 `v3_structure_nowd_cs` seed0 체크포인트를 그대로 복사해둔
  파일(`TFT_MODEL_PATH` 환경변수로 다른 경로 지정 가능, 미지정 시 이 파일이 기본값).
- `TFT_MODEL_VERSION` 환경변수(또는 파일명)가 매 예측에 붙는 모델 버전 문자열 — 백엔드가 `AIPredictionHistory`/
  실시간 조회 API에 그대로 저장·노출한다.
- 프로세스당 1회만 로드하는 전역 싱글턴(`get_model()`, lock으로 보호) — `api_server.py`와
  `inference_pipeline.py`가 공유.
- `STATIC_CARDINALITIES = [21, 3]` (sector_id 0-20, market_id 0-2)는 `training/run_stage1_search.py`와
  중복 선언(무거운 Optuna/W&B 의존성을 서빙에 끌어오지 않기 위함) — 학습 쪽을 바꾸면 여기도 같이 바꿔야 함.

## 입력 조립 (`feature_builder.py`) — 서빙의 핵심

학습 때 쓰는 `TickerDayDataset(align="today")`의 60행(=59 실제 거래일 + "오늘" 1행)을 그대로 재현한다.

1. **과거 59행**: `stock_db_v2.feature_pool`에서 이미 계산되어 있는 33개 챔피언 컬럼 값을 그대로 읽음(재계산 없음).
2. **"오늘" 1행**: 아직 장이 끝나지 않은 당일 데이터를, 운영 `stock_db`의 실시간 5분봉(`intraday_5min`)으로부터
   조립. 33개 컬럼 전부가 다음 3가지 정책 중 정확히 하나로 분류되며(분류 안 된 컬럼이 있으면 즉시 예외 —
   새 챔피언 컬럼이 조용히 잘못된 기본값으로 서빙되는 걸 막음):
   - **LIVE** (`log_ret`, `disparity_5d/20d/60d`, `rsi_14`, `volatility_20d`, `day_of_week`): 당일 실시간 가격으로
     직접 계산.
   - **FFILL** (sector_* 6개, 매크로 11개): 실시간 피드가 없는 값은 가장 최근 feature_pool 값을 그대로 이어씀
     (학습 때의 결측 처리 정책과 동일).
   - **ZERO_DEFAULT** (`is_dividend`, `is_bonus_issue`, `is_rights_offering`, `is_split`, `is_vi_triggered`):
     당일 발생 여부를 알 수 없는 이벤트 플래그는 0.
3. 당일 첫 틱 가격은 실시간 5분봉의 첫 `open`(없으면 전일 종가) — 과거엔 `encoder_df.iloc[-1]["close"]`를 썼다가
   항상 없는 컬럼이라 조용히 0으로 깔리던 버그가 있었음(이미 수정됨, 모듈 docstring 참고).
4. 히스토리가 59일보다 짧으면(신규 상장 등) 예외를 던진다 — LSTM은 짧은 시퀀스도 조용히 돌아가버려서, 막지
   않으면 이상하지만 "그럴듯한" 예측이 나온다.

학습-서빙 패리티(이 조립이 학습 때 텐서와 수치까지 일치하는지)는 `docs/serving_parity.md`에 별도 검증돼 있음
(33열 전체 일치 확인, float32 rtol/atol=1e-5).

## 추론 (`inference.py`)

- `assemble_model_inputs`: `encoder_df`(60행) -> `{historical_ts_numeric, future_ts_numeric,
  static_feats_categorical}` 텐서(batch=1). static 텐서는 `[batch, num_static]` 2D여야 함(3D로 만들면 조용히
  틀린 shape으로 forward가 돌아감 — 실제로 한 번 겪은 버그).
- `run_inference`: forward pass -> `class_logits` -> softmax -> `{pred_label(0/1/2), pred_str(매수/관망/매도),
  prob_buy, prob_hold, prob_sell}`.
- 모델/전처리 관련 가정은 전부 `feature_builder`가 끝낸 뒤라 이 모듈은 순수 forward pass만 담당.

## 실시간 서빙 루프 (`api_server.py` + `inference_pipeline.py`)

백엔드와의 연동 방식 — 두 가지 경로:

1. **5분 주기 push (크론, `inference_pipeline.py`)**: `serving/active_tickers.json`에 등록된 전체 활성 종목에
   대해 추론을 돌리고, 결과를 백엔드 `POST {BACKEND_WEBHOOK_URL}/ai/realtime`로 push (`X-API-Key` 헤더 인증,
   `AI_SERVER_API_KEY` 공유 비밀값 — AI/백엔드 양쪽 기본값 통일됨, 2026-10-02). 실패한 종목은 결과 배열에서
   그냥 빠짐(백엔드가 요청한 개수와 받은 개수를 대조하지는 않음).
2. **명령 기반 (`api_server.py`의 `POST /command`)**: 백엔드가 자동매매 시작/중지/1회 실행을 요청하는
   경로. 단, 백엔드(`app/routes_trade.py`)는 이 엔드포인트를 직접 호출하지 않고 자기 메모리
   `command_queue`에 쌓아두기만 한다(`GET /ai/commands/pending`로 조회 가능) — 그 큐를 이 `/command`로
   그대로 중계하는 게 `serving/poll_commands.py`다(크론, 1분마다 `--once`). 이 폴러가 없으면 `/trade/once`
   등으로 넣은 요청이 큐에 쌓인 채 아무도 안 가져가서 조용히 아무 일도 안 일어난다 — 운영 원본
   `/home/user/poll_commands.py`는 운영 `api_server.py`를 직접 import하는 방식이라 이 프로젝트와는
   무관했고, 실제로 서버를 다 띄워서 end-to-end로 돌려보기 전까지는 이 연결 자체가 없다는 걸 몰랐다
   (2026-10-03 실제 테스트로 발견, `serving/poll_commands.py` 신설로 해결).
   - `START`: `req.tickers`를 `active_tickers.json`에 등록(user_id별로 쌓이고 합쳐짐) — 이후 5분 주기 push가
     이 종목들을 포함.
   - `STOP`: 해당 user_id 등록 제거.
   - `ONCE`: 백그라운드 태스크로 즉시 추론 후 `req.callback_url`(백엔드의 `/ai/callback`)로 결과 POST. ONCE
     결과에는 `interpretability` 키가 없음(5분 push와 다른 점 — 과거 백엔드가 이를 몰라서 ONCE 콜백이 항상
     500 나던 버그가 있었음, 2026-10-02 수정).
   - `CommandRequest`에 `warmup` 필드가 없어서, 백엔드가 자동매매 시작 시 함께 보내는 "과거 5분봉 워밍업
     요청"은 조용히 무시된다 — AI 서버도 KIS도 여러 날치 5분봉 과거 데이터를 가진 소스가 없어서(KIS REST는
     당일 1분봉만 제공), 이건 필드 연결이 아니라 데이터 소스 자체가 없는 구조적 제약(2026-10-02 확인, 백엔드
     쪽 타임아웃을 90초->5초로 줄여 낭비만 줄임).
   - `/health`: 헬스체크.

인증 비대칭에 주의: 실시간 시세 조회(`get_current_price` 등)는 `KIS_MOCK` 설정과 무관하게 항상
`REAL_BASE_URL`/실계좌 토큰을 쓴다(모의계좌는 시세 품질이 떨어지는 KIS 정책 때문) — 매매 자체는 모의계좌로
나가지만 시세는 실계좌 인증으로 조회한다는 뜻.

## 배포 (crontab, 2026-10-03)

세 가지 모두 `crontab -l`에 등록돼 있고, 전역 crontab 맨 위에 선언된 `BACKEND_WEBHOOK_URL`(GCP,
`34.64.252.181:8000`)/`AI_SERVER_API_KEY`(`dev-ai-key`) — 운영 파이프라인·이 프로젝트가 GCP로
배포됐을 때를 위한 값 — 를 전혀 쓰지 않고 각자 이 레포의 `.env`(`set -a && . ./.env && set +a`)를
명시적으로 source해서 **로컬 백엔드(`localhost:8000`)와 프로젝트 전용 키로만** 통신한다:

- `*/5 * * * 1-5 serving.inference_pipeline` — 5분 push.
- `@reboot` + `*/5 * * * * (헬스체크)` — `serving.api_server:app`(8001)을 띄우고, 죽어 있으면 재시작.
- `* * * * * serving.poll_commands --once` — 백엔드 커맨드 큐 중계.

**GCP로 옮겨갈 때(지금 계획엔 없지만, 나중에 그렇게 되면)**: 이 코드/스크립트 자체는 손댈 필요가 없다 —
`BACKEND_WEBHOOK_URL`/`AI_SERVER_API_KEY`가 전부 환경변수로 읽히므로, GCP 쪽 머신의 crontab(또는
systemd 등)에서 그 환경변수를 GCP 백엔드 주소/키로 설정하기만 하면 된다. 지금 이 로컬 crontab 파일을
그대로 옮겨 쓰는 게 아니라 GCP 쪽에 별도로 등록해야 하는 것도 그 때 할 일 — 지금 로컬 엔트리를
GCP 값으로 바꿔두는 것은 아무 의미가 없다(로컬 드라이런이 깨질 뿐).

## 배치 랭킹 (`universe_batch.py`) — 실시간 루프와 별개

- 하루 1회(권장: 장 마감 후, feature_pool 일배치가 끝난 저녁), 전체 `ticker_universe`(200종목)에 대해
  `build_encoder_df_for_ticker` -> `run_inference`를 순차 실행.
- `score = p_buy - p_sell`, 그날 하루 안에서 cross-sectional rank percentile(`rank_pct`, 0~1, 1이 가장
  매수 우세)로 변환.
- 출력: `training/artifacts/universe_scores_<asof>.json` (scores/excluded/failed + 모델 경로·버전·git
  commit 메타데이터).
- CPU로 동작, 운영 DB에는 SELECT만(쓰기 없음). 아직 crontab에 설치되지는 않음(`docs/universe_batch.md` 참고).
- 실시간 자동매매 신호(`/ai/realtime`)와는 완전히 독립된 산출물 — 지금은 리서치/모니터링용.

## AI가 실제로 커버하는 종목

`ticker_universe` 테이블(= `stock_db_v2`, 코스피 시가총액 상위 200종목, `is_kospi` 플래그) 딱 200개뿐이다.
코스닥은 전혀 포함되지 않는다. 백엔드의 "종목 레포트/추천"(`rec_report.py`, Gemini 기반, 코스피200+코스닥150=
350종목 풀)은 이 AI와 완전히 분리된 별도 시스템이며 TFT를 호출하지 않는다 — 두 시스템을 혼동하지 않도록
백엔드 쪽에 `app/ai_universe.py`(이 `ticker_universe` 테이블을 그대로 읽는 조회 함수)를 두어 Basket 등록/
`/ai/agreement` 비교 시점에 구분한다(2026-10-02, `docs/integration_audit_2026-10-02.md` 참고).

## 알려진 설계상 제약

- **5분봉 워밍업 데이터 없음**: 위 설명대로, 전략이 요구하는 240개(=20시간, 여러 거래일치) 5분봉 히스토리를
  채워줄 데이터 소스가 AI/KIS 어느 쪽에도 없다. 전략들은 라이브 데이터가 실시간으로 쌓이면서 자연스럽게
  (수 시간~하루 단위로) 워밍업된다 — 즉시 전체 워밍업은 구조적으로 불가능.
- **ONCE 경로는 interpretability를 보내지 않음**: 피처 중요도/어텐션 기반 리포트 생성은 5분 push 경로에만
  구현돼 있고 ONCE는 아직 없음(의도적 스코프 축소, 버그 아님 — 단, 백엔드가 이 차이를 전제해야 함).
- **실패한 종목은 결과 배열에서 조용히 빠짐**: 특정 종목 추론이 계속 실패해도 백엔드가 "요청한 개수 대비
  부족"을 감지하는 로직이 없음(개선 여지, `docs/integration_audit_2026-10-02.md` 항목 참고).

## AI 데이터(stock_db_v2) 일별 갱신 (2026-10-06부터)

운영 수집 크론(`collector_*`, `build_batch_features`)은 **운영 `stock_db`(350종목)** 만 채운다. AI가 읽는
`stock_db_v2`(200종목, `feature_pool` 포함)는 별도로 갱신해야 하며, 평일 16:45에
`scripts/daily_v2_refresh.sh`가 다음을 순서대로 실행한다(2026-10-06 이전에는 아무 것도 하지 않음):

1. `data_collection/sync_v2_from_prod.py` — 운영 `stock_db`에서 AI 200종목 공통 테이블 9개를 증분 복사
2. `data_collection/run_leverage_backfill.py` — v2 전용 `leverage_daily` 갱신 (VI 이벤트는 KIS에 과거 조회 API가 없어 제외)
3. `features/build_features.py --start <90일 전> --end <오늘>` — `feature_pool` 재빌드(롤링 피처 lookback 확보)

로그: `serving/daily_v2_refresh.log`. 이 갱신이 멈추면 AI 추론이 "stale history"로 전부 실패하므로 가장 먼저 볼 것.
