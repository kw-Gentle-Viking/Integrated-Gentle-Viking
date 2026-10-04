# AI_Gentle_Viking_RE

KOSPI 상위 200 종목 대상, 익일 매수/관망/매도 3분류 예측 AI (TFT). 백엔드(`Back-Gentle-Viking`)·프론트엔드
(`Front-Gentle-Viking`)는 별도 레포로 관리됨.

## 확정 모델 (E4 최종 후보)

### 식별 정보

| | |
|---|---|
| Recipe | `v3_structure_nowd_cs` (E4), seed 0 |
| Checkpoint | `training/artifacts/checkpoints/e4-v3_structure_nowd_cs.pt` |
| Code commit | `ab5e52e` |
| Data version | `adj1` |
| Results record | `training/artifacts/e4_results.json["v3_structure_nowd_cs"]` |

### 라벨

Cross-sectional quantile 라벨(`cs`). 매 거래일, `next_day_return`이 유효한 종목들 중 상위 30% -> BUY(0),
하위 30% -> SELL(2), 나머지 -> HOLD(1). 평균 순위 백분위 `p = (rank - 0.5)/n` 기준 `p >= 0.7`면 BUY,
`p <= 0.3`이면 SELL(동률 수익률은 라벨 공유). 해당일 유효 종목이 20개 미만이면 라벨 NaN. Glitch 행
(`|next_day_return| > 0.31`)은 랭크 계산 전에 train 프레임에서 마스킹.

### 아키텍처 / 하이퍼파라미터

| | |
|---|---|
| 모델 | TFT (`tft-torch` fork), 3-class `class_logits` head |
| state_size | 32 |
| attention_heads | 8 |
| lstm_layers | 2 |
| dropout | 0.16577574724588015 |
| lr | 0.0002905890080008017 |
| weight_decay | 0.0 (Adam) |
| batch_size | 128 |
| encoder_len | 60 |
| align | `today` |
| seed | 0 |
| static ids (`sector_id`, `market_id`) | 그대로 유지(constant 아님) |
| 입력 컬럼 | 전처리 없는 챔피언 원본 33컬럼 (`log_ret`, `disparity_5d/20d/60d`, `rsi_14`, `volatility_20d`, `sector_ret_1d/5d/20d`, `sector_ma_ratio_20d`, `sector_volatility`, `sector_volume_ratio`, `is_dividend`, `is_bonus_issue`, `is_rights_offering`, `is_split`, `day_of_week`, `lev_total_aum`, `lev_aum_to_mktcap`, `est_rebalancing_flow`, `is_vi_triggered`, `vi_count_recent5d`, `kospi_ret`, `kosdaq_ret`, `snp500_ret`, `nasdaq_ret`, `phlx_semi_ret`, `vix_chg`, `usd_krw_chg`, `us_10y_yield_chg`, `rate_spread_us_kr`, `wti_ret`, `gold_ret`) |

### 학습

| | |
|---|---|
| Train samples | 234,226건 (glitch-masked 0건) |
| Class weights (buy/hold/sell) | 1.110 / 0.834 / 1.111 |
| Selection metric | val 2024 timing IC |
| Patience | 4 |
| 종료 | early_stopping, epoch 9 (best epoch 5) |

### 성능 (라벨 비종속 신호 지표; score = p_buy - p_sell)

| window | raw IC | fixed-effect IC | **timing IC** | timing SE | timing IR | macro F1 (자체 라벨) |
|---|---|---|---|---|---|---|
| val_2024 (선택 윈도우) | +0.0738 | +0.0243 | **+0.0732** | 0.0100 | 0.537 | 0.403 |
| oot_2026 (확인용, 1회만 채점) | +0.0533 | +0.0296 | **+0.0293** | 0.0124 | 0.183 | 0.346 |

timing IC = (score - 해당 종목 자체 평균 score)의 일별 cross-sectional rank IC vs 익일 수익률; OOT의
fixed effect는 이 체크포인트의 val 윈도우 종목별 평균을 그대로 사용(look-ahead 없음). seed 0/1/2 전부에서
안정적으로 재현됨(OOT timing IC 0.029 / 0.023 / 0.021; val 0.073 / 0.066 / 0.068)이고, 3-시드 확률평균
앙상블(val +0.073, OOT +0.029)과도 성능이 동일 — 이 단일 체크포인트 대비 앙상블의 추가 이득 없음.

## 시스템 워크플로우 (AI 추론 ↔ 백엔드)

관련 레포: 이 레포(`serving/`, 데이터 파이프라인), `Back-Gentle-Viking`(FastAPI 백엔드, 자동매매 루프).
프론트엔드 `Front-Gentle-Viking`은 백엔드 API만 호출한다.

```
[운영 수집 - stock_db (350종목)]
  collector_realtime 08:55  1분봉 → intraday_1min/5min
  collector_batch    16:00  일봉/5분봉 정리
  collector_kis      15:50  수급·밸류에이션
  collector_yf/dart 08:00  매크로·공시
        │
        ▼
[AI DB - stock_db_v2 (AI 200종목)]          ← 10/6부터 매일 16:45
  sync_v2_from_prod      운영 DB에서 9개 공통 테이블 증분 복사
  run_leverage_backfill  leverage_daily 갱신
  build_features         feature_pool 재빌드 (롤링 피처용 90일 lookback)
        │
        ▼
[AI 추론 서버 - serving]
  serving/active_tickers.json   (START 명령으로 등록, 최대 5개)
  inference_pipeline  */5 평일 09:00~15:30
  poll_commands       매분 (백엔드 큐 → /command 중계)
  api_server (8001)   /command: START · STOP · ONCE
        │  POST /ai/realtime (5분마다, X-API-Key)
        │  POST /ai/callback (ONCE 결과)
        ▼
[백엔드 - FastAPI (8000)]
  realtime_predictions · ai_prediction_history · once_results
  command_queue ◀── /basket, /trade/start, /trade/once, /trade/stop
        │
        ▼
[자동매매 루프 trading_loop (유저별 태스크)]
  KIS 웹소켓 시세 구독 → ai_signal_event 대기 → 사이클
        │
        ▼
[KIS 모의계좌] 잔고 조회 · 주문
```

### 1. AI 추론 (`serving/`)

**5분 추론 (`inference_pipeline.py`)**
1. 크론(`*/5`, 평일)이 실행된다. 09:00~15:30 밖이면 건너뛴다.
2. `active_tickers.json`에서 종목을 읽는다(5개 초과면 앞 5개만).
3. 종목마다:
   - `stock_db.intraday_5min`에서 오늘 봉을 가져와 미완성 일봉을 만든다(고가·저가·거래량 누적).
   - `stock_db_v2.feature_pool`에서 과거 59거래일을 가져온다.
   - 60행 입력을 조립하고(LIVE / FFILL / ZERO_DEFAULT 정책), 모델을 돌려 확률을 얻는다.
4. 결과를 백엔드 `POST /ai/realtime`으로 보낸다. 실패한 종목은 결과에서 빠진다.

**명령 처리 (`poll_commands.py`, 매분)**
1. 백엔드 `GET /ai/commands/pending`으로 큐를 가져온다(조회 시 백엔드에서 delivered 처리).
2. 명령을 그대로 serving `POST /command`로 전달한다.
   - `START`: 사용자별 종목을 등록(합쳐서 5개 초과면 400).
   - `STOP`: 사용자 종목 제거.
   - `ONCE`: 백그라운드 스레드에서 한 번 추론하고 `callback_url`(백엔드 `/ai/callback`)로 결과를 보낸다.

**모델**: `serving/best_model_state_dict.pt` = 확정 모델 `v3_structure_nowd_cs` seed0. 모든 결과에 `TFT_MODEL_VERSION`(`v3_structure_nowd_cs-seed0`)이 붙는다.

### 2. 백엔드 (`Back-Gentle-Viking`)

**AI → 백엔드 입력**
- `/ai/realtime`: 종목별 결과를 `realtime_predictions`에 저장(ONCE가 채운 리포트 필드는 유지), `ai_prediction_history`에 기록, `ai_signal_event`를 set해서 자동매매 루프를 깨운다.
- `/ai/callback`: Gemini로 리포트를 만든다(키가 없으면 "보고서 생성 실패" 문자열로 대체), 리포트를 저장하고 `once_results[job_id]`에 넣는다. 프론트는 `GET /ai/once/{job_id}`로 완료를 확인한다.

**사용자 명령**
- `POST /basket`: 중복, 상한 5개, AI 유니버스(`ticker_universe`) 포함 여부를 확인한다.
- `POST /trade/start`: 총 자본을 정하고(요청값 또는 KIS 잔고), 바구니를 DB와 동기화하고(상한 확인), 직접매매 잠금·AI 미커버 종목을 제외한다. 전략 ID를 확인하고(모르는 ID는 conservative로 대체, 응답에 표시), START를 큐에 넣고, 자동매매 태스크를 띄운다.
- `POST /trade/once`: 바구니를 확인하고 ONCE를 큐에 넣은 뒤 `job_id`를 돌려준다.
- `POST /trade/stop`: 태스크를 취소하고 STOP을 큐에 넣는다.

**자동매매 루프 (`trading_loop`)**
1. KIS 웹소켓으로 종목 시세를 구독하고 워밍업을 기다린다(현재 5초).
2. `ai_signal_event`를 기다린다(최대 600초). AI 푸시마다 한 사이클을 돈다.
3. 한 사이클:
   1. KIS 실제 잔고로 보유 수량을 읽는다(실패하면 이번 사이클을 건너뛴다).
   2. 종목별 예측: 30분 넘은 예측이나 없는 예측은 HOLD(확신도 0)로 본다.
   3. `allocate_portfolio`: BUY 신호 중 확신도 기준을 넘는 종목만 비중으로 나눈다(종목당 최대 30%).
   4. 종목별로: 시세 없으면 SKIP → 확신도 미달이면 SKIP → 전략 주문이 없으면 HOLD → AI·전략 신호 불일치면 HOLD → 수량 계산 → `RiskManager` 확인(종목당·1회 주문·전체 노출) → KIS 주문(실패 시 10초 간격 3회 재시도) → 체결이면 포지션 갱신, `TradeLog`·`AutoTradeDecision` 기록, 종목마다 커밋.
   5. 예외는 종목 단위로 막고 다음 종목으로 진행한다.

### 3. 현재 제약

- **KIS 계정 불일치**: 잔고 조회와 주문이 `INVALID_CHECK_ACNO`로 실패한다. 포지션 조회와 주문 단계가 실제로는 동작하지 않는다.
- **수집 대상 파일**: 운영 수집기는 `/home/user/active_tickers.json`을 읽는다. 추론 대상으로 등록한 종목은 이 파일에도 있어야 수집된다.
- **5분봉 워밍업**: 백엔드 요청을 AI가 처리하지 않아 사실상 비어 있다. 대기는 5초로 줄여 두었다.
- **ONCE 경로**: 폴러(`poll_commands`)가 떠 있어야 동작한다.
- **상한 5종목**: 추론(serving)·등록(serving, 백엔드 `/basket`·`/trade/*`)·프론트 장바구니에 적용돼 있다.
