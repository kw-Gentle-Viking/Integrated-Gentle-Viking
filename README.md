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
