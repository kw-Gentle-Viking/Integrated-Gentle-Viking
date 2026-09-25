# Signal baseline (deployed Stage-1 champion, fixed label)

> **정정 (2026-09-26, 독립 리뷰 지적 — 이 문서의 해석 일부는 무효):**
> 1. **TFT 정렬 결함**: `TickerDayDataset`은 인코더로 d−60…d−1일을 넣고 라벨은 d일 종가→d+1일 수익률(행 d의 라벨)을 씀 — TFT는 **당일(d) 종가 정보 없이** 예측했음(사실상 2일 앞 예측). 반면 표형 모델/반전 규칙은 d일 정보를 봄. 따라서 아래의 "TFT vs GBM/반전 IC 격차"는 **모델 능력 차이가 아니라 정보량이 다른 비교**이며, "배포 모델의 신호가 ≈0"은 "현재 정렬 그대로 학습·평가된 모델" 한정 결론으로만 유효함(한 칸 늦은 반전 규칙의 IC는 val −0.003 / OOT +0.035로 TFT와 비슷 — 리뷰어 측정). 서빙은 마지막 인코더 스텝이 '오늘'이라 학습과 어긋나 있었음. 수정 작업 중(align="today").
> 2. **가격 데이터가 수정주가 미반영**(FID_ORG_ADJ_PRC="1")이었음 — 이 문서의 모든 수치는 오염된 데이터 기준. 재백필/재빌드 후 재산출 예정.
> 3. "val 2024" 표본의 실제 타깃일 범위는 2024-03-29..2024-12-30(앞 60행은 워밍업). 일별 IC의 SE는 일 간 독립을 가정.
> 4. 레버리지 피처 3종(lev_total_aum, lev_aum_to_mktcap, est_rebalancing_flow)은 전 기간 0인 죽은 컬럼.


Plan S3 reference measurement: how much real, label-agnostic signal the current champion has.
Code commit: `a9a7325`. Champion: see `training/champion_config.json` (`stage1-remove_lev_total_volume`, checkpoint `stage1-ablation-remove_lev_total_volume.pt`).

Score = p_buy - p_sell. Rank IC = per-day cross-sectional Spearman(score, next_day_return), days with < 20 names skipped; IC IR = mean/std of daily IC (ddof=1). argmax L/S = mean return of predicted-buy minus predicted-sell (pooled). Quantile L/S = per-day top-20% minus bottom-20% by score, averaged over days. Returns are raw next_day_return (t to t+1).

Windows: the 2025 test window is never scored.

## val_2024: targets 2024-01-01..2024-12-31

Samples: 36729 (scored with returns: 36729); feature rows loaded 2024-01-01..2024-12-31; actual target-date range 2024-03-29..2024-12-30; mean names/day 199.6.

| Scorer | mean rank IC | IC std | IC IR | days | argmax L/S | n buy / n sell | quantile L/S | macro F1 | pred share buy/hold/sell |
|---|---|---|---|---|---|---|---|---|---|
| champion | 0.0143 | 0.1466 | 0.0979 | 184 | 0.29% | 9938 / 12864 | 0.14% | 0.3823 | 0.271 / 0.379 / 0.350 |
| random | -0.0020 | 0.0745 | -0.0270 | 184 | -0.05% | 18305 / 18424 | -0.01% | 0.2297 | 0.498 / 0.000 / 0.502 |
| reversal | 0.0625 | 0.1446 | 0.4320 | 184 | 0.13% | 18779 / 16786 | 0.27% | 0.2638 | 0.511 / 0.032 / 0.457 |
| momentum | -0.0625 | 0.1446 | -0.4320 | 184 | -0.13% | 16786 / 18779 | -0.27% | 0.2391 | 0.457 / 0.032 / 0.511 |

Champion per-class (0=buy, 1=hold, 2=sell):

| class | precision | recall | F1 |
|---|---|---|---|
| 0 | 0.2952 | 0.3161 | 0.3053 |
| 1 | 0.5437 | 0.4412 | 0.4871 |
| 2 | 0.3189 | 0.3989 | 0.3544 |

accuracy 0.3977, MCC 0.0832. Mean next-day return by argmax group: buy 0.10%, hold 0.06%, sell -0.19% (n 9938 / 13927 / 12864).

## oot_2026: targets 2026-01-01..2026-09-08

Samples: 33364 (scored with returns: 33364); feature rows loaded 2025-09-01..2026-09-08; actual target-date range 2026-01-02..2026-09-07; mean names/day 199.8.

| Scorer | mean rank IC | IC std | IC IR | days | argmax L/S | n buy / n sell | quantile L/S | macro F1 | pred share buy/hold/sell |
|---|---|---|---|---|---|---|---|---|---|
| champion | -0.0049 | 0.1987 | -0.0247 | 167 | -0.54% | 9987 / 19101 | -0.13% | 0.3420 | 0.299 / 0.128 / 0.573 |
| random | 0.0064 | 0.0738 | 0.0868 | 167 | 0.04% | 16640 / 16724 | 0.04% | 0.2727 | 0.499 / 0.000 / 0.501 |
| reversal | 0.0427 | 0.2009 | 0.2125 | 167 | 0.19% | 16624 / 16062 | 0.29% | 0.2954 | 0.498 / 0.020 / 0.481 |
| momentum | -0.0427 | 0.2009 | -0.2125 | 167 | -0.19% | 16062 / 16624 | -0.29% | 0.2741 | 0.481 / 0.020 / 0.498 |

Champion per-class (0=buy, 1=hold, 2=sell):

| class | precision | recall | F1 |
|---|---|---|---|
| 0 | 0.3253 | 0.2913 | 0.3074 |
| 1 | 0.4719 | 0.1914 | 0.2724 |
| 2 | 0.3595 | 0.5885 | 0.4463 |

accuracy 0.3637, MCC 0.0373. Mean next-day return by argmax group: buy -0.23%, hold 0.19%, sell 0.31% (n 9987 / 4276 / 19101).

References (macro F1 is meaningless for them since they never predict hold; they only give context for the IC/spread scale): `random` = seeded Gaussian score; `reversal` = -log_ret of the target day; `momentum` = +log_ret of the target day.
