# Signal baseline (deployed Stage-1 champion, fixed label)

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
