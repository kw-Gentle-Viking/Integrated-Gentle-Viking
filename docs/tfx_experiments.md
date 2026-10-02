# TFX: TFT retraining recipes (input standardisation / label_vn / cross-sectional rank inputs)

Plan: `docs/superpowers/plans/2026-09-26-signal-improvement-plan.md`, section "E0 결과 이후 수정". Code commit: `e9aa760+dirty` (`training/run_tfx_experiments.py`, `training/preprocess.py`). Raw output: `training/artifacts/tfx_results.json` (gitignored); preprocessing artifacts `training/artifacts/preproc_<recipe>.json`.

**Selection by val IC; OOT is confirmation only.** Model/recipe choice uses the 2024 val window (mean daily rank IC). The 2026 OOT window was scored once per recipe (3 recipes) and is NOT a selection criterion. The 2025 test window is never scored. Bar to clear: OOT IC >= ~0.044 (HGB F3 level).

## Protocol

- Architecture/hparams from `training/champion_config.json` (state 32, heads 8, layers 2, dropout 0.1658, lr 0.000291), batch 128, max 12 epochs, val-loss early stopping (patience 3), best-val-loss checkpoint, balanced class weights from each recipe's train label distribution.
- **Alignment**: all recipes use `TickerDayDataset(align="today")`: label / future / static features come from the last encoder row (the model sees the labelled day's own features, as serving does). The legacy dataset took them from the row AFTER the encoder (effective 2-day-ahead forecast). Sample date = date of the last encoder row, so the aligned sample sets contain a few hundred more samples than S3/E0 (first window per ticker); they are NOT asserted equal to S3's 36729 / 33364 (that legacy reproduction is checked separately with `--align legacy --check-windows`).
- R1-R3 preprocessing fitted on train rows dated <= 2023-12-31 only (clip at train 0.5/99.5 quantiles, z-score of the clipped values; binary 0/1 untouched; zero-variance columns dropped); val/OOT apply-only. R0 feeds raw values.
- Train labels of glitch rows (|next_day_return| > 0.31) masked to NaN; evaluation sample sets gated by the fixed-label NULL set for every recipe (same across recipes). Spreads shown raw and with returns clipped to +-30%; IC is rank-based (primary). Comparison rows (TFT champion, reversal from S3; HGB F3 from E0) carry raw spreads only (clipped = n/a) and are **legacy-aligned TFT / old-data tabular; not like-for-like** (different sample sets and price data version; clean-data aligned tabular baselines are re-run separately).
- Score = p_buy - p_sell; rank IC = per-day cross-sectional Spearman with next_day_return (days with < 20 names skipped); SE = IC std / sqrt(days). Macro F1 / per-class P/R are against each recipe's own label, so R1 (fixed label) vs R2/R3 (label_vn) F1 values are NOT comparable with each other.

## Recipes

| recipe | tag | alignment | preprocessing | label | model inputs | dropped (constant in train) | epochs run / stop | best val loss | class weights (buy/hold/sell) | train samples (glitch-masked rows) |
|---|---|---|---|---|---|---|---|---|---|---|
| aligned | R0 | today | raw | label | 33 (0 rank) | none | 4 / early_stopping | 1.0926 | 1.386 / 0.669 / 1.276 | 234226 (0) |
| std | R1 | today | standardised | label | 30 (0 rank) | lev_total_aum, lev_aum_to_mktcap, est_rebalancing_flow | 4 / early_stopping | 1.0744 | 1.386 / 0.669 / 1.276 | 234226 (0) |
| std_vn | R2 | today | standardised | label_vn | 30 (0 rank) | lev_total_aum, lev_aum_to_mktcap, est_rebalancing_flow | 4 / early_stopping | 1.0884 | 1.388 / 0.667 / 1.281 | 234226 (0) |
| std_vn_csr | R3 | today | standardised | label_vn | 36 (6 rank) | lev_total_aum, lev_aum_to_mktcap, est_rebalancing_flow | 4 / early_stopping | 1.0886 | 1.388 / 0.667 / 1.281 | 234226 (0) |

Recipe descriptions: **R0 aligned**: raw inputs (no preprocessing, all 33 champion columns), fixed label; only the alignment fix vs the legacy champion recipe; **R1 std**: R0 + input standardisation (train<=2023 clip 0.5/99.5 + z-score, binary untouched, constant cols dropped); **R2 std_vn**: R1 + volatility-normalised label (label_vn); **R3 std_vn_csr**: R2 + per-date cross-sectional rank inputs (log_ret, disparity_5d/20d/60d, rsi_14, volatility_20d).

## val_2024 (selection window): targets 2024-01-01..2024-12-31

Samples 36929 (with returns 36929), actual target dates 2024-03-28..2024-12-30, 185 scored days, mean names/day 199.6; feature rows loaded 2024-01-01..2024-12-31. Legacy-aligned S3/E0 counts were 36729 (aligned sets are larger). Samples with |next_day_return| > 30%: 0.

| scorer | mean rank IC | SE | IC IR | days | quantile L/S | quantile L/S (clip30) | argmax L/S | argmax L/S (clip30) | macro F1 | pred share buy/hold/sell |
|---|---|---|---|---|---|---|---|---|---|---|
| **R0 aligned** | 0.0463 | 0.0116 | 0.293 | 185 | 0.10% | 0.10% | 0.23% | 0.23% | 0.3681 (36929 lbl) | 0.109 / 0.596 / 0.295 |
| **R1 std** | 0.0366 | 0.0084 | 0.319 | 185 | 0.15% | 0.15% | 0.18% | 0.18% | 0.4073 (36929 lbl) | 0.256 / 0.436 / 0.308 |
| **R2 std_vn** | 0.0142 | 0.0093 | 0.112 | 185 | 0.13% | 0.13% | -0.20% | -0.20% | 0.3611 (36929 lbl) | 0.084 / 0.621 / 0.294 |
| **R3 std_vn_csr** | 0.0165 | 0.0104 | 0.117 | 185 | 0.10% | 0.10% | 0.14% | 0.14% | 0.3737 (36929 lbl) | 0.139 / 0.610 / 0.251 |
| TFT champion (S3, legacy-aligned) | 0.0143 | 0.0108 | 0.098 | 184 | 0.14% | n/a | 0.29% | n/a | 0.3823 | n/a |
| 1-day reversal (S3, legacy sample set) | 0.0625 | 0.0107 | 0.432 | 184 | 0.27% | n/a | 0.13% | n/a | n/a | n/a |
| HGB reg (z), F3 (E0v2) | 0.0564 | 0.0099 | 0.417 | 185 | 0.21% | n/a | 0.14% | n/a | n/a | n/a |
| HGB clf fixed label, F3 (E0v2) | 0.0578 | 0.0092 | 0.463 | 185 | 0.21% | n/a | 0.30% | n/a | n/a | n/a |

Per-class precision / recall (0=buy, 1=hold, 2=sell; against each recipe's own label):

| recipe | buy P / R | hold P / R | sell P / R | accuracy | MCC |
|---|---|---|---|---|---|
| aligned | 0.288 / 0.125 | 0.518 / 0.661 | 0.341 / 0.358 | 0.4408 | 0.0879 |
| std | 0.303 / 0.308 | 0.571 / 0.532 | 0.349 / 0.383 | 0.4340 | 0.1230 |
| std_vn | 0.272 / 0.098 | 0.555 / 0.684 | 0.308 / 0.347 | 0.4587 | 0.0853 |
| std_vn_csr | 0.283 / 0.168 | 0.552 / 0.667 | 0.312 / 0.300 | 0.4544 | 0.0832 |

## oot_2026 (confirmation only, not a selection criterion): targets 2026-01-01..2026-09-08

Samples 33364 (with returns 33364), actual target dates 2026-01-02..2026-09-07, 167 scored days, mean names/day 199.8; feature rows loaded 2025-09-01..2026-09-08. Legacy-aligned S3/E0 counts were 33364 (aligned sets are larger). Samples with |next_day_return| > 30%: 0.

| scorer | mean rank IC | SE | IC IR | days | quantile L/S | quantile L/S (clip30) | argmax L/S | argmax L/S (clip30) | macro F1 | pred share buy/hold/sell |
|---|---|---|---|---|---|---|---|---|---|---|
| **R0 aligned** | 0.0395 | 0.0136 | 0.224 | 167 | -0.01% | -0.01% | 0.07% | 0.07% | 0.3860 (33364 lbl) | 0.227 / 0.327 / 0.446 |
| **R1 std** | 0.0300 | 0.0121 | 0.192 | 167 | 0.14% | 0.14% | -0.02% | -0.02% | 0.3316 (33364 lbl) | 0.695 / 0.213 / 0.092 |
| **R2 std_vn** | 0.0006 | 0.0168 | 0.003 | 167 | 0.21% | 0.21% | 0.46% | 0.46% | 0.3401 (33364 lbl) | 0.076 / 0.786 / 0.138 |
| **R3 std_vn_csr** | 0.0176 | 0.0131 | 0.104 | 167 | 0.17% | 0.17% | -0.11% | -0.11% | 0.3412 (33364 lbl) | 0.095 / 0.718 / 0.186 |
| TFT champion (S3, legacy-aligned) | -0.0049 | 0.0154 | -0.025 | 167 | -0.13% | n/a | -0.54% | n/a | 0.3420 | n/a |
| 1-day reversal (S3, legacy sample set) | 0.0427 | 0.0155 | 0.212 | 167 | 0.29% | n/a | 0.19% | n/a | n/a | n/a |
| HGB reg (z), F3 (E0v2) | 0.0308 | 0.0129 | 0.184 | 167 | 0.11% | n/a | -0.61% | n/a | n/a | n/a |
| HGB clf fixed label, F3 (E0v2) | 0.0371 | 0.0135 | 0.212 | 167 | 0.09% | n/a | -0.38% | n/a | n/a | n/a |

Per-class precision / recall (0=buy, 1=hold, 2=sell; against each recipe's own label):

| recipe | buy P / R | hold P / R | sell P / R | accuracy | MCC |
|---|---|---|---|---|---|
| aligned | 0.363 / 0.247 | 0.424 / 0.438 | 0.387 / 0.493 | 0.3934 | 0.0890 |
| std | 0.346 / 0.718 | 0.457 / 0.309 | 0.384 / 0.101 | 0.3728 | 0.0736 |
| std_vn | 0.346 / 0.102 | 0.516 / 0.838 | 0.322 / 0.172 | 0.4762 | 0.0854 |
| std_vn_csr | 0.298 / 0.110 | 0.519 / 0.770 | 0.290 / 0.209 | 0.4551 | 0.0659 |

## Columns

Dropped as constant in train (<= 2023-12-31): est_rebalancing_flow, lev_aum_to_mktcap, lev_total_aum. Kept/dropped column lists per recipe and the fitted clip bounds / mean / std are in `training/artifacts/preproc_<recipe>.json`.
