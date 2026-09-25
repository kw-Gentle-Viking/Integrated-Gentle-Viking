# Tabular baseline (E0): is there learnable signal in the champion features?

> **정정 (2026-09-26, 독립 리뷰 지적 — 이 문서의 해석 일부는 무효):**
> 1. **TFT 정렬 결함**: `TickerDayDataset`은 인코더로 d−60…d−1일을 넣고 라벨은 d일 종가→d+1일 수익률(행 d의 라벨)을 씀 — TFT는 **당일(d) 종가 정보 없이** 예측했음(사실상 2일 앞 예측). 반면 표형 모델/반전 규칙은 d일 정보를 봄. 따라서 아래의 "TFT vs GBM/반전 IC 격차"는 **모델 능력 차이가 아니라 정보량이 다른 비교**이며, "배포 모델의 신호가 ≈0"은 "현재 정렬 그대로 학습·평가된 모델" 한정 결론으로만 유효함(한 칸 늦은 반전 규칙의 IC는 val −0.003 / OOT +0.035로 TFT와 비슷 — 리뷰어 측정). 서빙은 마지막 인코더 스텝이 '오늘'이라 학습과 어긋나 있었음. 수정 작업 중(align="today").
> 2. **가격 데이터가 수정주가 미반영**(FID_ORG_ADJ_PRC="1")이었음 — 이 문서의 모든 수치는 오염된 데이터 기준. 재백필/재빌드 후 재산출 예정.
> 3. "val 2024" 표본의 실제 타깃일 범위는 2024-03-29..2024-12-30(앞 60행은 워밍업). 일별 IC의 SE는 일 간 독립을 가정.
> 4. 레버리지 피처 3종(lev_total_aum, lev_aum_to_mktcap, est_rebalancing_flow)은 전 기간 0인 죽은 컬럼.


Model/metric code commit: `af077f8` (results produced by it; later commit `4a6b643` only added doc rendering) (`training/run_tabular_baseline.py`, `training/tabular_features.py`). Raw output: `training/artifacts/tabular_baseline.json` (gitignored, like the other artifacts).

## Protocol

- Same sample sets and metrics as S3 (`docs/signal_baseline.md`). The samples were re-enumerated with `TickerDayDataset` + `filter_index_by_target_date` on the S3 caches and asserted identical, in order, to the (ticker, date) list S3 stored: **val_2024** targets 2024-01-01..2024-12-31, 36729 samples, 184 scored days; **oot_2026** targets 2026-01-01..2026-09-07, 33364 samples, 167 scored days. `next_day_return` of every sample was asserted equal to S3's.
- Score = model output (regression prediction, or p_buy - p_sell for the classifier); converted to the S3 pseudo-probs form via `score_to_probs` and scored with `compute_signal_metrics`. IC SE = IC std / sqrt(n_days).
- Training rows: 2019-01-02.., a row needs >= 60 prior rows (mirrors the TFT encoder), finite `next_day_return` and `volatility_20d`. Hyper-parameter selection used only a time-ordered split inside the train era: fit on rows <= 2022-12-31, select on 2023 rows (mean daily rank IC), refit on rows <= 2023-12-31. 2025 rows are never used.
- Target z = next_day_return / max(volatility_20d, floor=0.007630) clipped to [-5, 5] (floor from `training/threshold_vn.json`). Classifier target = `label_vn` (classes 0=buy, 1=hold, 2=sell).
- Feature sets. F1: the 33 champion columns at day t. F2: F1 + log_ret lags 1..5 (per-ticker shift). F3: F2 + per-date cross-sectional percentile ranks of log_ret, disparity_5d/20d/60d, rsi_14, volatility_20d and the 5 lags. Event flags NULL -> 0; other NaN handled per model.
- Ridge: winsorise at train 0.5/99.5 pct, NaN -> train median, standardise (all train statistics). HGB (sklearn `HistGradientBoosting`; lightgbm is not installed): lr 0.05, max_depth 4, min_samples_leaf 500, l2 10, max_bins 64, native NaN, no early stopping.
- Configs tried: 8 (3 Ridge, 3 HGB regressors, 1 HGB regressor on the date-demeaned z, 1 HGB classifier). The date-demeaned-z config removes the market-wide component from the regression target (the plain regressor's first trees split on date-level columns, giving a constant score per day, so its IC is undefined at 25 iterations). All 8 configs are reported.

## Selection on 2023 (fit on <= 2022)

| config | grid searched (mean daily IC on 2023) | chosen |
|---|---|---|
| ridge_F1 | 10.0: 0.033, 1000.0: 0.033, 100000.0: 0.025, 1000000.0: 0.002 | alpha=10 |
| hgb_reg_F1 | 25: nan, 50: -0.012, 100: 0.030, 200: 0.035, 300: 0.039, 400: 0.042 | n_iter=400 |
| ridge_F2 | 10.0: 0.034, 1000.0: 0.034, 100000.0: 0.028, 1000000.0: 0.004 | alpha=10 |
| hgb_reg_F2 | 25: nan, 50: -0.015, 100: 0.021, 200: 0.037, 300: 0.044, 400: 0.039 | n_iter=300 |
| ridge_F3 | 10.0: 0.041, 1000.0: 0.041, 100000.0: 0.051, 1000000.0: 0.045 | alpha=100000 |
| hgb_reg_F3 | 25: nan, 50: -0.015, 100: 0.038, 200: 0.051, 300: 0.048, 400: 0.048 | n_iter=200 |
| hgb_reg_csz_F3 | 25: 0.059, 50: 0.058, 100: 0.054, 200: 0.050, 300: 0.046, 400: 0.040 | n_iter=25 |
| hgb_clf_F3 | 25: 0.011, 50: 0.032, 100: 0.038, 200: 0.040, 300: 0.044, 400: 0.045 | n_iter=400 |

Caveats: `ridge_F3` picked alpha at 1e5 (interior of the grid, 1e6 is worse); `hgb_reg_csz_F3` picked the smallest n_iter in the grid (25) with IC still falling as iterations increase, so its optimum may be lower still; `hgb_clf_F3` picked the largest (400) with IC still rising.

## val_2024 (targets 2024-01-01..2024-12-31; 36729 samples, 184 days)

| scorer | mean rank IC | SE | IC IR | quantile L/S | argmax L/S | IC minus TFT (SE of diff) | IC minus reversal (SE of diff) |
|---|---|---|---|---|---|---|---|
| Ridge, F1 | 0.0277 | 0.0093 | 0.220 | 0.17% | 0.03% | +0.0133 (0.0131) | -0.0348 (0.0080) |
| Ridge, F2 | 0.0267 | 0.0093 | 0.211 | 0.14% | -0.01% | +0.0123 (0.0132) | -0.0358 (0.0087) |
| Ridge, F3 | 0.0415 | 0.0095 | 0.322 | 0.23% | -0.08% | +0.0272 (0.0127) | -0.0209 (0.0065) |
| HGB reg (z), F1 | 0.0443 | 0.0088 | 0.371 | 0.21% | 0.22% | +0.0300 (0.0133) | -0.0181 (0.0087) |
| HGB reg (z), F2 | 0.0496 | 0.0087 | 0.419 | 0.22% | 0.14% | +0.0353 (0.0131) | -0.0129 (0.0075) |
| HGB reg (z), F3 | 0.0576 | 0.0096 | 0.444 | 0.24% | 0.16% | +0.0432 (0.0135) | -0.0049 (0.0068) |
| HGB reg (date-demeaned z), F3 | 0.0536 | 0.0097 | 0.406 | 0.24% | 0.19% | +0.0393 (0.0127) | -0.0088 (0.0039) |
| HGB clf (label_vn), F3 | 0.0493 | 0.0089 | 0.409 | 0.22% | 0.25% | +0.0350 (0.0126) | -0.0132 (0.0073) |
| TFT champion (S3) | 0.0143 | 0.0108 | 0.098 | 0.14% | 0.19% | - | - |
| reversal (S3) | 0.0625 | 0.0107 | 0.432 | 0.27% | 0.13% | - | - |
| random (S3) | -0.0020 | 0.0055 | -0.027 | -0.01% | -0.05% | - | - |

## oot_2026 (targets 2026-01-01..2026-09-07; 33364 samples, 167 days)

| scorer | mean rank IC | SE | IC IR | quantile L/S | argmax L/S | IC minus TFT (SE of diff) | IC minus reversal (SE of diff) |
|---|---|---|---|---|---|---|---|
| Ridge, F1 | 0.0211 | 0.0139 | 0.117 | 0.32% | -0.20% | +0.0260 (0.0235) | -0.0216 (0.0131) |
| Ridge, F2 | 0.0130 | 0.0138 | 0.073 | 0.28% | -0.15% | +0.0179 (0.0234) | -0.0297 (0.0141) |
| Ridge, F3 | 0.0177 | 0.0149 | 0.092 | 0.22% | -0.17% | +0.0226 (0.0245) | -0.0250 (0.0141) |
| HGB reg (z), F1 | 0.0286 | 0.0120 | 0.185 | 0.21% | -0.55% | +0.0335 (0.0199) | -0.0141 (0.0130) |
| HGB reg (z), F2 | 0.0165 | 0.0118 | 0.109 | 0.11% | -0.45% | +0.0214 (0.0200) | -0.0262 (0.0118) |
| HGB reg (z), F3 | 0.0444 | 0.0115 | 0.297 | 0.09% | -0.53% | +0.0493 (0.0180) | +0.0017 (0.0141) |
| HGB reg (date-demeaned z), F3 | 0.0443 | 0.0141 | 0.243 | 0.30% | 0.13% | +0.0492 (0.0227) | +0.0016 (0.0057) |
| HGB clf (label_vn), F3 | 0.0332 | 0.0134 | 0.191 | 0.27% | -0.38% | +0.0381 (0.0219) | -0.0095 (0.0128) |
| TFT champion (S3) | -0.0049 | 0.0154 | -0.025 | -0.13% | -0.44% | - | - |
| reversal (S3) | 0.0427 | 0.0155 | 0.212 | 0.29% | 0.19% | - | - |
| random (S3) | 0.0064 | 0.0057 | 0.087 | 0.04% | 0.04% | - | - |

The TFT / reversal / random rows are recomputed from S3's saved probabilities and asserted equal to `signal_baseline.json` (mean IC to 1e-9); the paired differences use the per-day IC series over days present in both (same days for all scorers), so their SE is much tighter than that of two independent means.

## Production tickers (noisy)

Time-series Spearman(score, next_day_return) over the window's days for one ticker, and sign hit-rate (sign(score) == sign(return), zero-return days excluded; the base rate of up days is shown for context). n is about 170-185 days, so SE of a correlation is about 0.07-0.08 and hit-rate SE about 0.037: **none of these per-ticker numbers is distinguishable from zero / 50%.**

| scorer | window | 005930 rho | 005930 hit (up-rate) | 000660 rho | 000660 hit (up-rate) |
|---|---|---|---|---|---|
| Ridge, F1 | val_2024 | -0.064 | 0.489 (0.44), n=184 | -0.059 | 0.483 (0.52), n=184 |
| Ridge, F2 | val_2024 | -0.070 | 0.483 (0.44), n=184 | -0.047 | 0.467 (0.52), n=184 |
| Ridge, F3 | val_2024 | -0.058 | 0.500 (0.44), n=184 | -0.047 | 0.467 (0.52), n=184 |
| HGB reg (z), F1 | val_2024 | 0.123 | 0.534 (0.44), n=184 | -0.001 | 0.467 (0.52), n=184 |
| HGB reg (z), F2 | val_2024 | 0.064 | 0.517 (0.44), n=184 | 0.020 | 0.506 (0.52), n=184 |
| HGB reg (z), F3 | val_2024 | 0.075 | 0.534 (0.44), n=184 | 0.033 | 0.533 (0.52), n=184 |
| HGB reg (date-demeaned z), F3 | val_2024 | -0.032 | 0.523 (0.44), n=184 | 0.024 | 0.539 (0.52), n=184 |
| HGB clf (label_vn), F3 | val_2024 | 0.053 | 0.557 (0.44), n=184 | 0.011 | 0.483 (0.52), n=184 |
| TFT champion | val_2024 | 0.016 | 0.494 (0.44), n=184 | 0.065 | 0.550 (0.52), n=184 |
| reversal | val_2024 | 0.003 | 0.511 (0.44), n=184 | 0.097 | 0.517 (0.52), n=184 |
| Ridge, F1 | oot_2026 | -0.040 | 0.558 (0.56), n=167 | -0.041 | 0.542 (0.58), n=167 |
| Ridge, F2 | oot_2026 | -0.040 | 0.564 (0.56), n=167 | -0.033 | 0.566 (0.58), n=167 |
| Ridge, F3 | oot_2026 | -0.072 | 0.539 (0.56), n=167 | -0.064 | 0.542 (0.58), n=167 |
| HGB reg (z), F1 | oot_2026 | -0.037 | 0.521 (0.56), n=167 | -0.070 | 0.488 (0.58), n=167 |
| HGB reg (z), F2 | oot_2026 | -0.085 | 0.503 (0.56), n=167 | -0.134 | 0.506 (0.58), n=167 |
| HGB reg (z), F3 | oot_2026 | -0.112 | 0.479 (0.56), n=167 | -0.137 | 0.482 (0.58), n=167 |
| HGB reg (date-demeaned z), F3 | oot_2026 | 0.107 | 0.515 (0.56), n=167 | -0.011 | 0.488 (0.58), n=167 |
| HGB clf (label_vn), F3 | oot_2026 | -0.013 | 0.491 (0.56), n=167 | -0.040 | 0.488 (0.58), n=167 |
| TFT champion | oot_2026 | 0.024 | 0.527 (0.56), n=167 | 0.041 | 0.488 (0.58), n=167 |
| reversal | oot_2026 | 0.106 | 0.515 (0.56), n=167 | 0.072 | 0.476 (0.58), n=167 |

## Interpretation (SE-aware)

Differences are called real only when they are >= ~2 paired SE.

1. **The features contain a small amount of learnable cross-sectional signal.** Every config has a positive
   mean IC on both windows; the best (HGB regressor, F3) has IC 0.058 (val, SE 0.010) and 0.044 (OOT, SE 0.012), about
   6 SE from zero on val and ~4 SE on OOT. IC IR (HGB) 0.4-0.45 on val and 0.2-0.3 on OOT, i.e. small in absolute terms. Quantile L/S is about 0.1-0.3% per day
   gross, before any transaction costs; this says nothing yet about tradability.
2. **The deployed TFT is well below what a simple tabular model gets from the same 33 columns.** Paired IC difference vs TFT on
   val: HGB F3 +0.043 (SE 0.0135, 3.2 SE), F2 +0.035 (2.7 SE), F1 +0.030 (2.3 SE), clf +0.035 (2.8 SE), demeaned-z +0.039 (3.1 SE);
   ridge F1/F2 (+0.012/+0.013, ~1 SE) are not distinguishable from TFT. On OOT only HGB F3 (+0.049, SE 0.018, 2.7 SE) and
   demeaned-z (+0.049, SE 0.023, 2.2 SE) clear 2 SE; the others are +0.02-0.038 with SE 0.02-0.025 (about 1-1.7 SE), i.e.
   same direction, not individually conclusive. The TFT was selected on val 2024 macro F1, so the val comparison, if anything, favours it.
   Because the tabular F1 models use the same columns as the TFT, the gap is attributable to the model/training setup, not to missing features.
3. **No tabular model beats the trivial 1-day reversal rule; the best ones match it.** IC minus reversal is negative or
   ~0 everywhere (best: HGB F3 -0.005 +/- 0.007 val, +0.002 +/- 0.014 OOT; ridge F1/F2 are significantly below on val, -0.035 +/- 0.008).
   `log_ret` (the reversal input) is in every feature set, so we did not test whether the HGB signal is anything beyond
   reversal plus noise; that decomposition (e.g. drop log_ret and its lags/ranks) is not done here.
4. **Cross-sectional normalisation looks helpful but is not established.** Within HGB and ridge the F3 IC is higher than F1/F2
   on val (ridge +0.014 vs F1, HGB +0.013 vs F1) but we did not compute paired SEs for feature-set differences and on OOT the
   ordering is inconsistent (ridge F1 0.021 vs F3 0.018; HGB F1 0.029 vs F3 0.044). Treat as a hypothesis for the next experiment.
5. **Val to OOT decay** (best HGB 0.058 to 0.044, ridge and clf similar) is within noise (SE ~0.01-0.015 per window) and cannot
   be distinguished from stationarity. argmax L/S (pooled, sign of the score) is negative for most tabular models on OOT because
   the score sign is not calibrated per day; IC and quantile L/S are the meaningful columns.
6. **Production tickers (005930, 000660):** all |rho| <= 0.14 (< 2 SE) and hit-rates are within ~2 SE of 50% (SE ~0.07 / ~0.037); nothing can be
   said per ticker. A cross-sectional IC of ~0.05 says the ranking helps on average across 200 names, not that any single name is predictable.
7. **Selection caveat:** the 8 configs were all scored on val and OOT, and "best" above is picked with hindsight on those windows
   (mildly optimistic, ~8 tries); OOT was scored once per config. HGB hyper-parameters other than n_iter (or ridge alpha) were fixed a priori and untuned.

**What this supports:** the features are not empty and TFT-as-trained is extracting less than gradient boosting does; label/regularisation
experiments on the TFT are worth running with IC (not macro F1) as the criterion, and HGB-F3 / reversal (IC ~0.05-0.06) is the bar to clear.
**What it does not support:** a claim that tabular beats reversal, that the signal survives costs, or that cross-sectional ranks are the cause of the gain.
