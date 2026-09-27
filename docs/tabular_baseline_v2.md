# Tabular baseline v2 (E0v2): like-for-like reference for the TFT recipes R0..R3

Code commit: `9523e24+dirty` (`training/run_tabular_baseline.py`, `training/tabular_features.py`). Raw output: `training/artifacts/tabular_baseline_v2.json` (gitignored). Supersedes the numbers of [`tabular_baseline.md`](tabular_baseline.md) (old E0: unadjusted prices + legacy sample alignment; see the errata at its top and in [`signal_baseline.md`](signal_baseline.md)); those files are unchanged. Data rebuild: [`data_adjustment_report.md`](data_adjustment_report.md).

## What changed vs old E0

- **Data**: adjusted prices, `DATA_VERSION = adj1` (feature_pool/labels rebuilt; caches `__adj1`).
- **Sample set**: exactly the (ticker, date) list the TFT runner scores with `TickerDayDataset(align="today")` (sample date = the day the prediction is made = last encoder row; features at that day, target = its `next_day_return`). **val_2024**: 36929 samples (targets 2024-03-28..2024-12-30, 185 scored days); **oot_2026**: 33364 samples (targets 2026-01-02..2026-09-07, 167 scored days). The lists were asserted identical, in order, to `run_tfx_experiments.build_window_dataset` output and to the known counts (36,929 / 33,364). Old E0 used the legacy enumeration (36,729 / 33,364, dates shifted one row).
- Everything else is the old E0 protocol: train <= 2023-12-31; hyper-parameters selected on 2023 (fit <= 2022); val 2024 scored, OOT 2026 scored once per final config; 2025 rows never used; same feature sets F1/F2/F3, Ridge / HGB (sklearn HistGradientBoosting), fixed HGB params, `random_state=0`, random reference `default_rng(0)`.
- Added config `hgb_clf_label_F3` (HGB classifier on the fixed-threshold `label`; `hgb_clf_F3` uses `label_vn`) so both label versions are on record. Macro F1 (auxiliary) exists only for the classifiers: argmax class vs that label on samples with a non-null label; regressors have no class output.
- Reproducibility: seed 0, floor 0.007630 (`threshold_vn.json`), returns are raw `next_day_return` (`|ret|>30%` counts: val_2024 0, oot_2026 0; a clipped-return variant is in the JSON as `signal_clip30`).

## Selection on 2023 (fit on <= 2022)

| config | grid (mean daily IC on 2023) | chosen |
|---|---|---|
| ridge_F1 | 10.0: 0.030, 1000.0: 0.029, 100000.0: 0.021, 1000000.0: -0.002 | alpha=10 |
| hgb_reg_F1 | 25: nan, 50: -0.003, 100: 0.026, 200: 0.036, 300: 0.036, 400: 0.033 | n_iter=200 |
| ridge_F2 | 10.0: 0.031, 1000.0: 0.031, 100000.0: 0.024, 1000000.0: -0.000 | alpha=10 |
| hgb_reg_F2 | 25: nan, 50: -0.009, 100: 0.019, 200: 0.035, 300: 0.039, 400: 0.036 | n_iter=300 |
| ridge_F3 | 10.0: 0.038, 1000.0: 0.038, 100000.0: 0.048, 1000000.0: 0.040 | alpha=100000 |
| hgb_reg_F3 | 25: nan, 50: 0.002, 100: 0.055, 200: 0.048, 300: 0.049, 400: 0.046 | n_iter=100 |
| hgb_reg_csz_F3 | 25: 0.058, 50: 0.058, 100: 0.054, 200: 0.048, 300: 0.044, 400: 0.045 | n_iter=25 |
| hgb_clf_F3 | 25: 0.017, 50: 0.035, 100: 0.041, 200: 0.045, 300: 0.044, 400: 0.043 | n_iter=200 |
| hgb_clf_label_F3 | 25: 0.041, 50: 0.051, 100: 0.061, 200: 0.067, 300: 0.066, 400: 0.063 | n_iter=200 |

## val_2024 (36929 samples, 185 scored days)

Signal metrics (not UTIL). IC minus reversal is a paired difference over days (SE of the difference). Old IC = old E0 (unadjusted, legacy alignment; not like-for-like).

| scorer | mean rank IC | SE | IC IR | quantile L/S | argmax L/S | macro F1 | IC minus reversal (SE) | old E0 IC |
|---|---|---|---|---|---|---|---|---|
| Ridge, F1 | 0.0266 | 0.0095 | 0.207 | 0.14% | 0.03% | n/a | -0.0359 (0.0082) | 0.0277 |
| Ridge, F2 | 0.0261 | 0.0095 | 0.203 | 0.13% | -0.02% | n/a | -0.0364 (0.0087) | 0.0267 |
| Ridge, F3 | 0.0399 | 0.0096 | 0.305 | 0.21% | -0.07% | n/a | -0.0226 (0.0067) | 0.0415 |
| HGB reg (z), F1 | 0.0499 | 0.0092 | 0.400 | 0.18% | 0.19% | n/a | -0.0126 (0.0079) | 0.0443 |
| HGB reg (z), F2 | 0.0415 | 0.0089 | 0.345 | 0.16% | 0.22% | n/a | -0.0210 (0.0076) | 0.0496 |
| HGB reg (z), F3 | 0.0564 | 0.0099 | 0.417 | 0.21% | 0.14% | n/a | -0.0061 (0.0049) | 0.0576 |
| HGB reg (date-demeaned z), F3 | 0.0514 | 0.0097 | 0.390 | 0.19% | 0.20% | n/a | -0.0111 (0.0044) | 0.0536 |
| HGB clf (label_vn), F3 | 0.0456 | 0.0096 | 0.348 | 0.22% | 0.22% | 0.290 | -0.0169 (0.0068) | 0.0493 |
| HGB clf (fixed label), F3 | 0.0578 | 0.0092 | 0.463 | 0.21% | 0.30% | 0.340 | -0.0047 (0.0075) | - |
| 1-day reversal (-log_ret) | 0.0625 | 0.0106 | 0.433 | 0.27% | 0.13% | n/a | - | 0.0625 |
| random | -0.0027 | 0.0047 | -0.043 | -0.00% | 0.02% | n/a | - | - |

## oot_2026 (33364 samples, 167 scored days)

Signal metrics (not UTIL). IC minus reversal is a paired difference over days (SE of the difference). Old IC = old E0 (unadjusted, legacy alignment; not like-for-like).

| scorer | mean rank IC | SE | IC IR | quantile L/S | argmax L/S | macro F1 | IC minus reversal (SE) | old E0 IC |
|---|---|---|---|---|---|---|---|---|
| Ridge, F1 | 0.0186 | 0.0146 | 0.099 | 0.34% | -0.19% | n/a | -0.0237 (0.0137) | 0.0211 |
| Ridge, F2 | 0.0115 | 0.0142 | 0.063 | 0.28% | -0.17% | n/a | -0.0308 (0.0145) | 0.0130 |
| Ridge, F3 | 0.0147 | 0.0154 | 0.074 | 0.26% | -0.20% | n/a | -0.0276 (0.0149) | 0.0177 |
| HGB reg (z), F1 | 0.0233 | 0.0115 | 0.158 | 0.20% | -0.58% | n/a | -0.0190 (0.0138) | 0.0286 |
| HGB reg (z), F2 | 0.0261 | 0.0118 | 0.172 | 0.19% | -0.42% | n/a | -0.0162 (0.0131) | 0.0165 |
| HGB reg (z), F3 | 0.0308 | 0.0129 | 0.184 | 0.11% | -0.61% | n/a | -0.0115 (0.0107) | 0.0444 |
| HGB reg (date-demeaned z), F3 | 0.0408 | 0.0140 | 0.225 | 0.30% | 0.17% | n/a | -0.0015 (0.0060) | 0.0443 |
| HGB clf (label_vn), F3 | 0.0458 | 0.0140 | 0.254 | 0.35% | -0.69% | 0.260 | +0.0035 (0.0119) | 0.0332 |
| HGB clf (fixed label), F3 | 0.0371 | 0.0135 | 0.212 | 0.09% | -0.38% | 0.356 | -0.0052 (0.0122) | - |
| 1-day reversal (-log_ret) | 0.0423 | 0.0155 | 0.211 | 0.29% | 0.19% | n/a | - | 0.0427 |
| random | 0.0063 | 0.0057 | 0.086 | 0.05% | 0.04% | n/a | - | - |

## Change vs old E0 (mean rank IC, same config)

| config | val old | val v2 | oot old | oot v2 |
|---|---|---|---|---|
| Ridge, F1 | 0.0277 | 0.0266 | 0.0211 | 0.0186 |
| Ridge, F2 | 0.0267 | 0.0261 | 0.0130 | 0.0115 |
| Ridge, F3 | 0.0415 | 0.0399 | 0.0177 | 0.0147 |
| HGB reg (z), F1 | 0.0443 | 0.0499 | 0.0286 | 0.0233 |
| HGB reg (z), F2 | 0.0496 | 0.0415 | 0.0165 | 0.0261 |
| HGB reg (z), F3 | 0.0576 | 0.0564 | 0.0444 | 0.0308 |
| HGB reg (date-demeaned z), F3 | 0.0536 | 0.0514 | 0.0443 | 0.0408 |
| HGB clf (label_vn), F3 | 0.0493 | 0.0456 | 0.0332 | 0.0458 |
| HGB clf (fixed label), F3 | - | 0.0578 | - | 0.0371 |
| reversal | 0.0625 | 0.0625 | 0.0427 | 0.0423 |

## Reference values for the TFT recipes (R0..R3)

Compare a TFT recipe's `mean_daily_rank_ic` / `ic_ir` / quantile L/S on the same window against the rows above (bar: best tabular = `hgb_clf_label_F3`, val IC 0.0578, OOT IC 0.0371; reversal val 0.0625, OOT 0.0423). Selection of the bar row uses val only (OOT shown for confirmation).

`training/artifacts/tfx_results.json` not present at run time: TFT recipes not yet run, so no TFT rows.

## Production tickers (noisy)

Time-series Spearman(score, next_day_return) and sign hit-rate per ticker (n about 170-260 days; none distinguishable from 0 / 50%).

| scorer | window | 005930 rho | 005930 hit (up-rate) | 000660 rho | 000660 hit (up-rate) |
|---|---|---|---|---|---|
| Ridge, F1 | val_2024 | -0.070 | 0.491 (0.45), n=185 | -0.058 | 0.492 (0.52), n=185 |
| Ridge, F2 | val_2024 | -0.077 | 0.497 (0.45), n=185 | -0.054 | 0.475 (0.52), n=185 |
| Ridge, F3 | val_2024 | -0.065 | 0.503 (0.45), n=185 | -0.054 | 0.453 (0.52), n=185 |
| HGB reg (z), F1 | val_2024 | 0.087 | 0.549 (0.45), n=185 | -0.034 | 0.503 (0.52), n=185 |
| HGB reg (z), F2 | val_2024 | 0.104 | 0.577 (0.45), n=185 | 0.035 | 0.497 (0.52), n=185 |
| HGB reg (z), F3 | val_2024 | 0.044 | 0.531 (0.45), n=185 | -0.042 | 0.514 (0.52), n=185 |
| HGB reg (date-demeaned z), F3 | val_2024 | -0.058 | 0.509 (0.45), n=185 | 0.050 | 0.552 (0.52), n=185 |
| HGB clf (label_vn), F3 | val_2024 | 0.050 | 0.554 (0.45), n=185 | 0.039 | 0.470 (0.52), n=185 |
| HGB clf (fixed label), F3 | val_2024 | 0.085 | 0.571 (0.45), n=185 | 0.087 | 0.492 (0.52), n=185 |
| reversal | val_2024 | -0.004 | 0.509 (0.45), n=185 | 0.102 | 0.519 (0.52), n=185 |
| Ridge, F1 | oot_2026 | -0.043 | 0.533 (0.56), n=167 | -0.045 | 0.560 (0.58), n=167 |
| Ridge, F2 | oot_2026 | -0.045 | 0.539 (0.56), n=167 | -0.038 | 0.566 (0.58), n=167 |
| Ridge, F3 | oot_2026 | -0.075 | 0.539 (0.56), n=167 | -0.070 | 0.536 (0.58), n=167 |
| HGB reg (z), F1 | oot_2026 | -0.102 | 0.485 (0.56), n=167 | -0.128 | 0.506 (0.58), n=167 |
| HGB reg (z), F2 | oot_2026 | -0.062 | 0.509 (0.56), n=167 | -0.102 | 0.500 (0.58), n=167 |
| HGB reg (z), F3 | oot_2026 | -0.103 | 0.497 (0.56), n=167 | -0.083 | 0.524 (0.58), n=167 |
| HGB reg (date-demeaned z), F3 | oot_2026 | 0.095 | 0.564 (0.56), n=167 | 0.006 | 0.554 (0.58), n=167 |
| HGB clf (label_vn), F3 | oot_2026 | -0.083 | 0.479 (0.56), n=167 | -0.070 | 0.494 (0.58), n=167 |
| HGB clf (fixed label), F3 | oot_2026 | 0.003 | 0.497 (0.56), n=167 | -0.042 | 0.470 (0.58), n=167 |
| reversal | oot_2026 | 0.106 | 0.515 (0.56), n=167 | 0.072 | 0.476 (0.58), n=167 |

## Interpretation (SE-aware; differences are called real only at >= ~2 paired SE)

1. **Bar for the TFT recipes (like-for-like: adjusted data, `align="today"` sample sets, features at day d predicting d->d+1).**
   Best tabular by val IC = HGB classifier on the fixed label, F3: val IC 0.058 (SE 0.009), OOT 0.037 (SE 0.014).
   HGB regressor F3: 0.056 / 0.031. Reversal: 0.0625 / 0.0423. Quantile L/S is about 0.2% per day gross on val for the
   best rows (0.26% for reversal), 0.1-0.35% on OOT; costs not considered. A TFT recipe has to beat these, not the old E0 numbers.
2. **The rebuild changed little.** Old vs v2 IC of the same config moves by at most ~0.015 (HGB reg F3 OOT 0.044 -> 0.031,
   HGB reg F2 OOT 0.017 -> 0.026), i.e. inside the ~0.01-0.015 SE of one window; the sample set and price adjustment do not
   alter the picture. No adjusted-data return exceeds 30% in either window (`n_ret_over_30pct` = 0).
   The unchanged reversal IC (0.0625 val) is expected: rank IC is insensitive to the few adjusted corporate-action days.
3. **No tabular model is distinguishable from the 1-day reversal rule.** IC minus reversal is negative or ~0 everywhere; on val
   Ridge F1/F2/F3 and HGB reg F2, csz and clf(label_vn) are 2-4 paired SE below it, HGB reg F3 -0.006 (1.2 SE), clf(fixed label)
   -0.005 (0.6 SE). On OOT all are within ~2 SE (best clf(label_vn) +0.004 +/- 0.012). `log_ret` and its lags are inputs to every
   feature set, and no decomposition (dropping the reversal inputs) was done, so we do not know whether the HGB signal is more than
   reversal plus noise.
4. **Ranking of configs is unstable across windows** (val best: clf fixed label 0.058; OOT best: clf label_vn 0.046, csz 0.041),
   the per-window SE is 0.009-0.015 and config differences were not tested with paired SEs; treat the ordering as noise. Classifier
   argmax L/S is negative on OOT for most models (uncalibrated sign), so IC / quantile L/S are the meaningful columns.
5. **Label versions.** The fixed-threshold-label classifier is not worse than the label_vn classifier on val (0.058 vs 0.046) and
   slightly worse on OOT (0.037 vs 0.046); both differences are inside one SE. Macro F1 (auxiliary, own label each): fixed label
   0.340 val / 0.356 OOT, label_vn 0.290 / 0.260; not comparable across the two label definitions.
6. **Selection caveat.** 9 configs were all scored on val and OOT; the bar row is chosen on val with hindsight over ~9 tries
   (mildly optimistic). HGB hyper-parameters other than n_iter (and ridge alpha) were fixed a priori. `hgb_reg_csz_F3` again
   picked the smallest n_iter of the grid (25) with IC still falling, so its optimum may be lower.
7. **Provenance.** Numbers were produced by the working-tree version of `run_tabular_baseline.py` on top of commit `9523e24`
   (JSON `code_commit` reads `9523e24+dirty` because the runner was not yet committed); the committed runner differs only by
   this interpretation text. TFT recipes had not been run when this was written, so no TFT row is present;
   `--render-only` re-reads `tfx_results.json` and adds the TFT rows once it exists.
