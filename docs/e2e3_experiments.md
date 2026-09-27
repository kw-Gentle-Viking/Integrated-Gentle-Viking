# E2/E3: seed variance and overfitting suppression (R0 TFT recipe)

Plan: `docs/superpowers/plans/2026-09-26-signal-improvement-plan.md`. Code commit: `1c5a8f0` (`training/run_e2e3_experiments.py`, reuses `training/run_tfx_experiments.py`). Raw output: `training/artifacts/e2e3_results.json` (gitignored). Baseline (R0, seed 0) = `aligned` record of `training/artifacts/tfx_results.json`, not retrained.

**Selection discipline.** Variants are chosen on the val 2024 mean daily rank IC ONLY. The 2026 OOT window is scored exactly ONCE per recipe, on its final selected checkpoint, and is recorded for confirmation; it is NOT used to select an epoch, a variant or a hyper-parameter (it is never scored per epoch). The 2025 test window is never scored.

## Protocol

- All recipes: R0 = raw 33 champion columns (no preprocessing), fixed label, `align="today"`, adjusted prices, batch 128, max 12 epochs, class weights from the train label distribution, same val/OOT sample sets and metrics as TFX (SE = IC std / sqrt(days)).
- **E3** (seeds 1, 2): identical to R0 - val-loss checkpoint selection, patience 3. The val IC per epoch is recorded only (no effect on training). Seed 0 is the recorded R0 run.
- **E2** (V1..V6, seed 0): Adam with the stated lr / dropout / `weight_decay` (plain Adam L2, not AdamW), checkpoint = epoch with the highest val IC (patience 4); val loss recorded alongside.
- **Caveat**: R0/E3 select by val loss, E2 variants by val IC, so E2-vs-R0 differences mix the hyper-parameter change with the selection rule. Single-seed differences below ~2 SE (about 0.02 for val IC, see E3 seed spread) are within noise.

## E3: seed variance (R0 recipe)

| seed | source | val IC | val SE | OOT IC | OOT SE | best epoch (val loss) | epochs run |
|---|---|---|---|---|---|---|---|
| 0 | tfx_results.json `aligned` (reused) | 0.0463 | 0.0116 | 0.0395 | 0.0136 | 0 | 4 |
| 1 | e2e3 `e3_seed1` | 0.0244 | 0.0102 | 0.0294 | 0.0108 | 0 | 4 |
| 2 | e2e3 `e3_seed2` | 0.0033 | 0.0104 | 0.0137 | 0.0142 | 0 | 4 |

Across 3 seed(s): val IC mean 0.0246 +- 0.0215 (std, ddof=1), OOT IC mean 0.0276 +- 0.0130.

## val_2024 (selection window)

| scorer | mean rank IC | SE | IC IR | days | quantile L/S | quantile L/S (clip30) | argmax L/S | argmax L/S (clip30) | macro F1 | best epoch | epochs run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **R0 seed 0 (tfx)** | 0.0463 | 0.0116 | 0.293 | 185 | 0.10% | 0.10% | 0.23% | 0.23% | 0.3681 | 0 | 4 |
| **E3-s1 e3_seed1** | 0.0244 | 0.0102 | 0.175 | 185 | 0.11% | 0.11% | 0.31% | 0.31% | 0.3745 | 0 | 4 |
| **E3-s2 e3_seed2** | 0.0033 | 0.0104 | 0.023 | 185 | 0.05% | 0.05% | 0.17% | 0.17% | 0.3629 | 0 | 4 |
| **V4 v4_lr_do** | 0.0541 | 0.0116 | 0.342 | 185 | 0.27% | 0.27% | 0.29% | 0.29% | 0.3888 | 11 | 12 |
| **V1 v1_lr** | 0.0449 | 0.0117 | 0.282 | 185 | 0.14% | 0.14% | 0.41% | 0.41% | 0.3846 | 5 | 10 |
| **V2 v2_do** | 0.0467 | 0.0107 | 0.320 | 185 | 0.09% | 0.09% | -0.01% | -0.01% | 0.3816 | 5 | 10 |
| **V3 v3_wd** | 0.0412 | 0.0142 | 0.214 | 185 | 0.05% | 0.05% | n/a | n/a | 0.3023 | 3 | 8 |
| **V5 v5_lr_do_wd** | 0.0414 | 0.0138 | 0.220 | 185 | 0.10% | 0.10% | n/a | n/a | 0.3054 | 2 | 7 |
| **V3-s1 v3_wd_seed1** | 0.0395 | 0.0139 | 0.210 | 185 | 0.05% | 0.05% | -0.41% | -0.41% | 0.3389 | 7 | 12 |
| V4-s1 v4_lr_do_seed1 | _pending_ | | | | | | | | | | |
| V3-s2 v3_wd_seed2 | _pending_ | | | | | | | | | | |
| V4-s2 v4_lr_do_seed2 | _pending_ | | | | | | | | | | |
| TFT champion (S3, legacy-aligned) | 0.0143 | 0.0108 | 0.098 | 184 | 0.14% | n/a | 0.29% | n/a | 0.3823 | n/a | n/a |
| 1-day reversal (S3, legacy sample set) | 0.0625 | 0.0107 | 0.432 | 184 | 0.27% | n/a | 0.13% | n/a | n/a | n/a | n/a |
| HGB reg (z), F3 (E0v2) | 0.0564 | 0.0099 | 0.417 | 185 | 0.21% | n/a | 0.14% | n/a | n/a | n/a | n/a |
| HGB clf fixed label, F3 (E0v2) | 0.0578 | 0.0092 | 0.463 | 185 | 0.21% | n/a | 0.30% | n/a | n/a | n/a | n/a |

Delta IC vs R0 seed 0: E3-s1 -0.0219, E3-s2 -0.0429, V4 0.0078, V1 -0.0014, V2 0.0004, V3 -0.0050, V5 -0.0048, V3-s1 -0.0068.

## oot_2026 (final checkpoint scored once; confirmation only, not a selection criterion)

| scorer | mean rank IC | SE | IC IR | days | quantile L/S | quantile L/S (clip30) | argmax L/S | argmax L/S (clip30) | macro F1 | best epoch | epochs run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **R0 seed 0 (tfx)** | 0.0395 | 0.0136 | 0.224 | 167 | -0.01% | -0.01% | 0.07% | 0.07% | 0.3860 | 0 | 4 |
| **E3-s1 e3_seed1** | 0.0294 | 0.0108 | 0.210 | 167 | 0.19% | 0.19% | 0.34% | 0.34% | 0.3536 | 0 | 4 |
| **E3-s2 e3_seed2** | 0.0137 | 0.0142 | 0.075 | 167 | 0.11% | 0.11% | -0.31% | -0.31% | 0.3526 | 0 | 4 |
| **V4 v4_lr_do** | 0.0348 | 0.0152 | 0.177 | 167 | 0.12% | 0.12% | 0.13% | 0.13% | 0.3764 | 11 | 12 |
| **V1 v1_lr** | 0.0338 | 0.0148 | 0.177 | 167 | 0.09% | 0.09% | 0.03% | 0.03% | 0.3313 | 5 | 10 |
| **V2 v2_do** | 0.0275 | 0.0138 | 0.154 | 167 | 0.04% | 0.04% | -0.19% | -0.19% | 0.3603 | 5 | 10 |
| **V3 v3_wd** | 0.0392 | 0.0184 | 0.165 | 167 | 0.16% | 0.16% | n/a | n/a | 0.2888 | 3 | 8 |
| **V5 v5_lr_do_wd** | 0.0217 | 0.0139 | 0.121 | 167 | -0.11% | -0.11% | n/a | n/a | 0.2697 | 2 | 7 |
| **V3-s1 v3_wd_seed1** | 0.0392 | 0.0177 | 0.171 | 167 | 0.16% | 0.16% | n/a | n/a | 0.2843 | 7 | 12 |
| V4-s1 v4_lr_do_seed1 | _pending_ | | | | | | | | | | |
| V3-s2 v3_wd_seed2 | _pending_ | | | | | | | | | | |
| V4-s2 v4_lr_do_seed2 | _pending_ | | | | | | | | | | |
| TFT champion (S3, legacy-aligned) | -0.0049 | 0.0154 | -0.025 | 167 | -0.13% | n/a | -0.54% | n/a | 0.3420 | n/a | n/a |
| 1-day reversal (S3, legacy sample set) | 0.0427 | 0.0155 | 0.212 | 167 | 0.29% | n/a | 0.19% | n/a | n/a | n/a | n/a |
| HGB reg (z), F3 (E0v2) | 0.0308 | 0.0129 | 0.184 | 167 | 0.11% | n/a | -0.61% | n/a | n/a | n/a | n/a |
| HGB clf fixed label, F3 (E0v2) | 0.0371 | 0.0135 | 0.212 | 167 | 0.09% | n/a | -0.38% | n/a | n/a | n/a | n/a |

Delta IC vs R0 seed 0: E3-s1 -0.0100, E3-s2 -0.0257, V4 -0.0047, V1 -0.0056, V2 -0.0119, V3 -0.0002, V5 -0.0177, V3-s1 -0.0003.

## Hyper-parameters, selected epochs and val curves (all recipes)

| recipe | lr | dropout | weight_decay | state | selection | best epoch | best val IC | min val loss | stop |
|---|---|---|---|---|---|---|---|---|---|
| E3-s1 e3_seed1 | 0.000290589 | 0.1658 | 0 | 32 | val_loss | 0 | 0.0244 | 1.0899 | early_stopping |
| E3-s2 e3_seed2 | 0.000290589 | 0.1658 | 0 | 32 | val_loss | 0 | 0.0033 | 1.0893 | early_stopping |
| V4 v4_lr_do | 0.0001 | 0.3 | 0 | 32 | val_ic | 11 | 0.0541 | 1.0886 | max_epochs |
| V1 v1_lr | 0.0001 | 0.1658 | 0 | 32 | val_ic | 5 | 0.0449 | 1.0900 | early_stopping |
| V2 v2_do | 0.000290589 | 0.3 | 0 | 32 | val_ic | 5 | 0.0467 | 1.0969 | early_stopping |
| V3 v3_wd | 0.000290589 | 0.1658 | 0.001 | 32 | val_ic | 3 | 0.0412 | 1.0949 | early_stopping |
| V5 v5_lr_do_wd | 0.0001 | 0.3 | 0.001 | 32 | val_ic | 2 | 0.0414 | 1.0947 | early_stopping |
| V3-s1 v3_wd_seed1 | 0.000290589 | 0.1658 | 0.001 | 32 | val_ic | 7 | 0.0395 | 1.0942 | early_stopping |
| V4-s1 v4_lr_do_seed1 | _pending_ | | | | | | | | |
| V3-s2 v3_wd_seed2 | _pending_ | | | | | | | | |
| V4-s2 v4_lr_do_seed2 | _pending_ | | | | | | | | |

Per-epoch val curves (`*` = selected epoch):

- **E3-s1 e3_seed1**: 0*: IC 0.0244 / loss 1.0899; 1: IC 0.0358 / loss 1.0990; 2: IC 0.0270 / loss 1.1097; 3: IC 0.0290 / loss 1.1331
- **E3-s2 e3_seed2**: 0*: IC 0.0033 / loss 1.0893; 1: IC 0.0328 / loss 1.1147; 2: IC 0.0314 / loss 1.1195; 3: IC 0.0419 / loss 1.1210
- **V4 v4_lr_do**: 0: IC 0.0190 / loss 1.0973; 1: IC 0.0171 / loss 1.0961; 2: IC 0.0393 / loss 1.0913; 3: IC 0.0427 / loss 1.0948; 4: IC 0.0359 / loss 1.0957; 5: IC 0.0441 / loss 1.0886; 6: IC 0.0407 / loss 1.0940; 7: IC 0.0501 / loss 1.0958; 8: IC 0.0457 / loss 1.0932; 9: IC 0.0433 / loss 1.0952; 10: IC 0.0517 / loss 1.0927; 11*: IC 0.0541 / loss 1.1028
- **V1 v1_lr**: 0: IC 0.0283 / loss 1.0950; 1: IC 0.0299 / loss 1.1000; 2: IC 0.0361 / loss 1.0944; 3: IC 0.0381 / loss 1.1037; 4: IC 0.0289 / loss 1.0992; 5*: IC 0.0449 / loss 1.0900; 6: IC 0.0262 / loss 1.0961; 7: IC 0.0315 / loss 1.0998; 8: IC 0.0252 / loss 1.0987; 9: IC 0.0356 / loss 1.1023
- **V2 v2_do**: 0: IC 0.0353 / loss 1.0969; 1: IC 0.0423 / loss 1.1028; 2: IC 0.0248 / loss 1.0991; 3: IC 0.0222 / loss 1.1205; 4: IC 0.0184 / loss 1.1449; 5*: IC 0.0467 / loss 1.1319; 6: IC 0.0348 / loss 1.1675; 7: IC 0.0342 / loss 1.1809; 8: IC 0.0302 / loss 1.1647; 9: IC 0.0297 / loss 1.1871
- **V3 v3_wd**: 0: IC 0.0262 / loss 1.1000; 1: IC 0.0395 / loss 1.1012; 2: IC 0.0359 / loss 1.0998; 3*: IC 0.0412 / loss 1.1015; 4: IC 0.0404 / loss 1.1081; 5: IC 0.0398 / loss 1.0949; 6: IC 0.0402 / loss 1.0979; 7: IC 0.0405 / loss 1.0997
- **V5 v5_lr_do_wd**: 0: IC 0.0384 / loss 1.0947; 1: IC 0.0414 / loss 1.1012; 2*: IC 0.0414 / loss 1.0995; 3: IC 0.0397 / loss 1.1000; 4: IC 0.0388 / loss 1.1086; 5: IC 0.0394 / loss 1.0973; 6: IC 0.0395 / loss 1.0992
- **V3-s1 v3_wd_seed1**: 0: IC 0.0386 / loss 1.1027; 1: IC 0.0386 / loss 1.0980; 2: IC 0.0376 / loss 1.1002; 3: IC 0.0381 / loss 1.1034; 4: IC 0.0386 / loss 1.0965; 5: IC 0.0373 / loss 1.0989; 6: IC 0.0387 / loss 1.0998; 7*: IC 0.0395 / loss 1.0982; 8: IC 0.0392 / loss 1.0987; 9: IC 0.0380 / loss 1.0985; 10: IC 0.0109 / loss 1.1018; 11: IC 0.0387 / loss 1.0942

Comparison rows (reversal, HGB) are from the S3 / E0v2 JSON files; legacy-aligned or old-data, raw spreads only, **not like-for-like** with the aligned TFT rows.
