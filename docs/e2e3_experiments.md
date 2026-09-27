# E2/E3: seed variance and overfitting suppression (R0 TFT recipe)

Plan: `docs/superpowers/plans/2026-09-26-signal-improvement-plan.md`. Code commit: `d4a82f6` (`training/run_e2e3_experiments.py`, reuses `training/run_tfx_experiments.py`). Raw output: `training/artifacts/e2e3_results.json` (gitignored). Baseline (R0, seed 0) = `aligned` record of `training/artifacts/tfx_results.json`, not retrained.

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

Across 1 seed(s): val IC mean 0.0463 +- n/a (std, ddof=1), OOT IC mean 0.0395 +- n/a. (fewer than 3 seeds recorded so far)

## val_2024 (selection window)

| scorer | mean rank IC | SE | IC IR | days | quantile L/S | quantile L/S (clip30) | argmax L/S | argmax L/S (clip30) | macro F1 | best epoch | epochs run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **R0 seed 0 (tfx)** | 0.0463 | 0.0116 | 0.293 | 185 | 0.10% | 0.10% | 0.23% | 0.23% | 0.3681 | 0 | 4 |
| E3-s1 e3_seed1 | _pending_ | | | | | | | | | | |
| E3-s2 e3_seed2 | _pending_ | | | | | | | | | | |
| V4 v4_lr_do | _pending_ | | | | | | | | | | |
| V1 v1_lr | _pending_ | | | | | | | | | | |
| V2 v2_do | _pending_ | | | | | | | | | | |
| V3 v3_wd | _pending_ | | | | | | | | | | |
| V5 v5_lr_do_wd | _pending_ | | | | | | | | | | |
| TFT champion (S3, legacy-aligned) | 0.0143 | 0.0108 | 0.098 | 184 | 0.14% | n/a | 0.29% | n/a | 0.3823 | n/a | n/a |
| 1-day reversal (S3, legacy sample set) | 0.0625 | 0.0107 | 0.432 | 184 | 0.27% | n/a | 0.13% | n/a | n/a | n/a | n/a |
| HGB reg (z), F3 (E0v2) | 0.0564 | 0.0099 | 0.417 | 185 | 0.21% | n/a | 0.14% | n/a | n/a | n/a | n/a |
| HGB clf fixed label, F3 (E0v2) | 0.0578 | 0.0092 | 0.463 | 185 | 0.21% | n/a | 0.30% | n/a | n/a | n/a | n/a |

Delta IC vs R0 seed 0: none finished yet.

## oot_2026 (final checkpoint scored once; confirmation only, not a selection criterion)

| scorer | mean rank IC | SE | IC IR | days | quantile L/S | quantile L/S (clip30) | argmax L/S | argmax L/S (clip30) | macro F1 | best epoch | epochs run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **R0 seed 0 (tfx)** | 0.0395 | 0.0136 | 0.224 | 167 | -0.01% | -0.01% | 0.07% | 0.07% | 0.3860 | 0 | 4 |
| E3-s1 e3_seed1 | _pending_ | | | | | | | | | | |
| E3-s2 e3_seed2 | _pending_ | | | | | | | | | | |
| V4 v4_lr_do | _pending_ | | | | | | | | | | |
| V1 v1_lr | _pending_ | | | | | | | | | | |
| V2 v2_do | _pending_ | | | | | | | | | | |
| V3 v3_wd | _pending_ | | | | | | | | | | |
| V5 v5_lr_do_wd | _pending_ | | | | | | | | | | |
| TFT champion (S3, legacy-aligned) | -0.0049 | 0.0154 | -0.025 | 167 | -0.13% | n/a | -0.54% | n/a | 0.3420 | n/a | n/a |
| 1-day reversal (S3, legacy sample set) | 0.0427 | 0.0155 | 0.212 | 167 | 0.29% | n/a | 0.19% | n/a | n/a | n/a | n/a |
| HGB reg (z), F3 (E0v2) | 0.0308 | 0.0129 | 0.184 | 167 | 0.11% | n/a | -0.61% | n/a | n/a | n/a | n/a |
| HGB clf fixed label, F3 (E0v2) | 0.0371 | 0.0135 | 0.212 | 167 | 0.09% | n/a | -0.38% | n/a | n/a | n/a | n/a |

Delta IC vs R0 seed 0: none finished yet.

## Hyper-parameters, selected epochs and val curves (all recipes)

| recipe | lr | dropout | weight_decay | state | selection | best epoch | best val IC | min val loss | stop |
|---|---|---|---|---|---|---|---|---|---|
| E3-s1 e3_seed1 | _pending_ | | | | | | | | |
| E3-s2 e3_seed2 | _pending_ | | | | | | | | |
| V4 v4_lr_do | _pending_ | | | | | | | | |
| V1 v1_lr | _pending_ | | | | | | | | |
| V2 v2_do | _pending_ | | | | | | | | |
| V3 v3_wd | _pending_ | | | | | | | | |
| V5 v5_lr_do_wd | _pending_ | | | | | | | | |

Per-epoch val curves (`*` = selected epoch):


Comparison rows (reversal, HGB) are from the S3 / E0v2 JSON files; legacy-aligned or old-data, raw spreads only, **not like-for-like** with the aligned TFT rows.
