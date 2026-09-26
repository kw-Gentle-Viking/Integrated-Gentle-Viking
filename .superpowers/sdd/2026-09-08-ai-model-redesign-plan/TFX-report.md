# TFX report: TFT retraining recipes R0..R3 (CPU-side build, GPU run not launched)

## Alignment change (read first)
`training/dataset.py::TickerDayDataset` had a horizon misalignment: encoder = rows [t, t+60) but label/future/static came
from row t+60, so the model never saw the labelled day's own features (effective 2-day-ahead forecast) while serving feeds
59 days + today as the 60th step. Added `align="legacy"|"today"` (+ `target_offset` property, `target_offset_of(ds)` helper).
- `legacy` (default) is byte-for-byte the old behaviour; the real-cache check below reproduces S3's 36,729 / 33,364 sample lists exactly.
- `today`: label/future/static from the last encoder row; windows = len - 60 + 1; NaN labels skipped.
- Consumers switched to `target_offset`: `training/signal_data.py` (`sample_meta`, `filter_index_by_target_date`),
  `run_stage1_champion_selection.dates_for_dataset`, `stage1_data.compute_label_distribution` (only my hunk; DATAADJ's own
  uncommitted DATA_VERSION edits to stage1_data.py were left untouched). `run_tabular_baseline.py` builds legacy datasets with the default, so it is unchanged and still legacy-correct.
- Serving-consistency test (`training/test_dataset_alignment.py`): a 60-row `serving.feature_builder.build_encoder_df` frame gives identical hist/fut/static tensors (shape, dtype, values) from an `align="today"` sample and from `serving.inference.run_inference`.
- `training/run_stage2_final_live.py --align {legacy,today}` (default legacy = unchanged). `today` adds run_name suffix `-align-today`, shrinks the held-out buffer to 59 rows and never auto-promotes to serving. **The final Stage-2 retrain must use `--align today` later.**

## What was built (commits, oldest first)
| SHA | content |
|---|---|
| 9d78d63 | `training/preprocess.py` (+14 tests): `fit_preprocessor` / `apply_preprocessor` / save / `load_preprocessor`, `add_cs_rank_inputs` |
| 345f4c9 | `run_training`: optional `early_stopping_patience`, `should_stop` hook (budget), `epoch_log`; patience counter persisted in the epoch checkpoint (`epochs_since_improve`, only written when patience set; old checkpoints resume with 0); returns `stopped_reason`, `last_epoch` (+9 tests) |
| fadbad1 | Dataset `align` + consumers + tests (5 dataset, 5 alignment/serving-consistency) |
| 2fff3dc | `run_stage2_final_live --align` |
| ca94321 | `training/run_tfx_experiments.py` + tests (12, incl. full CPU dry run) |
| 4d3cf0d | `training/run_tfx_watch.sh` + tests (6) |

## Recipes (all align="today", champion arch state 32/heads 8/layers 2/dropout .1658/lr .000291, batch 128, max 12 epochs, patience 3, balanced class weights from each recipe's glitch-masked TRAIN label counts)
- R0 `aligned`: raw 33 columns, fixed label (isolates the alignment fix). No preprocessing artifact.
- R1 `std`: train-fit clip(0.5/99.5)+z-score, binary untouched, constant columns dropped -> 30 inputs.
- R2 `std_vn`: R1 + `label_vn`.
- R3 `std_vn_csr`: R2 + 6 centred per-date rank inputs (log_ret, disparity_5d/20d/60d, rsi_14, volatility_20d; via `tabular_features.add_cs_ranks`) -> 36 inputs.
Train glitch masking (|next_day_return|>0.31 -> label NaN) is train-only; `next_day_return` is joined onto the frames after cache load (`fetch_next_day_returns`, read-only), so default training behaviour of other scripts is unchanged.
Eval sample set = fixed-label NULL gate for every recipe (rows where only `label_vn` is NULL stay in the set with a placeholder label and are excluded from F1 via `label_ok`), so all recipes are scored on identical samples.

## Kept / dropped columns (real train frames, rows <= 2023-12-28, 246,026 rows)
Dropped as constant: `lev_total_aum`, `lev_aum_to_mktcap`, `est_rebalancing_flow`. Kept 30: everything else (binary flags is_dividend, is_bonus_issue, is_rights_offering, is_split, is_vi_triggered untouched; day_of_week and the rest continuous). R3 adds 6 `_csr` columns. NaN -> 0.0 after transform; note the cached frames already carry NaN -> raw 0.0 from `build_ticker_dfs`, so that fill only matters for newly created columns (fit is on those cached values, i.e. consistent with pipeline fill semantics). If a continuous column's quantile clip degenerates (>99.5% zeros) the bounds fall back to min/max instead of killing it.

## Sample-count / window assertions (real cached frames, structural only; old pre-adjustment caches, no DB, no model)
- `--align legacy`: val_2024 36,729 samples, OOT 33,364, and the (ticker, date) lists identical to S3's npz (R0 and R1).
- `--align today` (used by all recipes): val_2024 36,929 (+200 = first window per ticker), OOT 33,364 (unchanged; extra windows fall in the lookback and are dropped by the date filter); val target dates 2024-03-28..2024-12-30, OOT 2026-01-02..2026-09-07, no 2025-dated sample, all dates >= window start. Aligned recipes do NOT assert legacy counts; the runner asserts date bounds + no-2025 always, legacy counts/lists only when `--align legacy`.
- Preprocessing fit is guarded (`max_date=2023-12-31`); real fit max date 2023-12-28.

## TDD / dry-run evidence
- Red then green shown for preprocess (ImportError, then 1 failing test fixed), early stopping (KeyError 'stopped_reason' before the change), dataset align (5 failures before, 9 pass after); consumer switch verified by temporarily reverting the two consumers (3 alignment tests fail) then restoring.
- CPU end-to-end (`training/test_run_tfx_experiments.py::test_full_dry_run_interrupt_resume_and_skip`, synthetic frames through the real runner: real window logic, real training, tiny model, 2 epochs, temp output paths): invocation 1 budget-stops after epoch 0 of R0 (resumable checkpoint, nothing recorded, no model_versions file); invocation 2 crashes during validation; invocation 3 resumes R0 from epoch 1 (exactly 1 epoch trained, then 2 each for R1..R3) and writes results JSON with 4 recipes, doc, 4 model_versions rows in order; invocation 4 skips everything (0 epochs trained, model_versions unchanged). Also: `--recipes` filter, window-count mismatch fails BEFORE training starts, `--check-windows` mode, budget stop only after the epoch checkpoint is saved.
- Watcher: `bash -n` ok; tests stub nvidia-smi/python: done-check, abort after 3 quick failures, cooldown+resume after rc 143, COMPLETE line.
- Full suite: see final line below.

## Outputs of a real run (written by the runner; paths overridable via CLI)
`training/artifacts/tfx_results.json` (atomic write), `training/artifacts/preproc_<recipe>.json` (R1..R3), `docs/tfx_experiments.md` (recipe table with alignment/preprocessing, val + OOT tables with IC/SE/IR/quantile+argmax spreads raw and clip30/F1/pred share, per-class P/R, comparison rows, windows/sample counts, dropped columns, commit, "Selection by val IC; OOT is confirmation only"), one `docs/model_versions.md` row per recipe (`tfx-<recipe>`), per-epoch log lines.
Comparison rows (TFT champion + reversal from `signal_baseline.json`, HGB F3 from `tabular_baseline.json`) are labelled "legacy-aligned TFT / old-data tabular; not like-for-like".

## Launch command for the watcher (NOT launched; wait until DATAADJ finishes the rebuild)
```
tmux new-session -d -s tfx-watch 'bash /home/user/AI_Gentle_Viking_RE/.worktrees/ai-model-redesign/training/run_tfx_watch.sh 2>&1 | tee -a /home/user/AI_Gentle_Viking_RE/.worktrees/ai-model-redesign/training/artifacts/tfx_watch.log'
```
Free the GPU: `pkill -f "training/[r]un_tfx_experiments"` (bracket trick; see script header). Completion line: `tfx experiments COMPLETE`.

## Deviations / open concerns
- Recipe set is R0..R3 (R0 added per coordinator); watcher waits for 4 recipes.
- label_vn frames and the adjusted-data caches (`__adj1`) do not exist yet: the first real invocation builds them from the DB via `load_or_build_ticker_dfs` (read-only). **Do not launch until the DATAADJ rebuild is complete**, otherwise a cache built mid-rebuild would be frozen under the `adj1` name. `threshold_vn.json` must also be regenerated on adjusted data first (label_vn in the DB depends on it).
- Old caches are still at their plain names (not moved yet when checked); my structural check used them directly.
- Early-stopping/val loss uses the 2024 val window (as the plan specifies); selection metric remains val IC.
- After a resume, DataLoader shuffle order differs from an uninterrupted run (seed set once per recipe start); harmless but not bit-reproducible across interruptions.
- Val-loss uses recipe-natural labels (label_vn NaN rows skipped); F1 for R2/R3 is against label_vn and not comparable with R0/R1.
- `argmax` spreads for label-free comparison rows use raw returns only.
- Stage-2 final retrain must later run with `--align today`; serving consistency of R1-R3 additionally needs scaler artifact loading in serving (and a rank-input design for R3), see plan.

## Full suite
`CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -m pytest serving/ training/ evaluation/ features/ -q` -> 221 passed (31m41s; includes DATAADJ's uncommitted stage1_data edits/tests present in the working tree at that time).
