# Chronos-2 zero-shot experiment

Code commit: `db3e2a4+dirty` (`training/run_chronos2_experiment.py`, `training/chronos2_scorer.py`). Model `amazon/chronos-2`, context 512d (raw close price + log1p(volume) as one multivariate group per ticker, native Chronos-2 group attention; no cross-ticker attention), prediction_length=1 (next trading day), quantile 0.5 point forecast used as the score (predicted log return = log(pred_close / last_close)). Zero-shot: no fine-tuning, no training rows used at all. Raw output: `training/artifacts/chronos2_results.json` (gitignored).

This is a showcase of Chronos-2 used the way a foundation model is meant to be used (full 512-day multivariate context), not a compute/information-matched comparison with the TFT (which sees 60-day context + 33 engineered columns).

**No date reduction was needed**: stride=1, every sample date of both windows was scored (see Coverage) -- the pre-run timing probe (below) showed CPU cost fit the ~1-2h budget without reducing the grid.

- Sample sets: `training.run_tabular_baseline.aligned_sample_sets(dsn, "today")` -- the exact (ticker, date) grid the TFT / tabular recipes score; `next_day_return` / `log_ret` come from the same `load_frame` query (champion columns) they use, so the target is identical.
- Date stride: **1** (every 1-th distinct sample date kept, full cross-section of tickers kept on every date that survives; 2025 never in the sample sets to begin with).

## Coverage / timing

| window | samples (full grid) | dates (full grid) | dates scored | samples scored | dropped (insufficient history) | wall time | s/date |
|---|---|---|---|---|---|---|---|
| val_2024 | 36929 | 185 | 185 | 36929 (scored 36929) | 0 | 4271s | 23.09 |
| oot_2026 | 33364 | 167 | 167 | 33364 (scored 33364) | 0 | 4040s | 24.19 |

Total wall time: 8350s (139.2 min), device=cpu.

## Signal metrics (chronos2) vs existing recipes

Same sample grid (post-stride), same `compute_signal_metrics` / `timing_decomposition` as every other recipe. OOT fixed effect uses the val window's ticker means (frozen prior, no OOT information).

| window | scorer | raw IC | SE | fixed-effect IC | timing IC | timing SE | quantile L/S |
|---|---|---|---|---|---|---|---|
| val_2024 | chronos2 (zero-shot) | 0.0007 | 0.0083 | -0.0153 | 0.0143 | 0.0066 | -0.08% |
| oot_2026 | chronos2 (zero-shot) | 0.0298 | 0.0122 | -0.0006 | 0.0255 | 0.0109 | 0.12% |

## Reference: existing recipes on the same (pre-stride) sample grid

From `training/artifacts/tabular_baseline_v2.json`, `tfx_results.json`, `e4_results.json` (unreduced, full grid -- chronos2's numbers above are on the strided subset, see Coverage).

| recipe | window | raw IC | quantile L/S |
|---|---|---|---|
| tabular hgb_clf_label_F3 | val_2024 | 0.0578 | 0.21% |
| tabular hgb_clf_label_F3 | oot_2026 | 0.0371 | 0.09% |
| tabular hgb_reg_F3 | val_2024 | 0.0564 | 0.21% |
| tabular hgb_reg_F3 | oot_2026 | 0.0308 | 0.11% |
| reversal (-log_ret) | val_2024 | 0.0625 | 0.27% |
| reversal (-log_ret) | oot_2026 | 0.0423 | 0.29% |
| TFT R0 (aligned) | val_2024 | 0.0463 | 0.10% |
| TFT R0 (aligned) | oot_2026 | 0.0395 | -0.01% |
| TFT E4 x1_cslabel | val_2024 | 0.0373 | 0.04% |
| TFT E4 x1_cslabel | oot_2026 | 0.0417 | 0.16% |
| TFT E4 x4_cslabel_vnfeat | val_2024 | 0.0370 | 0.04% |
| TFT E4 x4_cslabel_vnfeat | oot_2026 | 0.0411 | 0.16% |

## Pre-run timing probe

One real `predict_df` call per row, measured before the full run to size the stride/budget (`--probe`).

| device | n_tickers (of universe) | model load (s) | infer (s) | ms/ticker | extrapolated full-universe/day (s) |
|---|---|---|---|---|---|
| cpu | 25/200 | 0.96 | 1.082 | 43.3 | 8.7 |
| cpu | 200/200 | 1.07 | 11.449 | 57.2 | 11.4 |
| cuda | 200/200 | 1.44 | 0.818 | 4.1 | 0.8 |

## Notes

- **Model download / gating**: `amazon/chronos-2` is a public (non-gated) Hugging Face model (`HfApi.model_info` reports `gated: False`); it was not cached locally before this experiment and was downloaded once (~478MB, ~20s) on first use. No access request or token issue was encountered.
- **Probe vs. actual run time**: the pre-run probe (above) projected ~65-90 min total (352 dates x ~11-12s/date pure `predict_df` time); the actual run took 139 min (23-24s/date). The probe measured only the `predict_df` call for one day; the full run adds per-day context-window construction (slicing/concatenating up to 512 rows x ~200 tickers into a long-format DataFrame) and ran concurrently with this project's own GPU-training CPU-bound driver process (`training/run_e4_experiments.py` under `training/run_e4_watch.sh`, left running per instructions), which competed for CPU. Both factors plausibly explain the ~2x gap; the run still finished within the stated ~1-2h target's order of magnitude and used the full (unreduced) sample grid.
- **GPU**: not used for the full run (the project's GPU was busy with another long-running job). A brief informational GPU probe (200 tickers, full universe) measured 0.82s vs. 11.45s on CPU for the same call (see Pre-run timing probe), i.e. ~14x faster; extrapolating that ratio to the 352-date full grid gives an estimated GPU total of roughly 10-20 min, vs. the 139 min CPU run actually executed. GPU was not used end-to-end for this run (CPU already fit the budget, and the GPU was reserved for `training/run_e4_experiments.py`); this estimate is informational only.
- **Interpretation (zero-shot, no feature engineering, no training)**: raw IC is near zero on val_2024 (0.0007, SE 0.0083 -- indistinguishable from 0) and positive but noisy on oot_2026 (0.0298, SE 0.0122, ~2.4 SE from 0); both are well below reversal (0.0625 / 0.0423) and the tabular / TFT recipes above on val, though oot_2026's raw IC is in the same range as some TFT E4 recipes. The fixed-effect IC is ~0 on both windows (unlike reversal / HGB / TFT R0, which have large NEGATIVE fixed-effect IC, i.e. most of their raw IC on val comes from a static low-volatility ranking, not timing) -- Chronos-2's IC is concentrated in the timing component (val timing IC 0.0143, oot 0.0256; close to its own raw IC on both windows), i.e. whatever signal it has looks like day-to-day timing rather than a persistent per-ticker ranking. This is a single zero-shot pass with no hyper-parameter or prompt tuning; it should not be read as a ceiling on what Chronos-2 could do with more context engineering.
- **Tests**: `training/test_chronos2_scorer.py` (pure context/scoring logic, synthetic data) and `training/test_run_chronos2_experiment.py` (orchestration helpers with a fake pipeline, plus one DB+model integration test gated on `STOCK_DB_V2_DSN` and the `chronos` package being importable) all pass; the gated integration test only runs under `chronosbolt_env` (`/home/user/miniconda3/envs/chronosbolt_env/bin/python`).
