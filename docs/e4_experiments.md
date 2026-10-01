# E4: timing-signal experiments (TFT, R0 base)

Background: `docs/signal_diagnosis.md`. Code commit: `db3e2a4` (`training/run_e4_experiments.py`, `training/signal_diagnostics.py`, reuses `training/run_tfx_experiments.py`). Raw output: `training/artifacts/e4_results.json` (gitignored). Baseline row = V3 seed 0 (`v3_wd` of `training/artifacts/e2e3_results.json`), not retrained.

**Goal.** V3's IC of ~0.04 is entirely a fixed per-ticker ranking (identical to a low-volatility factor); its timing IC (score minus the ticker's mean score) is negative: val -0.0253, OOT -0.0393. The fixed label threshold (+-1.238%) makes volatile tickers BUY/SELL-heavy, so ticker volatility is the easiest thing to learn. E4 keeps the TFT and changes the label / inputs so that the only thing left to learn is timing. Success criterion: timing IC > 0 (val AND OOT), not raw IC.

**Selection discipline.** Epoch selection and early stopping use the val 2024 TIMING IC only. The 2026 OOT window is scored exactly ONCE per recipe, on its final selected checkpoint, for confirmation; it is never used to select an epoch, a recipe or a hyper-parameter (never scored per epoch). The 2025 test window is never scored.

## Protocol

- All recipes: R0 = raw 33 champion columns (no preprocessing), `align="today"`, adjusted prices, champion architecture/hparams (`training/champion_config.json`), Adam `weight_decay` 0.001 (L2, as V3), seed 0, batch 128, max 12 epochs, patience 4, class weights = balanced from each recipe's own train label distribution (after glitch masking). Same val/OOT sample sets as V3 (the fixed-label gate); samples the recipe label cannot label are kept for the IC and marked out of the F1.
- **Timing IC.** score = p_buy - p_sell. fixed effect = the ticker's mean score (val window: its own mean over the window; OOT: the val-window means of the SAME final checkpoint - scores only, never returns). timing = score - fixed effect. All three ICs (raw, fixed-effect, timing) use the `evaluation.evaluate` daily Spearman rule (>= 20 names per day, constant days skipped; SE = std / sqrt(days)). Per-epoch val curve: timing / raw IC / val loss.
- **Cross-sectional label (X1, X2, X4).** Per trading day, over the names with a valid next_day_return: top 30% = BUY(0), bottom 30% = SELL(2), rest HOLD(1). Average-rank percentile p = (rank - 0.5)/n; BUY if p >= 0.7, SELL if p <= 0.3 (tied returns share one label; no dependence on ticker order). Days with fewer than 20 eligible names -> NaN. Glitch rows (|next_day_return| > 0.31) are masked out of the TRAIN frames BEFORE the ranks are formed. The label replaces the `label` column of the train / val-loss / eval frames, so `TickerDayDataset` is unchanged. Macro F1 is against each recipe's own label: X1/X2/X4 (cs label) and X3 (label_vn) F1 values, and V3's fixed-label F1, are NOT comparable with each other.
- **Constant static ids (X2, X3).** `sector_id` and `market_id` are set to 0 in every frame; the model still has its two static embeddings (cardinalities unchanged), they just always receive id 0.
- **Volatility inputs as cross-sectional ranks (X4).** In every frame (train / val / OOT) these columns of the 33 are replaced, under the same column name, by their per-date percentile rank in (0, 1] over the tickers of that date (average ranks; same-date information only): `volatility_20d`, `sector_volatility`. `volatility_20d` is the ticker's 20-day std of log returns; `sector_volatility` = (high - low) / close of the ticker's sector index (it identifies the sector's range). Deliberately NOT ranked: `vix_chg` and the other market-wide columns (identical for all tickers on a date, a rank would be constant), `is_vi_triggered` / `vi_count_recent5d` (sparse event flags), `log_ret` / `disparity_*` / `rsi_14` (returns / momentum).

## Recipes

`min train_loss` = the lowest train_loss over the run's epoch_curve; stuck at ln(3)~=1.0986 (3-class random-guess cross-entropy) past epoch 1 flags a degenerate class-prior fit rather than real learning (see the module docstring; this is what x2_cslabel_nostatic / x3_vn_nostatic showed).

| recipe | tag | label | static ids | vol inputs | description | epochs run / stop | best epoch | best val timing IC | min train_loss | class weights (buy/hold/sell) | train samples (glitch-masked rows) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| x1_cslabel | X1 | cs quantile 30/30 | as is | raw | per-date cross-sectional quantile label (top 30% BUY / bottom 30% SELL / rest HOLD) | 7 / early_stopping | 2 | +0.0258 | 1.0935 | 1.110 / 0.834 / 1.111 | 234226 (0) |
| x2_cslabel_nostatic | X2 | cs quantile 30/30 | const 0 | raw | X1 + sector_id / market_id constant 0 | 5 / early_stopping | 0 | +0.0153 | 1.0989 | 1.110 / 0.834 / 1.111 | 234226 (0) |
| x3_vn_nostatic | X3 | label_vn | const 0 | raw | volatility-normalised label (label_vn) + constant static ids | 5 / early_stopping | 0 | +0.0261 | 1.0987 | 1.388 / 0.667 / 1.281 | 234226 (0) |
| x4_cslabel_vnfeat | X4 | cs quantile 30/30 | as is | cs rank | X1 + volatility_20d / sector_volatility as per-date cross-sectional percentile ranks | 7 / early_stopping | 2 | +0.0273 | 1.0935 | 1.110 / 0.834 / 1.111 | 234226 (0) |
| x2_nostatic_nowd | X2wd0 | cs quantile 30/30 | const 0 | raw | X2 (cs label, static ids const 0) + weight_decay=0 -- does removing Adam L2 stop the 1-epoch train_loss freeze at ln(3)? | 10 / early_stopping | 5 | +0.0745 | 1.0780 | 1.110 / 0.834 / 1.111 | 234226 (0) |
| x3_vn_nostatic_nowd | X3wd0 | label_vn | const 0 | raw | X3 (label_vn, static ids const 0) + weight_decay=0 -- does removing Adam L2 stop the 1-epoch train_loss freeze at ln(3)? | 5 / early_stopping | 0 | +0.0446 | 1.0197 | 1.388 / 0.667 / 1.281 | 234226 (0) |
| x2_nostatic_seed1 | X2s1 | cs quantile 30/30 | const 0 | raw | X2 (cs label, static ids const 0), seed=1 -- is the 1-epoch freeze seed-independent? | 12 / max_epochs | 9 | +0.0077 | 1.0988 | 1.110 / 0.834 / 1.111 | 234226 (0) |
| x3_vn_nostatic_seed1 | X3s1 | label_vn | const 0 | raw | X3 (label_vn, static ids const 0), seed=1 -- is the 1-epoch freeze seed-independent? | 5 / early_stopping | 0 | +0.0195 | 1.0987 | 1.388 / 0.667 / 1.281 | 234226 (0) |
| x2_nostatic_seed2 | X2s2 | cs quantile 30/30 | const 0 | raw | X2 (cs label, static ids const 0), seed=2 -- is the 1-epoch freeze seed-independent? | 5 / early_stopping | 0 | +0.0145 | 1.0989 | 1.110 / 0.834 / 1.111 | 234226 (0) |
| x3_vn_nostatic_seed2 | X3s2 | label_vn | const 0 | raw | X3 (label_vn, static ids const 0), seed=2 -- is the 1-epoch freeze seed-independent? | 5 / early_stopping | 0 | +0.0073 | 1.0987 | 1.388 / 0.667 / 1.281 | 234226 (0) |

## val_2024 (selection window; timing IC of the selected epoch)

| scorer | raw IC | raw SE | raw IR | fixed-effect IC | timing IC | timing SE | timing IR | days | quantile L/S (raw score) | quantile L/S (timing score) | macro F1 (own label) | best epoch | epochs run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V3 seed 0 (`v3_wd`, fixed label)** | 0.0412 | 0.0142 | 0.214 | +0.0409† | -0.0253† | n/a | n/a | 185 | 0.05% | n/a | 0.3023 (fixed label) | 3 | 8 |
| **X1 x1_cslabel** | +0.0373 | 0.0140 | 0.197 | +0.0372 | +0.0258 | 0.0132 | 0.144 | 185 | 0.04% | 0.07% | 0.2929 (36929 lbl) | 2 | 7 |
| **X2 x2_cslabel_nostatic** | -0.0003 | 0.0067 | -0.004 | -0.0080 | +0.0153 | 0.0059 | 0.190 | 185 | -0.04% | 0.10% | 0.2392 (36929 lbl) | 0 | 5 |
| **X3 x3_vn_nostatic** | +0.0088 | 0.0065 | 0.099 | -0.0086 | +0.0261 | 0.0066 | 0.290 | 185 | 0.00% | 0.20% | 0.2189 (36929 lbl) | 0 | 5 |
| **X4 x4_cslabel_vnfeat** | +0.0370 | 0.0138 | 0.198 | +0.0369 | +0.0273 | 0.0135 | 0.148 | 185 | 0.04% | 0.08% | 0.2852 (36929 lbl) | 2 | 7 |
| **X2wd0 x2_nostatic_nowd** | +0.0710 | 0.0101 | 0.518 | +0.0201 | +0.0745 | 0.0103 | 0.534 | 185 | 0.19% | 0.42% | 0.4034 (36929 lbl) | 5 | 10 |
| **X3wd0 x3_vn_nostatic_nowd** | +0.0087 | 0.0123 | 0.052 | -0.0572 | +0.0446 | 0.0092 | 0.357 | 185 | 0.08% | 0.29% | 0.3166 (36929 lbl) | 0 | 5 |
| **X2s1 x2_nostatic_seed1** | -0.0126 | 0.0189 | -0.126 | -0.0087 | +0.0077 | 0.0067 | 0.085 | 28 | 0.24% | 0.08% | 0.1903 (36929 lbl) | 9 | 12 |
| **X3s1 x3_vn_nostatic_seed1** | +0.0029 | 0.0093 | 0.023 | -0.0280 | +0.0195 | 0.0065 | 0.222 | 185 | 0.02% | 0.14% | 0.1380 (36929 lbl) | 0 | 5 |
| **X2s2 x2_nostatic_seed2** | +0.0030 | 0.0071 | 0.030 | -0.0041 | +0.0145 | 0.0060 | 0.176 | 185 | -0.02% | 0.10% | 0.1539 (36929 lbl) | 0 | 5 |
| **X3s2 x3_vn_nostatic_seed2** | +0.0080 | 0.0075 | 0.079 | +0.0130 | +0.0073 | 0.0073 | 0.073 | 185 | 0.10% | 0.10% | 0.2924 (36929 lbl) | 0 | 5 |

† V3 fixed-effect / timing IC are quoted from `docs/signal_diagnosis.md` (computed from the V3 score dump); raw IC / IR / quantile L/S / F1 are read from the results JSON. 

## oot_2026 (final checkpoint scored once; confirmation only, not a selection criterion)

| scorer | raw IC | raw SE | raw IR | fixed-effect IC | timing IC | timing SE | timing IR | days | quantile L/S (raw score) | quantile L/S (timing score) | macro F1 (own label) | best epoch | epochs run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V3 seed 0 (`v3_wd`, fixed label)** | 0.0392 | 0.0184 | 0.165 | +0.0392† | -0.0393† | 0.0180 | n/a | 167 | 0.16% | n/a | 0.2888 (fixed label) | 3 | 8 |
| **X1 x1_cslabel** | +0.0417 | 0.0182 | 0.177 | +0.0417 | -0.0448 | 0.0181 | -0.191 | 167 | 0.16% | -0.19% | 0.2661 (33364 lbl) | 2 | 7 |
| **X2 x2_cslabel_nostatic** | +0.0314 | 0.0091 | 0.267 | +0.0044 | +0.0237 | 0.0123 | 0.149 | 167 | 0.28% | 0.04% | 0.1539 (33364 lbl) | 0 | 5 |
| **X3 x3_vn_nostatic** | +0.0280 | 0.0100 | 0.216 | +0.0031 | +0.0244 | 0.0117 | 0.161 | 167 | 0.14% | 0.02% | 0.1370 (33364 lbl) | 0 | 5 |
| **X4 x4_cslabel_vnfeat** | +0.0411 | 0.0180 | 0.176 | +0.0412 | -0.0441 | 0.0181 | -0.189 | 167 | 0.16% | -0.19% | 0.2608 (33364 lbl) | 2 | 7 |
| **X2wd0 x2_nostatic_nowd** | +0.0412 | 0.0158 | 0.201 | +0.0236 | +0.0223 | 0.0132 | 0.131 | 167 | 0.09% | 0.10% | 0.3303 (33364 lbl) | 5 | 10 |
| **X3wd0 x3_vn_nostatic_nowd** | +0.0055 | 0.0164 | 0.026 | -0.0145 | +0.0134 | 0.0150 | 0.069 | 167 | -0.04% | -0.04% | 0.3419 (33364 lbl) | 0 | 5 |
| **X2s1 x2_nostatic_seed1** | n/a | n/a | n/a | -0.0077 | +0.0077 | 0.0074 | 0.080 | 0 | n/a | -0.07% | 0.1905 (33364 lbl) | 9 | 12 |
| **X3s1 x3_vn_nostatic_seed1** | -0.0057 | 0.0119 | -0.037 | -0.0112 | -0.0006 | 0.0088 | -0.006 | 167 | -0.18% | -0.09% | 0.1370 (33364 lbl) | 0 | 5 |
| **X2s2 x2_nostatic_seed2** | +0.0152 | 0.0122 | 0.097 | -0.0081 | +0.0133 | 0.0108 | 0.096 | 167 | 0.02% | 0.02% | 0.1539 (33364 lbl) | 0 | 5 |
| **X3s2 x3_vn_nostatic_seed2** | +0.0291 | 0.0091 | 0.246 | +0.0142 | +0.0103 | 0.0098 | 0.082 | 167 | 0.25% | 0.09% | 0.1370 (33364 lbl) | 0 | 5 |

† V3 fixed-effect / timing IC are quoted from `docs/signal_diagnosis.md` (computed from the V3 score dump); raw IC / IR / quantile L/S / F1 are read from the results JSON. 

## Per-epoch val curves (`*` = selected epoch; timing IC is the selection metric)

- **X1 x1_cslabel**: 0: timing -0.0182 / raw +0.0395 / vloss 1.0937 / tloss 1.0961; 1: timing +0.0223 / raw +0.0364 / vloss 1.0945 / tloss 1.0944; 2*: timing +0.0258 / raw +0.0373 / vloss 1.0938 / tloss 1.0939; 3: timing -0.0189 / raw +0.0386 / vloss 1.0929 / tloss 1.0937; 4: timing -0.0216 / raw +0.0397 / vloss 1.0931 / tloss 1.0938; 5: timing -0.0272 / raw +0.0395 / vloss 1.0930 / tloss 1.0936; 6: timing -0.0266 / raw +0.0378 / vloss 1.0940 / tloss 1.0935
- **X2 x2_cslabel_nostatic**: 0*: timing +0.0153 / raw -0.0003 / vloss 1.0993 / tloss 1.0994; 1: timing +0.0132 / raw -0.0539 / vloss 1.1006 / tloss 1.0991; 2: timing +0.0051 / raw -0.0539 / vloss 1.0999 / tloss 1.0989; 3: timing +0.0078 / raw -0.0126 / vloss 1.0997 / tloss 1.0989; 4: timing +0.0051 / raw -0.0539 / vloss 1.0987 / tloss 1.0989
- **X3 x3_vn_nostatic**: 0*: timing +0.0261 / raw +0.0088 / vloss 1.1004 / tloss 1.0994; 1: timing -0.0144 / raw +0.0342 / vloss 1.0992 / tloss 1.0988; 2: timing -0.0144 / raw +0.0342 / vloss 1.0995 / tloss 1.0988; 3: timing -0.0144 / raw +0.0342 / vloss 1.0991 / tloss 1.0987; 4: timing -0.0144 / raw +0.0342 / vloss 1.0993 / tloss 1.0987
- **X4 x4_cslabel_vnfeat**: 0: timing -0.0155 / raw +0.0395 / vloss 1.0935 / tloss 1.0956; 1: timing +0.0154 / raw +0.0356 / vloss 1.0945 / tloss 1.0945; 2*: timing +0.0273 / raw +0.0370 / vloss 1.0938 / tloss 1.0939; 3: timing -0.0181 / raw +0.0392 / vloss 1.0932 / tloss 1.0937; 4: timing -0.0212 / raw +0.0397 / vloss 1.0930 / tloss 1.0938; 5: timing -0.0262 / raw +0.0389 / vloss 1.0930 / tloss 1.0935; 6: timing -0.0263 / raw +0.0378 / vloss 1.0939 / tloss 1.0935
- **X2wd0 x2_nostatic_nowd**: 0: timing +0.0478 / raw +0.0292 / vloss 1.0856 / tloss 1.0948; 1: timing +0.0480 / raw +0.0393 / vloss 1.0828 / tloss 1.0879; 2: timing +0.0620 / raw +0.0634 / vloss 1.0881 / tloss 1.0849; 3: timing +0.0698 / raw +0.0704 / vloss 1.0749 / tloss 1.0810; 4: timing +0.0541 / raw +0.0668 / vloss 1.0744 / tloss 1.0802; 5*: timing +0.0745 / raw +0.0710 / vloss 1.0736 / tloss 1.0795; 6: timing +0.0712 / raw +0.0702 / vloss 1.0780 / tloss 1.0792; 7: timing +0.0624 / raw +0.0693 / vloss 1.0748 / tloss 1.0790; 8: timing +0.0610 / raw +0.0671 / vloss 1.0753 / tloss 1.0784; 9: timing +0.0709 / raw +0.0666 / vloss 1.0746 / tloss 1.0780
- **X3wd0 x3_vn_nostatic_nowd**: 0*: timing +0.0446 / raw +0.0087 / vloss 1.1021 / tloss 1.0925; 1: timing +0.0252 / raw +0.0112 / vloss 1.1252 / tloss 1.0746; 2: timing +0.0378 / raw +0.0211 / vloss 1.1255 / tloss 1.0540; 3: timing +0.0238 / raw +0.0147 / vloss 1.1295 / tloss 1.0378; 4: timing +0.0174 / raw +0.0159 / vloss 1.1358 / tloss 1.0197
- **X2s1 x2_nostatic_seed1**: 0: timing -0.0080 / raw -0.0133 / vloss 1.1007 / tloss 1.0994; 1: timing -0.0146 / raw -0.0126 / vloss 1.0998 / tloss 1.0990; 2: timing -0.0145 / raw -0.0126 / vloss 1.0994 / tloss 1.0990; 3: timing -0.0039 / raw -0.0126 / vloss 1.0989 / tloss 1.0989; 4: timing -0.0037 / raw -0.0126 / vloss 1.0991 / tloss 1.0989; 5: timing +0.0076 / raw -0.0126 / vloss 1.0989 / tloss 1.0989; 6: timing -0.0037 / raw -0.0126 / vloss 1.0986 / tloss 1.0989; 7: timing +0.0076 / raw -0.0126 / vloss 1.0989 / tloss 1.0989; 8: timing +0.0076 / raw -0.0126 / vloss 1.0989 / tloss 1.0988; 9*: timing +0.0077 / raw -0.0126 / vloss 1.0986 / tloss 1.0988; 10: timing -0.0038 / raw -0.0126 / vloss 1.0988 / tloss 1.0988; 11: timing -0.0038 / raw -0.0126 / vloss 1.0988 / tloss 1.0988
- **X3s1 x3_vn_nostatic_seed1**: 0*: timing +0.0195 / raw +0.0029 / vloss 1.0994 / tloss 1.0993; 1: timing -0.0182 / raw -0.0126 / vloss 1.0992 / tloss 1.0989; 2: timing -0.0182 / raw -0.0126 / vloss 1.0989 / tloss 1.0988; 3: timing -0.0181 / raw -0.0126 / vloss 1.0997 / tloss 1.0988; 4: timing -0.0075 / raw -0.0126 / vloss 1.0991 / tloss 1.0987
- **X2s2 x2_nostatic_seed2**: 0*: timing +0.0145 / raw +0.0030 / vloss 1.1007 / tloss 1.0995; 1: timing -0.0078 / raw +0.0126 / vloss 1.0997 / tloss 1.0991; 2: timing -0.0078 / raw +0.0126 / vloss 1.0989 / tloss 1.0990; 3: timing -0.0078 / raw +0.0126 / vloss 1.0990 / tloss 1.0989; 4: timing -0.0078 / raw +0.0126 / vloss 1.0992 / tloss 1.0989
- **X3s2 x3_vn_nostatic_seed2**: 0*: timing +0.0073 / raw +0.0080 / vloss 1.0994 / tloss 1.0994; 1: timing -0.0051 / raw +0.0539 / vloss 1.0987 / tloss 1.0989; 2: timing -0.0078 / raw +0.0126 / vloss 1.0981 / tloss 1.0989; 3: timing -0.0078 / raw +0.0126 / vloss 1.0983 / tloss 1.0988; 4: timing -0.0052 / raw +0.0539 / vloss 1.0987 / tloss 1.0987

Reading guide: a recipe helps only if its timing IC is clearly positive on val AND OOT (SE ~0.014 val / ~0.018 OOT; differences below ~2 SE are noise). A high raw IC with fixed-effect IC ~ raw IC and timing IC <= 0 means the model still learned only a ticker ranking.
