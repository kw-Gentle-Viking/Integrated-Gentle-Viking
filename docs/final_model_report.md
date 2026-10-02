# Final TFT model (E4 confirmed candidate)

## Identity

| | |
|---|---|
| Recipe | `v3_structure_nowd_cs` (E4), seed 0 |
| Checkpoint | `training/artifacts/checkpoints/e4-v3_structure_nowd_cs.pt` |
| Code commit | `ab5e52e` |
| Data version | `adj1` |
| Results record | `training/artifacts/e4_results.json["v3_structure_nowd_cs"]` |

## Label

Cross-sectional quantile label (`cs`), computed per trading day over names with a valid `next_day_return`:
top 30% -> BUY(0), bottom 30% -> SELL(2), rest -> HOLD(1). Average-rank percentile `p = (rank - 0.5)/n`;
BUY if `p >= 0.7`, SELL if `p <= 0.3` (tied returns share one label). Days with fewer than 20 eligible
names -> label NaN. Glitch rows (`|next_day_return| > 0.31`) are masked out of the train frames before ranks
are formed.

## Architecture / hyperparameters

| | |
|---|---|
| Model | TFT (`tft-torch` fork), 3-class `class_logits` head |
| state_size | 32 |
| attention_heads | 8 |
| lstm_layers | 2 |
| dropout | 0.16577574724588015 |
| lr | 0.0002905890080008017 |
| weight_decay | 0.0 (Adam) |
| batch_size | 128 |
| encoder_len | 60 |
| align | `today` |
| seed | 0 |
| static ids (`sector_id`, `market_id`) | kept as-is (not constant) |
| Input columns | 33 raw champion columns, no preprocessing (`log_ret`, `disparity_5d/20d/60d`, `rsi_14`, `volatility_20d`, `sector_ret_1d/5d/20d`, `sector_ma_ratio_20d`, `sector_volatility`, `sector_volume_ratio`, `is_dividend`, `is_bonus_issue`, `is_rights_offering`, `is_split`, `day_of_week`, `lev_total_aum`, `lev_aum_to_mktcap`, `est_rebalancing_flow`, `is_vi_triggered`, `vi_count_recent5d`, `kospi_ret`, `kosdaq_ret`, `snp500_ret`, `nasdaq_ret`, `phlx_semi_ret`, `vix_chg`, `usd_krw_chg`, `us_10y_yield_chg`, `rate_spread_us_kr`, `wti_ret`, `gold_ret`) |

## Training

| | |
|---|---|
| Train samples | 234,226 (0 glitch-masked) |
| Class weights (buy/hold/sell) | 1.110 / 0.834 / 1.111 |
| Selection metric | val 2024 timing IC |
| Patience | 4 |
| Stopped | early_stopping, epoch 9 (best epoch 5) |

## Performance (label-agnostic signal metrics; score = p_buy - p_sell)

| window | raw IC | fixed-effect IC | **timing IC** | timing SE | timing IR | macro F1 (own label) |
|---|---|---|---|---|---|---|
| val_2024 (selection window) | +0.0738 | +0.0243 | **+0.0732** | 0.0100 | 0.537 | 0.403 |
| oot_2026 (confirmation only, scored once) | +0.0533 | +0.0296 | **+0.0293** | 0.0124 | 0.183 | 0.346 |

Timing IC = daily cross-sectional rank IC of (score minus the ticker's own mean score) against next-day
returns; OOT fixed effect uses the val-window ticker means of this same checkpoint (no look-ahead). Confirmed
stable across seeds 0/1/2 (OOT timing IC 0.029 / 0.023 / 0.021; val 0.073 / 0.066 / 0.068) and matched by the
3-seed probability-average ensemble (val +0.073, OOT +0.029) -- no additional benefit from ensembling over this
single checkpoint.
