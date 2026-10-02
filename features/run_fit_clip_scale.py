#!/usr/bin/env python
"""
Fit train-split-only outlier-clipping bounds and a StandardScaler for feature_pool's
continuous numeric feature columns, using ONLY the 1단계 train split
(2019-01-02 ~ 2023-12-31) to avoid leakage into val/test/2단계/serving.

Column selection (live feature_pool had 61 columns as of Task 10's fix round 2, and now
has 72 after Task 10.5 added 11 macro derived columns (2026-09-09) — NOT the column
list this task's brief was originally drafted against; see
docs/superpowers/plans/2026-09-08-ai-model-redesign-plan.md's Task 13
HISTORICAL_COLS_DEFAULT for the downstream design intent this selection follows):

INCLUDED (19 columns) — already-derived, relative/stationary continuous features that
Task 13's HISTORICAL_COLS_DEFAULT names verbatim (or a same-shape analogue of):
  log_ret, disparity_5d, disparity_20d, disparity_60d, rsi_14, volume_ratio,
  volatility_20d, hl_range, lev_total_volume, lev_total_aum, lev_aum_to_mktcap,
  est_rebalancing_flow, sector_ret_1d, sector_ret_5d, sector_ret_20d,
  sector_ma_ratio_20d, sector_volatility, sector_volume_ratio, vi_count_recent5d

INCLUDED (11 more columns, added by this fix round) — macro derived features computed
by Task 10.5 (commit 5eb4a26, see docs/data_units.md's "feature_pool 매크로 파생 컬럼
11개" section) directly into feature_pool. These did not exist when Task 11 was first
written (hence the original EXCLUDED note below, now stale for these names). All are
stationary by construction — percentage returns (pct_change), absolute changes (diff),
or a same-day rate spread — not raw trending levels, so quantile-clipping/scaling them
is legitimate:
  kospi_ret, kosdaq_ret, snp500_ret, nasdaq_ret, phlx_semi_ret, vix_chg, usd_krw_chg,
  us_10y_yield_chg, rate_spread_us_kr, wti_ret, gold_ret

EXCLUDED, and why:
  - ticker, trade_date: identifiers, not features
  - label, next_day_return: prediction target, must never be clipped/scaled as an input
  - sector_id, market_id: categorical ids (Task 13's STATIC_COLS) — embeddings, not scaling
  - day_of_week: categorical (0-6)
  - is_market_open, is_holiday, is_short_selling_banned, is_bok, is_fomc,
    is_witching_kr, is_witching_us, is_dividend, is_bonus_issue, is_split,
    is_rights_offering, is_vi_triggered: binary 0/1 event/calendar flags — left as-is,
    clipping/scaling a binary indicator is meaningless
  - open_price, high_price, low_price, close_price, volume, turnover,
    shares_outstanding, sma_5d, sma_20d, sma_60d, volume_sma_20d: RAW price/volume
    LEVELS. These are ticker-dependent absolute magnitudes (a KRW 1,000 stock vs a
    KRW 500,000 stock), not stationary/relative — their information content is already
    captured in normalized form by log_ret/disparity_Xd/volume_ratio. Task 13's
    HISTORICAL_COLS_DEFAULT never references any of these raw columns directly (it
    references rel_close/rel_high/rel_low/vol_ratio instead), confirming raw price/
    volume levels are not intended as direct model input.
  - snp500_close, nasdaq_close, phlx_semi_close, vix, wti_crude_oil, gold_price,
    usd_krw, us_10y_yield, fed_rate, kr_base_rate, index_0001, index_1001: RAW macro/
    index LEVELS. These trend/shift regime over the 2019-2026 span (e.g. fed_rate
    ~2.5% in 2019 vs ~5%+ later) — quantile-clipping bounds fit only on 2019-2023
    train would clip out genuinely-shifted-regime future values as if they were
    outliers, which is exactly the failure mode train-only fitting is supposed to
    avoid causing downstream. This remains correctly excluded and is NOT affected by
    Task 10.5: their *_ret/*_chg derived counterparts (see the second INCLUDED list
    above) are now in feature_pool and included, but these raw levels are still
    non-stationary and stay out.

NaN handling for the 11 new macro-derived columns: 10 of the 11 (all except
rate_spread_us_kr, which is a same-day level difference with no prior-day dependency)
are NULL for exactly the first trading day in the whole dataset (2019-01-02, 200 rows
— one per ticker, a structural pandas pct_change()/diff() first-element artifact
documented in docs/data_units.md, not a data gap). This is the same shape of NaN
pattern log_ret already has and already handles correctly via fit_clip_bounds's
~np.isnan filtering and StandardScaler's native NaN-aware fit — no new NaN-handling
code was needed for these columns; see the Task 11 fix-round-1 report for how this was
verified.

Usage:
  set -a && source .env && set +a
  python features/run_fit_clip_scale.py
"""

import os
import sys
import psycopg2
import pandas as pd

from features.clip_scale import fit_clip_bounds, apply_clip, fit_scaler, save_artifacts

STAGE1_TRAIN_START = "2019-01-02"
STAGE1_TRAIN_END = "2023-12-31"

CLIP_SCALE_COLUMNS = [
    "log_ret", "disparity_5d", "disparity_20d", "disparity_60d", "rsi_14",
    "volume_ratio", "volatility_20d", "hl_range",
    "lev_total_volume", "lev_total_aum", "lev_aum_to_mktcap", "est_rebalancing_flow",
    "sector_ret_1d", "sector_ret_5d", "sector_ret_20d", "sector_ma_ratio_20d",
    "sector_volatility", "sector_volume_ratio",
    "vi_count_recent5d",
    # Task 10.5 macro derived columns (added by this fix round; see docstring above)
    "kospi_ret", "kosdaq_ret", "snp500_ret", "nasdaq_ret", "phlx_semi_ret",
    "vix_chg", "usd_krw_chg", "us_10y_yield_chg", "rate_spread_us_kr",
    "wti_ret", "gold_ret",
]

ARTIFACT_PATH = "features/artifacts/stage1"


def load_train_split(dsn: str) -> pd.DataFrame:
    """Load only the 1단계 train split (2019-01-02~2023-12-31) from feature_pool."""
    conn = psycopg2.connect(dsn)
    try:
        cols_sql = ", ".join(CLIP_SCALE_COLUMNS)
        query = (
            f"SELECT {cols_sql} FROM feature_pool "
            f"WHERE trade_date >= %s AND trade_date <= %s"
        )
        df = pd.read_sql(query, conn, params=(STAGE1_TRAIN_START, STAGE1_TRAIN_END))
    finally:
        conn.close()
    return df


def main() -> None:
    dsn = os.environ.get("STOCK_DB_V2_DSN")
    if not dsn:
        print("ERROR: STOCK_DB_V2_DSN environment variable must be set")
        sys.exit(1)

    print(f"Loading 1단계 train split ({STAGE1_TRAIN_START}~{STAGE1_TRAIN_END}) from feature_pool...")
    train_df = load_train_split(dsn)
    print(f"  Loaded {len(train_df)} rows x {len(train_df.columns)} columns")

    # Coerce to float64 (psycopg2/pandas may return Decimal/object dtype for numeric columns)
    for col in CLIP_SCALE_COLUMNS:
        train_df[col] = train_df[col].astype("float64")

    null_report = train_df[CLIP_SCALE_COLUMNS].isna().mean()
    flagged_nulls = null_report[null_report > 0]
    if len(flagged_nulls) > 0:
        print("  NaN present (not dropped — will be forward-filled upstream per project policy,"
              " fit_clip_bounds/fit_scaler are NaN-aware):")
        for col, ratio in flagged_nulls.items():
            print(f"    - {col}: {ratio * 100:.4f}%")

    print(f"Fitting quantile clip bounds (0.5%~99.5%) for {len(CLIP_SCALE_COLUMNS)} columns...")
    bounds: dict[str, tuple[float, float]] = {}
    clipped_df = train_df.copy()
    for col in CLIP_SCALE_COLUMNS:
        lower, upper = fit_clip_bounds(train_df[col].to_numpy(), lower_q=0.005, upper_q=0.995)
        bounds[col] = (lower, upper)
        clipped_df[col] = apply_clip(train_df[col].to_numpy(), (lower, upper))
        print(f"  {col}: [{lower:.6g}, {upper:.6g}]")

    print("Fitting StandardScaler on clipped train data...")
    scaler = fit_scaler(clipped_df, columns=CLIP_SCALE_COLUMNS)

    os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
    save_artifacts(bounds, scaler, ARTIFACT_PATH)
    print(f"Saved artifacts: {ARTIFACT_PATH}.bounds.json, {ARTIFACT_PATH}.scaler.pkl")


if __name__ == "__main__":
    main()
