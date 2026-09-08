#!/usr/bin/env python
"""
Validation runner for feature_pool table.
Runs data integrity checks and generates validation_report.md.
"""

import os
import sys
from datetime import datetime
import psycopg2
import pandas as pd
from features.validate_feature_pool import (
    check_market_cap_consistency, check_value_ranges, check_null_rates, check_trading_day_gaps,
)

# check_market_cap_consistency's tolerance is a *relative* difference: abs(a-b)/max(a,b), which
# saturates near 1.0 for any large discrepancy (e.g. a=1, b=1000 -> rel_diff=0.999). We're
# comparing ticker_universe.market_cap (2019-01-02 snapshot) against a market cap freshly computed
# from price_daily on the most recent trade_date — 7+ years apart, so ordinary price appreciation
# or decline (even a multi-bagger, even a stock down 90%) is expected and NOT a bug. A genuine unit
# bug (off by 1e3x, 1e6x — e.g. mixing 원 and 백만원) shows up as a ~1000x+ magnitude gap, which
# maps to rel_diff > 0.999. tolerance=0.99 (~100x magnitude gap) is deliberately generous: it
# tolerates decades of organic price/split drift while still catching any real order-of-thousands
# unit bug. See features/test_validate_feature_pool.py's original tolerance=0.1 test, which is a
# same-day cross-check (different scenario) and is unaffected by this constant.
MARKET_CAP_CROSS_CHECK_TOLERANCE = 0.99


def load_feature_pool(dsn: str) -> pd.DataFrame:
    """Load feature_pool table from database."""
    conn = psycopg2.connect(dsn)
    query = "SELECT * FROM feature_pool ORDER BY ticker, trade_date"
    df = pd.read_sql(query, conn)
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df


def load_ticker_universe(dsn: str) -> pd.DataFrame:
    """Load ticker_universe table."""
    conn = psycopg2.connect(dsn)
    query = "SELECT * FROM ticker_universe"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def adapt_market_cap_cross_check(feature_pool_df: pd.DataFrame,
                                  ticker_universe_df: pd.DataFrame) -> tuple[list[str], str]:
    """
    Adapted market cap cross-check: daily_valuation is intentionally empty (confirmed dead end,
    KIS free API has no historical support), so this compares two INDEPENDENTLY-computed market
    caps instead:
      - universe_market_cap: ticker_universe.market_cap, computed at the 2019-01-02 snapshot
      - valuation_market_cap: price_daily.close_price x price_daily.shares_outstanding on the
        most recent trade_date in feature_pool (7+ years later)

    These will legitimately differ (different dates, share count may have changed via splits/
    buybacks, multi-year price moves) so this is a sanity check for unit bugs (off by 1e3x, 1e6x),
    not a same-day cross-check — see MARKET_CAP_CROSS_CHECK_TOLERANCE above for the reasoning.
    Actually runs check_market_cap_consistency() and returns its real result (previously this
    function computed values but never compared them, so `issues` was unconditionally empty).
    """
    # Get most recent date per ticker
    latest_dates = feature_pool_df.groupby('ticker')['trade_date'].max().reset_index()
    latest_rows = feature_pool_df.merge(latest_dates, on=['ticker', 'trade_date'], how='inner')

    valuation_df = latest_rows[['ticker', 'trade_date', 'close_price', 'shares_outstanding']].copy()
    valuation_df['valuation_market_cap'] = valuation_df['close_price'] * valuation_df['shares_outstanding']

    universe_df = ticker_universe_df[['ticker', 'market_cap']].rename(
        columns={'market_cap': 'universe_market_cap'})

    compare_df = valuation_df.merge(universe_df, on='ticker', how='inner')
    compare_df = compare_df.dropna(subset=['valuation_market_cap', 'universe_market_cap'])

    issues = check_market_cap_consistency(compare_df, tolerance=MARKET_CAP_CROSS_CHECK_TOLERANCE)

    samples = [
        f"{r.ticker} on {r.trade_date.strftime('%Y-%m-%d')}: universe={r.universe_market_cap:.2e}, "
        f"valuation(recomputed)={r.valuation_market_cap:.2e}"
        for r in compare_df.head(3).itertuples()
    ]
    note_text = (
        f"Market cap cross-check: compared {len(compare_df)} tickers "
        f"(universe snapshot 2019-01-02 vs price_daily-recomputed on latest trade_date), "
        f"tolerance={MARKET_CAP_CROSS_CHECK_TOLERANCE} (~100x magnitude gap). "
        f"Sample: {samples if samples else 'no data'}"
    )
    return issues, note_text


def check_expected_trading_days(feature_pool_df: pd.DataFrame, start_date: str, end_date: str) -> list[str]:
    """Calculate expected minimum trading days based on calendar."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    business_days = pd.bdate_range(start=start, end=end)
    expected_min_days = int(len(business_days) * 0.90)  # 90% of business days
    return expected_min_days


def run_validation(dsn: str = None) -> None:
    """Run all validation checks and generate report."""

    if dsn is None:
        dsn = os.environ.get("STOCK_DB_V2_DSN")
        if not dsn:
            print("ERROR: STOCK_DB_V2_DSN environment variable must be set")
            sys.exit(1)

    print("Loading feature_pool...")
    feature_pool_df = load_feature_pool(dsn)
    print(f"  Loaded {len(feature_pool_df)} rows, {len(feature_pool_df.columns)} columns")

    print("Loading ticker_universe...")
    ticker_universe_df = load_ticker_universe(dsn)
    print(f"  Loaded {len(ticker_universe_df)} rows")

    # Get date range
    min_date = feature_pool_df['trade_date'].min()
    max_date = feature_pool_df['trade_date'].max()
    print(f"  Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

    # Count of tickers
    n_tickers = feature_pool_df['ticker'].nunique()
    print(f"  Number of tickers: {n_tickers}")

    report_lines = [
        "# Feature Pool Validation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        f"- feature_pool rows: {len(feature_pool_df):,}",
        f"- Columns: {len(feature_pool_df.columns)}",
        f"- Tickers: {n_tickers}",
        f"- Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}",
        "",
        "## Validation Checks",
        "",
    ]

    # Check 1: Market cap consistency (adapted — real comparison, see adapt_market_cap_cross_check)
    print("Running market cap consistency check (adapted for empty daily_valuation)...")
    issues, note_text = adapt_market_cap_cross_check(feature_pool_df, ticker_universe_df)
    report_lines.append("### Market Cap Consistency")
    if issues:
        report_lines.append(f"**Issues found: {len(issues)}**")
        for ticker in issues:
            report_lines.append(f"  - {ticker}")
    else:
        report_lines.append("✓ No major discrepancies found (real cross-check against ticker_universe.market_cap)")
    report_lines.append(f"  {note_text}")
    report_lines.append("")

    # Check 2: Value ranges
    print("Checking value ranges...")
    value_issues = check_value_ranges(feature_pool_df)
    report_lines.append("### Value Ranges")
    if value_issues:
        report_lines.append(f"**Issues found in: {', '.join(value_issues)}**")
        for col in value_issues:
            report_lines.append(f"  - {col}: found negative or invalid values")
    else:
        report_lines.append("✓ All values within expected ranges")
    report_lines.append("")

    # Check 3: Null rates
    print("Checking null rates...")
    null_issues = check_null_rates(feature_pool_df, max_null_ratio=0.5)  # Lenient threshold
    report_lines.append("### Null Rates")
    if null_issues:
        report_lines.append(f"**High null rate columns (> 50%): {len(null_issues)}**")
        for col, ratio in null_issues.items():
            report_lines.append(f"  - {col}: {ratio*100:.1f}%")
    else:
        report_lines.append("✓ All columns within acceptable null rate (< 50%)")
    report_lines.append("")

    # Check 4: Trading day gaps
    print("Checking trading day gaps...")
    expected_min_days = check_expected_trading_days(feature_pool_df, min_date.strftime('%Y-%m-%d'), max_date.strftime('%Y-%m-%d'))
    print(f"  Expected minimum days (90%): {expected_min_days}")
    gap_issues = check_trading_day_gaps(feature_pool_df, expected_min_days)
    report_lines.append("### Trading Day Gaps")
    if gap_issues:
        report_lines.append(f"**Tickers with insufficient days (< {expected_min_days}): {len(gap_issues)}**")
        for ticker in gap_issues[:20]:  # Show first 20
            count = len(feature_pool_df[feature_pool_df['ticker'] == ticker])
            report_lines.append(f"  - {ticker}: {count} days (expected {expected_min_days})")
        if len(gap_issues) > 20:
            report_lines.append(f"  ... and {len(gap_issues) - 20} more")
    else:
        report_lines.append("✓ All tickers have sufficient trading days")
    report_lines.append("")

    # Check 5: Label completeness
    print("Checking label completeness...")
    report_lines.append("### Label Completeness")
    if 'label' in feature_pool_df.columns:
        label_null_rate = feature_pool_df['label'].isna().mean()
        label_counts = feature_pool_df['label'].value_counts()
        report_lines.append(f"- Label null rate: {label_null_rate*100:.1f}%")
        report_lines.append("- Label distribution:")
        for label_val, count in label_counts.items():
            report_lines.append(f"  - {label_val}: {count:,} rows")
    else:
        report_lines.append("⚠ Label column not found")
    report_lines.append("")

    # Check 6: Technical indicators
    print("Checking technical indicators...")
    report_lines.append("### Technical Indicators")
    tech_cols = ['log_ret', 'sma_5d', 'sma_20d', 'sma_60d', 'disparity_5d', 'disparity_20d',
                 'disparity_60d', 'rsi_14', 'volume_sma_20d', 'volume_ratio', 'volatility_20d', 'hl_range']
    missing_cols = [col for col in tech_cols if col not in feature_pool_df.columns]
    if missing_cols:
        report_lines.append(f"⚠ Missing columns: {', '.join(missing_cols)}")
    else:
        report_lines.append("✓ All technical indicators present")
    report_lines.append("")

    # Check 7: Leverage features
    print("Checking leverage features...")
    report_lines.append("### Leverage Features")
    leverage_cols = ['lev_total_volume', 'lev_total_aum', 'lev_aum_to_mktcap', 'est_rebalancing_flow']
    missing_leverage = [col for col in leverage_cols if col not in feature_pool_df.columns]
    if missing_leverage:
        report_lines.append(f"⚠ Missing columns: {', '.join(missing_leverage)}")
    else:
        # Show stats on leverage features
        for col in leverage_cols:
            non_zero = (feature_pool_df[col] != 0).sum()
            report_lines.append(f"- {col}: {non_zero:,} non-zero values out of {len(feature_pool_df):,} ({non_zero*100/len(feature_pool_df):.1f}%)")
    report_lines.append("")

    # Summary
    report_lines.append("## Overall Status")
    if not issues and not value_issues and not null_issues and not gap_issues:
        report_lines.append("✓ **All checks passed. Feature pool is ready for use.**")
    else:
        all_issues = len(issues) + len(value_issues) + len(null_issues) + len(gap_issues)
        report_lines.append(f"⚠ **Found {all_issues} issue(s). Review details above.**")

    # Write report
    report_path = "features/validation_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\nValidation report written to {report_path}")
    print('\n'.join(report_lines[-5:]))  # Print last few lines


if __name__ == "__main__":
    run_validation()
