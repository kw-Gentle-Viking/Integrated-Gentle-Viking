# Feature Pool Validation Report
Generated: 2026-09-08 23:36:40

## Summary
- feature_pool rows: 376,682
- Columns: 154
- Tickers: 200
- Date range: 2019-01-02 to 2026-09-08

## Validation Checks

### Market Cap Consistency
✓ No major discrepancies found (daily_valuation is intentionally empty)
  Market cap cross-check: daily_valuation is empty (intentional). Computed market caps from price_daily for 200 tickers. Sample: ['000080 on 2026-09-08: computed market cap = 1.08e+12 won', '000100 on 2026-09-08: computed market cap = 5.82e+12 won', '000120 on 2026-09-08: computed market cap = 1.68e+12 won']

### Value Ranges
✓ All values within expected ranges

### Null Rates
**High null rate columns (> 50%): 1**
  - sector_id: 100.0%

### Trading Day Gaps
**Tickers with insufficient days (< 1804): 1**
  - 214370: 1600 days (expected 1804)

### Label Completeness
- Label null rate: 0.1%
- Label distribution:
  - 1.0: 180,597 rows
  - 2.0: 101,064 rows
  - 0.0: 94,821 rows

### Technical Indicators
✓ All technical indicators present

### Leverage Features
- lev_total_volume: 144 non-zero values out of 376,682 (0.0%)
- lev_total_aum: 0 non-zero values out of 376,682 (0.0%)
- lev_aum_to_mktcap: 0 non-zero values out of 376,682 (0.0%)
- est_rebalancing_flow: 0 non-zero values out of 376,682 (0.0%)

## Overall Status
⚠ **Found 1 issue(s). Review details above.**