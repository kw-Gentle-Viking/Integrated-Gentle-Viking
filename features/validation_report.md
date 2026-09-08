# Feature Pool Validation Report
Generated: 2026-09-09 00:47:41

## Summary
- feature_pool rows: 376,682
- Columns: 155
- Tickers: 200
- Date range: 2019-01-02 to 2026-09-08

## Validation Checks

### Market Cap Consistency
✓ No major discrepancies found (real cross-check against ticker_universe.market_cap)
  Market cap cross-check: compared 200 tickers (universe snapshot 2019-01-02 vs price_daily-recomputed on latest trade_date), tolerance=0.99 (~100x magnitude gap). Sample: ['000080 on 2026-09-08: universe=1.14e+12, valuation(recomputed)=1.08e+12', '000100 on 2026-09-08: universe=1.54e+13, valuation(recomputed)=5.82e+12', '000120 on 2026-09-08: universe=3.80e+12, valuation(recomputed)=1.68e+12']

### Value Ranges
✓ All values within expected ranges

### Null Rates
✓ All columns within acceptable null rate (< 50%)

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