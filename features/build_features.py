#!/usr/bin/env python
"""
Feature pool builder: joins all tables (price_daily, market_global, indices, sector, events, leverage, labels)
and calculates derived features (technical indicators, leverage signals, VI events, stock events).
Outputs: feature_pool table with ~200 tickers × ~1887 trading days per ticker.
"""

import os
import sys
import argparse
from datetime import datetime
import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from features.leverage_features import aggregate_leverage_signals

# market_id cardinality-3 design (see training/config.py STATIC_COLS / static_categorical_cardinalities):
# 0=KOSDAQ, 1=KOSPI, 2=unclassified (reserved fallback; ticker_universe.is_kospi is NOT NULL for all
# 200 tickers today, so 2 is never actually hit, but kept for forward-compatibility with new tickers).
MARKET_ID_KOSPI = 1
MARKET_ID_KOSDAQ = 0
MARKET_ID_UNCLASSIFIED = 2

# sector_id cardinality-21 design: 0-19 are the 20 known "업종지수" sectors (see
# data_collection/backfill_sector_id.py SECTOR_NAMES), 20 = unclassified fallback.
SECTOR_ID_UNCLASSIFIED = 20


def derive_market_id(is_kospi) -> int:
    """ticker_universe.is_kospi(bool) -> market_id(0/1/2). Pure function, unit-testable."""
    if is_kospi is True:
        return MARKET_ID_KOSPI
    if is_kospi is False:
        return MARKET_ID_KOSDAQ
    return MARKET_ID_UNCLASSIFIED


def load_tickers(dsn: str) -> list[str]:
    """Load all tickers from ticker_universe."""
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT ticker FROM ticker_universe ORDER BY ticker")
        tickers = [row[0] for row in cur.fetchall()]
    conn.close()
    return tickers


def load_price_daily(dsn: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load price_daily table."""
    conn = psycopg2.connect(dsn)
    query = """
        SELECT ticker, trade_date, open_price, high_price, low_price, close_price, volume, turnover, shares_outstanding
        FROM price_daily
        WHERE trade_date >= %s AND trade_date <= %s
        ORDER BY ticker, trade_date
    """
    df = pd.read_sql(query, conn, params=(start_date, end_date))
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df


def load_market_global(dsn: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load market_global table."""
    conn = psycopg2.connect(dsn)
    query = """
        SELECT trade_date, snp500_close, nasdaq_close, phlx_semi_close, vix,
               wti_crude_oil, gold_price, usd_krw, us_10y_yield, fed_rate, kr_base_rate
        FROM market_global
        WHERE trade_date >= %s AND trade_date <= %s
        ORDER BY trade_date
    """
    df = pd.read_sql(query, conn, params=(start_date, end_date))
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df


def load_market_index_daily(dsn: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load market_index_daily table (KOSPI, KOSDAQ, etc.)."""
    conn = psycopg2.connect(dsn)
    query = """
        SELECT index_code, trade_date, close_price
        FROM market_index_daily
        WHERE trade_date >= %s AND trade_date <= %s
        ORDER BY index_code, trade_date
    """
    df = pd.read_sql(query, conn, params=(start_date, end_date))
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    # Pivot to get KOSPI, KOSDAQ, etc. as columns
    df_pivot = df.pivot_table(index='trade_date', columns='index_code', values='close_price', aggfunc='first')
    df_pivot.columns = [f"index_{col}" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    return df_pivot


def load_sector_daily_ohlcv(dsn: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load sector_daily_ohlcv table."""
    conn = psycopg2.connect(dsn)
    query = """
        SELECT sector_code, trade_date, open, high, low, close, volume
        FROM sector_daily_ohlcv
        WHERE trade_date >= %s AND trade_date <= %s
        ORDER BY sector_code, trade_date
    """
    df = pd.read_sql(query, conn, params=(start_date, end_date))
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    # Pivot to get sector codes as columns
    df_pivot = df.pivot_table(index='trade_date', columns='sector_code',
                              values=['open', 'high', 'low', 'close', 'volume'], aggfunc='first')
    df_pivot.columns = [f"sector_{col[0]}_{col[1]}" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    return df_pivot


def load_calendar(dsn: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load calendar table."""
    conn = psycopg2.connect(dsn)
    query = """
        SELECT base_date, day_of_week, is_market_open, is_holiday, is_short_selling_banned
        FROM calendar
        WHERE base_date >= %s AND base_date <= %s
        ORDER BY base_date
    """
    df = pd.read_sql(query, conn, params=(start_date, end_date))
    conn.close()
    df['base_date'] = pd.to_datetime(df['base_date'])
    df.rename(columns={'base_date': 'trade_date'}, inplace=True)
    return df


def load_market_events(dsn: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load market_events table."""
    conn = psycopg2.connect(dsn)
    query = """
        SELECT DISTINCT event_date, is_bok, is_fomc, is_witching_kr, is_witching_us
        FROM market_events
        WHERE event_date >= %s AND event_date <= %s
        ORDER BY event_date
    """
    df = pd.read_sql(query, conn, params=(start_date, end_date))
    conn.close()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df.rename(columns={'event_date': 'trade_date'}, inplace=True)
    return df


def load_stock_events(dsn: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load stock_events table and create event flags."""
    conn = psycopg2.connect(dsn)
    query = """
        SELECT ticker, event_date, event_type, description
        FROM stock_events
        WHERE event_date >= %s AND event_date <= %s
        ORDER BY ticker, event_date, event_type
    """
    df = pd.read_sql(query, conn, params=(start_date, end_date))
    conn.close()
    if df.empty:
        # Return empty DataFrame with expected structure
        return pd.DataFrame(columns=['ticker', 'trade_date', 'is_dividend', 'is_bonus_issue', 'is_split', 'is_rights_offering'])

    df['event_date'] = pd.to_datetime(df['event_date'])
    df.rename(columns={'event_date': 'trade_date'}, inplace=True)

    # Create event flags
    events_pivot = df.pivot_table(
        index=['ticker', 'trade_date'],
        columns='event_type',
        values='event_type',
        aggfunc='count'
    )
    events_pivot = events_pivot.fillna(0)
    events_pivot = (events_pivot > 0).astype(int)

    # Ensure all expected columns exist
    expected_cols = ['배당', '유상증자', '무상증자', '액면분할', '액면병합']
    for col in expected_cols:
        if col not in events_pivot.columns:
            events_pivot[col] = 0

    # Rename to English
    event_mapping = {
        '배당': 'is_dividend',
        '유상증자': 'is_rights_offering',
        '무상증자': 'is_bonus_issue',
        '액면분할': 'is_split',
        '액면병합': 'is_reverse_split'
    }
    events_pivot = events_pivot[[col for col in expected_cols if col in events_pivot.columns]]
    events_pivot = events_pivot.rename(columns=event_mapping)

    events_pivot = events_pivot.reset_index()
    return events_pivot


def load_vi_events(dsn: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load vi_events table and create VI flags."""
    conn = psycopg2.connect(dsn)
    query = """
        SELECT ticker, triggered_at, vi_type
        FROM vi_events
        WHERE triggered_at::date >= %s AND triggered_at::date <= %s
        ORDER BY ticker, triggered_at
    """
    df = pd.read_sql(query, conn, params=(start_date, end_date))
    conn.close()
    if df.empty:
        # Return empty DataFrame with expected structure
        return pd.DataFrame(columns=['ticker', 'trade_date', 'is_vi_triggered', 'vi_count_recent5d'])

    df['triggered_at'] = pd.to_datetime(df['triggered_at'])
    df['trade_date'] = df['triggered_at'].dt.date
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    # Create VI flags (count per day, and mark as triggered if count > 0)
    vi_daily = df.groupby(['ticker', 'trade_date']).size().reset_index(name='vi_count')
    vi_daily['is_vi_triggered'] = (vi_daily['vi_count'] > 0).astype(int)

    # Calculate 5-day rolling count
    vi_daily = vi_daily.sort_values(['ticker', 'trade_date']).reset_index(drop=True)
    vi_daily['vi_count_recent5d'] = vi_daily.groupby('ticker')['vi_count'].rolling(window=5, min_periods=1).sum().reset_index(0, drop=True)

    return vi_daily[['ticker', 'trade_date', 'is_vi_triggered', 'vi_count_recent5d']]


def load_leverage_daily_and_products(dsn: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load leverage_daily and leverage_products tables."""
    conn = psycopg2.connect(dsn)

    # Load leverage_products
    query_products = "SELECT * FROM leverage_products WHERE product_type IS NOT NULL ORDER BY code"
    df_products = pd.read_sql(query_products, conn)

    # Load leverage_daily
    query_daily = """
        SELECT code, trade_date, close_price, volume, turnover, nav, aum
        FROM leverage_daily
        WHERE trade_date >= %s AND trade_date <= %s
        ORDER BY code, trade_date
    """
    df_daily = pd.read_sql(query_daily, conn, params=(start_date, end_date))

    conn.close()
    df_daily['trade_date'] = pd.to_datetime(df_daily['trade_date'])
    if 'listed_date' in df_products.columns:
        df_products['listed_date'] = pd.to_datetime(df_products['listed_date'])

    return df_daily, df_products


def load_labels(dsn: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load labels table (Task 9 output)."""
    conn = psycopg2.connect(dsn)
    query = """
        SELECT ticker, trade_date, next_day_return, label
        FROM labels
        WHERE trade_date >= %s AND trade_date <= %s
        ORDER BY ticker, trade_date
    """
    df = pd.read_sql(query, conn, params=(start_date, end_date))
    conn.close()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators per ticker."""
    df = df.sort_values(['ticker', 'trade_date']).copy()

    # Log returns
    df['log_ret'] = df.groupby('ticker')['close_price'].transform(
        lambda x: np.log(x / x.shift(1))
    )

    # 5-day, 20-day, 60-day simple moving averages
    for window in [5, 20, 60]:
        df[f'sma_{window}d'] = df.groupby('ticker')['close_price'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

    # Disparity: (close - sma) / sma
    for window in [5, 20, 60]:
        df[f'disparity_{window}d'] = (
            (df['close_price'] - df[f'sma_{window}d']) / df[f'sma_{window}d']
        )

    # RSI (14-day)
    def calculate_rsi(group):
        delta = group['close_price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    df['rsi_14'] = df.groupby('ticker', group_keys=False).apply(calculate_rsi)

    # Volume indicators
    df['volume_sma_20d'] = df.groupby('ticker')['volume'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20d'] + 1e-10)

    # Volatility (20-day rolling std of returns)
    df['volatility_20d'] = df.groupby('ticker')['log_ret'].transform(
        lambda x: x.rolling(window=20, min_periods=1).std()
    )

    # High-Low range
    df['hl_range'] = (df['high_price'] - df['low_price']) / df['close_price']

    return df


def merge_features(price_df: pd.DataFrame, market_global_df: pd.DataFrame,
                   market_index_df: pd.DataFrame, sector_df: pd.DataFrame,
                   calendar_df: pd.DataFrame, market_events_df: pd.DataFrame,
                   stock_events_df: pd.DataFrame, vi_events_df: pd.DataFrame,
                   leverage_daily_df: pd.DataFrame, leverage_products_df: pd.DataFrame,
                   ticker_universe_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Merge all feature tables together."""

    # Start with price_daily as base
    feature_pool = price_df.copy()

    # Join market_global (macro indicators)
    feature_pool = feature_pool.merge(market_global_df, on='trade_date', how='left')

    # Join market indices
    feature_pool = feature_pool.merge(market_index_df, on='trade_date', how='left')

    # Join sector data (need to get sector_id from ticker_universe first)
    ticker_to_sector = ticker_universe_df[['ticker', 'sector_id']].drop_duplicates()
    feature_pool = feature_pool.merge(ticker_to_sector, on='ticker', how='left')
    feature_pool = feature_pool.merge(sector_df, on='trade_date', how='left')
    # Any ticker missing a sector_id match (should not happen post-backfill, but defensive
    # for future new tickers) falls back to the documented "unclassified" bucket, not NULL.
    feature_pool['sector_id'] = feature_pool['sector_id'].fillna(SECTOR_ID_UNCLASSIFIED).astype(int)

    # Join market_id (derived from ticker_universe.is_kospi — see derive_market_id())
    ticker_to_market = ticker_universe_df[['ticker', 'is_kospi']].drop_duplicates().copy()
    ticker_to_market['market_id'] = ticker_to_market['is_kospi'].apply(derive_market_id)
    feature_pool = feature_pool.merge(ticker_to_market[['ticker', 'market_id']], on='ticker', how='left')
    feature_pool['market_id'] = feature_pool['market_id'].fillna(MARKET_ID_UNCLASSIFIED).astype(int)

    # Join calendar
    # Convert boolean columns to int before merging
    bool_cols = calendar_df.select_dtypes(include='bool').columns
    for col in bool_cols:
        calendar_df[col] = calendar_df[col].astype('int')
    feature_pool = feature_pool.merge(calendar_df, on='trade_date', how='left')

    # Join market events
    # Convert boolean columns to int before merging
    bool_cols = market_events_df.select_dtypes(include='bool').columns
    for col in bool_cols:
        market_events_df[col] = market_events_df[col].astype('int')
    feature_pool = feature_pool.merge(market_events_df, on='trade_date', how='left')

    # Join stock events
    if not stock_events_df.empty:
        # Convert boolean columns to int before merging
        bool_cols = stock_events_df.select_dtypes(include='bool').columns
        for col in bool_cols:
            stock_events_df[col] = stock_events_df[col].astype('int')
        feature_pool = feature_pool.merge(stock_events_df, on=['ticker', 'trade_date'], how='left')
    else:
        # Add empty columns with 0 values
        for col in ['is_dividend', 'is_bonus_issue', 'is_split', 'is_rights_offering']:
            feature_pool[col] = 0

    # Join VI events
    if not vi_events_df.empty:
        # Convert boolean columns to int before merging
        bool_cols = vi_events_df.select_dtypes(include='bool').columns
        for col in bool_cols:
            vi_events_df[col] = vi_events_df[col].astype('int')
        feature_pool = feature_pool.merge(vi_events_df, on=['ticker', 'trade_date'], how='left')
    else:
        feature_pool['is_vi_triggered'] = 0
        feature_pool['vi_count_recent5d'] = 0

    # Join labels
    if not labels_df.empty:
        feature_pool = feature_pool.merge(labels_df, on=['ticker', 'trade_date'], how='left')
    else:
        feature_pool['label'] = None
        feature_pool['next_day_return'] = None

    # Forward-fill macro and market-level features (not ticker-specific)
    macro_cols = [col for col in feature_pool.columns if col.startswith('index_') or
                  col in ['snp500_close', 'nasdaq_close', 'phlx_semi_close', 'vix',
                         'wti_crude_oil', 'gold_price', 'usd_krw', 'us_10y_yield',
                         'fed_rate', 'kr_base_rate', 'is_bok', 'is_fomc',
                         'is_witching_kr', 'is_witching_us']]
    feature_pool[macro_cols] = feature_pool[macro_cols].fillna(method='ffill')

    return feature_pool


def aggregate_leverage_features(leverage_daily_df: pd.DataFrame, leverage_products_df: pd.DataFrame,
                                price_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate leverage signals per underlying ticker per day."""

    if leverage_daily_df.empty or leverage_products_df.empty:
        # Return DataFrame with all zeros
        dates = price_df[['ticker', 'trade_date']].drop_duplicates()
        dates['lev_total_volume'] = 0.0
        dates['lev_total_aum'] = 0.0
        dates['lev_aum_to_mktcap'] = 0.0
        dates['est_rebalancing_flow'] = 0.0
        return dates

    # Join leverage_daily with leverage_products to get underlying ticker and multiple
    lev_merged = leverage_daily_df.merge(leverage_products_df[['code', 'underlying_ticker', 'multiple']],
                                        left_on='code', right_on='code', how='left')

    # Remove rows where underlying_ticker is missing
    lev_merged = lev_merged.dropna(subset=['underlying_ticker'])

    # Calculate per-underlying-per-day leverage signals
    result_rows = []
    for (underlying_ticker, trade_date), group in lev_merged.groupby(['underlying_ticker', 'trade_date']):
        # Get price of underlying on that date for return calculation
        underlying_price_row = price_df[(price_df['ticker'] == underlying_ticker) &
                                       (price_df['trade_date'] == trade_date)]

        if underlying_price_row.empty or underlying_price_row.iloc[0]['close_price'] is None:
            continue

        current_price = underlying_price_row.iloc[0]['close_price']
        prev_price_row = price_df[(price_df['ticker'] == underlying_ticker) &
                                  (price_df['trade_date'] < trade_date)].sort_values('trade_date').tail(1)

        if prev_price_row.empty or prev_price_row.iloc[0]['close_price'] is None:
            underlying_return = 0.0
        else:
            prev_price = prev_price_row.iloc[0]['close_price']
            underlying_return = (current_price - prev_price) / prev_price

        # Market cap from ticker_universe (use latest available)
        ticker_universe_row = price_df[(price_df['ticker'] == underlying_ticker)].iloc[-1]
        shares_outstanding = ticker_universe_row['shares_outstanding']
        if shares_outstanding is None or shares_outstanding == 0:
            underlying_market_cap = current_price * 1e9  # Rough estimate
        else:
            underlying_market_cap = current_price * shares_outstanding

        # Build product_rows for aggregate_leverage_signals
        product_rows = []
        for _, lev_row in group.iterrows():
            product_rows.append({
                'code': lev_row['code'],
                'multiple': lev_row['multiple'],
                'close_price': lev_row['close_price'] or 0.0,
                'volume': lev_row['volume'] or 0,
                'aum': lev_row['aum'] or 0.0,
            })

        # Aggregate leverage signals
        signals = aggregate_leverage_signals(product_rows, underlying_return, underlying_market_cap)
        result_rows.append({
            'ticker': underlying_ticker,
            'trade_date': trade_date,
            'lev_total_volume': signals['lev_total_volume'],
            'lev_total_aum': signals['lev_total_aum'],
            'lev_aum_to_mktcap': signals['lev_aum_to_mktcap'],
            'est_rebalancing_flow': signals['est_rebalancing_flow'],
        })

    result_df = pd.DataFrame(result_rows)

    # Merge with all ticker-date combinations and fill missing with 0
    all_dates = price_df[['ticker', 'trade_date']].drop_duplicates()
    if not result_df.empty:
        leverage_features = all_dates.merge(result_df, on=['ticker', 'trade_date'], how='left')
    else:
        leverage_features = all_dates.copy()

    leverage_features['lev_total_volume'] = leverage_features['lev_total_volume'].fillna(0.0)
    leverage_features['lev_total_aum'] = leverage_features['lev_total_aum'].fillna(0.0)
    leverage_features['lev_aum_to_mktcap'] = leverage_features['lev_aum_to_mktcap'].fillna(0.0)
    leverage_features['est_rebalancing_flow'] = leverage_features['est_rebalancing_flow'].fillna(0.0)

    return leverage_features


def build_feature_pool(dsn: str, start_date: str, end_date: str) -> None:
    """Build feature_pool table by joining all features."""

    print(f"Loading data from {start_date} to {end_date}...")

    # Load all data
    price_df = load_price_daily(dsn, start_date, end_date)
    print(f"  Loaded {len(price_df)} price_daily rows")

    market_global_df = load_market_global(dsn, start_date, end_date)
    print(f"  Loaded {len(market_global_df)} market_global rows")

    market_index_df = load_market_index_daily(dsn, start_date, end_date)
    print(f"  Loaded {len(market_index_df)} market_index_daily rows")

    sector_df = load_sector_daily_ohlcv(dsn, start_date, end_date)
    print(f"  Loaded {len(sector_df)} sector_daily_ohlcv rows")

    calendar_df = load_calendar(dsn, start_date, end_date)
    print(f"  Loaded {len(calendar_df)} calendar rows")

    market_events_df = load_market_events(dsn, start_date, end_date)
    print(f"  Loaded {len(market_events_df)} market_events rows")

    stock_events_df = load_stock_events(dsn, start_date, end_date)
    print(f"  Loaded {len(stock_events_df)} stock_events rows")

    vi_events_df = load_vi_events(dsn, start_date, end_date)
    print(f"  Loaded {len(vi_events_df)} vi_events rows")

    leverage_daily_df, leverage_products_df = load_leverage_daily_and_products(dsn, start_date, end_date)
    print(f"  Loaded {len(leverage_daily_df)} leverage_daily rows, {len(leverage_products_df)} products")

    labels_df = load_labels(dsn, start_date, end_date)
    print(f"  Loaded {len(labels_df)} labels rows")

    # Load ticker_universe for sector mapping
    conn = psycopg2.connect(dsn)
    ticker_universe_df = pd.read_sql("SELECT * FROM ticker_universe", conn)
    conn.close()
    print(f"  Loaded {len(ticker_universe_df)} ticker_universe rows")

    # Calculate technical indicators
    print("Calculating technical indicators...")
    price_df = calculate_technical_indicators(price_df)

    # Aggregate leverage features
    print("Aggregating leverage features...")
    leverage_features_df = aggregate_leverage_features(leverage_daily_df, leverage_products_df, price_df)

    # Merge all features
    print("Merging all features...")
    feature_pool = merge_features(
        price_df, market_global_df, market_index_df, sector_df,
        calendar_df, market_events_df, stock_events_df, vi_events_df,
        leverage_daily_df, leverage_products_df, ticker_universe_df, labels_df
    )

    # Merge leverage features
    feature_pool = feature_pool.merge(
        leverage_features_df,
        on=['ticker', 'trade_date'],
        how='left'
    )

    # Fill missing leverage features with 0
    leverage_cols = ['lev_total_volume', 'lev_total_aum', 'lev_aum_to_mktcap', 'est_rebalancing_flow']
    for col in leverage_cols:
        if col in feature_pool.columns:
            feature_pool[col] = feature_pool[col].fillna(0.0)

    # Deduplicate: keep the row with the most non-null values for each (ticker, trade_date)
    print(f"Deduplicating feature_pool (rows before dedup: {len(feature_pool)})...")
    feature_pool['_non_null_count'] = feature_pool.notna().sum(axis=1)
    feature_pool = feature_pool.loc[feature_pool.groupby(['ticker', 'trade_date'])['_non_null_count'].idxmax()]
    feature_pool = feature_pool.drop('_non_null_count', axis=1)

    print(f"Built feature_pool with {len(feature_pool)} rows, {len(feature_pool.columns)} columns")

    # Upsert to database
    print("Upserting to feature_pool table...")
    upsert_feature_pool(dsn, feature_pool)
    print("Feature pool build complete!")


def upsert_feature_pool(dsn: str, df: pd.DataFrame) -> None:
    """Upsert feature_pool DataFrame to database using a parameterized, batched upsert
    (psycopg2.extras.execute_values) rather than string-concatenating values into SQL text.
    String concatenation was fragile to inf/-inf floats (e.g. from divide-by-zero in disparity/
    volume_ratio calcs) producing invalid SQL literals — those now round-trip safely as NULL."""
    conn = psycopg2.connect(dsn)

    non_pk_cols = [c for c in df.columns if c not in ('ticker', 'trade_date')]

    # Create feature_pool table if not exists, and add any newly-introduced columns
    # (e.g. market_id) to an already-existing table.
    with conn.cursor() as cur:
        columns_sql = []
        for col in non_pk_cols:
            dtype = df[col].dtype
            if col in ('sector_id', 'market_id', 'day_of_week'):
                col_type = 'INT'
            elif dtype == 'float64':
                col_type = 'NUMERIC'
            elif dtype == 'int64' or dtype == 'int32':
                col_type = 'BIGINT'
            else:
                col_type = 'NUMERIC'
            columns_sql.append((col, col_type))

        columns_sql_str = ", ".join(f"{c} {t}" for c, t in columns_sql)
        if columns_sql_str:
            columns_sql_str = ", " + columns_sql_str

        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS feature_pool (
                ticker VARCHAR(6) NOT NULL,
                trade_date DATE NOT NULL
                {columns_sql_str},
                PRIMARY KEY (ticker, trade_date)
            )
        """
        cur.execute(create_table_sql)

        for col, col_type in columns_sql:
            cur.execute(f"ALTER TABLE feature_pool ADD COLUMN IF NOT EXISTS {col} {col_type}")
    conn.commit()

    # Replace +/-inf (e.g. divide-by-zero in disparity/volume_ratio) with NULL, and NaN with
    # None so psycopg2 can bind them as parameters rather than needing string literals.
    print("Preparing data for parameterized batch upsert...")
    insert_cols = list(df.columns)
    clean_df = df[insert_cols].replace([np.inf, -np.inf], np.nan)
    clean_df = clean_df.astype(object).where(pd.notna(clean_df), None)
    values = list(clean_df.itertuples(index=False, name=None))

    conflict_update = ", ".join(f"{c} = EXCLUDED.{c}" for c in non_pk_cols)
    insert_sql = f"""
        INSERT INTO feature_pool ({', '.join(insert_cols)})
        VALUES %s
        ON CONFLICT (ticker, trade_date) DO UPDATE SET {conflict_update}
    """

    print("Executing parameterized batch upsert...")
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, insert_sql, values, page_size=1000)
    conn.commit()

    conn.close()
    print(f"Batch upsert complete: {len(df)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build feature_pool table")
    parser.add_argument("--start", default="2019-01-02", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD)")
    parser.add_argument("--dsn", default=None, help="Database DSN")

    args = parser.parse_args()

    if args.dsn is None:
        args.dsn = os.environ.get("STOCK_DB_V2_DSN")
        if not args.dsn:
            print("ERROR: STOCK_DB_V2_DSN environment variable must be set")
            sys.exit(1)

    build_feature_pool(args.dsn, args.start, args.end)
