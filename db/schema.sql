CREATE TABLE IF NOT EXISTS ticker_universe (
    ticker VARCHAR(6) PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    market_cap NUMERIC NOT NULL,
    rank INT NOT NULL,
    is_kospi BOOLEAN NOT NULL,
    sector_id INT,
    listing_date DATE
);

CREATE TABLE IF NOT EXISTS price_daily (
    ticker VARCHAR(6) NOT NULL,
    trade_date DATE NOT NULL,
    open_price NUMERIC, high_price NUMERIC, low_price NUMERIC, close_price NUMERIC,
    volume BIGINT, turnover NUMERIC, shares_outstanding BIGINT,
    PRIMARY KEY (ticker, trade_date)
);

CREATE TABLE IF NOT EXISTS daily_valuation (
    ticker VARCHAR(6) NOT NULL, trade_date DATE NOT NULL,
    per NUMERIC, pbr NUMERIC, market_cap NUMERIC,
    PRIMARY KEY (ticker, trade_date)
);

CREATE TABLE IF NOT EXISTS investor_flow_daily (
    ticker VARCHAR(6) NOT NULL, trade_date DATE NOT NULL,
    individual_net_amt NUMERIC, foreign_net_amt NUMERIC, inst_net_amt NUMERIC, market_cap NUMERIC,
    PRIMARY KEY (ticker, trade_date)
);

CREATE TABLE IF NOT EXISTS market_index_daily (
    index_code VARCHAR(10) NOT NULL, trade_date DATE NOT NULL, close_price NUMERIC,
    PRIMARY KEY (index_code, trade_date)
);

CREATE TABLE IF NOT EXISTS market_global (
    trade_date DATE PRIMARY KEY,
    snp500_close NUMERIC, nasdaq_close NUMERIC, phlx_semi_close NUMERIC, vix NUMERIC,
    wti_crude_oil NUMERIC, gold_price NUMERIC, usd_krw NUMERIC,
    us_10y_yield NUMERIC, fed_rate NUMERIC, kr_base_rate NUMERIC
);

CREATE TABLE IF NOT EXISTS sector_daily_ohlcv (
    sector_code VARCHAR(4) NOT NULL, trade_date DATE NOT NULL,
    open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC, volume BIGINT,
    PRIMARY KEY (sector_code, trade_date)
);

CREATE TABLE IF NOT EXISTS stock_events (
    ticker VARCHAR(6) NOT NULL, event_date DATE NOT NULL, event_type VARCHAR(20) NOT NULL,
    description TEXT,
    PRIMARY KEY (ticker, event_date, event_type)
);

CREATE TABLE IF NOT EXISTS calendar (
    base_date DATE PRIMARY KEY,
    day_of_week INT, is_market_open BOOLEAN, is_holiday BOOLEAN, is_short_selling_banned BOOLEAN
);

CREATE TABLE IF NOT EXISTS market_events (
    event_date DATE NOT NULL, event_type VARCHAR(20) NOT NULL,
    is_bok BOOLEAN, is_fomc BOOLEAN, is_witching_kr BOOLEAN, is_witching_us BOOLEAN,
    PRIMARY KEY (event_date, event_type)
);

CREATE TABLE IF NOT EXISTS leverage_products (
    code VARCHAR(6) PRIMARY KEY,
    product_name VARCHAR(80) NOT NULL,
    underlying_ticker VARCHAR(6) NOT NULL,
    multiple NUMERIC NOT NULL,          -- +2.0 or -2.0
    product_type VARCHAR(3) NOT NULL,   -- 'ETF' or 'ETN'
    issuer VARCHAR(20),
    listed_date DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS leverage_daily (
    code VARCHAR(6) NOT NULL, trade_date DATE NOT NULL,
    close_price NUMERIC, volume BIGINT, turnover NUMERIC, nav NUMERIC, aum NUMERIC,
    PRIMARY KEY (code, trade_date)
);

CREATE TABLE IF NOT EXISTS vi_events (
    ticker VARCHAR(6) NOT NULL, triggered_at TIMESTAMP NOT NULL,
    released_at TIMESTAMP, vi_type VARCHAR(10),
    PRIMARY KEY (ticker, triggered_at)
);

CREATE TABLE IF NOT EXISTS intraday_5min (
    ticker VARCHAR(6) NOT NULL, datetime TIMESTAMP NOT NULL,
    open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC, volume BIGINT,
    PRIMARY KEY (ticker, datetime)
);

CREATE TABLE IF NOT EXISTS intraday_1min (
    ticker VARCHAR(6) NOT NULL, datetime TIMESTAMP NOT NULL,
    open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC, volume BIGINT,
    PRIMARY KEY (ticker, datetime)
);

CREATE TABLE IF NOT EXISTS labels (
    ticker VARCHAR(6) NOT NULL, trade_date DATE NOT NULL,
    next_day_return NUMERIC, label SMALLINT,  -- 0=매수, 1=관망, 2=매도, NULL=마지막 거래일(라벨 없음)
    PRIMARY KEY (ticker, trade_date)
);
