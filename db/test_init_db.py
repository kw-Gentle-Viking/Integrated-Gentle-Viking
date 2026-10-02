import re
from pathlib import Path


def test_schema_defines_all_required_tables():
    ddl = Path(__file__).parent.joinpath("schema.sql").read_text()
    required_tables = [
        "ticker_universe", "price_daily", "daily_valuation", "investor_flow_daily",
        "market_index_daily", "market_global", "sector_daily_ohlcv", "stock_events",
        "calendar", "market_events", "leverage_products", "leverage_daily",
        "vi_events", "intraday_5min", "intraday_1min", "labels",
    ]
    found = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", ddl))
    missing = [t for t in required_tables if t not in found]
    assert not missing, f"schema.sql is missing tables: {missing}"
