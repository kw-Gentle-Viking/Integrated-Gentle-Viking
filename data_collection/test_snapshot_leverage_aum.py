from data_collection.snapshot_leverage_aum import estimate_aum, parse_etf_nav, parse_listed_shares


def test_parses_nav_and_listed_shares():
    assert parse_etf_nav({"output": {"nav": "13393.81"}}) == 13393.81
    assert parse_listed_shares({"output": {"lstn_stcn": "115775000"}}) == 115775000


def test_missing_fields_give_none_not_zero():
    assert parse_etf_nav({"output": {}}) is None
    assert estimate_aum(None, 100) is None


def test_estimated_aum_is_nav_times_shares():
    assert round(estimate_aum(13393.81, 115775000)) == 1550668352750
