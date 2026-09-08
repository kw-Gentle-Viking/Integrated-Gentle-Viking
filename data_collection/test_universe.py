from data_collection.universe import compute_top_n_snapshot


def test_ranks_by_market_cap_descending():
    candidates = [
        {"ticker": "AAA", "close_price": 100.0, "shares_outstanding": 10, "is_kospi": True},   # cap 1000
        {"ticker": "BBB", "close_price": 50.0, "shares_outstanding": 100, "is_kospi": True},    # cap 5000
        {"ticker": "CCC", "close_price": 10.0, "shares_outstanding": 10, "is_kospi": False},    # cap 100
    ]
    result = compute_top_n_snapshot(candidates, n=2)
    assert [c["ticker"] for c in result] == ["BBB", "AAA"]
    assert result[0]["rank"] == 1 and result[0]["market_cap"] == 5000.0
    assert result[1]["rank"] == 2 and result[1]["market_cap"] == 1000.0


def test_truncates_to_n():
    candidates = [
        {"ticker": f"T{i}", "close_price": float(i), "shares_outstanding": 1, "is_kospi": True}
        for i in range(10)
    ]
    result = compute_top_n_snapshot(candidates, n=3)
    assert len(result) == 3
    assert [c["ticker"] for c in result] == ["T9", "T8", "T7"]


def test_excludes_tickers_missing_price_or_shares():
    candidates = [
        {"ticker": "OK", "close_price": 10.0, "shares_outstanding": 10, "is_kospi": True},
        {"ticker": "NO_PRICE", "close_price": None, "shares_outstanding": 10, "is_kospi": True},
        {"ticker": "NO_SHARES", "close_price": 10.0, "shares_outstanding": None, "is_kospi": True},
    ]
    result = compute_top_n_snapshot(candidates, n=10)
    assert [c["ticker"] for c in result] == ["OK"]
