import csv
import tempfile
from pathlib import Path

from data_collection.universe import compute_top_n_snapshot, fetch_snapshot_prices, load_candidate_tickers


class _FakeClientOneTickerFails:
    def request(self, path, tr_id, params):
        if params["FID_INPUT_ISCD"] == "BAD":
            raise Exception("500 Server Error")
        return {"output2": [{"stck_clpr": "1000", "lstn_stcn": "10"}]}


def test_fetch_snapshot_prices_skips_ticker_whose_request_errors():
    result = fetch_snapshot_prices(_FakeClientOneTickerFails(), ["005930", "BAD", "000660"], "20190102")
    assert [r["ticker"] for r in result] == ["005930", "000660"]


def test_load_candidate_tickers_skips_non_numeric_codes():
    with tempfile.TemporaryDirectory() as tmp:
        kospi_path = Path(tmp) / "kospi.csv"
        kosdaq_path = Path(tmp) / "kosdaq.csv"
        with open(kospi_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ticker", "ticker_name"])
            writer.writerow(["005930", "삼성전자"])
            writer.writerow(["005935", "삼성전자우"])  # 우선주지만 숫자만이라 통과되는 케이스는 허용
        with open(kosdaq_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ticker", "ticker_name"])
            writer.writerow(["A123Z0", "특수종목"])  # 영문 포함 -> 제외돼야 함
            writer.writerow(["035760", "CJ ENM"])

        result = load_candidate_tickers(str(kospi_path), str(kosdaq_path))
        assert [c["ticker"] for c in result] == ["005930", "005935", "035760"]


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
