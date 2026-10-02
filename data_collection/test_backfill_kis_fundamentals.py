from data_collection.backfill_kis_fundamentals import (
    parse_valuation_response, parse_investor_flow_response,
)


def test_parse_valuation_extracts_per_pbr_market_cap():
    raw = {"output": {"per": "12.5", "pbr": "1.8", "hts_avls": "450000"}}
    result = parse_valuation_response(raw, ticker="005930", trade_date="2019-01-02")
    assert result == {"ticker": "005930", "trade_date": "2019-01-02",
                        "per": 12.5, "pbr": 1.8, "market_cap": 450000.0 * 1_000_000}


def test_parse_valuation_returns_none_when_output_missing():
    assert parse_valuation_response({}, ticker="005930", trade_date="2019-01-02") is None


def test_parse_investor_flow_extracts_net_amounts():
    raw = {"output": [{"frgn_ntby_qty": "100", "frgn_ntby_tr_pbmn": "5000",
                         "orgn_ntby_tr_pbmn": "3000", "prsn_ntby_tr_pbmn": "-8000"}]}
    result = parse_investor_flow_response(raw, ticker="005930", trade_date="2019-01-02")
    assert result == {"ticker": "005930", "trade_date": "2019-01-02",
                        "individual_net_amt": -8000.0 * 1_000_000,
                        "foreign_net_amt": 5000.0 * 1_000_000,
                        "inst_net_amt": 3000.0 * 1_000_000}
