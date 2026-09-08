from data_collection.backfill_daily_price import parse_daily_price_response


def test_parses_output2_rows_into_price_rows():
    raw = {
        "output2": [
            {"stck_bsop_date": "20190102", "stck_oprc": "45500", "stck_hgpr": "45650",
             "stck_lwpr": "45250", "stck_clpr": "45400", "acml_vol": "10000000",
             "acml_tr_pbmn": "454000000000", "lstn_stcn": "5969782550"},
        ]
    }
    rows = parse_daily_price_response(raw, ticker="005930")
    assert rows == [{
        "ticker": "005930", "trade_date": "2019-01-02",
        "open_price": 45500.0, "high_price": 45650.0, "low_price": 45250.0, "close_price": 45400.0,
        "volume": 10000000, "turnover": 454000000000.0, "shares_outstanding": 5969782550,
    }]


def test_skips_zero_volume_rows():
    raw = {"output2": [{"stck_bsop_date": "20190105", "stck_oprc": "0", "stck_hgpr": "0",
                          "stck_lwpr": "0", "stck_clpr": "0", "acml_vol": "0",
                          "acml_tr_pbmn": "0", "lstn_stcn": "5969782550"}]}
    assert parse_daily_price_response(raw, ticker="005930") == []


def test_returns_empty_list_for_missing_output2():
    assert parse_daily_price_response({}, ticker="005930") == []
