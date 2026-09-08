from data_collection.backfill_daily_price import parse_daily_price_response


def test_parses_output2_rows_into_price_rows():
    # 실제 KIS 응답 형태: output1(현재가, lstn_stcn 포함)과 output2(일별 시세, lstn_stcn 없음)가
    # 같은 응답 안에 함께 옴 — 2026-09-08 라이브 백필 중 발견됨.
    raw = {
        "output1": {"lstn_stcn": "5969782550"},
        "output2": [
            {"stck_bsop_date": "20190102", "stck_oprc": "45500", "stck_hgpr": "45650",
             "stck_lwpr": "45250", "stck_clpr": "45400", "acml_vol": "10000000",
             "acml_tr_pbmn": "454000000000"},
        ],
    }
    rows = parse_daily_price_response(raw, ticker="005930")
    assert rows == [{
        "ticker": "005930", "trade_date": "2019-01-02",
        "open_price": 45500.0, "high_price": 45650.0, "low_price": 45250.0, "close_price": 45400.0,
        "volume": 10000000, "turnover": 454000000000.0, "shares_outstanding": 5969782550,
    }]


def test_applies_same_shares_outstanding_to_all_rows_in_chunk():
    raw = {
        "output1": {"lstn_stcn": "100"},
        "output2": [
            {"stck_bsop_date": "20190103", "stck_oprc": "1", "stck_hgpr": "1",
             "stck_lwpr": "1", "stck_clpr": "1", "acml_vol": "1", "acml_tr_pbmn": "1"},
            {"stck_bsop_date": "20190102", "stck_oprc": "1", "stck_hgpr": "1",
             "stck_lwpr": "1", "stck_clpr": "1", "acml_vol": "1", "acml_tr_pbmn": "1"},
        ],
    }
    rows = parse_daily_price_response(raw, ticker="005930")
    assert [r["shares_outstanding"] for r in rows] == [100, 100]


def test_shares_outstanding_is_none_when_output1_missing():
    raw = {"output2": [{"stck_bsop_date": "20190102", "stck_oprc": "1", "stck_hgpr": "1",
                          "stck_lwpr": "1", "stck_clpr": "1", "acml_vol": "1", "acml_tr_pbmn": "1"}]}
    rows = parse_daily_price_response(raw, ticker="005930")
    assert rows[0]["shares_outstanding"] is None


def test_skips_zero_volume_rows():
    raw = {"output1": {"lstn_stcn": "5969782550"},
           "output2": [{"stck_bsop_date": "20190105", "stck_oprc": "0", "stck_hgpr": "0",
                         "stck_lwpr": "0", "stck_clpr": "0", "acml_vol": "0", "acml_tr_pbmn": "0"}]}
    assert parse_daily_price_response(raw, ticker="005930") == []


def test_returns_empty_list_for_missing_output2():
    assert parse_daily_price_response({}, ticker="005930") == []
