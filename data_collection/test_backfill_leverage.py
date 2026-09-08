from data_collection.backfill_leverage import parse_vi_event_response, parse_leverage_daily_response


def test_parses_vi_trigger_and_release():
    raw = {"output": [{"vi_bsop_time": "20260602103000", "vi_rls_time": "20260602104500",
                         "vi_type_cd": "1"}]}
    result = parse_vi_event_response(raw, ticker="005930")
    assert result == [{"ticker": "005930", "triggered_at": "2026-06-02 10:30:00",
                         "released_at": "2026-06-02 10:45:00", "vi_type": "1"}]


def test_release_none_when_still_active():
    raw = {"output": [{"vi_bsop_time": "20260602103000", "vi_rls_time": "", "vi_type_cd": "1"}]}
    result = parse_vi_event_response(raw, ticker="005930")
    assert result[0]["released_at"] is None


def test_parses_leverage_daily_response():
    raw = {"output2": [
        {
            "stck_bsop_date": "20260610",
            "stck_clpr": "20975",
            "acml_vol": "64294119",
            "acml_tr_pbmn": "1378953345819",
        },
        {
            "stck_bsop_date": "20260609",
            "stck_clpr": "24115",
            "acml_vol": "71869987",
            "acml_tr_pbmn": "1609506998893",
        },
    ]}
    result = parse_leverage_daily_response(raw, code="0193W0")
    assert len(result) == 2
    # Verify first record
    assert result[0]["code"] == "0193W0"
    assert result[0]["trade_date"] == "2026-06-10"
    assert result[0]["close_price"] == 20975.0
    assert result[0]["volume"] == 64294119
    assert result[0]["turnover"] == 1378953345819.0
    assert result[0]["nav"] is None
    assert result[0]["aum"] is None
    # Verify second record
    assert result[1]["trade_date"] == "2026-06-09"
    assert result[1]["close_price"] == 24115.0


def test_leverage_daily_response_with_empty_output():
    raw = {"output2": []}
    result = parse_leverage_daily_response(raw, code="0193W0")
    assert result == []
