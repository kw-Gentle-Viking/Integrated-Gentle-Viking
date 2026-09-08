from data_collection.backfill_dart_calendar_sector import (
    classify_dart_event, build_calendar_rows, parse_dart_reports, parse_sector_daily_response,
)


def test_classifies_known_keywords():
    assert classify_dart_event("유상증자 결정") == "유상증자"
    assert classify_dart_event("무상증자 결정") == "무상증자"
    assert classify_dart_event("현금ㆍ현물배당 결정") == "배당"
    assert classify_dart_event("주식분할 결정") == "액면분할"
    assert classify_dart_event("연결재무제표기준영업(잠정)실적") == "실적발표"
    assert classify_dart_event("합병 결정") == "합병"


def test_returns_none_for_unrelated_report():
    assert classify_dart_event("최대주주변경") is None


def test_build_calendar_marks_weekends_as_closed():
    rows = build_calendar_rows("2019-01-01", "2019-01-07", short_selling_ban_periods=[])
    by_date = {r["base_date"]: r for r in rows}
    assert by_date["2019-01-05"]["is_market_open"] is False  # 토
    assert by_date["2019-01-06"]["is_market_open"] is False  # 일
    assert by_date["2019-01-02"]["is_market_open"] is True   # 수


def test_build_calendar_flags_short_selling_ban_period():
    rows = build_calendar_rows("2019-01-01", "2019-01-03",
                                short_selling_ban_periods=[("2019-01-01", "2019-01-03")])
    assert all(r["is_short_selling_banned"] for r in rows if r["day_of_week"] < 5)


def test_parse_dart_reports_classifies_and_filters():
    raw_reports = [
        {"rcept_dt": "20190315", "report_nm": "유상증자 결정"},
        {"rcept_dt": "20190316", "report_nm": "최대주주변경"},  # 매핑 안 되는 건 제외
    ]
    rows = parse_dart_reports(raw_reports, ticker="005930")
    assert rows == [{"ticker": "005930", "event_date": "2019-03-15",
                       "event_type": "유상증자", "description": "유상증자 결정"}]


def test_parse_sector_daily_response_extracts_ohlcv():
    raw = {"output2": [{"stck_bsop_date": "20190102", "bstp_nmix_oprc": "1050.5",
                          "bstp_nmix_hgpr": "1055.0", "bstp_nmix_lwpr": "1048.0",
                          "bstp_nmix_prpr": "1052.0", "acml_vol": "500000"}]}
    rows = parse_sector_daily_response(raw, sector_code="0005")
    assert rows == [{"sector_code": "0005", "trade_date": "2019-01-02",
                       "open": 1050.5, "high": 1055.0, "low": 1048.0,
                       "close": 1052.0, "volume": 500000}]


def test_smoke_dart_calendar_sector_imports():
    """Smoke test: verify all module imports work correctly."""
    from data_collection.run_dart_calendar_sector_backfill import (
        SECTOR_CODES, backfill_calendar, backfill_sector_daily_ohlcv,
    )
    assert len(SECTOR_CODES) == 25, f"Expected 25 sector codes, got {len(SECTOR_CODES)}"
    assert all(isinstance(code, str) and len(code) == 4 for code in SECTOR_CODES)


def test_calendar_includes_multiple_ban_periods():
    """Verify calendar handles multiple overlapping ban periods correctly."""
    rows = build_calendar_rows(
        "2020-03-15", "2020-03-20",
        short_selling_ban_periods=[
            ("2020-03-16", "2020-03-18"),  # First ban
            ("2020-03-17", "2020-03-19"),  # Overlapping ban
        ]
    )
    by_date = {r["base_date"]: r for r in rows}
    # Dates outside bans
    assert not by_date["2020-03-15"]["is_short_selling_banned"]
    # Dates in first ban
    assert by_date["2020-03-16"]["is_short_selling_banned"]
    assert by_date["2020-03-17"]["is_short_selling_banned"]
    # Date in overlap
    assert by_date["2020-03-18"]["is_short_selling_banned"]
    # Date only in second ban
    assert by_date["2020-03-19"]["is_short_selling_banned"]
    # Date after bans
    assert not by_date["2020-03-20"]["is_short_selling_banned"]
