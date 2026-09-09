"""Tests for backfill_vi_status.py, using real captured KIS FHPST01390000 response samples
(captured live 2026-09-09 against mrkt=K, date=2026-05-27, ticker 007120)."""

from data_collection.backfill_vi_status import parse_vi_status_response, raw_row_count

# Real sample captured live: FID_INPUT_ISCD=007120, FID_MRKT_CLS_CODE=K, FID_INPUT_DATE_1=20260527
REAL_SAMPLE_007120 = {
    "output": [
        {"hts_kor_isnm": "미래아이앤지", "mksc_shrn_iscd": "007120", "vi_cls_code": "N",
         "bsop_date": "20260527", "cntg_vi_hour": "180030", "vi_cncl_hour": "180238",
         "vi_kind_code": "2", "vi_prc": "1496", "vi_stnd_prc": "0", "vi_dprt": "0.00",
         "vi_dmc_stnd_prc": "1361", "vi_dmc_dprt": "9.92", "vi_count": "4"},
        {"hts_kor_isnm": "미래아이앤지", "mksc_shrn_iscd": "007120", "vi_cls_code": "N",
         "bsop_date": "20260527", "cntg_vi_hour": "173030", "vi_cncl_hour": "173232",
         "vi_kind_code": "2", "vi_prc": "1361", "vi_stnd_prc": "0", "vi_dprt": "0.00",
         "vi_dmc_stnd_prc": "1496", "vi_dmc_dprt": "-9.02", "vi_count": "3"},
        {"hts_kor_isnm": "미래아이앤지", "mksc_shrn_iscd": "007120", "vi_cls_code": "N",
         "bsop_date": "20260527", "cntg_vi_hour": "171030", "vi_cncl_hour": "171226",
         "vi_kind_code": "2", "vi_prc": "1497", "vi_stnd_prc": "0", "vi_dprt": "0.00",
         "vi_dmc_stnd_prc": "1361", "vi_dmc_dprt": "9.99", "vi_count": "2"},
        {"hts_kor_isnm": "미래아이앤지", "mksc_shrn_iscd": "007120", "vi_cls_code": "N",
         "bsop_date": "20260527", "cntg_vi_hour": "093149", "vi_cncl_hour": "093407",
         "vi_kind_code": "2", "vi_prc": "1470", "vi_stnd_prc": "0", "vi_dprt": "0.00",
         "vi_dmc_stnd_prc": "1371", "vi_dmc_dprt": "7.22", "vi_count": "1"},
    ],
    "rt_cd": "0", "msg_cd": "MCA00000", "msg1": "정상처리 되었습니다.",
}

EMPTY_SAMPLE = {"output": [], "rt_cd": "0", "msg_cd": "MCA00000", "msg1": "정상처리 되었습니다."}


def test_parses_all_rows_when_ticker_in_universe():
    rows = parse_vi_status_response(REAL_SAMPLE_007120, universe_tickers={"007120"})
    assert len(rows) == 4


def test_filters_out_tickers_not_in_universe():
    rows = parse_vi_status_response(REAL_SAMPLE_007120, universe_tickers={"005930"})
    assert rows == []


def test_triggered_and_released_timestamps_built_correctly():
    rows = parse_vi_status_response(REAL_SAMPLE_007120, universe_tickers={"007120"})
    first = rows[0]  # cntg_vi_hour=180030, vi_cncl_hour=180238, bsop_date=20260527
    assert first["triggered_at"] == "2026-05-27 18:00:30"
    assert first["released_at"] == "2026-05-27 18:02:38"


def test_vi_kind_code_mapped_to_static_dynamic():
    rows = parse_vi_status_response(REAL_SAMPLE_007120, universe_tickers={"007120"})
    assert all(r["vi_type"] == "DYNAMIC" for r in rows)  # all sample rows are vi_kind_code=2


def test_unknown_vi_kind_code_flagged_not_silently_dropped():
    raw = {"output": [dict(REAL_SAMPLE_007120["output"][0], vi_kind_code="9")]}
    rows = parse_vi_status_response(raw, universe_tickers={"007120"})
    assert rows[0]["vi_type"] == "UNKNOWN_9"


def test_vi_kind_code_3_mapped_to_both():
    # Real sample captured live 2026-09-09: 140410 (메지온), 2019-06-28 15:30:22, KOSDAQ.
    # vi_stnd_prc=90700 (static ref, non-zero) AND vi_dmc_stnd_prc=82700 (dynamic ref,
    # non-zero) both populated -- confirms code '3' means static+dynamic triggered together.
    raw = {"output": [{"hts_kor_isnm": "메지온", "mksc_shrn_iscd": "140410", "vi_cls_code": "N",
                        "bsop_date": "20190628", "cntg_vi_hour": "153022",
                        "vi_cncl_hour": "153228", "vi_kind_code": "3", "vi_prc": "74100",
                        "vi_stnd_prc": "90700", "vi_dprt": "-18.30",
                        "vi_dmc_stnd_prc": "82700", "vi_dmc_dprt": "-10.40", "vi_count": "2"}]}
    rows = parse_vi_status_response(raw, universe_tickers={"140410"})
    assert rows[0]["vi_type"] == "BOTH"


def test_missing_release_time_yields_none():
    raw = {"output": [dict(REAL_SAMPLE_007120["output"][0], vi_cncl_hour="")]}
    rows = parse_vi_status_response(raw, universe_tickers={"007120"})
    assert rows[0]["released_at"] is None


def test_all_zero_release_time_treated_as_none():
    # Real anomaly observed live 2026-09-09: 014680, 2026-06-08 09:02:29 trigger came back with
    # vi_cncl_hour="000000" once; a re-query moments later returned the correct "090342". A VI
    # release literally at midnight is physically impossible (trading/extended hours are
    # 09:00-18:00), so "000000" must be treated as "no real release time", not a real timestamp.
    raw = {"output": [dict(REAL_SAMPLE_007120["output"][0], vi_cncl_hour="000000")]}
    rows = parse_vi_status_response(raw, universe_tickers={"007120"})
    assert rows[0]["released_at"] is None


def test_empty_output_yields_no_rows():
    assert parse_vi_status_response(EMPTY_SAMPLE, universe_tickers={"007120"}) == []


def test_raw_row_count():
    assert raw_row_count(REAL_SAMPLE_007120) == 4
    assert raw_row_count(EMPTY_SAMPLE) == 0
