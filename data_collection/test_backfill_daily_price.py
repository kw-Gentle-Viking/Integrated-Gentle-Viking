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


# ---- adjusted-price backfill (2026-09-26 DATAADJ) ----
import os
import psycopg2
import pytest
from data_collection import backfill_daily_price as bdp


class _FakeClient:
    def __init__(self):
        self.calls = []

    def request(self, path, tr_id, params):
        self.calls.append(params)
        return {"output1": {}, "output2": []}


def _run_backfill(monkeypatch, **kw):
    monkeypatch.setattr(bdp, "upsert_price_daily", lambda dsn, rows: None)
    monkeypatch.setattr(bdp.time, "sleep", lambda s: None)
    client = _FakeClient()
    bdp.backfill_ticker(client, "dsn", "005930", start_date="20240101", end_date="20240110", **kw)
    return client.calls


def test_backfill_defaults_to_adjusted_prices(monkeypatch):
    calls = _run_backfill(monkeypatch)
    assert calls and all(c["FID_ORG_ADJ_PRC"] == "0" for c in calls)


def test_backfill_adjusted_false_requests_raw_prices(monkeypatch):
    calls = _run_backfill(monkeypatch, adjusted=False)
    assert calls and all(c["FID_ORG_ADJ_PRC"] == "1" for c in calls)


@pytest.mark.skipif(not os.environ.get("STOCK_DB_V2_DSN"), reason="needs stock_db_v2")
def test_upsert_overwrites_every_price_and_volume_column(monkeypatch):
    """A second upsert with different values must replace ALL columns (raw rows must not stay raw)."""
    conn = psycopg2.connect(os.environ["STOCK_DB_V2_DSN"])
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE price_daily (LIKE public.price_daily INCLUDING ALL)")

    class _Conn:  # hand the temp-table connection to upsert; ignore its close()
        def cursor(self, *a, **k): return conn.cursor(*a, **k)
        def commit(self): pass
        def close(self): pass

    monkeypatch.setattr(bdp.psycopg2, "connect", lambda dsn: _Conn())
    base = {"ticker": "T00001", "trade_date": "2021-09-01", "open_price": 100.0, "high_price": 110.0,
            "low_price": 90.0, "close_price": 105.0, "volume": 10, "turnover": 1000.0,
            "shares_outstanding": 5}
    bdp.upsert_price_daily("x", [base])
    new = {**base, "open_price": 10.0, "high_price": 11.0, "low_price": 9.0, "close_price": 10.5,
           "volume": 100, "turnover": 1001.0, "shares_outstanding": 50}
    bdp.upsert_price_daily("x", [new])
    with conn.cursor() as cur:
        cur.execute("SELECT open_price, high_price, low_price, close_price, volume, turnover, "
                    "shares_outstanding FROM price_daily WHERE ticker='T00001'")
        rows = cur.fetchall()
    conn.rollback(); conn.close()
    assert len(rows) == 1
    assert [float(x) for x in rows[0]] == [10.0, 11.0, 9.0, 10.5, 100.0, 1001.0, 50.0]
