"""허술한 부분을 드러내는 탐침 테스트 (2026-10-05). 실패가 예상되는 항목은 xfail(strict=True)."""
import logging
from datetime import datetime

import pytest

import serving.inference_pipeline as ip


def test_weekend_is_skipped():
    assert ip.is_within_market_hours(datetime(2026, 10, 3, 10, 0)) is False


def test_calendar_unknown_falls_back_to_weekday_rule(monkeypatch):
    """달력을 못 읽으면 평일 기준으로 진행한다(의도된 동작)."""
    monkeypatch.delenv("PROD_STOCK_DB_DSN", raising=False)
    assert ip.is_within_market_hours(datetime(2026, 10, 5, 10, 0)) is True


def test_missing_ticker_file_is_logged_not_silent(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(ip, "TICKERS_FILE", str(tmp_path / "missing.json"), raising=False)
    caplog.set_level(logging.WARNING)
    import inspect
    fn = next(getattr(ip, n) for n in ("load_active_tickers",) if hasattr(ip, n))
    params = inspect.signature(fn).parameters
    kwargs = {next(iter(params)): str(tmp_path / "missing.json")} if params else {}
    out = fn(**kwargs)
    assert out == []
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_registry_write_is_atomic_for_readers(tmp_path, monkeypatch):
    """쓰는 도중에도 읽는 쪽은 항상 완전한 JSON을 본다(임시 파일 + os.replace)."""
    import json
    import serving.api_server as api
    path = tmp_path / "active_tickers.json"
    monkeypatch.setattr(api, "TICKERS_FILE", str(path))
    api.save_tickers({"users": {"1": ["005930"]}, "all_tickers": ["005930"]})
    assert not (tmp_path / "active_tickers.json.tmp").exists()
    assert json.loads(path.read_text(encoding="utf-8"))["all_tickers"] == ["005930"]


def test_holiday_blocks_inference_when_calendar_says_closed(monkeypatch):
    from datetime import datetime as dt
    monkeypatch.setattr(ip, "is_market_open_day", lambda d: False)
    assert ip.is_within_market_hours(dt(2026, 10, 5, 10, 0)) is False
