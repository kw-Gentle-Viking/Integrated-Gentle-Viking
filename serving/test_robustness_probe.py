"""허술한 부분을 드러내는 탐침 테스트 (2026-10-05). 실패가 예상되는 항목은 xfail(strict=True)."""
import logging
from datetime import datetime

import pytest

import serving.inference_pipeline as ip


def test_weekend_is_skipped():
    assert ip.is_within_market_hours(datetime(2026, 10, 3, 10, 0)) is False


@pytest.mark.xfail(strict=True, reason="추론 가드가 평일·시간만 보고 달력 휴장일을 모름 (10/5 대체공휴일에 추론 실행됨)")
def test_holiday_is_skipped():
    assert ip.is_within_market_hours(datetime(2026, 10, 5, 10, 0)) is False


@pytest.mark.xfail(strict=True, reason="종목 파일이 없으면 경고 없이 빈 목록을 돌려줌 -> 추론이 조용히 멈춤")
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
