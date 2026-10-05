from datetime import date, datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import app.market_calendar as mc
import app.routes_trade as rt


def test_weekend_is_closed_without_db(monkeypatch):
    monkeypatch.delenv("PROD_STOCK_DB_DSN", raising=False)
    assert mc.is_market_open_day(date(2026, 10, 3)) is False


def test_unknown_when_calendar_not_configured(monkeypatch):
    monkeypatch.delenv("PROD_STOCK_DB_DSN", raising=False)
    assert mc.is_market_open_day(date(2026, 10, 6)) is None


def test_holiday_is_reported_from_calendar(monkeypatch):
    monkeypatch.setenv("PROD_STOCK_DB_DSN", "postgresql://fake")
    monkeypatch.setattr(mc, "_lookup", lambda dsn, day: day != date(2026, 10, 5))
    assert mc.is_market_open_day(date(2026, 10, 5)) is False
    assert mc.is_market_open_day(date(2026, 10, 6)) is True


def test_start_is_refused_on_holiday(monkeypatch):
    monkeypatch.setattr(rt, "is_market_open_day", lambda d: False)
    with pytest.raises(HTTPException) as exc:
        rt._ensure_market_open_today()
    assert exc.value.status_code == 400 and "휴장일" in exc.value.detail


def test_start_proceeds_when_calendar_unknown(monkeypatch):
    monkeypatch.setattr(rt, "is_market_open_day", lambda d: None)
    rt._ensure_market_open_today()  # 예외 없이 통과


def test_stop_when_idle_does_not_queue_stop(monkeypatch):
    import asyncio
    monkeypatch.setattr(rt, "active_tasks", {})
    monkeypatch.setattr(rt, "active_demo_trades", set())
    queue = []
    monkeypatch.setattr(rt, "command_queue", queue)
    out = asyncio.run(rt.stop_trading(current_user=SimpleNamespace(id=1)))
    assert out["status"] == "NOT_RUNNING" and queue == []
