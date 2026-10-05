"""허술한 부분을 드러내는 탐침 테스트 (2026-10-05). 실패가 예상되는 항목은 xfail(strict=True)로 표시한다.
고치면 XPASS가 되어 실패하므로, 그때 마커를 지우면 된다."""
import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest

import app.routes_ai_command as rac
import app.routes_trade as rt


def _user(uid=1):
    return SimpleNamespace(id=uid)


@pytest.mark.xfail(strict=True, reason="실행 중이 아니어도 STOP 을 큐에 넣음 (10/5 사고 원인 후보)")
def test_stop_when_nothing_running_still_queues_stop(monkeypatch):
    """실행 중이 아닐 때 /trade/stop 이 STOP 을 큐에 넣으면, 추론 서버 등록이 이유 없이 지워진다(10/5 사고)."""
    monkeypatch.setattr(rt, "active_tasks", {})
    monkeypatch.setattr(rt, "active_demo_trades", set())
    monkeypatch.setattr(rac, "command_queue", [])
    monkeypatch.setattr(rt, "command_queue", rac.command_queue)
    asyncio.run(rt.stop_trading(current_user=_user()))
    assert [c for c in rac.command_queue if c["command"] == "STOP"] == []


@pytest.mark.xfail(strict=True, reason="장 휴일 여부(calendar)를 자동매매 시작에서 확인하지 않음")
def test_start_trading_checks_market_calendar():
    import inspect
    src = inspect.getsource(rt.start_trading)
    assert "calendar" in src or "is_market_open" in src or "is_trading_day" in src


def test_ai_prediction_with_out_of_range_label_is_flagged(capsys):
    """pred_label 이 0/1/2 밖이면 WARNING 이 남아야 한다(모델·서빙 버그를 조용히 덮지 않도록)."""
    from app.routes_ai_webhook import parse_prediction
    res = SimpleNamespace(ticker="005930", pred_str="BUY", pred_label=7,
                          prob_buy=0.5, prob_hold=0.3, prob_sell=0.2,
                          trade_datetime=datetime.now(), model_version="probe")
    parse_prediction(res)
    assert "WARNING" in capsys.readouterr().out
