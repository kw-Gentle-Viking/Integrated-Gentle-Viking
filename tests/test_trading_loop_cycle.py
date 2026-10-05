"""자동매매 루프 한 사이클을 실제 코드 경로로 돌린다: AI 예측 → 배분 → 전략 주문 → 리스크 확인 →
KIS 주문(mojito 시그니처) → 포지션 갱신 → TradeLog/AutoTradeDecision 기록.
외부 의존성(KIS 웹소켓, 브로커, 잔고, DB 세션)만 가짜로 바꾸고 나머지는 실제 코드다."""
import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.routes_trade as rt
from app.db import Base
from app.models import AutoTradeDecision, TradeLog
from app.schemas import AllocationConfig
from backtest.engine.risk import Order, OrderType, Side


class FakeWS:
    def __init__(self, *a, **k):
        pass

    async def connect(self, tickers):
        return None

    async def disconnect(self):
        return None

    def get_price(self, ticker):
        return {"price": 70_000, "high": 70_500, "low": 69_500, "volume": 1000}


class FakeBroker:
    def __init__(self):
        self.calls = []

    def create_order(self, **kwargs):
        self.calls.append(kwargs)
        return {"rt_cd": "0", "msg1": "OK", "output": {"ODNO": "0000123456", "KRX_FWDG_ORD_ORGNO": "00950"}}


class AlwaysBuyStrategy:
    def __init__(self, symbol):
        self.symbol = symbol

    def generate_orders(self, row, portfolio):
        return [Order(ts=row.name, symbol=self.symbol, side=Side.BUY, qty=1, order_type=OrderType.MARKET)]


@pytest.fixture()
def session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


def test_one_trading_cycle_places_order_and_records_it(monkeypatch, session_factory):
    broker = FakeBroker()
    monkeypatch.setattr(rt, "SessionLocal", session_factory)
    monkeypatch.setattr(rt, "KISWebSocket", FakeWS)
    monkeypatch.setattr(rt, "broker", broker)
    monkeypatch.setattr(rt, "kis_is_mock", lambda: True)
    monkeypatch.setattr(rt, "kis_real_trading_enabled", lambda: False)
    monkeypatch.setattr(rt, "load_live_positions", lambda b: {})
    monkeypatch.setattr(rt, "create_strategy", lambda symbol, sid, params=None: AlwaysBuyStrategy(symbol))
    monkeypatch.setattr(rt, "ai_signal_event", asyncio.Event())
    monkeypatch.setattr(rt, "FILL_CHECK_DELAY_SEC", 0)
    monkeypatch.setattr(rt, "inquire_order_fill", lambda b, odno, d, m: {
        "ticker": "x", "ordered_qty": 1, "filled_qty": 1, "avg_price": 70_250.0})
    from app.shared_state import realtime_predictions
    tickers = ["005930", "000660", "035420"]
    for t in tickers:
        monkeypatch.setitem(realtime_predictions, t, {
            "ticker": t, "signal": "BUY", "confidence": 0.9,
            "trade_datetime": datetime.now().isoformat(), "model_version": "v3_structure_nowd_cs-seed0",
        })
    user_id = 1
    rt.warmup_events[user_id] = asyncio.Event()
    rt.warmup_events[user_id].set()
    rt.warmup_requirements[user_id] = {t: 0 for t in tickers}
    rt.warmup_received[user_id] = {}

    async def scenario():
        task = asyncio.create_task(rt.trading_loop(
            user_id=user_id, tickers=tickers, persona_id=3, total_capital=10_000_000,
            config=AllocationConfig(), ticker_strategies=[]))
        await asyncio.sleep(4)  # 구독/워밍업 초기 대기 (3초)
        rt.ai_signal_event.set()
        for _ in range(100):
            await asyncio.sleep(0.1)
            if len(broker.calls) >= len(tickers):
                break
        await asyncio.sleep(0.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())

    assert len(broker.calls) == 3
    for call in broker.calls:
        assert call["side"] == "buy" and call["symbol"] in tickers
        assert call["quantity"] >= 1 and call["order_type"] == "01"

    db = session_factory()
    logs = db.query(TradeLog).all()
    assert len(logs) == 3
    assert all(l.side == "BUY" and l.status == "FILLED" and l.order_no == "0000123456" for l in logs)
    assert all(l.price == 70_250.0 for l in logs)  # 체결 조회 평균가로 기록 (결정 시점 가격 아님)
    decisions = [d.action for d in db.query(AutoTradeDecision).all()]
    assert decisions.count("ORDER_SUBMITTED") == 3
    db.close()


@pytest.mark.parametrize("policy", ["cap_cash", "redistribute"])
@pytest.mark.parametrize("n_buy", [1, 2, 3, 5])
def test_policy_passes_risk_check_for_every_signal_count(policy, n_buy):
    """정책 a(cap_cash)와 b(redistribute)는 신호가 1~5개여도 1회 주문이 리스크 한도를 통과해야 한다."""
    import pandas as pd
    from app.allocation_policy import per_order_notional_cap
    from app.services_allocation import allocate_portfolio
    from backtest.engine.risk import Portfolio, RiskLimits, RiskManager
    preds = [{"ticker": f"T{i}", "signal": "BUY", "confidence": 0.9} for i in range(n_buy)]
    alloc = allocate_portfolio(preds, persona_id=3, total_capital=10_000_000, max_weight=0.3,
                               cash_reserve=0.1, min_confidence=0.6, policy=policy)
    rm = RiskManager(RiskLimits(per_order_notional_cap=per_order_notional_cap(policy)))
    pf = Portfolio(cash=10_000_000, equity=10_000_000)
    for a in alloc:
        notional = (a["amount"] // 70_000) * 70_000
        assert rm.check_pretrade(pd.Timestamp.now(), pf, a["ticker"], notional)
