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
    decisions = [d.action for d in db.query(AutoTradeDecision).all()]
    assert decisions.count("ORDER_SUBMITTED") == 3
    db.close()


@pytest.mark.xfail(strict=True, reason="1~2개 BUY 신호면 재정규화 후 종목 비중이 30%를 넘어 리스크 확인에서 전부 거절됨")
def test_two_buy_signals_pass_risk_check():
    from backtest.engine.risk import Portfolio, RiskLimits, RiskManager
    from app.services_allocation import allocate_portfolio
    preds = [{"ticker": t, "signal": "BUY", "confidence": 0.9} for t in ("005930", "000660")]
    alloc = allocate_portfolio(preds, persona_id=3, total_capital=10_000_000, max_weight=0.3,
                               cash_reserve=0.1, min_confidence=0.6)
    rm = RiskManager(RiskLimits())
    pf = Portfolio(cash=10_000_000, equity=10_000_000)
    import pandas as pd
    for a in alloc:
        notional = (a["amount"] // 70_000) * 70_000
        assert rm.check_pretrade(pd.Timestamp.now(), pf, a["ticker"], notional)
