"""Basket(자동매매 대상 종목)에 AI가 전혀 모르는 종목(코스닥 등)이 검증 없이 들어가면 app.ai_client.
AIClient.predict()가 커버리지 부재를 confidence-0 HOLD 또는(AI_ALLOW_DUMMY_PREDICTIONS=true 시) 완전
무작위 BUY/SELL 로 조용히 치환해 실제 AI 신호와 구분이 안 됐다(2026-10-02 통합 감사). 이 모듈이 그 원인을
Basket 등록/자동매매 시작 시점에 막는다."""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def db():
    from app.db import Base
    import app.models  # noqa: F401
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


# ---- app.ai_universe ----
def test_is_ai_covered_ticker_allows_everything_when_dsn_unset(monkeypatch):
    import app.ai_universe as au
    monkeypatch.setattr(au, "STOCK_DB_V2_DSN", "")
    monkeypatch.setitem(au._cache, "tickers", None)
    assert au.is_ai_covered_ticker("999999") is True  # DSN 미설정 -> 과거 동작(검증 없음)과 동일


def test_is_ai_covered_ticker_checks_against_loaded_universe(monkeypatch):
    import app.ai_universe as au
    monkeypatch.setattr(au, "get_ai_universe", lambda force_refresh=False: {"005930", "000660"})
    assert au.is_ai_covered_ticker("005930") is True
    assert au.is_ai_covered_ticker("068270") is False  # 코스닥 종목 예시, AI 유니버스 밖


def test_get_ai_universe_caches_until_ttl_expires(monkeypatch):
    import app.ai_universe as au
    calls = []

    class FakeConn:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def execute(self, *_):
            calls.append(1)
            return SimpleNamespace(fetchall=lambda: [("005930",), ("000660",)])

    class FakeEngine:
        def connect(self):
            return FakeConn()

    monkeypatch.setattr(au, "STOCK_DB_V2_DSN", "postgresql://fake")
    monkeypatch.setattr(au, "_get_engine", lambda: FakeEngine())
    monkeypatch.setitem(au._cache, "tickers", None)
    monkeypatch.setitem(au._cache, "loaded_at", 0.0)

    first = au.get_ai_universe()
    second = au.get_ai_universe()
    assert first == {"005930", "000660"} == second
    assert len(calls) == 1  # 두 번째 호출은 캐시 사용, DB 재조회 없음


# ---- get_auto_trade_basket_items: AI 미커버 종목 제외 ----
def test_get_auto_trade_basket_items_excludes_ai_uncovered_tickers(monkeypatch, db):
    import app.routes_trade as rt
    from app.models import Basket

    db.add(Basket(user_id=1, ticker="005930", ticker_name="삼성전자"))
    db.add(Basket(user_id=1, ticker="068270", ticker_name="코스닥종목"))
    db.commit()

    monkeypatch.setattr(rt, "is_ai_covered_ticker", lambda t: t == "005930")

    auto_items, excluded_manual, excluded_unsupported = rt.get_auto_trade_basket_items(db, 1)
    assert [i.ticker for i in auto_items] == ["005930"]
    assert excluded_manual == []
    assert excluded_unsupported == ["068270"]


def test_no_auto_basket_message_reports_unsupported_tickers_too():
    from app.trade_helpers import no_auto_basket_message
    msg = no_auto_basket_message([], ["068270"])
    assert "068270" in msg and "AI" in msg


# ---- /basket POST rejects AI-uncovered tickers ----
def test_add_to_basket_rejects_ai_uncovered_ticker(monkeypatch, db):
    import app.routes_basket as rb

    monkeypatch.setattr(rb, "is_ai_covered_ticker", lambda t: t == "005930")
    user = SimpleNamespace(id=1)

    rb.add_to_basket(rb.BasketAdd(ticker="005930", ticker_name="삼성전자"), db=db, current_user=user)  # 통과

    with pytest.raises(HTTPException) as exc:
        rb.add_to_basket(rb.BasketAdd(ticker="068270", ticker_name="코스닥종목"), db=db, current_user=user)
    assert exc.value.status_code == 400
    assert "068270" in exc.value.detail


def test_seed_live_candles_is_noop_without_prod_dsn(monkeypatch, db):
    from app.intraday_seed import seed_live_candles
    monkeypatch.delenv("PROD_STOCK_DB_DSN", raising=False)
    assert seed_live_candles(db, ["005930"], {"005930": 240}) == {}
