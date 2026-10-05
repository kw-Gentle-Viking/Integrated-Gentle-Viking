"""mojito.KoreaInvestment.create_order 시그니처는 (side, symbol, price, quantity, order_type) 이고
side 는 소문자 'buy'/'sell', 시장가는 order_type '01'. 이전 코드는 qty= 키워드와 'BUY'/'market'을 보내서
모든 주문이 TypeError 또는 잘못된 매수/매도로 갔다(2026-10-05 실주문 점검 중 발견)."""
import inspect

import mojito
import pytest


def test_mojito_create_order_signature_matches_our_call():
    params = inspect.signature(mojito.KoreaInvestment.create_order).parameters
    for name in ("side", "symbol", "price", "quantity", "order_type"):
        assert name in params
    assert "qty" not in params


def test_routes_trade_uses_mojito_compatible_order_call():
    import app.routes_trade as rt
    src = inspect.getsource(rt)
    assert "quantity=qty" in src
    assert "side=order_signal.lower()" in src
    assert rt.MARKET_ORDER_TYPE == "01"


def test_smoke_order_refuses_production(monkeypatch):
    import asyncio
    import app.routes_trade as rt
    from types import SimpleNamespace
    from fastapi import HTTPException
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setattr(rt, "kis_is_mock", lambda: True)
    with __import__("pytest").raises(HTTPException) as e:
        asyncio.run(rt.smoke_order(rt.SmokeOrderRequest(ticker="005930", side="buy"), SimpleNamespace(id=1)))
    assert e.value.status_code == 403


def test_smoke_order_refuses_non_universe_ticker(monkeypatch):
    import asyncio
    import app.routes_trade as rt
    from types import SimpleNamespace
    from fastapi import HTTPException
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setattr(rt, "kis_is_mock", lambda: True)
    monkeypatch.setattr(rt, "is_ai_covered_ticker", lambda t: False)
    monkeypatch.setattr(rt, "broker", object())
    with __import__("pytest").raises(HTTPException) as e:
        asyncio.run(rt.smoke_order(rt.SmokeOrderRequest(ticker="247540", side="buy"), SimpleNamespace(id=1)))
    assert e.value.status_code == 400


def test_retry_only_transient_kis_failures():
    import requests
    from app.kis_errors import KISOrderError, is_retryable
    assert is_retryable(KISOrderError("토큰 발급 1분당 1회", "EGW00133"))
    assert is_retryable(KISOrderError("초당 거래건수 초과", "EGW00201"))
    assert is_retryable(requests.exceptions.Timeout("timeout"))
    assert is_retryable(ConnectionError("reset"))
    assert not is_retryable(KISOrderError("모의투자 영업일이 아닙니다.", "40100000"))
    assert not is_retryable(KISOrderError("주문가능금액 부족", "APBK0952"))
    assert not is_retryable(Exception("KIS broker is not configured"))


def test_trade_lock_is_exclusive_per_user_in_process():
    from app.trade_lock import acquire, release
    assert acquire(777) is True
    assert acquire(777) is False  # 같은 유저는 두 번 잡을 수 없다
    release(777)
    assert acquire(777) is True
    release(777)
