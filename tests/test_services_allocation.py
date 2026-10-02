"""min_confidence=0으로 설정하면 confidence 0.0짜리 BUY 신호도 배분 대상에 들어오는데, 전부 0이면
total_conf==0으로 나누기가 일어나 ZeroDivisionError가 trading_loop 전체(asyncio task)를 조용히
죽였다(2026-10-02 통합 감사)."""
from app.services_allocation import allocate_portfolio


def test_allocate_portfolio_returns_empty_instead_of_dividing_by_zero_confidence():
    preds = [{"ticker": "005930", "signal": "BUY", "confidence": 0.0}]
    result = allocate_portfolio(preds, persona_id=3, total_capital=10_000_000, min_confidence=0.0)
    assert result == []


def test_allocate_portfolio_still_allocates_normally_with_real_confidence():
    preds = [{"ticker": "005930", "signal": "BUY", "confidence": 0.8},
             {"ticker": "000660", "signal": "BUY", "confidence": 0.7}]
    result = allocate_portfolio(preds, persona_id=3, total_capital=10_000_000)
    assert len(result) == 2 and all(r["amount"] > 0 for r in result)
