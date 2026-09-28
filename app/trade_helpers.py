"""자동매매 라우트가 쓰는 순수 헬퍼 (KIS/mojito 의존 없이 테스트하려고 분리)."""


def trade_log_to_dict(log) -> dict:
    """/trade/history 응답 1건. status(FILLED/FAILED)를 포함해 실패한 주문이 체결로 보이지 않게 한다."""
    return {
        "ticker": log.ticker,
        "side": log.side,
        "qty": log.qty,
        "price": log.price,
        "amount": log.amount,
        "ai_signal": log.ai_signal,
        "ai_confidence": log.ai_confidence,
        "strategy_id": log.strategy_id,
        "status": log.status or "FILLED",   # 컬럼 도입 전 행은 체결로 간주
        "created_at": log.created_at.isoformat(),
    }


def no_auto_basket_message(excluded_manual_tickers: list[str]) -> str:
    """직접매매 종목만 있을 때의 400 detail. 객체가 아니라 문자열이어야 프론트가 그대로 보여줄 수 있다."""
    tickers = ", ".join(excluded_manual_tickers)
    return f"자동매매 가능한 바구니 종목이 없습니다. 직접매매 종목은 자동매매에서 제외됩니다. (제외: {tickers})"
