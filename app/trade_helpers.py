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
        "order_no": log.order_no,
        "created_at": log.created_at.isoformat(),
    }


def no_auto_basket_message(excluded_manual_tickers: list[str], excluded_unsupported_tickers: list[str] | None = None) -> str:
    """자동매매 가능한 바구니 종목이 하나도 없을 때의 400 detail. 객체가 아니라 문자열이어야 프론트가
    그대로 보여줄 수 있다."""
    reasons = []
    if excluded_manual_tickers:
        reasons.append(f"직접매매 종목이라 제외: {', '.join(excluded_manual_tickers)}")
    if excluded_unsupported_tickers:
        reasons.append(f"AI가 분석하지 않는 종목이라 제외: {', '.join(excluded_unsupported_tickers)}")
    detail = " / ".join(reasons) if reasons else "바구니가 비어있습니다"
    return f"자동매매 가능한 바구니 종목이 없습니다. ({detail})"
