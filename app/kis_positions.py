# app/kis_positions.py
"""trading_loop/run_once는 매 사이클 `Portfolio()`를 새로 만들고 체결 후에도 `positions`를 갱신하지
않아서 '보유 중인지' 판단이 영원히 거짓이었다(2026-10-02 통합 감사) -- 같은 종목 반복 매수가 무한정
가능했고, 매도(청산) 조건은 거의 발화하지 않았다. 이 모듈은 KIS 실제 잔고(output1)를 그대로 읽어
Portfolio.positions를 채운다 -- 메모리에 따로 추적하는 상태가 아니라 매 사이클 KIS 쪽 진짜 보유
현황을 그대로 가져오므로, 프로세스 재시작/수동매매/부분체결 등으로 상태가 깨질 일이 없다."""
from backtest.engine.risk import Position


def parse_balance_positions(balance: dict) -> dict[str, Position]:
    """KIS fetch_balance() 응답의 output1(보유종목 리스트)을 Portfolio.positions 형태로 변환.
    보유수량 0(전량 매도된 과거 종목이 계속 잡히는 경우)인 행은 제외한다."""
    positions: dict[str, Position] = {}
    for row in (balance or {}).get("output1") or []:
        qty = int(row.get("hldg_qty") or 0)
        ticker = row.get("pdno")
        if qty <= 0 or not ticker:
            continue
        entry_price = int(float(row.get("pchs_avg_pric") or 0))
        positions[ticker] = Position(symbol=ticker, qty=qty, entry_price=entry_price)
    return positions


def apply_fill(portfolio, ticker: str, side: str, qty: float, price: float) -> None:
    """체결 직후 portfolio.positions를 그 자리에서 갱신한다 -- 다음 KIS 잔고 조회(다음 사이클) 전까지,
    같은 사이클의 뒤 종목들이 참조하는 risk_mgr.check_pretrade/전략의 current_qty 판단에 즉시 반영되도록."""
    pos = portfolio.ensure_pos(ticker)
    if side == "BUY":
        new_qty = pos.qty + qty
        pos.entry_price = price if pos.qty <= 0 else int((pos.qty * pos.entry_price + qty * price) / new_qty)
        pos.qty = new_qty
    else:
        pos.qty = max(pos.qty - qty, 0)
        if pos.qty == 0:
            pos.entry_price = 0


def load_live_positions(broker) -> dict[str, Position]:
    """실패하면 예외를 그대로 올린다 -- 포지션을 모르는 채로 '빈 상태'로 조용히 넘어가면 이 모듈이
    고치려는 버그(포지션 추적 없음)가 그대로 재현된다. 호출부는 이 사이클을 스킵해야 한다."""
    if not broker:
        raise RuntimeError("KIS broker is not configured")
    balance = broker.fetch_balance()
    return parse_balance_positions(balance)
