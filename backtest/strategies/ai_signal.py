from typing import Dict, List, Optional

import pandas as pd

from backtest.engine.risk import Order, OrderType, Portfolio, Side


class AISignalStrategy:
    """AI 서버가 낸 BUY/HOLD/SELL 신호 시계열을 재생하는 전략 (일봉 기준).

    - signals: {'YYYY-MM-DD': 'BUY'|'HOLD'|'SELL'}  (예: app.ai_history.load_signals 결과)
    - BUY  & 무포지션 -> position_size 비율만큼 정수 주 매수
    - SELL & 보유 중  -> 전량 매도 (공매도 없음, 롱온리)
    - HOLD / 신호 없음 / 이미 보유 중 BUY / 무포지션 SELL -> 주문 없음
    - lag: 신호를 낸 날로부터 몇 봉 뒤에 체결할지 (기본 0 = 신호 계산에 쓴 종가와 같은 봉 종가에 체결.
      엔진이 봉의 종가에 체결하므로, 종가 이후에 나온 신호를 쓰려면 lag=1 로 둔다)
    """

    def __init__(self, symbol: str, signals: Optional[Dict[str, str]] = None, lag: int = 0,
                 position_size: float = 0.1):
        self.symbol = symbol
        self.signals = {str(k)[:10]: str(v).upper() for k, v in (signals or {}).items()}
        self.lag = int(lag)
        self.position_size = float(position_size)
        self._dates: List[str] = []

    def _signal_for_current_bar(self) -> Optional[str]:
        i = len(self._dates) - 1 - self.lag
        if i < 0:
            return None
        return self.signals.get(self._dates[i])

    def generate_orders(self, row: pd.Series, portfolio: Portfolio) -> List[Order]:
        self._dates.append(pd.Timestamp(row.name).strftime("%Y-%m-%d"))
        sig = self._signal_for_current_bar()
        if sig not in ("BUY", "SELL"):
            return []

        close = float(row["close"])
        pos = portfolio.positions.get(self.symbol)
        qty_held = pos.qty if pos else 0

        if sig == "BUY" and qty_held <= 0:
            qty = int(portfolio.equity * self.position_size // close)
            if qty <= 0:
                return []
            return [Order(ts=row.name, symbol=self.symbol, side=Side.BUY, qty=qty, order_type=OrderType.MARKET)]
        if sig == "SELL" and qty_held > 0:
            return [Order(ts=row.name, symbol=self.symbol, side=Side.SELL, qty=qty_held, order_type=OrderType.MARKET)]
        return []
