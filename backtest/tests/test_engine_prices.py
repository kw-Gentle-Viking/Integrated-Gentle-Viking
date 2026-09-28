"""백테스트 엔진의 주가 관련 가정 검증.

이전 엔진은 (1) 종가가 10만원 이상이면 루프를 조용히 break해서 성과 0으로 반환했고
(2) 종가가 100만원을 넘으면 예외, (3) 포지션이 100주를 넘으면(저가주) 예외를 던졌다.
한국 주식은 수백 원짜리부터 수백만 원(삼성바이오로직스 등)까지 있으므로 세 경우 모두 정상 시세다.
"""
import numpy as np
import pandas as pd
import pytest

from backtest.engine.backtester import Backtester, performance_from_curve
from backtest.engine.execution import CostModelCfg, ExecutionModel
from backtest.engine.risk import Order, OrderType, RiskLimits, RiskManager, Side


class BuyOnceThenHold:
    """첫 봉에 자본의 10%를 매수하고 이후 보유만 하는 결정적 전략(가격 범위 검증 전용)."""

    def __init__(self, symbol):
        self.symbol = symbol
        self._done = False

    def generate_orders(self, row, portfolio):
        if self._done:
            return []
        self._done = True
        qty = int(portfolio.equity * 0.1 // float(row["close"]))
        if qty <= 0:
            return []
        return [Order(ts=row.name, symbol=self.symbol, side=Side.BUY, qty=qty, order_type=OrderType.MARKET)]


def _run(start_price, n=30, cap=10_000_000, closes=None):
    if closes is None:
        rng = np.random.default_rng(0)
        closes = start_price * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.bdate_range("2024-01-02", periods=len(closes))
    df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1e6}, index=idx)
    cfg = CostModelCfg()
    bt = Backtester(df, "X", ExecutionModel(cfg), cfg, RiskManager(RiskLimits()), BuyOnceThenHold("X"))
    bt.portfolio.cash = cap
    bt.portfolio.equity = cap
    return bt, bt.run()


@pytest.mark.parametrize("price", [1_000, 60_000, 300_000, 2_000_000])
def test_runs_over_the_full_series_for_any_realistic_krw_price(price):
    # 자본은 최소 1주를 살 수 있게 주가에 비례해 잡는다 (200만원짜리에 1천만원의 10%로는 0주).
    bt, curve = _run(price, n=30, cap=max(10_000_000, int(price * 50)))
    assert len(curve) == 31, "곡선이 입력 봉 수(+초기점)만큼 이어져야 함 -- 가격대 때문에 잘리면 안 된다"
    assert len(bt.trades) == 1
    perf = performance_from_curve(curve["equity"], "1d")
    assert perf, "성과 지표가 비어 있으면(곡선 1점) 조용한 실패"


def test_non_positive_or_non_finite_price_raises_instead_of_silently_truncating():
    closes = np.array([50_000.0, 50_500.0, 0.0, 51_000.0])
    with pytest.raises(ValueError, match="close"):
        _run(None, closes=closes)
    closes = np.array([50_000.0, np.nan, 51_000.0])
    with pytest.raises(ValueError, match="close"):
        _run(None, closes=closes)
