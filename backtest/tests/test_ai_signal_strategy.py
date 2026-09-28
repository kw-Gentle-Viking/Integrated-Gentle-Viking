"""AI 신호(BUY/HOLD/SELL 시계열)를 백테스트 엔진에 재생하는 전략."""
import numpy as np
import pandas as pd
import pytest

from backtest.engine.backtester import Backtester
from backtest.engine.execution import CostModelCfg, ExecutionModel
from backtest.engine.risk import RiskLimits, RiskManager


def _run(signals, n=20, lag=0, cap=10_000_000, position_size=0.1):
    from backtest.strategies.ai_signal import AISignalStrategy
    idx = pd.bdate_range("2026-01-05", periods=n)
    close = 50_000 * (1 + 0.001 * np.arange(n))
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1e6}, index=idx)
    cfg = CostModelCfg()
    strat = AISignalStrategy("X", signals=signals, lag=lag, position_size=position_size)
    bt = Backtester(df, "X", ExecutionModel(cfg), cfg, RiskManager(RiskLimits()), strat)
    bt.portfolio.cash = cap
    bt.portfolio.equity = cap
    curve = bt.run()
    return bt, curve, idx


def test_buy_then_sell_signals_produce_one_round_trip_with_integer_shares():
    idx = pd.bdate_range("2026-01-05", periods=20)
    signals = {idx[3].strftime("%Y-%m-%d"): "BUY", idx[10].strftime("%Y-%m-%d"): "SELL"}
    bt, curve, idx = _run(signals)
    sides = [(t["side"], pd.Timestamp(t["ts"])) for t in bt.trades]
    assert sides == [("BUY", idx[3]), ("SELL", idx[10])]
    assert all(float(t["qty"]).is_integer() for t in bt.trades), "한국 주식은 정수 주 단위"
    assert bt.trades[0]["qty"] == bt.trades[1]["qty"]


def test_hold_and_missing_signals_do_nothing():
    idx = pd.bdate_range("2026-01-05", periods=20)
    bt, _, _ = _run({idx[2].strftime("%Y-%m-%d"): "HOLD"})
    assert bt.trades == []


def test_buy_signal_while_already_long_does_not_add_and_sell_while_flat_does_nothing():
    idx = pd.bdate_range("2026-01-05", periods=20)
    d = lambda i: idx[i].strftime("%Y-%m-%d")   # noqa: E731
    bt, _, _ = _run({d(1): "SELL", d(3): "BUY", d(4): "BUY", d(5): "BUY"})
    assert [t["side"] for t in bt.trades] == ["BUY"], "SELL(무포지션)과 중복 BUY 는 주문이 없어야 함(공매도/물타기 없음)"


def test_lag_delays_execution_by_that_many_bars():
    idx = pd.bdate_range("2026-01-05", periods=20)
    bt, _, _ = _run({idx[3].strftime("%Y-%m-%d"): "BUY"}, lag=1)
    assert [pd.Timestamp(t["ts"]) for t in bt.trades] == [idx[4]]


def test_service_loads_ai_signal_strategy_with_explicit_signals():
    pytest.importorskip("sqlalchemy")
    from backtest.service import BacktestService
    s = BacktestService()._load_strategy("ai_signal", "005930", {"signals": {"2026-01-07": "BUY"}})
    assert s.__class__.__name__ == "AISignalStrategy"
