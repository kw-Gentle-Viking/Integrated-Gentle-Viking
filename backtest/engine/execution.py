from __future__ import annotations
from typing import List, Optional
import dataclasses as dc
import pandas as pd
import numpy as np

from backtest.engine.risk import Side, OrderType, Fill

MAX_KRW_PRICE = 1e9  # backtester.MAX_KRW_PRICE 와 같은 값 (순환 import 방지용 복제)


@dc.dataclass
class CostModelCfg:
    # taker_fee_bps: float = 1.47 # 매수 / 매도 수수료 0.0147%
    # maker_fee_bps: float = 1.47
    commission_bps: float = 1.47  # 어차피 같으니간 통합
    sell_tax_bps: float = 18.0  # 증권거래세

    slip_bps_mkt: float = 2.0  # slipage
    slip_bps_lmt: float = 0.5
    maker_fill_prob: float = 0.6


class ExecutionModel:
    def __init__(self, cost_cfg: CostModelCfg):
        self.cost_cfg = cost_cfg
        self._next_id = 1

    def _new_id(self) -> int:
        order_id = self._next_id
        self._next_id += 1
        return order_id

    def _bps(self, px: float, bps: float) -> float:
        return px * (bps / 10_000.0)

    def simulate(
        self,
        ts: pd.Timestamp,
        symbol: str,
        side: Side,
        qty: int,
        order_type: OrderType,
        mid: float,
        spread_bps: float = 1.0,
        limit_price: Optional[float] = None,
    ) -> List[Fill]:
        if qty == 0:
            return []

        if (
            isinstance(mid, (pd.Timestamp, np.datetime64))
            or (not np.isfinite(mid))
            or (not (0 < float(mid) < MAX_KRW_PRICE))
        ):
            raise ValueError(f"simulate(): mid invalid @ {ts} -> {mid}")

        mid = float(mid)
        fills: List[Fill] = []
        half_spread = self._bps(mid, spread_bps / 2)

        if order_type == OrderType.MARKET:
            impact = self._bps(mid, self.cost_cfg.slip_bps_mkt)
            px = mid + (half_spread + impact) * (1 if side == Side.BUY else -1)

            fee = int(abs(qty) * px * (self.cost_cfg.commission_bps / 10_000.0))
            if side == Side.SELL:
                fee += abs(qty) * px * (self.cost_cfg.sell_tax_bps / 10_000.0)

            fills.append(Fill(ts, self._new_id(), symbol, side, qty, px, fee, False))

            if not np.isfinite(px) or not (0 < float(px) < MAX_KRW_PRICE):
                raise ValueError(f"simulate(): px invalid @ {ts} -> {px}")

            return fills

        # Limit order
        assert limit_price is not None
        crossed = (side == Side.BUY and limit_price >= mid - half_spread) or (
            side == Side.SELL and limit_price <= mid + half_spread
        )

        if crossed and (np.random.rand() < self.cost_cfg.maker_fill_prob):
            slip = self._bps(mid, self.cost_cfg.slip_bps_lmt)
            px = limit_price + (slip if side == Side.BUY else -slip)
            fee = abs(qty) * px * (self.cost_cfg.commission_bps / 10_000.0)

            if side == Side.SELL:
                fee += abs(qty) * px * (self.cost_cfg.sell_tax_bps / 10_000.0)

            fills.append(Fill(ts, self._new_id(), symbol, side, qty, px, fee, True))

            if not np.isfinite(px) or not (0 < float(px) < MAX_KRW_PRICE):
                raise ValueError(f"simulate(): px invalid @ {ts} -> {px}")

        return fills
