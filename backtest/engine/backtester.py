from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import math
import pandas as pd
import numpy as np

from backtest.engine.execution import ExecutionModel, CostModelCfg
from backtest.engine.risk import (
    RiskManager,
    RiskLimits,
    Portfolio,
    Order,
    OrderType,
    Side,
    Fill,
)


# 원화 주가 sanity 범위. 한국 주식은 수백 원짜리부터 수백만 원(삼성바이오로직스 등)까지 정상 시세이므로
# 상·하한을 "단위 오류를 잡는 정도"로만 둔다. (이전에는 종가 10만원 이상이면 루프를 조용히 break,
# 100만원 초과는 예외, 포지션 100주 초과도 예외였다.)
MAX_KRW_PRICE = 1e9
# 포지션 평가액이 시작 자본의 이 배수를 넘으면 엔진 버그로 본다 (주식 수 상한 대신 금액 기준).
MAX_POSITION_NOTIONAL_MULT = 100.0


def _valid_price(px) -> bool:
    try:
        px = float(px)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(px) and 0 < px < MAX_KRW_PRICE)


class Backtester:
    def __init__(
        self,
        base_df: pd.DataFrame,
        symbol: str,
        exec_model: ExecutionModel,
        cost_cfg: CostModelCfg,
        risk_mng: RiskManager,
        strategy: Any,
    ):
        self.base_df = base_df
        self.symbol = symbol
        self.exec_model = exec_model
        self.cost_cfg = cost_cfg
        self.risk_mng = risk_mng
        self.strategy = strategy

        self.portfolio = Portfolio()
        self.fills: List[Fill] = []
        self.trades: List[Dict[str, Any]] = []
        self._start_equity = float(self.portfolio.equity)

    def _mid_and_spreadbps(self, row: pd.Series) -> Tuple[float, float]:
        close_val = row["close"]
        close = float(close_val)

        if not _valid_price(close):
            raise ValueError(f"Invalid close value detected: {close}")

        spread_bps = float(row.get("spread_bps", 2.0))
        return close, spread_bps

    def _mark_to_market(self, px: float):
        px = float(px)
        eq = self.portfolio.cash

        for pos in self.portfolio.positions.values():
            eq += pos.qty * px
            # 주식 수가 아니라 금액으로 sanity check (저가주는 정상적으로 수백~수천 주가 된다).
            if abs(pos.qty) * px > MAX_POSITION_NOTIONAL_MULT * max(self._start_equity, 1):
                raise ValueError(f"Position notional too big: qty={pos.qty} px={px}")

        if eq < 0:
            eq = 0.0
            raise ValueError("Stopped due to negative equity")

        self.portfolio.equity = int(eq)

    def _apply_fill(self, fill: Fill):
        pos = self.portfolio.ensure_pos(fill.symbol)
        signed_qty = float(fill.qty if fill.side == Side.BUY else -fill.qty)
        notional = fill.qty * fill.price

        if (pos.qty == 0) or (np.sign(pos.qty) == np.sign(signed_qty)):
            new_qty = pos.qty + signed_qty
            numerator = pos.entry_price * abs(pos.qty) + fill.price * abs(signed_qty)
            pos.entry_price = numerator / max(abs(new_qty), 1e-9)
            pos.qty = new_qty
        else:
            new_qty = pos.qty + signed_qty
            pos.qty = new_qty
            if new_qty == 0:
                pos.entry_price = 0.0

        self.portfolio.cash += (
            -notional if fill.side == Side.BUY else notional
        ) - fill.fee

        self.trades.append(
            {
                "ts": fill.ts,
                "symbol": fill.symbol,
                "side": fill.side.value,
                "qty": fill.qty,
                "price": fill.price,
                "fee": fill.fee,
                "is_maker": fill.is_maker,
            }
        )

    def run(self) -> pd.DataFrame:
        self._start_equity = float(self.portfolio.equity)
        eq_curve = []
        eq_curve.append({"ts": self.base_df.index[0], "equity": self.portfolio.equity})

        for i, (ts, row) in enumerate(self.base_df.iterrows(), start=1):
            # self._refresh_sr_if_needed(i, ts)

            # 잘못된 시세는 조용히 루프를 끊지 않고 명시적으로 실패시킨다 (끊으면 곡선이 1점이 되어
            # 성과가 전부 0으로 보이는데, 실제로는 데이터/가격대 문제였다).
            if not _valid_price(row["close"]):
                raise ValueError(f"Invalid close at {ts}: {row['close']}")

            orders: List[Order] = self.strategy.generate_orders(row, self.portfolio)
            mid, spread_bps = self._mid_and_spreadbps(row)

            if (
                isinstance(mid, (pd.Timestamp, np.datetime64))
                or (not np.isfinite(mid))
                or (not _valid_price(mid))
            ):
                raise ValueError(f"mid invalid at {ts}: {mid}")

            for o in orders:
                notional = abs(o.qty) * (o.limit_price or mid)
                if not self.risk_mng.check_pretrade(
                    ts, self.portfolio, o.symbol, notional
                ):
                    continue

                fills = self.exec_model.simulate(
                    ts,
                    o.symbol,
                    o.side,
                    o.qty,
                    o.order_type,
                    mid,
                    spread_bps,
                    o.limit_price,
                )

                for fill in fills:
                    self._apply_fill(fill)

            self._mark_to_market(px=mid)
            self.risk_mng.check_intraday_dd(ts, self.portfolio.equity)
            eq_curve.append({"ts": ts, "equity": self.portfolio.equity})

        curve = pd.DataFrame(eq_curve).set_index("ts")
        self.curve = curve
        return curve

    def tca_report(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()

        t = pd.DataFrame(self.trades)
        t["notional"] = t["qty"] * t["price"]
        t["fee_bps"] = (t["fee"] / t["notional"]).replace(
            [np.inf, -np.inf], np.nan
        ) * 10_000
        by_side = t.groupby("side").agg(
            {"notional": "sum", "fee": "sum", "fee_bps": "mean"}
        )
        by_maker = t.groupby("is_maker").agg({"notional": "sum", "fee": "sum"})
        overall = t.agg({"notional": "sum", "fee": "sum"})
        return pd.concat(
            {"overall": overall, "by_side": by_side, "by_maker": by_maker}, axis=0
        )

    def trades_report(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades).set_index("ts")


def performance_from_curve(c_eq: pd.Series, timeframe: str = "1d") -> Dict[str, float]:
    c = c_eq.ffill()
    ret = c.pct_change().dropna()

    if ret.empty:
        return {}

    ann_map = {
        "1d": 252,
        "5m": 252 * 78,
        "15m": 252 * 26,
        "1m": 252 * 390,
    }
    ann = ann_map.get(timeframe, 252)  # 분 단위 연환산
    mean = float(ret.mean() * ann)
    vol = float(ret.std(ddof=1) * math.sqrt(ann))
    sharpe = mean / (vol + 1e-12)
    cum = float(c.iloc[-1] / c.iloc[0] - 1)
    roll_max = c.cummax()
    dd = c / roll_max - 1
    mdd = float(dd.min())
    calmar = mean / (abs(mdd) + 1e-12)

    return {
        "Cumulative": cum,
        "Sharpe": sharpe,
        "Vol_ann": vol,
        "MaxDD": mdd,
        "Calmar": calmar,
    }
