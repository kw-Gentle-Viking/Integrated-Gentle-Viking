from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime
import asyncio

from backtest.schemas import BacktestRequest, BacktestResponse, BacktestListResponse
from backtest.db import get_market_data, save_to_clickhouse, get_from_clickhouse
from backtest.engine.backtester import Backtester, performance_from_curve
from backtest.engine.execution import ExecutionModel, CostModelCfg
from backtest.engine.risk import RiskManager, RiskLimits
from backtest.strategies.ma_cross import MACrossStrategy
from backtest.strategies.aggressive import AggressiveStrategy
from backtest.strategies.balanced import BalancedStrategy
from backtest.strategies.conservative import ConservativeStrategy
from backtest.strategies.ultra_safe import UltraSafeStrategy
from backtest.strategies.rsi_reversal import RSIReversalStrategy
from backtest.strategies.ai_signal import AISignalStrategy


# 인메모리 작업 상태 (나중에 Redis로 교체 가능)
_job_status: Dict[str, Dict] = {}
_job_results: Dict[str, BacktestResponse] = {}


class BacktestService:

    async def run(self, request: BacktestRequest) -> BacktestResponse:
        """백테스트 실행"""

        # 1. market-db에서 데이터 로딩
        df = await get_market_data(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            timeframe=request.timeframe,
        )

        if df.empty:
            raise ValueError(f"No data found for {request.symbol}")

        # 2. 엔진 설정
        cost_cfg = CostModelCfg()
        exec_model = ExecutionModel(cost_cfg)
        risk_limits = RiskLimits(
            # max_leverage=request.max_leverage, 주식시장에서는 안 씀
            intraday_dd_limit=request.max_drawdown
        )
        risk_mng = RiskManager(risk_limits)

        # 3. 전략 로딩
        strategy = self._load_strategy(
            request.strategy, request.symbol, self._resolve_strategy_params(request)
        )

        # 4. 백테스터 생성 & 실행
        bt = Backtester(
            base_df=df,
            symbol=request.symbol,
            exec_model=exec_model,
            cost_cfg=cost_cfg,
            risk_mng=risk_mng,
            strategy=strategy,
        )

        # 초기 자본 설정
        bt.portfolio.cash = request.initial_capital
        bt.portfolio.equity = request.initial_capital

        # 실행
        curve = bt.run()

        # 5. 성과 계산
        perf = performance_from_curve(curve["equity"], request.timeframe)

        # 6. 응답 생성
        response = BacktestResponse(
            cumulative=perf.get("Cumulative", 0.0),
            sharpe=perf.get("Sharpe", 0.0),
            volatility=perf.get("Vol_ann", 0.0),
            max_drawdown=perf.get("MaxDD", 0.0),
            calmar=perf.get("Calmar", 0.0),
            total_trades=len(bt.trades),
            symbol=request.symbol,
            strategy=request.strategy,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            final_equity=bt.portfolio.equity,
            equity_curve=curve.reset_index().to_dict(orient="records"),
            trades=bt.trades,
        )

        # 7. ClickHouse에 결과 저장
        await save_to_clickhouse(response)

        return response

    async def run_and_save(self, job_id: str, request: BacktestRequest):
        """백그라운드에서 백테스트 실행 후 저장"""
        _job_status[job_id] = {
            "status": "running",
            "started_at": datetime.now().isoformat(),
        }

        try:
            result = await self.run(request)
            _job_results[job_id] = result
            _job_status[job_id] = {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
            }
        except Exception as e:
            _job_status[job_id] = {"status": "failed", "error": str(e)}

    async def get_status(self, job_id: str) -> Optional[Dict]:
        """작업 상태 조회"""
        return _job_status.get(job_id)

    async def get_result(self, job_id: str) -> Optional[BacktestResponse]:
        """작업 결과 조회"""
        return _job_results.get(job_id)

    async def get_history(
        self, symbol: Optional[str], strategy: Optional[str], limit: int, offset: int
    ) -> BacktestListResponse:
        """백테스트 기록 조회"""
        results, total = await get_from_clickhouse(symbol, strategy, limit, offset)
        return BacktestListResponse(results=results, total=total)

    def _resolve_strategy_params(self, request: BacktestRequest, session_factory=None) -> Optional[Dict]:
        """ai_signal 전략은 요청에 signals 가 없으면 저장된 AI 예측 이력(ai_prediction_history)에서 불러온다."""
        params = request.strategy_params
        if request.strategy != "ai_signal" or (params or {}).get("signals"):
            return params
        from app.ai_history import load_signals

        own = session_factory is None   # 주입된 세션(테스트)은 호출자가 소유하므로 닫지 않는다
        if own:
            from app.db import SessionLocal as session_factory
        db = session_factory()
        try:
            signals = load_signals(db, request.symbol, request.start_date, request.end_date)
        finally:
            if own:
                db.close()
        if not signals:
            raise ValueError(
                f"AI 예측 이력이 없습니다: {request.symbol} {request.start_date}~{request.end_date} "
                "(strategy_params.signals 로 직접 넘기거나, AI 서버 push 후 다시 시도)"
            )
        return {**(params or {}), "signals": signals}

    def _load_strategy(
        self, strategy_name: str, symbol: str, params: Optional[Dict] = None
    ):
        """전략 클래스 로딩"""

        strategies = {
            "ma_cross": MACrossStrategy,
            "rsi_reversal": RSIReversalStrategy,
            "ai_signal": AISignalStrategy,
            # /backtest/strategies 가 광고하는 4종 -- 예전엔 여기 없어서 고르면 400 "Unknown strategy" 였다
            "aggressive": AggressiveStrategy,
            "balanced": BalancedStrategy,
            "conservative": ConservativeStrategy,
            "ultra_safe": UltraSafeStrategy,
        }

        strategy_cls = strategies.get(strategy_name)
        if not strategy_cls:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        if params:
            return strategy_cls(symbol, **params)
        return strategy_cls(symbol)
