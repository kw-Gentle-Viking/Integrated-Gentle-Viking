# app/strategy_factory.py
from backtest.strategies.ultra_safe import UltraSafeStrategy
from backtest.strategies.conservative import ConservativeStrategy
from backtest.strategies.balanced import BalancedStrategy
from backtest.strategies.aggressive import AggressiveStrategy

STRATEGY_CONFIG = {
    "ultra_safe": {
        "class": UltraSafeStrategy,
        "timeframe": "5m",
        "warmup": lambda p: 240,  # 1시간봉 20개 = 5분봉 240개
    },
    "conservative": {
        "class": ConservativeStrategy,
        "timeframe": "5m",
        "warmup": lambda p: 63,   # 15분봉 21개 = 5분봉 63개
    },
    "balanced": {
        "class": BalancedStrategy,
        "timeframe": "5m",
        "warmup": lambda p: 81,   # 15분봉 27개 = 5분봉 81개
    },
    "aggressive": {
        "class": AggressiveStrategy,
        "timeframe": "5m",
        "warmup": lambda p: 21,   # 5분봉 21개만
    },
}


def _get_strategy_config(strategy_id: str):
    return STRATEGY_CONFIG.get(strategy_id) or STRATEGY_CONFIG["conservative"]


def resolve_strategy_id(strategy_id: str) -> dict:
    """요청된 전략 ID 가 실제로 어떤 전략으로 실행되는지. 미지원 ID(프론트의 rsi_reversal 등 구 이름 포함)는
    conservative 로 대체되는데, 이 사실을 응답/로그에 드러내려고 대체 여부를 함께 돌려준다."""
    known = strategy_id in STRATEGY_CONFIG
    return {"requested": strategy_id, "resolved": strategy_id if known else "conservative", "fallback": not known}


def create_strategy(symbol: str, strategy_id: str, params: dict = None):
    cfg = _get_strategy_config(strategy_id)
    return cfg["class"](symbol=symbol)


def get_warmup_count(strategy_id: str, params: dict = None) -> int:
    cfg = _get_strategy_config(strategy_id)
    return cfg["warmup"](params or {})


def get_timeframe(strategy_id: str) -> str:
    cfg = _get_strategy_config(strategy_id)
    return cfg["timeframe"]

## multiframe 전략으로 진화 시킬 것 