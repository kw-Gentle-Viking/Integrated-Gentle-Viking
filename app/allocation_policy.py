"""배분 정책 선택 (ALLOCATION_POLICY 환경변수).

a / cap_cash     : 종목당 max_weight 엄격 준수, 남는 돈은 현금. 리스크 한도 1회 주문 30% 유지.
b / redistribute : 잘린 몫을 남은 종목에 재분배. 그 대신 1회 주문 리스크 한도를 투자 가능 금액 전체(100%)로 완화.
"""
import os

POLICIES = ("cap_cash", "redistribute")
_ALIASES = {"a": "cap_cash", "b": "redistribute"}


def allocation_policy() -> str:
    raw = os.getenv("ALLOCATION_POLICY", "cap_cash").strip().lower()
    raw = _ALIASES.get(raw, raw)
    return raw if raw in POLICIES else "cap_cash"


def per_order_notional_cap(policy: str) -> float:
    return 1.0 if policy == "redistribute" else 0.3
