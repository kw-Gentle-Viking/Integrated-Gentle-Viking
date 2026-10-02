"""ONCE 콜백(/ai/callback)이 보내는 결과에는 interpretability 키가 없다(serving/api_server.py 자체 docstring이
명시) -- build_user_prompt가 이를 {}로 안전하게 처리하지 못하면 매 ONCE 요청이 500으로 터진다."""
from app.services_report import build_user_prompt


def _result(**overrides):
    base = {"ticker": "005930", "pred_str": "매수", "prob_buy": 0.6, "prob_hold": 0.3, "prob_sell": 0.1,
            "trade_datetime": "2026-10-02T15:30:00"}
    base.update(overrides)
    return base


def test_build_user_prompt_handles_missing_interpretability_key():
    prompt = build_user_prompt(_result())  # interpretability 키 자체가 없음 (ONCE 콜백의 실제 payload shape)
    assert "005930" in prompt


def test_build_user_prompt_handles_interpretability_none():
    prompt = build_user_prompt(_result(interpretability=None))  # 명시적으로 None이 저장된 경우도 안전해야 함
    assert "005930" in prompt


def test_build_user_prompt_handles_empty_interpretability_dict():
    prompt = build_user_prompt(_result(interpretability={}))
    assert "005930" in prompt
