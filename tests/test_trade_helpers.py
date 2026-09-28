"""프론트 연동에서 발견된 백엔드 쪽 결함 수정 테스트 (실패 주문 표시 / 전략 ID 대체 / 에러 detail 형식 / Google 콜백)."""
from datetime import datetime
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse


def _log(status="FILLED"):
    return SimpleNamespace(ticker="005930", side="BUY", qty=3, price=70000.0, amount=210000.0, ai_signal="BUY",
                           ai_confidence=0.7, strategy_id="ConservativeStrategy",
                           created_at=datetime(2026, 9, 28, 10, 0), status=status)


# 1) 실패한 주문이 체결처럼 보이던 문제: /trade/history 응답에 status 를 포함한다
def test_trade_log_dict_includes_status_so_failed_orders_are_distinguishable():
    from app.trade_helpers import trade_log_to_dict
    assert trade_log_to_dict(_log("FAILED"))["status"] == "FAILED"
    assert trade_log_to_dict(_log("FILLED"))["status"] == "FILLED"
    d = trade_log_to_dict(_log())
    assert d["created_at"] == "2026-09-28T10:00:00" and d["qty"] == 3 and d["ticker"] == "005930"


def test_trade_log_dict_defaults_missing_status_to_filled_for_legacy_rows():
    from app.trade_helpers import trade_log_to_dict
    log = _log(); log.status = None
    assert trade_log_to_dict(log)["status"] == "FILLED"


# 2) 프론트가 보내는 strategy_id(rsi_reversal 등)가 조용히 conservative 로 바뀌던 문제: 대체 사실을 드러낸다
def test_resolve_strategy_id_reports_fallback_for_legacy_or_unknown_ids():
    from app.strategy_factory import resolve_strategy_id
    assert resolve_strategy_id("balanced") == {"requested": "balanced", "resolved": "balanced", "fallback": False}
    assert resolve_strategy_id("rsi_reversal") == {"requested": "rsi_reversal", "resolved": "conservative", "fallback": True}
    assert resolve_strategy_id("nope")["fallback"] is True


# 3) 에러 detail 이 객체/배열이라 프론트에서 [object Object] 로 보이던 문제: 문자열 메시지로 만든다
def test_no_auto_basket_message_is_a_plain_string_listing_excluded_tickers():
    from app.trade_helpers import no_auto_basket_message
    msg = no_auto_basket_message(["005930", "000660"])
    assert isinstance(msg, str) and "005930" in msg and "000660" in msg and "직접매매" in msg


# 4) Google 로그인: 프론트는 ?access_token=&refresh_token= 리다이렉트를 기대한다
def test_frontend_redirect_url_carries_tokens_as_query_params():
    from app.auth_redirect import frontend_redirect_url
    url = frontend_redirect_url("https://front.example/auth/google/callback", "AT", "RT")
    q = parse_qs(urlparse(url).query)
    assert q == {"access_token": ["AT"], "refresh_token": ["RT"]}
    assert url.startswith("https://front.example/auth/google/callback?")


def test_frontend_redirect_url_preserves_existing_query_and_encodes_values():
    from app.auth_redirect import frontend_redirect_url
    url = frontend_redirect_url("https://f/cb?x=1", "a b", "c&d")
    q = parse_qs(urlparse(url).query)
    assert q["x"] == ["1"] and q["access_token"] == ["a b"] and q["refresh_token"] == ["c&d"]
