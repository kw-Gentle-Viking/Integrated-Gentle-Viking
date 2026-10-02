"""/ai/agreement가 TFT 유니버스(코스피 200종목) 밖 종목도 tft_signal을 "HOLD"로 정규화해 "TFT는
관망 의견"이라는 그럴듯하지만 허위인 해석을 만들어내고 있었다(2026-10-02 통합 감사)."""
from types import SimpleNamespace

import app.routes_ai_webhook as webhook
from app.schemas import AgreementAnalysisRequest


def _payload(**overrides):
    base = dict(ticker="247540", recommendation_signal="BUY", tft_signal="", confidence=0.0,
                prob_buy=0.0, prob_hold=0.0, prob_sell=0.0)
    base.update(overrides)
    return AgreementAnalysisRequest(**base)


def test_agreement_reports_no_coverage_instead_of_fabricating_a_hold_opinion(monkeypatch):
    monkeypatch.setattr(webhook, "is_ai_covered_ticker", lambda t: t == "005930")
    out = webhook.analyze_model_agreement(_payload(ticker="247540"), current_user=SimpleNamespace(id=1))
    assert out["status"] == "no_tft_coverage"
    assert out["tft_signal"] is None
    assert "247540" in out["summary"]


def test_agreement_runs_normally_for_ai_covered_tickers(monkeypatch):
    monkeypatch.setattr(webhook, "is_ai_covered_ticker", lambda t: t == "005930")
    monkeypatch.setattr(webhook, "generate_agreement_analysis", lambda payload: {"status": "fallback"})
    out = webhook.analyze_model_agreement(_payload(ticker="005930"), current_user=SimpleNamespace(id=1))
    assert out == {"status": "fallback"}
