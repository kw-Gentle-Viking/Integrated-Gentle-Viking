import json
from datetime import datetime

import pandas as pd
import pytest

import serving.inference_pipeline as pipeline


def test_is_within_market_hours_true_during_session():
    assert pipeline.is_within_market_hours(datetime(2026, 9, 18, 10, 0)) is True  # Friday


def test_is_within_market_hours_false_before_open_after_close_and_weekend():
    assert pipeline.is_within_market_hours(datetime(2026, 9, 18, 8, 59)) is False
    assert pipeline.is_within_market_hours(datetime(2026, 9, 18, 15, 31)) is False
    assert pipeline.is_within_market_hours(datetime(2026, 9, 19, 10, 0)) is False  # Saturday


def test_load_active_tickers_missing_file_returns_empty(tmp_path):
    missing = tmp_path / "no_such_file.json"
    assert pipeline.load_active_tickers(str(missing)) == []


def test_load_active_tickers_reads_all_tickers_field(tmp_path):
    f = tmp_path / "active_tickers.json"
    f.write_text(json.dumps({"updated_at": "x", "users": {"u1": ["005930"]}, "all_tickers": ["005930", "000660"]}))
    assert pipeline.load_active_tickers(str(f)) == ["005930", "000660"]


def test_run_for_tickers_skips_failing_ticker_and_keeps_others(monkeypatch):
    def fake_build(ticker, v2_dsn, prod_dsn, hist, fut, static, now=None):
        if ticker == "BAD":
            raise ValueError("no history")
        return pd.DataFrame({"x": [1]})

    def fake_run_inference(ticker, encoder_df, model, hist, fut, static):
        return {"pred_label": 0, "pred_str": "매수", "prob_buy": 0.7, "prob_hold": 0.2, "prob_sell": 0.1}

    monkeypatch.setattr(pipeline, "build_encoder_df_for_ticker", fake_build)
    monkeypatch.setattr(pipeline, "run_inference", fake_run_inference)
    monkeypatch.setattr(pipeline, "get_model", lambda: object())
    monkeypatch.setattr(pipeline, "get_feature_columns", lambda: {"historical": [], "future": [], "static": []})

    results = pipeline.run_for_tickers(["GOOD", "BAD"], "v2dsn", "proddsn", now=datetime(2026, 9, 18, 10, 0))
    assert len(results) == 1
    assert results[0]["ticker"] == "GOOD"
    assert results[0]["pred_str"] == "매수"


def test_push_results_skips_when_no_webhook_url():
    assert pipeline.push_results([{"ticker": "005930"}], webhook_url="") is False


def test_push_results_posts_payload_shape(monkeypatch):
    captured = {}

    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResp()

    monkeypatch.setattr(pipeline.requests, "post", fake_post)
    results = [{"ticker": "005930", "pred_label": 0, "pred_str": "매수",
                "prob_buy": 0.7, "prob_hold": 0.2, "prob_sell": 0.1}]
    ok = pipeline.push_results(results, webhook_url="http://example.com/webhook")
    assert ok is True
    assert captured["url"] == "http://example.com/webhook"
    assert set(captured["json"].keys()) == {"job_id", "user_id", "inference_at", "results"}
    assert captured["json"]["results"] == results
