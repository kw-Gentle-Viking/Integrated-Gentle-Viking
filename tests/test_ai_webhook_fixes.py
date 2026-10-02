"""2026-10-02 통합 감사에서 발견된 /ai 웹훅의 세 가지 작은 결함 수정 테스트:
1) pred_label이 0/1/2 밖이면 로그 없이 조용히 HOLD로 폴백 -- 경고를 남기도록.
2) ONCE 콜백 결과는 realtime push와 달리 AIPredictionHistory에 전혀 기록되지 않았음.
3) AI가 매번 보내는 job_id/user_id가 RealtimePayload 스키마에 없어 조용히 드롭됨."""
import pytest

from app.schemas import OnceCallbackPayload, PredictionResult, RealtimePayload


def _pred(**overrides):
    base = dict(ticker="005930", trade_datetime="2026-10-02T15:30:00", pred_label=0, pred_str="매수",
                prob_buy=0.6, prob_hold=0.3, prob_sell=0.1)
    base.update(overrides)
    return PredictionResult(**base)


# ---- 1) pred_label 이상값: 폴백은 유지하되 경고를 남긴다 ----
def test_parse_prediction_falls_back_silently_mapped_but_logs_warning_on_bad_label(capsys):
    from app.routes_ai_webhook import parse_prediction
    result = parse_prediction(_pred(pred_label=99, pred_str="매수"))
    assert result["signal"] == "BUY"  # pred_str 기준 폴백 결과는 그대로 유지
    assert "WARNING" in capsys.readouterr().out


def test_parse_prediction_no_warning_for_normal_labels(capsys):
    from app.routes_ai_webhook import parse_prediction
    parse_prediction(_pred(pred_label=0, pred_str="매수"))
    assert "WARNING" not in capsys.readouterr().out


# ---- 2) ONCE 콜백도 realtime push처럼 AIPredictionHistory에 남아야 한다 ----
def test_once_callback_records_predictions_for_history(monkeypatch):
    import app.routes_ai_webhook as webhook

    monkeypatch.setattr(webhook, "generate_report", lambda result: "리포트 더미")
    monkeypatch.setattr(webhook, "save_report", lambda *a, **k: None)
    recorded = []
    monkeypatch.setattr(webhook, "record_predictions", lambda parsed: recorded.append(parsed))

    payload = OnceCallbackPayload(
        job_id="job-1", user_id="1", inference_at="2026-10-02T15:30:00",
        results=[_pred()],
    )
    out = webhook.receive_once_callback(payload, x_api_key=webhook.AI_API_KEY)

    assert out["status"] == "ok"
    assert len(recorded) == 1 and recorded[0][0]["ticker"] == "005930"


# ---- 3) RealtimePayload가 job_id/user_id를 받아도 거절하지 않는다 ----
def test_realtime_payload_accepts_job_id_and_user_id():
    payload = RealtimePayload(inference_at="2026-10-02T15:30:00", results=[_pred()],
                              job_id="pipeline_20261002_153000", user_id="pipeline")
    assert payload.job_id == "pipeline_20261002_153000" and payload.user_id == "pipeline"


def test_realtime_payload_still_works_without_job_id_and_user_id():
    payload = RealtimePayload(inference_at="2026-10-02T15:30:00", results=[_pred()])
    assert payload.job_id == "" and payload.user_id == ""
