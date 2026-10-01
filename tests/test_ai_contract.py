"""AI 서버(TFT 서빙) <-> 백엔드 계약 테스트."""
from app.schemas import PredictionResult


def test_prediction_result_accepts_payload_without_model_version():
    # 우리 서빙 페이로드는 model_version 을 보내지 않았다 -> 예전 스키마(필수)에서는 /ai/realtime 이 422 로 거절됐다.
    ours = {"ticker": "005930", "trade_datetime": "2026-09-28T10:00:00", "pred_label": 1, "pred_str": "관망",
            "prob_buy": 0.31, "prob_hold": 0.38, "prob_sell": 0.31}
    r = PredictionResult(**ours)
    assert r.model_version == "unknown"


def test_prediction_result_keeps_explicit_model_version():
    r = PredictionResult(ticker="005930", trade_datetime="2026-09-28T10:00:00", pred_label=0, pred_str="매수",
                         prob_buy=0.4, prob_hold=0.3, prob_sell=0.3, model_version="v3_wd-seed0")
    assert r.model_version == "v3_wd-seed0"


# ---- 예측 신선도 (오래된 예측으로 매매하지 않기) ----
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")


def test_is_stale_uses_kst_for_naive_timestamps_and_respects_max_age():
    from app.ai_client import is_stale
    now = datetime(2026, 9, 28, 10, 30, tzinfo=KST)
    assert is_stale("2026-09-28T10:20:00", now=now, max_age_min=30) is False       # 10분 전(naive=KST)
    assert is_stale("2026-09-28T09:40:00", now=now, max_age_min=30) is True        # 50분 전
    assert is_stale("2026-09-28T09:40:00+09:00", now=now, max_age_min=30) is True  # tz 명시도 동일
    assert is_stale("2026-09-28T01:20:00+00:00", now=now, max_age_min=30) is False # UTC 10:20 KST
    assert is_stale("2026-09-28T09:40:00", now=now, max_age_min=0) is False        # 0 이면 검사 끔


def test_is_stale_treats_missing_or_unparseable_timestamp_as_stale():
    from app.ai_client import is_stale
    now = datetime(2026, 9, 28, 10, 30, tzinfo=KST)
    assert is_stale(None, now=now, max_age_min=30) is True
    assert is_stale("not-a-date", now=now, max_age_min=30) is True


def test_predict_downgrades_stale_prediction_to_hold_with_zero_confidence(monkeypatch):
    import app.ai_client as ac
    old = (datetime.now(KST) - timedelta(hours=3)).isoformat()
    ac.realtime_predictions["005930"] = {"ticker": "005930", "signal": "BUY", "confidence": 0.9, "prob_buy": 0.9,
                                         "prob_hold": 0.05, "prob_sell": 0.05, "trade_datetime": old, "model_version": "x"}
    monkeypatch.setenv("AI_PRED_MAX_AGE_MIN", "30")
    out = ac.AIClient().predict("005930")
    assert out["signal"] == "HOLD" and out["confidence"] == 0.0 and out.get("stale") is True


def test_predict_passes_through_fresh_prediction(monkeypatch):
    import app.ai_client as ac
    fresh = datetime.now(KST).isoformat()
    ac.realtime_predictions["000660"] = {"ticker": "000660", "signal": "BUY", "confidence": 0.9, "prob_buy": 0.9,
                                         "prob_hold": 0.05, "prob_sell": 0.05, "trade_datetime": fresh, "model_version": "x"}
    monkeypatch.setenv("AI_PRED_MAX_AGE_MIN", "30")
    out = ac.AIClient().predict("000660")
    assert out["signal"] == "BUY" and out["confidence"] == 0.9 and not out.get("stale")


# ---- 5분 push가 ONCE 리포트를 지우는 문제 / 조회 API에 신선도 미표시 ----
def test_merge_realtime_prediction_preserves_existing_report_fields():
    from app.ai_client import merge_realtime_prediction
    old = {"ticker": "005930", "signal": "BUY", "confidence": 0.8, "report": "보고서", "analysis": "분석",
          "interpretability": {"top_features": ["a"]}}
    new = {"ticker": "005930", "signal": "HOLD", "confidence": 0.5}   # 5분 push 는 report류 필드를 안 보낸다
    merged = merge_realtime_prediction(old, new)
    assert merged["signal"] == "HOLD" and merged["confidence"] == 0.5          # 새 신호/확신도는 갱신
    assert merged["report"] == "보고서" and merged["analysis"] == "분석"        # 리포트는 보존
    assert merged["interpretability"] == {"top_features": ["a"]}


def test_merge_realtime_prediction_lets_new_report_override_when_present():
    from app.ai_client import merge_realtime_prediction
    old = {"ticker": "005930", "report": "옛 보고서"}
    new = {"ticker": "005930", "signal": "BUY", "report": "새 보고서"}
    assert merge_realtime_prediction(old, new)["report"] == "새 보고서"


def test_merge_realtime_prediction_with_no_prior_value_is_just_new():
    from app.ai_client import merge_realtime_prediction
    new = {"ticker": "000660", "signal": "SELL"}
    assert merge_realtime_prediction(None, new) == new


def test_with_staleness_adds_stale_flag_using_trade_datetime():
    from app.ai_client import with_staleness
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    now = datetime(2026, 9, 28, 10, 30, tzinfo=ZoneInfo("Asia/Seoul"))
    fresh = with_staleness({"trade_datetime": "2026-09-28T10:20:00"}, now=now)
    stale = with_staleness({"trade_datetime": "2026-09-28T09:00:00"}, now=now)
    assert fresh["stale"] is False and stale["stale"] is True
