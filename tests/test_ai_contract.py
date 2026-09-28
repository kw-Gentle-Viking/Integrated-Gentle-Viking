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
