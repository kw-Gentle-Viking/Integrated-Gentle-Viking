# app/ai_client.py
import os
import random
from datetime import datetime
from zoneinfo import ZoneInfo

from app.shared_state import realtime_predictions

KST = ZoneInfo("Asia/Seoul")


def _max_age_min() -> float:
    try:
        return float(os.getenv("AI_PRED_MAX_AGE_MIN", "30"))
    except ValueError:
        return 30.0


def is_stale(trade_datetime, now=None, max_age_min=None) -> bool:
    """예측 시각(trade_datetime)이 max_age_min 분보다 오래됐거나 없거나 해석 불가면 True.

    AI 서버는 5분마다 push 하지만, 서버가 죽으면 마지막 예측이 메모리에 남아 계속 매매 판단에 쓰였다.
    tz 없는 시각은 KST 로 본다(서빙이 naive isoformat 을 보낸다). max_age_min<=0 이면 검사하지 않는다.
    """
    max_age = _max_age_min() if max_age_min is None else max_age_min
    if max_age <= 0:
        return False
    if not trade_datetime:
        return True
    try:
        ts = datetime.fromisoformat(str(trade_datetime))
    except ValueError:
        return True
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=KST)
    now = now or datetime.now(KST)
    return (now - ts).total_seconds() > max_age * 60


class AIClient:
    """AI 서버 클라이언트"""

    def predict(self, ticker: str) -> dict:
        """
        실시간 추론 결과가 있으면 사용, 없으면 더미
        """
        # AI 서버에서 push된 결과가 있으면 사용
        pred = realtime_predictions.get(ticker)
        if pred and is_stale(pred.get("trade_datetime")):
            # 오래된 예측으로는 매매하지 않는다: 신뢰도 0 의 HOLD 로 낮춘다 (min_confidence 게이트에서 걸러짐)
            return {
                "ticker": ticker, "signal": "HOLD", "confidence": 0.0,
                "prob_buy": 0.0, "prob_hold": 1.0, "prob_sell": 0.0,
                "trade_datetime": pred.get("trade_datetime"), "model_version": pred.get("model_version"),
                "stale": True,
            }
        if pred:
            return {
                "ticker": pred["ticker"],
                "signal": pred["signal"],
                "confidence": pred["confidence"],
                "prob_buy": pred.get("prob_buy", 0.0),
                "prob_hold": pred.get("prob_hold", 0.0),
                "prob_sell": pred.get("prob_sell", 0.0),
                "trade_datetime": pred.get("trade_datetime"),
                "model_version": pred.get("model_version"),
            }

        if os.getenv("AI_ALLOW_DUMMY_PREDICTIONS", "false").lower() not in {
            "1",
            "true",
            "yes",
            "y",
        }:
            return {
                "ticker": ticker,
                "signal": "HOLD",
                "confidence": 0.0,
                "prob_buy": 0.0,
                "prob_hold": 1.0,
                "prob_sell": 0.0,
            }

        # 더미 (AI 서버 미연결 시)
        signal = random.choice(["BUY", "HOLD", "SELL"])
        confidence = round(random.uniform(0.5, 0.95), 2)
        return {
            "ticker": ticker,
            "signal": signal,
            "confidence": confidence,
            "prob_buy": confidence if signal == "BUY" else 0.0,
            "prob_hold": confidence if signal == "HOLD" else 0.0,
            "prob_sell": confidence if signal == "SELL" else 0.0,
        }