"""AI 예측 이력 저장/조회 (백테스트에서 AI 신호를 재생하기 위한 기반)."""
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import and_

from app.models import AIPredictionHistory

logger = logging.getLogger(__name__)
KST = ZoneInfo("Asia/Seoul")


def _to_kst_naive(value: str) -> datetime:
    ts = datetime.fromisoformat(str(value))
    if ts.tzinfo is not None:
        ts = ts.astimezone(KST).replace(tzinfo=None)
    return ts


def record_prediction(db, pred: dict) -> bool:
    """예측 1건을 저장한다. 웹훅을 죽이지 않도록 실패는 삼키고 False 를 돌려준다."""
    try:
        row = AIPredictionHistory(
            ticker=pred["ticker"],
            trade_datetime=_to_kst_naive(pred["trade_datetime"]),
            signal=pred["signal"],
            prob_buy=float(pred.get("prob_buy", 0.0) or 0.0),
            prob_hold=float(pred.get("prob_hold", 0.0) or 0.0),
            prob_sell=float(pred.get("prob_sell", 0.0) or 0.0),
            model_version=str(pred.get("model_version") or "unknown"),
        )
        db.add(row)
        db.commit()
        return True
    except Exception as e:  # noqa: BLE001
        db.rollback()
        logger.warning("AI 예측 이력 저장 실패 (%s): %s", pred.get("ticker"), e)
        return False


def load_signals(db, ticker: str, start_date: str, end_date: str) -> dict[str, str]:
    """{'YYYY-MM-DD': signal}: 각 날짜의 '마지막' 예측 (종가 부근 예측이 그날의 신호). end_date 포함."""
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59)
    rows = (
        db.query(AIPredictionHistory)
        .filter(and_(AIPredictionHistory.ticker == ticker,
                     AIPredictionHistory.trade_datetime >= start,
                     AIPredictionHistory.trade_datetime <= end))
        .order_by(AIPredictionHistory.trade_datetime.asc())
        .all()
    )
    out: dict[str, str] = {}
    for r in rows:
        out[r.trade_datetime.strftime("%Y-%m-%d")] = r.signal   # 시간순이라 마지막 값이 남는다
    return out


def record_predictions(parsed: list[dict], session_factory=None) -> int:
    """웹훅에서 쓰는 일괄 저장 헬퍼. 저장된 건수를 돌려준다.

    session_factory 가 없으면 자체 세션을 열고 닫는다. 주입된 경우(테스트 등)에는 호출자가 세션을 소유하므로 닫지 않는다.
    """
    own = session_factory is None
    if own:
        from app.db import SessionLocal as session_factory
    db = session_factory()
    try:
        return sum(1 for p in parsed if record_prediction(db, p))
    finally:
        if own:
            db.close()
