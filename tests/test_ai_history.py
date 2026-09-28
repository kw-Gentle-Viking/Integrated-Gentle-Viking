"""AI 예측 이력 저장: 백엔드는 /ai/realtime 결과를 메모리에 '덮어쓰기'만 해서 과거 예측이 남지 않았고,
그래서 AI 신호를 백테스트에 재생할 수 없었다."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def db():
    from app.db import Base
    import app.models  # noqa: F401  (테이블 등록)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _pred(ticker, dt, signal, model_version="v3"):
    return {"ticker": ticker, "signal": signal, "confidence": 0.4, "prob_buy": 0.3, "prob_hold": 0.4,
            "prob_sell": 0.3, "trade_datetime": dt, "model_version": model_version}


def test_record_and_load_last_signal_per_day(db):
    from app.ai_history import load_signals, record_prediction
    record_prediction(db, _pred("005930", "2026-09-28T10:00:00", "HOLD"))
    record_prediction(db, _pred("005930", "2026-09-28T15:20:00", "BUY"))     # 그날 마지막 = 종가 부근 예측
    record_prediction(db, _pred("005930", "2026-09-29T15:20:00", "SELL"))
    record_prediction(db, _pred("000660", "2026-09-28T15:20:00", "SELL"))     # 다른 종목은 섞이지 않음
    sig = load_signals(db, "005930", "2026-09-28", "2026-09-30")
    assert sig == {"2026-09-28": "BUY", "2026-09-29": "SELL"}


def test_load_signals_respects_date_range(db):
    from app.ai_history import load_signals, record_prediction
    for d, s in (("2026-09-25", "BUY"), ("2026-09-28", "SELL"), ("2026-10-01", "BUY")):
        record_prediction(db, _pred("005930", f"{d}T15:20:00", s))
    assert load_signals(db, "005930", "2026-09-26", "2026-09-30") == {"2026-09-28": "SELL"}


def test_record_prediction_never_raises_on_bad_input(db):
    # 웹훅이 이력 저장 실패 때문에 죽으면 안 된다: 저장 실패는 False 를 돌려주고 삼킨다.
    from app.ai_history import record_prediction
    assert record_prediction(db, {"ticker": "005930"}) is False               # 필수 필드 없음
    assert record_prediction(db, _pred("005930", "garbage", "BUY")) is False   # 시각 해석 불가


def test_record_predictions_opens_its_own_session_and_saves_all(db):
    from app.ai_history import load_signals, record_predictions
    saved = record_predictions(
        [_pred("005930", "2026-09-28T15:20:00", "BUY"), _pred("000660", "2026-09-28T15:20:00", "SELL")],
        session_factory=lambda: db,
    )
    assert saved == 2
    assert load_signals(db, "000660", "2026-09-28", "2026-09-28") == {"2026-09-28": "SELL"}


def test_backtest_service_resolves_ai_signals_from_history_when_not_given(db):
    from app.ai_history import record_prediction
    from backtest.schemas import BacktestRequest
    from backtest.service import BacktestService
    record_prediction(db, _pred("005930", "2026-09-28T15:20:00", "BUY"))
    req = BacktestRequest(symbol="005930", strategy="ai_signal", start_date="2026-09-01", end_date="2026-09-30")
    params = BacktestService()._resolve_strategy_params(req, session_factory=lambda: db)
    assert params["signals"] == {"2026-09-28": "BUY"}


def test_backtest_service_prefers_explicit_signals_and_errors_when_no_history(db):
    from backtest.schemas import BacktestRequest
    from backtest.service import BacktestService
    explicit = BacktestRequest(symbol="005930", strategy="ai_signal", start_date="2026-09-01", end_date="2026-09-30",
                               strategy_params={"signals": {"2026-09-02": "SELL"}})
    assert BacktestService()._resolve_strategy_params(explicit, session_factory=lambda: db)["signals"] == {"2026-09-02": "SELL"}
    empty = BacktestRequest(symbol="005930", strategy="ai_signal", start_date="2026-09-01", end_date="2026-09-30")
    with pytest.raises(ValueError, match="AI 예측 이력"):
        BacktestService()._resolve_strategy_params(empty, session_factory=lambda: db)


def test_non_ai_strategies_keep_their_params_untouched(db):
    from backtest.schemas import BacktestRequest
    from backtest.service import BacktestService
    req = BacktestRequest(symbol="005930", strategy="rsi_reversal", start_date="2026-09-01", end_date="2026-09-30",
                          strategy_params={"period": 10})
    assert BacktestService()._resolve_strategy_params(req, session_factory=lambda: db) == {"period": 10}
