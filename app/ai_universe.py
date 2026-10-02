# app/ai_universe.py
"""AI(TFT) 모델이 실제로 추론하는 종목 집합 -- `ticker_universe`(stock_db_v2, AI 레포가 관리하는 200개
KOSPI 티커)을 읽기 전용으로 조회한다. Basket에 AI가 전혀 모르는 종목(코스닥 등)이 들어가면
app.ai_client.AIClient.predict()가 AI 서버 커버리지 부재를 "AI_ALLOW_DUMMY_PREDICTIONS=true 시 완전
무작위 BUY/SELL, 아니면 confidence 0 HOLD"로 조용히 치환해 실제 AI 판단과 구분이 안 되는 문제가 있었다
(2026-10-02 통합 감사). 이 모듈은 Basket 등록 시점에 그 원인을 차단한다."""
import os
import time

from sqlalchemy import create_engine, text

STOCK_DB_V2_DSN = os.getenv("STOCK_DB_V2_DSN", "")
CACHE_TTL_SEC = 3600  # 유니버스는 하루 1회 스냅샷이라 자주 바뀌지 않는다.

_engine = None
_cache: dict = {"tickers": None, "loaded_at": 0.0}


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(STOCK_DB_V2_DSN, pool_pre_ping=True)
    return _engine


def get_ai_universe(force_refresh: bool = False) -> set[str] | None:
    """AI가 커버하는 티커 집합, 또는 STOCK_DB_V2_DSN이 설정되지 않았으면 None(검증 생략을 의미).

    DSN이 설정됐는데 조회 자체가 실패하면 예외를 그대로 올린다 -- 설정된 환경에서 조회 실패를 조용히
    '전부 허용'으로 처리하면 이 모듈이 막으려는 문제(검증 없음)가 그대로 재현되기 때문이다."""
    if not STOCK_DB_V2_DSN:
        return None
    now = time.monotonic()
    if not force_refresh and _cache["tickers"] is not None and now - _cache["loaded_at"] < CACHE_TTL_SEC:
        return _cache["tickers"]
    with _get_engine().connect() as conn:
        rows = conn.execute(text("SELECT ticker FROM ticker_universe")).fetchall()
    tickers = {r[0] for r in rows}
    _cache["tickers"] = tickers
    _cache["loaded_at"] = now
    return tickers


def is_ai_covered_ticker(ticker: str) -> bool:
    """AI 유니버스를 조회할 수 없으면(DSN 미설정) 과거 동작(검증 없음)과 동일하게 통과시킨다."""
    universe = get_ai_universe()
    if universe is None:
        return True
    return ticker in universe
