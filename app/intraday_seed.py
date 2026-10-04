"""운영 stock_db.intraday_5min 에 쌓인 과거 5분봉으로 전략 워밍업 데이터를 만든다.

전략은 최근 5분봉 N개(ultra_safe 240개 등)가 있어야 신호를 낸다. AI 서버는 /ai/warmup 요청을 처리하지
않으므로, 이미 적재된 운영 5분봉(읽기 전용)을 live_candles에 넣어서 기존 워밍업 로직이 쓰게 한다.
PROD_STOCK_DB_DSN 이 없으면 아무 것도 하지 않는다(기존 동작과 동일)."""
import os

import psycopg2
from sqlalchemy.orm import Session

from app.models import LiveCandle


def seed_live_candles(db: Session, tickers: list[str], counts: dict[str, int]) -> dict[str, int]:
    """종목별로 최근 counts[ticker] 개의 5분봉을 live_candles에 넣고, 실제로 넣은 개수를 돌려준다."""
    dsn = os.getenv("PROD_STOCK_DB_DSN", "")
    if not dsn or not tickers:
        return {}
    received: dict[str, int] = {}
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            for ticker in tickers:
                n = counts.get(ticker, 0)
                if n <= 0:
                    continue
                cur.execute(
                    "SELECT datetime, open, high, low, close, volume FROM intraday_5min "
                    "WHERE ticker = %s ORDER BY datetime DESC LIMIT %s",
                    (ticker, n),
                )
                rows = list(reversed(cur.fetchall()))
                for dt, o, h, l, c, v in rows:
                    db.add(LiveCandle(ticker=ticker, timeframe="5m", open=o, high=h, low=l, close=c,
                                      volume=v, trade_datetime=dt))
                received[ticker] = len(rows)
    finally:
        conn.close()
    db.commit()
    return received
