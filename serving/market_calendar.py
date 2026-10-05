"""휴장일 판단. 운영 DB calendar 테이블(is_market_open)이 기준이다.
달력을 못 읽으면 None을 돌려주고, 호출하는 쪽은 평일 기준으로 판단한다(백엔드 app/market_calendar.py와 같은 규칙)."""
import logging
import os
from datetime import date
from functools import lru_cache

import psycopg2

logger = logging.getLogger(__name__)


@lru_cache(maxsize=512)
def _lookup(dsn: str, day: date):
    try:
        with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT is_market_open FROM calendar WHERE base_date = %s", (day,))
            row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001
        logger.warning("휴장 달력 조회 실패 (%s): %s", day, exc)
        return None
    if row is None:
        logger.warning("휴장 달력에 %s 없음", day)
        return None
    return bool(row[0])


def is_market_open_day(day: date):
    """True=개장일, False=휴장일, None=모름."""
    if day.weekday() >= 5:
        return False
    dsn = os.getenv("PROD_STOCK_DB_DSN", "")
    if not dsn:
        return None
    return _lookup(dsn, day)
