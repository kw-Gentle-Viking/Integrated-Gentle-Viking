"""유저별 자동매매 실행 잠금. 워커/레플리카가 여러 개여도 같은 유저의 루프가 두 번 돌지 않게 한다.

Postgres 세션 advisory lock을 쓰고, 잠금을 잡은 커넥션은 루프가 살아 있는 동안 유지한다.
Postgres가 아니면(테스트용 sqlite) 프로세스 내 잠금만 본다."""
from sqlalchemy import text

from app.db import engine

_held: dict[int, object] = {}
_KEY_BASE = 9_100_000


def _is_postgres() -> bool:
    return engine.dialect.name == "postgresql"


def acquire(user_id: int) -> bool:
    if user_id in _held:
        return False
    if not _is_postgres():
        _held[user_id] = None
        return True
    conn = engine.connect()
    got = conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": _KEY_BASE + user_id}).scalar()
    if not got:
        conn.close()
        return False
    _held[user_id] = conn
    return True


def release(user_id: int) -> None:
    conn = _held.pop(user_id, None)
    if conn is None:
        return
    try:
        conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _KEY_BASE + user_id})
    finally:
        conn.close()
