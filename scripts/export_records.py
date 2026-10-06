"""추론·체결·자동매매 판단 기록을 파일 로그(JSONL)로 누적한다.

- 대상: ai_prediction_history, trade_logs, auto_trade_decisions (백엔드 DB)
- 저장: /home/user/trade_records/<table>.jsonl  (추가만 한다, 기존 줄은 고치지 않는다)
- 중복 방지: /home/user/trade_records/.watermark.json 에 테이블별 마지막 id 를 남긴다
- 실행 주기: 5분마다 (cron). 새 행이 없으면 아무것도 쓰지 않는다.
"""
import json
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import psycopg2

OUT_DIR = Path(os.getenv("TRADE_RECORDS_DIR", "/home/user/trade_records"))
WATERMARK = OUT_DIR / ".watermark.json"
TABLES = {
    "ai_prediction_history": "SELECT * FROM ai_prediction_history WHERE id > %s ORDER BY id LIMIT 5000",
    "trade_logs": "SELECT * FROM trade_logs WHERE id > %s ORDER BY id LIMIT 5000",
    "auto_trade_decisions": "SELECT * FROM auto_trade_decisions WHERE id > %s ORDER BY id LIMIT 5000",
}


def _jsonable(v):
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v


def _load_watermark() -> dict:
    try:
        return json.loads(WATERMARK.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def export(dsn: str) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    marks = _load_watermark()
    written = {}
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            for table, sql in TABLES.items():
                last = marks.get(table, 0)
                cur.execute(sql, (last,))
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
                if not rows:
                    written[table] = 0
                    continue
                with open(OUT_DIR / f"{table}.jsonl", "a", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps({c: _jsonable(v) for c, v in zip(cols, r)}, ensure_ascii=False) + "\n")
                marks[table] = rows[-1][0]
                written[table] = len(rows)
    finally:
        conn.close()
    tmp = WATERMARK.with_suffix(".tmp")
    tmp.write_text(json.dumps(marks), encoding="utf-8")
    tmp.replace(WATERMARK)
    return written


if __name__ == "__main__":
    print(export(os.environ["DATABASE_URL"]))
