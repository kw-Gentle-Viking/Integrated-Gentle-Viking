"""레버리지 ETF 일별 NAV·상장주식수·추정 AUM 스냅샷.

과거 AUM/NAV는 KIS 일봉 API에 없다(backfill_leverage.py 참고). 그래서 매일 현재가 조회 값을 따로 쌓는다.
- nav: ETF 현재가 API (tr_id FHPST02400000)
- listed_shares: 주식 현재가 API 의 lstn_stcn
- est_aum = nav * listed_shares  (AUM 필드는 KIS가 주지 않는다)

피처 테이블(feature_pool)에는 쓰지 않는다. 학습 때 AUM 피처는 0이었으므로, 라이브 값을 넣으면 입력 분포가 바뀐다.
"""
import json
import os
from datetime import date, datetime

import psycopg2
import requests

REAL_BASE_URL = "https://openapi.koreainvestment.com:9443"
TOKEN_FILE = os.getenv("KIS_REAL_TOKEN_FILE", "/home/user/team_repos/Back-Gentle-Viking/.cache/kis_real_token.json")


def parse_etf_nav(raw: dict) -> float | None:
    out = raw.get("output") or {}
    try:
        return float(out["nav"])
    except (KeyError, TypeError, ValueError):
        return None


def parse_listed_shares(raw: dict) -> int | None:
    out = raw.get("output") or {}
    try:
        return int(out["lstn_stcn"])
    except (KeyError, TypeError, ValueError):
        return None


def estimate_aum(nav: float | None, shares: int | None) -> float | None:
    if nav is None or shares is None:
        return None
    return nav * shares


def _real_token() -> str:
    try:
        data = json.load(open(TOKEN_FILE, encoding="utf-8"))
        if datetime.fromisoformat(data["expires_at"]) > datetime.now():
            return data["token"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    r = requests.post(f"{REAL_BASE_URL}/oauth2/tokenP", timeout=20, json={
        "grant_type": "client_credentials",
        "appkey": os.environ["KIS_APP_KEY"], "appsecret": os.environ["KIS_APP_SECRET"]})
    r.raise_for_status()
    return r.json()["access_token"]


def _get(token: str, path: str, tr_id: str, code: str) -> dict:
    r = requests.get(f"{REAL_BASE_URL}{path}", timeout=20, headers={
        "authorization": f"Bearer {token}", "appkey": os.environ["KIS_APP_KEY"],
        "appsecret": os.environ["KIS_APP_SECRET"], "tr_id": tr_id, "Content-Type": "application/json"},
        params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code})
    r.raise_for_status()
    return r.json()


def snapshot(dsn: str) -> int:
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("SELECT code FROM leverage_products ORDER BY code")
        codes = [r[0] for r in cur.fetchall()]
    rows = []
    token = _real_token()
    if True:
        for code in codes:
            try:
                nav = parse_etf_nav(_get(token, "/uapi/etfetn/v1/quotations/inquire-price", "FHPST02400000", code))
                shares = parse_listed_shares(_get(token, "/uapi/domestic-stock/v1/quotations/inquire-price", "FHKST01010100", code))
            except requests.RequestException as exc:
                print(f"WARN {code}: KIS 조회 실패, 이 종목은 건너뜀 ({exc})", flush=True)
                continue
            rows.append((code, date.today(), datetime.now(), nav, shares, estimate_aum(nav, shares)))
    with conn.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS leverage_aum_snapshot (
            code VARCHAR(16), snapshot_date DATE, captured_at TIMESTAMP,
            nav NUMERIC, listed_shares BIGINT, est_aum NUMERIC,
            PRIMARY KEY (code, snapshot_date))""")
        cur.executemany("""INSERT INTO leverage_aum_snapshot (code, snapshot_date, captured_at, nav, listed_shares, est_aum)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (code, snapshot_date) DO UPDATE SET captured_at=EXCLUDED.captured_at,
              nav=EXCLUDED.nav, listed_shares=EXCLUDED.listed_shares, est_aum=EXCLUDED.est_aum""", rows)
    conn.commit()
    conn.close()
    return len(rows)


if __name__ == "__main__":
    n = snapshot(os.environ["STOCK_DB_V2_DSN"])
    print(f"leverage AUM snapshot rows: {n}")
