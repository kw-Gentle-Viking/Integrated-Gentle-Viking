# AI 학습·추론 재설계 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 한국 주식 AI 자동매매 시스템의 학습·추론 파이프라인을 일봉 호라이즌 TFT 3진 분류로 전면 재설계하고, 2019~2025 클린 구간(1단계)과 2019~현재 레버리지 국면 포함 전체 구간(2단계)의 2단계 학습을 거쳐 배포 가능한 모델과 서빙 파이프라인을 완성한다.

**Architecture:** KIS API 백필 → `stock_db_v2`(기존 운영 DB와 완전 분리) → 조인된 feature_pool 테이블 → train-split 기준 클리핑/스케일링 → PyTorch `Dataset`(encoder 60일 + 오늘의 실시간 미완성 일봉) → 포크된 `tft-torch`(3-class 헤드) 학습(1단계 벤치마크 → 2단계 최종) → 평가(Macro F1 중심) → 기존 `api_server.py`/crontab 패턴을 재사용한 서빙.

**Tech Stack:** Python(`kis_collector` conda env 기준: pandas, psycopg2-binary, requests, omegaconf), PyTorch + 포크된 `tft-torch`, PostgreSQL(`stock_db_v2`), wandb, dvc, Optuna, pytest.

**Spec:** `/home/user/AI_Gentle_Viking_RE/docs/superpowers/specs/2026-09-08-ai-model-redesign-design.md` (커밋 `27d334b`)

## Global Constraints

- 기존 운영 중인 `stock_db`, crontab, `/home/user/*.py`는 절대 수정하지 않는다 — 전부 `stock_db_v2` + `/home/user/AI_Gentle_Viking_RE` 안에서만 작업
- KIS API 키는 기존과 공유하되, 신규 백필 스크립트는 장중 08:55~15:30, 15:50, 16:00 시간대를 피해서 실행한다(각 백필 스크립트에 시간대 가드 포함)
- 모델: TFT 고정, 3진 분류(매수=0/관망=1/매도=2) 고정, `tft-torch` 포크(`/home/user/AI_Gentle_Viking_RE/tft-torch`, `class_logits` 출력 키, `config.model.num_classes=3`) 사용
- 유니버스: 시총 상위 200(2019-01-02 종가 기준 스냅샷, 재계산 금지 — 미래정보 사용 방지)
- 1단계 기간: train 2019-01-02~2023-12-31 / val 2024-01-01~2024-12-31 / test 2025-01-01~2025-12-31 (2026 전체 배제)
- 2단계 기간: 2019-01-02~현재 전체(레버리지 국면 포함)
- 날짜 기준 시계열 분리만 사용, 랜덤 셔플 금지
- 클리핑/스케일링은 항상 해당 단계의 train split에서만 fit, val/test/서빙엔 그대로 적용(재계산 금지)
- 평가 핵심 지표: Macro F1(그 외 Accuracy/클래스별 P·R·F1/Confusion Matrix/MCC 병기), 1단계와 2단계(레버리지 국면) 성능은 항상 분리 리포트
- encoder 길이 60거래일 고정
- 신규 인프라(TorchServe, BentoML 등) 도입 금지 — 기존 `api_server.py`/crontab 패턴 재사용
- 금액/거래대금 관련 새 API 응답 필드는 대량 백필 전에 반드시 1건 샘플로 단위(원 vs 백만원 등)를 수동 확인한다 — 기존 KIS 필드(`hts_avls`, `*_tr_pbmn` 등)는 이미 pipeline_overview.md에 배율이 문서화돼 있지만, 처음 다루는 API(레버리지 ETF/ETN NAV·AUM 등)는 확인된 바 없음
- 데이터 정합성은 클리핑(이상치 처리, Task 11)과 별개 문제 — 독립 소스 간 교차검증(예: 두 가지 방식으로 계산한 시가총액 비교), 값 범위, 결측률, 종목별 거래일수 갭 검증을 Task 10에서 수행하고 통과해야 다음 단계로 진행

---

## Task 1: `stock_db_v2` 스키마 생성

**Files:**
- Create: `db/schema.sql`
- Create: `db/init_db.py`
- Test: `db/test_init_db.py`

**Interfaces:**
- Produces: `init_db.create_database(dsn_admin: str, db_name: str = "stock_db_v2") -> None`, `init_db.apply_schema(dsn: str, schema_path: str = "db/schema.sql") -> None`

- [ ] **Step 1: 스키마 DDL 작성**

`db/schema.sql`:

```sql
CREATE TABLE IF NOT EXISTS ticker_universe (
    ticker VARCHAR(6) PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    market_cap NUMERIC NOT NULL,
    rank INT NOT NULL,
    is_kospi BOOLEAN NOT NULL,
    sector_id INT,
    listing_date DATE
);

CREATE TABLE IF NOT EXISTS price_daily (
    ticker VARCHAR(6) NOT NULL,
    trade_date DATE NOT NULL,
    open_price NUMERIC, high_price NUMERIC, low_price NUMERIC, close_price NUMERIC,
    volume BIGINT, turnover NUMERIC, shares_outstanding BIGINT,
    PRIMARY KEY (ticker, trade_date)
);

CREATE TABLE IF NOT EXISTS daily_valuation (
    ticker VARCHAR(6) NOT NULL, trade_date DATE NOT NULL,
    per NUMERIC, pbr NUMERIC, market_cap NUMERIC,
    PRIMARY KEY (ticker, trade_date)
);

CREATE TABLE IF NOT EXISTS investor_flow_daily (
    ticker VARCHAR(6) NOT NULL, trade_date DATE NOT NULL,
    individual_net_amt NUMERIC, foreign_net_amt NUMERIC, inst_net_amt NUMERIC, market_cap NUMERIC,
    PRIMARY KEY (ticker, trade_date)
);

CREATE TABLE IF NOT EXISTS market_index_daily (
    index_code VARCHAR(10) NOT NULL, trade_date DATE NOT NULL, close_price NUMERIC,
    PRIMARY KEY (index_code, trade_date)
);

CREATE TABLE IF NOT EXISTS market_global (
    trade_date DATE PRIMARY KEY,
    snp500_close NUMERIC, nasdaq_close NUMERIC, phlx_semi_close NUMERIC, vix NUMERIC,
    wti_crude_oil NUMERIC, gold_price NUMERIC, usd_krw NUMERIC,
    us_10y_yield NUMERIC, fed_rate NUMERIC, kr_base_rate NUMERIC
);

CREATE TABLE IF NOT EXISTS sector_daily_ohlcv (
    sector_code VARCHAR(4) NOT NULL, trade_date DATE NOT NULL,
    open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC, volume BIGINT,
    PRIMARY KEY (sector_code, trade_date)
);

CREATE TABLE IF NOT EXISTS stock_events (
    ticker VARCHAR(6) NOT NULL, event_date DATE NOT NULL, event_type VARCHAR(20) NOT NULL,
    description TEXT,
    PRIMARY KEY (ticker, event_date, event_type)
);

CREATE TABLE IF NOT EXISTS calendar (
    base_date DATE PRIMARY KEY,
    day_of_week INT, is_market_open BOOLEAN, is_holiday BOOLEAN, is_short_selling_banned BOOLEAN
);

CREATE TABLE IF NOT EXISTS market_events (
    event_date DATE NOT NULL, event_type VARCHAR(20) NOT NULL,
    is_bok BOOLEAN, is_fomc BOOLEAN, is_witching_kr BOOLEAN, is_witching_us BOOLEAN,
    PRIMARY KEY (event_date, event_type)
);

CREATE TABLE IF NOT EXISTS leverage_products (
    code VARCHAR(6) PRIMARY KEY,
    product_name VARCHAR(80) NOT NULL,
    underlying_ticker VARCHAR(6) NOT NULL,
    multiple NUMERIC NOT NULL,          -- +2.0 or -2.0
    product_type VARCHAR(3) NOT NULL,   -- 'ETF' or 'ETN'
    issuer VARCHAR(20),
    listed_date DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS leverage_daily (
    code VARCHAR(6) NOT NULL, trade_date DATE NOT NULL,
    close_price NUMERIC, volume BIGINT, turnover NUMERIC, nav NUMERIC, aum NUMERIC,
    PRIMARY KEY (code, trade_date)
);

CREATE TABLE IF NOT EXISTS vi_events (
    ticker VARCHAR(6) NOT NULL, triggered_at TIMESTAMP NOT NULL,
    released_at TIMESTAMP, vi_type VARCHAR(10),
    PRIMARY KEY (ticker, triggered_at)
);

CREATE TABLE IF NOT EXISTS intraday_5min (
    ticker VARCHAR(6) NOT NULL, datetime TIMESTAMP NOT NULL,
    open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC, volume BIGINT,
    PRIMARY KEY (ticker, datetime)
);

CREATE TABLE IF NOT EXISTS intraday_1min (
    ticker VARCHAR(6) NOT NULL, datetime TIMESTAMP NOT NULL,
    open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC, volume BIGINT,
    PRIMARY KEY (ticker, datetime)
);

CREATE TABLE IF NOT EXISTS labels (
    ticker VARCHAR(6) NOT NULL, trade_date DATE NOT NULL,
    next_day_return NUMERIC, label SMALLINT,  -- 0=매수, 1=관망, 2=매도, NULL=마지막 거래일(라벨 없음)
    PRIMARY KEY (ticker, trade_date)
);
```

- [ ] **Step 2: DB 생성/스키마 적용 스크립트 작성**

`db/init_db.py`:

```python
import argparse
import psycopg2


def create_database(dsn_admin: str, db_name: str = "stock_db_v2") -> None:
    conn = psycopg2.connect(dsn_admin)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        if cur.fetchone() is None:
            cur.execute(f'CREATE DATABASE "{db_name}"')
    conn.close()


def apply_schema(dsn: str, schema_path: str = "db/schema.sql") -> None:
    with open(schema_path, "r") as f:
        ddl = f.read()
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--admin-dsn", required=True, help="admin DSN, e.g. postgresql://user:pw@localhost/postgres")
    parser.add_argument("--target-dsn", required=True, help="stock_db_v2 DSN")
    args = parser.parse_args()
    create_database(args.admin_dsn)
    apply_schema(args.target_dsn)
    print("stock_db_v2 ready")
```

- [ ] **Step 3: 테스트 작성 (스키마 파싱 및 필수 테이블 존재 검증)**

`db/test_init_db.py`:

```python
import re
from pathlib import Path


def test_schema_defines_all_required_tables():
    ddl = Path(__file__).parent.joinpath("schema.sql").read_text()
    required_tables = [
        "ticker_universe", "price_daily", "daily_valuation", "investor_flow_daily",
        "market_index_daily", "market_global", "sector_daily_ohlcv", "stock_events",
        "calendar", "market_events", "leverage_products", "leverage_daily",
        "vi_events", "intraday_5min", "intraday_1min", "labels",
    ]
    found = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", ddl))
    missing = [t for t in required_tables if t not in found]
    assert not missing, f"schema.sql is missing tables: {missing}"
```

- [ ] **Step 4: 테스트 실행**

Run: `cd /home/user/AI_Gentle_Viking_RE && /home/user/miniconda3/envs/kis_collector/bin/python -m pytest db/test_init_db.py -v`
Expected: PASS

- [ ] **Step 5: 실제 DB에 적용 (로컬 postgres, 장외 시간 무관 — DB 생성은 API 호출 아님)**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python db/init_db.py --admin-dsn postgresql://stock_user:<PW>@localhost/postgres --target-dsn postgresql://stock_user:<PW>@localhost/stock_db_v2`
Expected: 출력 "stock_db_v2 ready", 이후 `psql stock_db_v2 -c '\dt'`로 15개 테이블 확인

- [ ] **Step 6: Commit**

```bash
git add db/
git commit -m "feat: add stock_db_v2 schema and init script"
```

---

## Task 2: KIS API 공용 클라이언트 (인증 + 격리 스케줄 가드)

**Files:**
- Create: `data_collection/kis_client.py`
- Test: `data_collection/test_kis_client.py`

**Interfaces:**
- Produces: `KisClient(app_key: str, app_secret: str, base_url: str = "https://openapi.koreainvestment.com:9443")` with `.get_token() -> str`, `.request(path: str, tr_id: str, params: dict) -> dict`
- Produces: `assert_outside_production_window(now: datetime | None = None) -> None` — raises `RuntimeError` if called during 08:55~15:30, 15:50~16:00, 16:00~16:10 on a weekday

**Interfaces (used by Task 4-8 backfill scripts):**
- All backfill scripts import and call `assert_outside_production_window()` before making any KIS API call.

- [ ] **Step 1: 실패하는 테스트 작성 (스케줄 가드)**

`data_collection/test_kis_client.py`:

```python
from datetime import datetime
import pytest
from data_collection.kis_client import assert_outside_production_window


def test_raises_during_realtime_collection_window():
    weekday_0955 = datetime(2026, 9, 9, 9, 55)  # 수요일
    with pytest.raises(RuntimeError, match="production collection window"):
        assert_outside_production_window(weekday_0955)


def test_raises_during_kis_collector_window():
    weekday_1555 = datetime(2026, 9, 9, 15, 55)
    with pytest.raises(RuntimeError, match="production collection window"):
        assert_outside_production_window(weekday_1555)


def test_raises_during_batch_collector_window():
    weekday_1605 = datetime(2026, 9, 9, 16, 5)
    with pytest.raises(RuntimeError, match="production collection window"):
        assert_outside_production_window(weekday_1605)


def test_allows_evening_window():
    weekday_2000 = datetime(2026, 9, 9, 20, 0)
    assert_outside_production_window(weekday_2000)  # no raise


def test_allows_weekend():
    saturday_1000 = datetime(2026, 9, 12, 10, 0)
    assert_outside_production_window(saturday_1000)  # no raise
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest data_collection/test_kis_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_collection.kis_client'`

- [ ] **Step 3: 구현**

`data_collection/kis_client.py`:

```python
import time
from datetime import datetime, time as dtime
from typing import Optional
import requests

# 기존 운영 수집기가 KIS API를 사용 중인 시간대(장중 실시간, collector_kis, collector_batch) —
# 이 구간엔 신규 백필 스크립트를 절대 실행하지 않는다 (토큰 충돌 방지, 설계 §9.1 참고)
PRODUCTION_WINDOWS = [
    (dtime(8, 55), dtime(15, 30)),
    (dtime(15, 50), dtime(16, 0)),
    (dtime(16, 0), dtime(16, 10)),
]


def assert_outside_production_window(now: Optional[datetime] = None) -> None:
    now = now or datetime.now()
    if now.weekday() >= 5:  # 토(5)/일(6)
        return
    t = now.time()
    for start, end in PRODUCTION_WINDOWS:
        if start <= t < end:
            raise RuntimeError(
                f"{now.time()} falls inside a production collection window "
                f"({start}-{end}) — reschedule this backfill outside market hours."
            )


class KisClient:
    def __init__(self, app_key: str, app_secret: str,
                 base_url: str = "https://openapi.koreainvestment.com:9443"):
        self.app_key = app_key
        self.app_secret = app_secret
        self.base_url = base_url
        self._token: Optional[str] = None
        self._token_issued_at: float = 0.0

    def get_token(self) -> str:
        if self._token and (time.time() - self._token_issued_at) < 23 * 3600:
            return self._token
        resp = requests.post(
            f"{self.base_url}/oauth2/tokenP",
            json={"grant_type": "client_credentials",
                  "appkey": self.app_key, "appsecret": self.app_secret},
            timeout=10,
        )
        resp.raise_for_status()
        self._token = resp.json()["access_token"]
        self._token_issued_at = time.time()
        return self._token

    def request(self, path: str, tr_id: str, params: dict) -> dict:
        assert_outside_production_window()
        headers = {
            "authorization": f"Bearer {self.get_token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
        }
        resp = requests.get(f"{self.base_url}{path}", headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest data_collection/test_kis_client.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add data_collection/kis_client.py data_collection/test_kis_client.py
git commit -m "feat: add shared KIS API client with production-window guard"
```

---

## Task 3: 후보 유니버스 확보 + 2019-01-02 시총 스냅샷 → 상위 200 확정

**Files:**
- Create: `data_collection/universe.py`
- Test: `data_collection/test_universe.py`

**Interfaces:**
- Consumes: `KisClient` (Task 2)
- Produces: `compute_top_n_snapshot(candidates: list[dict], n: int = 200) -> list[dict]` where each candidate dict has `{"ticker": str, "close_price": float, "shares_outstanding": int, "is_kospi": bool}`, returns candidates sorted by `market_cap` desc with `rank` and `market_cap` added, truncated to top `n`
- Produces: `fetch_snapshot_prices(client: KisClient, tickers: list[str], snapshot_date: str) -> list[dict]` (calls KIS daily price API `FHKST03010100` for `snapshot_date` only)
- Produces: `save_universe(dsn: str, universe: list[dict], snapshot_date: str) -> None` (writes to `ticker_universe` table, Task 1)

**Note (simplification, documented per spec §5.1):** 후보 풀은 원본 캡스톤 레포의 `KOSPI200_종목리스트.csv` + `KOSDAQ150_종목리스트.csv`(350종목)를 재사용한다. 전체 KRX 상장종목이 아니라 이미 대형/중형주로 좁혀진 리스트라 완전한 시총 랭킹은 아니지만, 삼성전자·SK하이닉스를 포함한 상위 200이 이 풀 안에 있을 것이 명백하므로 시간 제약상 이 단순화를 채택한다.

- [ ] **Step 1: 실패하는 테스트 작성 (순수 함수 — 랭킹 로직)**

`data_collection/test_universe.py`:

```python
from data_collection.universe import compute_top_n_snapshot


def test_ranks_by_market_cap_descending():
    candidates = [
        {"ticker": "AAA", "close_price": 100.0, "shares_outstanding": 10, "is_kospi": True},   # cap 1000
        {"ticker": "BBB", "close_price": 50.0, "shares_outstanding": 100, "is_kospi": True},    # cap 5000
        {"ticker": "CCC", "close_price": 10.0, "shares_outstanding": 10, "is_kospi": False},    # cap 100
    ]
    result = compute_top_n_snapshot(candidates, n=2)
    assert [c["ticker"] for c in result] == ["BBB", "AAA"]
    assert result[0]["rank"] == 1 and result[0]["market_cap"] == 5000.0
    assert result[1]["rank"] == 2 and result[1]["market_cap"] == 1000.0


def test_truncates_to_n():
    candidates = [
        {"ticker": f"T{i}", "close_price": float(i), "shares_outstanding": 1, "is_kospi": True}
        for i in range(10)
    ]
    result = compute_top_n_snapshot(candidates, n=3)
    assert len(result) == 3
    assert [c["ticker"] for c in result] == ["T9", "T8", "T7"]


def test_excludes_tickers_missing_price_or_shares():
    candidates = [
        {"ticker": "OK", "close_price": 10.0, "shares_outstanding": 10, "is_kospi": True},
        {"ticker": "NO_PRICE", "close_price": None, "shares_outstanding": 10, "is_kospi": True},
        {"ticker": "NO_SHARES", "close_price": 10.0, "shares_outstanding": None, "is_kospi": True},
    ]
    result = compute_top_n_snapshot(candidates, n=10)
    assert [c["ticker"] for c in result] == ["OK"]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest data_collection/test_universe.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: 구현**

`data_collection/universe.py`:

```python
import csv
from pathlib import Path
from typing import Optional
import psycopg2
from data_collection.kis_client import KisClient


def load_candidate_tickers(kospi_csv: str, kosdaq_csv: str) -> list[dict]:
    candidates = []
    for path, is_kospi in [(kospi_csv, True), (kosdaq_csv, False)]:
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                candidates.append({"ticker": row["ticker"].zfill(6), "is_kospi": is_kospi})
    return candidates


def fetch_snapshot_prices(client: KisClient, tickers: list[str], snapshot_date: str) -> list[dict]:
    results = []
    for ticker in tickers:
        data = client.request(
            path="/uapi/domestic-stock/v1/quotations/inquire-daily-price",
            tr_id="FHKST03010100",
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker,
                    "FID_INPUT_DATE_1": snapshot_date, "FID_INPUT_DATE_2": snapshot_date,
                    "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "1"},
        )
        rows = data.get("output2", [])
        if not rows:
            continue
        row = rows[0]
        results.append({
            "ticker": ticker,
            "close_price": float(row.get("stck_clpr") or 0) or None,
            "shares_outstanding": int(row.get("lstn_stcn") or 0) or None,
        })
    return results


def compute_top_n_snapshot(candidates: list[dict], n: int = 200) -> list[dict]:
    valid = [
        c for c in candidates
        if c.get("close_price") is not None and c.get("shares_outstanding") is not None
    ]
    for c in valid:
        c["market_cap"] = c["close_price"] * c["shares_outstanding"]
    ranked = sorted(valid, key=lambda c: c["market_cap"], reverse=True)[:n]
    for i, c in enumerate(ranked):
        c["rank"] = i + 1
    return ranked


def save_universe(dsn: str, universe: list[dict], snapshot_date: str) -> None:
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for c in universe:
            cur.execute(
                """INSERT INTO ticker_universe (ticker, snapshot_date, market_cap, rank, is_kospi)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (ticker) DO UPDATE SET
                     snapshot_date = EXCLUDED.snapshot_date, market_cap = EXCLUDED.market_cap,
                     rank = EXCLUDED.rank, is_kospi = EXCLUDED.is_kospi""",
                (c["ticker"], snapshot_date, c["market_cap"], c["rank"], c.get("is_kospi", True)),
            )
    conn.commit()
    conn.close()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest data_collection/test_universe.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: 실제 실행 (장외 시간에)**

```python
# data_collection/run_universe_snapshot.py 로 별도 실행 스크립트 작성 후:
# candidates = load_candidate_tickers("KOSPI200_종목리스트.csv", "KOSDAQ150_종목리스트.csv")
# prices = fetch_snapshot_prices(client, [c["ticker"] for c in candidates], "20190102")
# merged = 후보candidates와 prices를 ticker로 merge
# universe = compute_top_n_snapshot(merged, n=200)
# save_universe(dsn, universe, "2019-01-02")
```

Run: `/home/user/miniconda3/envs/kis_collector/bin/python data_collection/run_universe_snapshot.py` (장외 시간에)
Expected: `ticker_universe` 테이블에 200행 적재, `SELECT count(*) FROM ticker_universe` → 200

- [ ] **Step 6: Commit**

```bash
git add data_collection/universe.py data_collection/test_universe.py data_collection/run_universe_snapshot.py
git commit -m "feat: compute look-ahead-free top-200 market cap universe snapshot"
```

---

## Task 4: 시총 200종목 일봉 OHLCV 백필 (2019-01-02~현재)

**Files:**
- Create: `data_collection/backfill_daily_price.py`
- Test: `data_collection/test_backfill_daily_price.py`

**Interfaces:**
- Consumes: `KisClient` (Task 2), `ticker_universe` table (Task 3)
- Produces: `parse_daily_price_response(raw: dict, ticker: str) -> list[dict]` (순수 변환 함수 — 실제 API 호출 로직과 분리해 테스트 가능하게 함), `upsert_price_daily(dsn: str, rows: list[dict]) -> None`

- [ ] **Step 1: 실패하는 테스트 작성 (응답 파싱 순수 함수)**

`data_collection/test_backfill_daily_price.py`:

```python
from data_collection.backfill_daily_price import parse_daily_price_response


def test_parses_output2_rows_into_price_rows():
    raw = {
        "output2": [
            {"stck_bsop_date": "20190102", "stck_oprc": "45500", "stck_hgpr": "45650",
             "stck_lwpr": "45250", "stck_clpr": "45400", "acml_vol": "10000000",
             "acml_tr_pbmn": "454000000000", "lstn_stcn": "5969782550"},
        ]
    }
    rows = parse_daily_price_response(raw, ticker="005930")
    assert rows == [{
        "ticker": "005930", "trade_date": "2019-01-02",
        "open_price": 45500.0, "high_price": 45650.0, "low_price": 45250.0, "close_price": 45400.0,
        "volume": 10000000, "turnover": 454000000000.0, "shares_outstanding": 5969782550,
    }]


def test_skips_zero_volume_rows():
    raw = {"output2": [{"stck_bsop_date": "20190105", "stck_oprc": "0", "stck_hgpr": "0",
                          "stck_lwpr": "0", "stck_clpr": "0", "acml_vol": "0",
                          "acml_tr_pbmn": "0", "lstn_stcn": "5969782550"}]}
    assert parse_daily_price_response(raw, ticker="005930") == []


def test_returns_empty_list_for_missing_output2():
    assert parse_daily_price_response({}, ticker="005930") == []
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest data_collection/test_backfill_daily_price.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: 구현**

`data_collection/backfill_daily_price.py`:

```python
from datetime import datetime, timedelta
import psycopg2
from data_collection.kis_client import KisClient


def parse_daily_price_response(raw: dict, ticker: str) -> list[dict]:
    rows = []
    for r in raw.get("output2", []):
        volume = int(r.get("acml_vol") or 0)
        if volume == 0:
            continue
        d = r["stck_bsop_date"]
        rows.append({
            "ticker": ticker,
            "trade_date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
            "open_price": float(r["stck_oprc"]), "high_price": float(r["stck_hgpr"]),
            "low_price": float(r["stck_lwpr"]), "close_price": float(r["stck_clpr"]),
            "volume": volume, "turnover": float(r["acml_tr_pbmn"]),
            "shares_outstanding": int(r["lstn_stcn"]),
        })
    return rows


def upsert_price_daily(dsn: str, rows: list[dict]) -> None:
    if not rows:
        return
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """INSERT INTO price_daily
                   (ticker, trade_date, open_price, high_price, low_price, close_price,
                    volume, turnover, shares_outstanding)
                   VALUES (%(ticker)s, %(trade_date)s, %(open_price)s, %(high_price)s,
                           %(low_price)s, %(close_price)s, %(volume)s, %(turnover)s,
                           %(shares_outstanding)s)
                   ON CONFLICT (ticker, trade_date) DO UPDATE SET
                     open_price = EXCLUDED.open_price, high_price = EXCLUDED.high_price,
                     low_price = EXCLUDED.low_price, close_price = EXCLUDED.close_price,
                     volume = EXCLUDED.volume, turnover = EXCLUDED.turnover,
                     shares_outstanding = EXCLUDED.shares_outstanding""",
                r,
            )
    conn.commit()
    conn.close()


def backfill_ticker(client: KisClient, dsn: str, ticker: str,
                     start_date: str = "20190102", end_date: str | None = None) -> None:
    """KIS는 1회 조회당 최대 약 100영업일만 반환하므로 청크 단위로 역순 페이징한다."""
    end_date = end_date or datetime.now().strftime("%Y%m%d")
    cursor_end = datetime.strptime(end_date, "%Y%m%d")
    start = datetime.strptime(start_date, "%Y%m%d")
    while cursor_end >= start:
        cursor_start = max(start, cursor_end - timedelta(days=140))
        raw = client.request(
            path="/uapi/domestic-stock/v1/quotations/inquire-daily-price",
            tr_id="FHKST03010100",
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker,
                    "FID_INPUT_DATE_1": cursor_start.strftime("%Y%m%d"),
                    "FID_INPUT_DATE_2": cursor_end.strftime("%Y%m%d"),
                    "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "1"},
        )
        upsert_price_daily(dsn, parse_daily_price_response(raw, ticker))
        cursor_end = cursor_start - timedelta(days=1)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest data_collection/test_backfill_daily_price.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: 200종목 전체 백필 실행 (장외 시간, 시간 소요 큼 — nohup 권장)**

```python
# data_collection/run_daily_price_backfill.py:
# tickers = SELECT ticker FROM ticker_universe
# for ticker in tickers: backfill_ticker(client, dsn, ticker)  (KIS_API_INTERVAL 0.056초 간격 유지)
```

Run: `nohup /home/user/miniconda3/envs/kis_collector/bin/python data_collection/run_daily_price_backfill.py >> backfill_daily_price.log 2>&1 &` (저녁~새벽 시간대)
Expected: `SELECT ticker, count(*) FROM price_daily GROUP BY ticker`로 200종목 전부 약 1700+ 행(2019-01-02~현재 영업일수) 확인

- [ ] **Step 6: Commit**

```bash
git add data_collection/backfill_daily_price.py data_collection/test_backfill_daily_price.py data_collection/run_daily_price_backfill.py
git commit -m "feat: backfill daily OHLCV for top-200 universe (2019-01-02~present)"
```

---

## Task 5: 밸류에이션/수급/지수 백필

**Files:**
- Create: `data_collection/backfill_kis_fundamentals.py`
- Test: `data_collection/test_backfill_kis_fundamentals.py`

**Interfaces:**
- Consumes: `KisClient` (Task 2)
- Produces: `parse_valuation_response(raw: dict, ticker: str, trade_date: str) -> dict | None`, `parse_investor_flow_response(raw: dict, ticker: str, trade_date: str) -> dict | None`, `upsert_daily_valuation`, `upsert_investor_flow_daily`, `upsert_market_index_daily`

- [ ] **Step 1: 실패하는 테스트 작성**

`data_collection/test_backfill_kis_fundamentals.py`:

```python
from data_collection.backfill_kis_fundamentals import (
    parse_valuation_response, parse_investor_flow_response,
)


def test_parse_valuation_extracts_per_pbr_market_cap():
    raw = {"output": {"per": "12.5", "pbr": "1.8", "hts_avls": "450000"}}
    result = parse_valuation_response(raw, ticker="005930", trade_date="2019-01-02")
    assert result == {"ticker": "005930", "trade_date": "2019-01-02",
                        "per": 12.5, "pbr": 1.8, "market_cap": 450000.0 * 1_000_000}


def test_parse_valuation_returns_none_when_output_missing():
    assert parse_valuation_response({}, ticker="005930", trade_date="2019-01-02") is None


def test_parse_investor_flow_extracts_net_amounts():
    raw = {"output": [{"frgn_ntby_qty": "100", "frgn_ntby_tr_pbmn": "5000",
                         "orgn_ntby_tr_pbmn": "3000", "prsn_ntby_tr_pbmn": "-8000"}]}
    result = parse_investor_flow_response(raw, ticker="005930", trade_date="2019-01-02")
    assert result == {"ticker": "005930", "trade_date": "2019-01-02",
                        "individual_net_amt": -8000.0 * 1_000_000,
                        "foreign_net_amt": 5000.0 * 1_000_000,
                        "inst_net_amt": 3000.0 * 1_000_000}
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest data_collection/test_backfill_kis_fundamentals.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`data_collection/backfill_kis_fundamentals.py`:

```python
import psycopg2
from data_collection.kis_client import KisClient


def parse_valuation_response(raw: dict, ticker: str, trade_date: str) -> dict | None:
    output = raw.get("output")
    if not output:
        return None
    return {
        "ticker": ticker, "trade_date": trade_date,
        "per": float(output["per"]), "pbr": float(output["pbr"]),
        "market_cap": float(output["hts_avls"]) * 1_000_000,
    }


def parse_investor_flow_response(raw: dict, ticker: str, trade_date: str) -> dict | None:
    rows = raw.get("output")
    if not rows:
        return None
    r = rows[0]
    return {
        "ticker": ticker, "trade_date": trade_date,
        "individual_net_amt": float(r["prsn_ntby_tr_pbmn"]) * 1_000_000,
        "foreign_net_amt": float(r["frgn_ntby_tr_pbmn"]) * 1_000_000,
        "inst_net_amt": float(r["orgn_ntby_tr_pbmn"]) * 1_000_000,
    }


def upsert_daily_valuation(dsn: str, row: dict) -> None:
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO daily_valuation (ticker, trade_date, per, pbr, market_cap)
               VALUES (%(ticker)s, %(trade_date)s, %(per)s, %(pbr)s, %(market_cap)s)
               ON CONFLICT (ticker, trade_date) DO UPDATE SET
                 per = EXCLUDED.per, pbr = EXCLUDED.pbr, market_cap = EXCLUDED.market_cap""",
            row,
        )
    conn.commit()
    conn.close()


def upsert_investor_flow_daily(dsn: str, row: dict) -> None:
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO investor_flow_daily
               (ticker, trade_date, individual_net_amt, foreign_net_amt, inst_net_amt)
               VALUES (%(ticker)s, %(trade_date)s, %(individual_net_amt)s,
                       %(foreign_net_amt)s, %(inst_net_amt)s)
               ON CONFLICT (ticker, trade_date) DO UPDATE SET
                 individual_net_amt = EXCLUDED.individual_net_amt,
                 foreign_net_amt = EXCLUDED.foreign_net_amt, inst_net_amt = EXCLUDED.inst_net_amt""",
            row,
        )
    conn.commit()
    conn.close()


def upsert_market_index_daily(dsn: str, index_code: str, trade_date: str, close_price: float) -> None:
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO market_index_daily (index_code, trade_date, close_price)
               VALUES (%s, %s, %s)
               ON CONFLICT (index_code, trade_date) DO UPDATE SET close_price = EXCLUDED.close_price""",
            (index_code, trade_date, close_price),
        )
    conn.commit()
    conn.close()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest data_collection/test_backfill_kis_fundamentals.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: 200종목 × 2019~현재 전체 백필 실행 (장외 시간, `run_kis_fundamentals_backfill.py` 별도 작성 후 nohup 실행)**

Expected: `daily_valuation`, `investor_flow_daily`, `market_index_daily`(KOSPI `0001`/KOSDAQ `1001`) 테이블에 각 종목·거래일별 데이터 적재 확인

- [ ] **Step 6: Commit**

```bash
git add data_collection/backfill_kis_fundamentals.py data_collection/test_backfill_kis_fundamentals.py
git commit -m "feat: backfill valuation, investor flow, and index data"
```

---

## Task 6: 매크로 데이터 백필 (yfinance + FRED)

**Files:**
- Create: `data_collection/backfill_yf_fred.py`
- Test: `data_collection/test_backfill_yf_fred.py`

**Interfaces:**
- Produces: `shift_us_date_to_kr(us_date: str) -> str` (미국 날짜 +1일 → 한국 반영일, 데이터 누수 방지), `upsert_market_global(dsn: str, row: dict) -> None`

- [ ] **Step 1: 실패하는 테스트 작성**

`data_collection/test_backfill_yf_fred.py`:

```python
from data_collection.backfill_yf_fred import shift_us_date_to_kr


def test_shifts_date_forward_by_one_day():
    assert shift_us_date_to_kr("2026-09-04") == "2026-09-05"


def test_shifts_across_month_boundary():
    assert shift_us_date_to_kr("2026-08-31") == "2026-09-01"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest data_collection/test_backfill_yf_fred.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`data_collection/backfill_yf_fred.py`:

```python
from datetime import datetime, timedelta
import psycopg2
import yfinance as yf
import pandas_datareader.data as web

YF_SYMBOLS = {"^GSPC": "snp500_close", "^IXIC": "nasdaq_close", "^SOX": "phlx_semi_close",
              "^VIX": "vix", "CL=F": "wti_crude_oil", "GC=F": "gold_price"}
FRED_SERIES = {"DEXKOUS": "usd_krw", "DGS10": "us_10y_yield",
               "DFEDTARL": "fed_rate", "INTDSRKRM193N": "kr_base_rate"}


def shift_us_date_to_kr(us_date: str) -> str:
    d = datetime.strptime(us_date, "%Y-%m-%d")
    return (d + timedelta(days=1)).strftime("%Y-%m-%d")


def fetch_yfinance(start: str, end: str) -> dict:
    out = {}
    for symbol, column in YF_SYMBOLS.items():
        df = yf.download(symbol, start=start, end=end, progress=False)["Close"]
        for us_date, value in df.items():
            kr_date = shift_us_date_to_kr(us_date.strftime("%Y-%m-%d"))
            out.setdefault(kr_date, {})[column] = float(value)
    return out


def fetch_fred(start: str, end: str) -> dict:
    out = {}
    for series, column in FRED_SERIES.items():
        df = web.DataReader(series, "fred", start, end)[series].dropna()
        for us_date, value in df.items():
            kr_date = shift_us_date_to_kr(us_date.strftime("%Y-%m-%d"))
            out.setdefault(kr_date, {})[column] = float(value)
    return out


def upsert_market_global(dsn: str, trade_date: str, row: dict) -> None:
    if not row:
        return
    columns = ", ".join(row.keys())
    placeholders = ", ".join(f"%({k})s" for k in row.keys())
    updates = ", ".join(f"{k} = COALESCE(EXCLUDED.{k}, market_global.{k})" for k in row.keys())
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO market_global (trade_date, {columns})
                VALUES (%(trade_date)s, {placeholders})
                ON CONFLICT (trade_date) DO UPDATE SET {updates}""",
            {**row, "trade_date": trade_date},
        )
    conn.commit()
    conn.close()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest data_collection/test_backfill_yf_fred.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: 실행 (yfinance/FRED는 KIS API가 아니라 production window 가드 무관 — 아무 때나 실행 가능)**

```python
# data_collection/run_yf_fred_backfill.py:
# yf_data = fetch_yfinance("2019-01-01", today)
# fred_data = fetch_fred("2019-01-01", today)
# merge by date, ffill 결측치, 주말 제거 후 upsert_market_global 반복 호출
```

Run: `/home/user/miniconda3/envs/kis_collector/bin/python data_collection/run_yf_fred_backfill.py`
Expected: `market_global` 테이블에 2019~현재 영업일 전체 적재

- [ ] **Step 6: Commit**

```bash
git add data_collection/backfill_yf_fred.py data_collection/test_backfill_yf_fred.py
git commit -m "feat: backfill macro data from yfinance and FRED"
```

---

## Task 7: 공시/캘린더/섹터 백필

**Files:**
- Create: `data_collection/backfill_dart_calendar_sector.py`
- Test: `data_collection/test_backfill_dart_calendar_sector.py`

**Interfaces:**
- Consumes: `KisClient` (Task 2)
- Produces: `classify_dart_event(report_name: str) -> str | None` (키워드 → 이벤트 타입 매핑, 원본 `init_dart.py` 로직 계승), `build_calendar_rows(start: str, end: str, short_selling_ban_periods: list[tuple[str, str]]) -> list[dict]`, `parse_dart_reports(raw_reports: list[dict], ticker: str) -> list[dict]`, `upsert_stock_events(dsn: str, rows: list[dict]) -> None`, `parse_sector_daily_response(raw: dict, sector_code: str) -> list[dict]`, `upsert_sector_daily_ohlcv(dsn: str, rows: list[dict]) -> None`, `upsert_calendar(dsn: str, rows: list[dict]) -> None`

- [ ] **Step 1: 실패하는 테스트 작성**

`data_collection/test_backfill_dart_calendar_sector.py`:

```python
from data_collection.backfill_dart_calendar_sector import classify_dart_event, build_calendar_rows


def test_classifies_known_keywords():
    assert classify_dart_event("유상증자 결정") == "유상증자"
    assert classify_dart_event("무상증자 결정") == "무상증자"
    assert classify_dart_event("현금ㆍ현물배당 결정") == "배당"
    assert classify_dart_event("주식분할 결정") == "액면분할"
    assert classify_dart_event("연결재무제표기준영업(잠정)실적") == "실적발표"
    assert classify_dart_event("합병 결정") == "합병"


def test_returns_none_for_unrelated_report():
    assert classify_dart_event("최대주주변경") is None


def test_build_calendar_marks_weekends_as_closed():
    rows = build_calendar_rows("2019-01-01", "2019-01-07", short_selling_ban_periods=[])
    by_date = {r["base_date"]: r for r in rows}
    assert by_date["2019-01-05"]["is_market_open"] is False  # 토
    assert by_date["2019-01-06"]["is_market_open"] is False  # 일
    assert by_date["2019-01-02"]["is_market_open"] is True   # 수


def test_build_calendar_flags_short_selling_ban_period():
    rows = build_calendar_rows("2019-01-01", "2019-01-03",
                                short_selling_ban_periods=[("2019-01-01", "2019-01-03")])
    assert all(r["is_short_selling_banned"] for r in rows if r["day_of_week"] < 5)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest data_collection/test_backfill_dart_calendar_sector.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`data_collection/backfill_dart_calendar_sector.py`:

```python
from datetime import datetime, timedelta

EVENT_KEYWORD_MAP = {
    "유상증자": "유상증자", "무상증자": "무상증자", "배당": "배당",
    "주식분할": "액면분할", "실적": "실적발표", "잠정": "실적발표",
    "매출액또는손익": "실적발표", "합병": "합병",
}


def classify_dart_event(report_name: str) -> str | None:
    for keyword, event_type in EVENT_KEYWORD_MAP.items():
        if keyword in report_name:
            return event_type
    return None


def build_calendar_rows(start: str, end: str,
                         short_selling_ban_periods: list[tuple[str, str]]) -> list[dict]:
    rows = []
    d = datetime.strptime(start, "%Y-%m-%d")
    end_d = datetime.strptime(end, "%Y-%m-%d")
    bans = [(datetime.strptime(s, "%Y-%m-%d"), datetime.strptime(e, "%Y-%m-%d"))
            for s, e in short_selling_ban_periods]
    while d <= end_d:
        dow = d.weekday()
        is_open = dow < 5
        banned = any(s <= d <= e for s, e in bans)
        rows.append({
            "base_date": d.strftime("%Y-%m-%d"), "day_of_week": dow,
            "is_market_open": is_open, "is_holiday": not is_open,
            "is_short_selling_banned": banned,
        })
        d += timedelta(days=1)
    return rows
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest data_collection/test_backfill_dart_calendar_sector.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: 추가 실패하는 테스트 작성 (stock_events/sector_daily_ohlcv 파싱)**

`data_collection/test_backfill_dart_calendar_sector.py`에 추가:

```python
from data_collection.backfill_dart_calendar_sector import (
    parse_dart_reports, parse_sector_daily_response,
)


def test_parse_dart_reports_classifies_and_filters():
    raw_reports = [
        {"rcept_dt": "20190315", "report_nm": "유상증자 결정"},
        {"rcept_dt": "20190316", "report_nm": "최대주주변경"},  # 매핑 안 되는 건 제외
    ]
    rows = parse_dart_reports(raw_reports, ticker="005930")
    assert rows == [{"ticker": "005930", "event_date": "2019-03-15",
                       "event_type": "유상증자", "description": "유상증자 결정"}]


def test_parse_sector_daily_response_extracts_ohlcv():
    raw = {"output2": [{"stck_bsop_date": "20190102", "bstp_nmix_oprc": "1050.5",
                          "bstp_nmix_hgpr": "1055.0", "bstp_nmix_lwpr": "1048.0",
                          "bstp_nmix_prpr": "1052.0", "acml_vol": "500000"}]}
    rows = parse_sector_daily_response(raw, sector_code="0005")
    assert rows == [{"sector_code": "0005", "trade_date": "2019-01-02",
                       "open": 1050.5, "high": 1055.0, "low": 1048.0,
                       "close": 1052.0, "volume": 500000}]
```

- [ ] **Step 6: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest data_collection/test_backfill_dart_calendar_sector.py -v`
Expected: FAIL (2개 새 테스트, `parse_dart_reports`/`parse_sector_daily_response` 미정의)

- [ ] **Step 7: 구현 — `stock_events`/`sector_daily_ohlcv`/`calendar`/`market_events` upsert 함수 추가**

`data_collection/backfill_dart_calendar_sector.py`에 추가:

```python
import psycopg2
from data_collection.kis_client import KisClient


def parse_dart_reports(raw_reports: list[dict], ticker: str) -> list[dict]:
    rows = []
    for r in raw_reports:
        event_type = classify_dart_event(r["report_nm"])
        if event_type is None:
            continue
        d = r["rcept_dt"]
        rows.append({"ticker": ticker, "event_date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
                      "event_type": event_type, "description": r["report_nm"]})
    return rows


def upsert_stock_events(dsn: str, rows: list[dict]) -> None:
    if not rows:
        return
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """INSERT INTO stock_events (ticker, event_date, event_type, description)
                   VALUES (%(ticker)s, %(event_date)s, %(event_type)s, %(description)s)
                   ON CONFLICT (ticker, event_date, event_type) DO NOTHING""",
                r,
            )
    conn.commit()
    conn.close()


def parse_sector_daily_response(raw: dict, sector_code: str) -> list[dict]:
    rows = []
    for r in raw.get("output2", []):
        d = r["stck_bsop_date"]
        rows.append({
            "sector_code": sector_code, "trade_date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
            "open": float(r["bstp_nmix_oprc"]), "high": float(r["bstp_nmix_hgpr"]),
            "low": float(r["bstp_nmix_lwpr"]), "close": float(r["bstp_nmix_prpr"]),
            "volume": int(r["acml_vol"]),
        })
    return rows


def upsert_sector_daily_ohlcv(dsn: str, rows: list[dict]) -> None:
    if not rows:
        return
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """INSERT INTO sector_daily_ohlcv (sector_code, trade_date, open, high, low, close, volume)
                   VALUES (%(sector_code)s, %(trade_date)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s)
                   ON CONFLICT (sector_code, trade_date) DO UPDATE SET
                     open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                     close = EXCLUDED.close, volume = EXCLUDED.volume""",
                r,
            )
    conn.commit()
    conn.close()


def upsert_calendar(dsn: str, rows: list[dict]) -> None:
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """INSERT INTO calendar (base_date, day_of_week, is_market_open, is_holiday, is_short_selling_banned)
                   VALUES (%(base_date)s, %(day_of_week)s, %(is_market_open)s, %(is_holiday)s, %(is_short_selling_banned)s)
                   ON CONFLICT (base_date) DO UPDATE SET
                     day_of_week = EXCLUDED.day_of_week, is_market_open = EXCLUDED.is_market_open,
                     is_holiday = EXCLUDED.is_holiday, is_short_selling_banned = EXCLUDED.is_short_selling_banned""",
                r,
            )
    conn.commit()
    conn.close()
```

- [ ] **Step 8: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest data_collection/test_backfill_dart_calendar_sector.py -v`
Expected: PASS (6 tests)

- [ ] **Step 9: 실행** — DART(`OpenDartReader`, KIS 아님)로 200종목 2019~현재 공시 수집 → `parse_dart_reports` → `upsert_stock_events`, 캘린더(`build_calendar_rows` → `upsert_calendar`, 공휴일 라이브러리 + 공매도 금지 기간 하드코딩: 2020-03-16~2021-05-02, 2023-11-06~2099-12-31, `market_events`도 동일 패턴으로 BOK/FOMC/위칭데이 하드코딩 일정 upsert), 섹터 일봉(KIS `FHKUP03500100`, 25개 섹터 코드 × 2019~현재, `parse_sector_daily_response` → `upsert_sector_daily_ohlcv`, 장외 시간)을 각각 `run_dart_calendar_sector_backfill.py`로 실행

Expected: `stock_events`, `calendar`, `market_events`, `sector_daily_ohlcv` 적재 확인

- [ ] **Step 6: Commit**

```bash
git add data_collection/backfill_dart_calendar_sector.py data_collection/test_backfill_dart_calendar_sector.py
git commit -m "feat: backfill DART events, trading calendar, and sector OHLCV"
```

---

## Task 8: 레버리지 ETF/ETN + VI 이벤트 백필

**Files:**
- Create: `data_collection/leverage_products.py`
- Create: `data_collection/backfill_leverage.py`
- Test: `data_collection/test_leverage_products.py`
- Test: `data_collection/test_backfill_leverage.py`

**Interfaces:**
- Produces: `LEVERAGE_PRODUCTS: list[dict]` (설계 §6.1 표, ETN 2종 코드는 `None`으로 두고 TODO 주석 — 구현 시 KRX 정보데이터시스템에서 확인 후 채움)
- Produces: `parse_vi_event_response(raw: dict, ticker: str) -> list[dict]`
- Produces: `save_leverage_products(dsn: str) -> None`, `upsert_leverage_daily(dsn: str, rows: list[dict]) -> None`, `upsert_vi_events(dsn: str, rows: list[dict]) -> None`, `parse_leverage_daily_response(raw: dict, code: str) -> dict | None` (Step 7에서 구현, 정확한 필드/배율은 Step 6의 수동 단위 검증 결과에 따름 — 이 시점에 미리 확정하지 않음)

- [ ] **Step 1: 실패하는 테스트 작성**

`data_collection/test_leverage_products.py`:

```python
from data_collection.leverage_products import LEVERAGE_PRODUCTS


def test_has_16_confirmed_etfs_and_2_pending_etns():
    etfs = [p for p in LEVERAGE_PRODUCTS if p["product_type"] == "ETF"]
    etns = [p for p in LEVERAGE_PRODUCTS if p["product_type"] == "ETN"]
    assert len(etfs) == 16
    assert len(etns) == 2
    assert all(p["code"] is not None for p in etfs)


def test_samsung_and_hynix_each_have_one_inverse_product():
    samsung_inverse = [p for p in LEVERAGE_PRODUCTS
                        if p["underlying_ticker"] == "005930" and p["multiple"] < 0]
    hynix_inverse = [p for p in LEVERAGE_PRODUCTS
                      if p["underlying_ticker"] == "000660" and p["multiple"] < 0]
    assert len(samsung_inverse) == 1
    assert len(hynix_inverse) == 1


def test_all_products_listed_2026_05_27():
    assert all(p["listed_date"] == "2026-05-27" for p in LEVERAGE_PRODUCTS)
```

`data_collection/test_backfill_leverage.py`:

```python
from data_collection.backfill_leverage import parse_vi_event_response


def test_parses_vi_trigger_and_release():
    raw = {"output": [{"vi_bsop_time": "20260602103000", "vi_rls_time": "20260602104500",
                         "vi_type_cd": "1"}]}
    result = parse_vi_event_response(raw, ticker="005930")
    assert result == [{"ticker": "005930", "triggered_at": "2026-06-02 10:30:00",
                         "released_at": "2026-06-02 10:45:00", "vi_type": "1"}]


def test_release_none_when_still_active():
    raw = {"output": [{"vi_bsop_time": "20260602103000", "vi_rls_time": "", "vi_type_cd": "1"}]}
    result = parse_vi_event_response(raw, ticker="005930")
    assert result[0]["released_at"] is None
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest data_collection/test_leverage_products.py data_collection/test_backfill_leverage.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`data_collection/leverage_products.py` (설계 §6.1 표 그대로 하드코딩 — 이 리스트는 18종 고정값이라 API로 조회할 필요 없음):

```python
LEVERAGE_PRODUCTS = [
    {"code": "0193W0", "product_name": "KODEX 삼성전자단일종목레버리지", "underlying_ticker": "005930", "multiple": 2.0, "product_type": "ETF", "issuer": "Samsung Asset", "listed_date": "2026-05-27"},
    {"code": "0195R0", "product_name": "TIGER 삼성전자단일종목레버리지", "underlying_ticker": "005930", "multiple": 2.0, "product_type": "ETF", "issuer": "Mirae Asset", "listed_date": "2026-05-27"},
    {"code": "0194M0", "product_name": "ACE 삼성전자단일종목레버리지", "underlying_ticker": "005930", "multiple": 2.0, "product_type": "ETF", "issuer": "Korea Investment", "listed_date": "2026-05-27"},
    {"code": "0192M0", "product_name": "RISE 삼성전자단일종목레버리지", "underlying_ticker": "005930", "multiple": 2.0, "product_type": "ETF", "issuer": "KB Asset", "listed_date": "2026-05-27"},
    {"code": "0193K0", "product_name": "PLUS 삼성전자단일종목레버리지", "underlying_ticker": "005930", "multiple": 2.0, "product_type": "ETF", "issuer": "Hanwha Asset", "listed_date": "2026-05-27"},
    {"code": "0194N0", "product_name": "KIWOOM 삼성전자선물단일종목레버리지", "underlying_ticker": "005930", "multiple": 2.0, "product_type": "ETF", "issuer": "Kiwoom Asset", "listed_date": "2026-05-27"},
    {"code": "0198B0", "product_name": "1Q 삼성전자선물단일종목레버리지", "underlying_ticker": "005930", "multiple": 2.0, "product_type": "ETF", "issuer": "Hana Asset", "listed_date": "2026-05-27"},
    {"code": "0193L0", "product_name": "PLUS 삼성전자선물단일종목인버스2X", "underlying_ticker": "005930", "multiple": -2.0, "product_type": "ETF", "issuer": "Hanwha Asset", "listed_date": "2026-05-27"},
    {"code": "0193T0", "product_name": "KODEX SK하이닉스단일종목레버리지", "underlying_ticker": "000660", "multiple": 2.0, "product_type": "ETF", "issuer": "Samsung Asset", "listed_date": "2026-05-27"},
    {"code": "0195S0", "product_name": "TIGER SK하이닉스단일종목레버리지", "underlying_ticker": "000660", "multiple": 2.0, "product_type": "ETF", "issuer": "Mirae Asset", "listed_date": "2026-05-27"},
    {"code": "0194T0", "product_name": "ACE SK하이닉스단일종목레버리지", "underlying_ticker": "000660", "multiple": 2.0, "product_type": "ETF", "issuer": "Korea Investment", "listed_date": "2026-05-27"},
    {"code": "0192L0", "product_name": "RISE SK하이닉스단일종목레버리지", "underlying_ticker": "000660", "multiple": 2.0, "product_type": "ETF", "issuer": "KB Asset", "listed_date": "2026-05-27"},
    {"code": "0197W0", "product_name": "SOL SK하이닉스단일종목레버리지", "underlying_ticker": "000660", "multiple": 2.0, "product_type": "ETF", "issuer": "Shinhan Asset", "listed_date": "2026-05-27"},
    {"code": "0194R0", "product_name": "KIWOOM SK하이닉스선물단일종목레버리지", "underlying_ticker": "000660", "multiple": 2.0, "product_type": "ETF", "issuer": "Kiwoom Asset", "listed_date": "2026-05-27"},
    {"code": "0198D0", "product_name": "1Q SK하이닉스선물단일종목레버리지", "underlying_ticker": "000660", "multiple": 2.0, "product_type": "ETF", "issuer": "Hana Asset", "listed_date": "2026-05-27"},
    {"code": "0197X0", "product_name": "SOL SK하이닉스선물단일종목인버스2X", "underlying_ticker": "000660", "multiple": -2.0, "product_type": "ETF", "issuer": "Shinhan Asset", "listed_date": "2026-05-27"},
    # TODO(구현 단계): KRX 정보데이터시스템 또는 KIS 종목마스터파일로 정확한 코드 확인 후 채울 것 (설계 §6.1, §14)
    {"code": None, "product_name": "TIGER 삼성전자레버리지", "underlying_ticker": "005930", "multiple": 2.0, "product_type": "ETN", "issuer": "Mirae Asset", "listed_date": "2026-05-27"},
    {"code": None, "product_name": "TIGER SK하이닉스레버리지", "underlying_ticker": "000660", "multiple": 2.0, "product_type": "ETN", "issuer": "Mirae Asset", "listed_date": "2026-05-27"},
]
```

`data_collection/backfill_leverage.py`:

```python
import psycopg2
from data_collection.kis_client import KisClient
from data_collection.leverage_products import LEVERAGE_PRODUCTS


def save_leverage_products(dsn: str) -> None:
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for p in LEVERAGE_PRODUCTS:
            if p["code"] is None:
                continue  # ETN 코드 미확인 — Step 6 이후 채워지면 재실행
            cur.execute(
                """INSERT INTO leverage_products
                   (code, product_name, underlying_ticker, multiple, product_type, issuer, listed_date)
                   VALUES (%(code)s, %(product_name)s, %(underlying_ticker)s, %(multiple)s,
                           %(product_type)s, %(issuer)s, %(listed_date)s)
                   ON CONFLICT (code) DO NOTHING""",
                p,
            )
    conn.commit()
    conn.close()


def upsert_leverage_daily(dsn: str, rows: list[dict]) -> None:
    if not rows:
        return
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """INSERT INTO leverage_daily (code, trade_date, close_price, volume, turnover, nav, aum)
                   VALUES (%(code)s, %(trade_date)s, %(close_price)s, %(volume)s,
                           %(turnover)s, %(nav)s, %(aum)s)
                   ON CONFLICT (code, trade_date) DO UPDATE SET
                     close_price = EXCLUDED.close_price, volume = EXCLUDED.volume,
                     turnover = EXCLUDED.turnover, nav = EXCLUDED.nav, aum = EXCLUDED.aum""",
                r,
            )
    conn.commit()
    conn.close()


def parse_vi_event_response(raw: dict, ticker: str) -> list[dict]:
    results = []
    for r in raw.get("output", []):
        t = r["vi_bsop_time"]
        triggered = f"{t[:4]}-{t[4:6]}-{t[6:8]} {t[8:10]}:{t[10:12]}:{t[12:]}"
        released = None
        if r.get("vi_rls_time"):
            rt = r["vi_rls_time"]
            released = f"{rt[:4]}-{rt[4:6]}-{rt[6:8]} {rt[8:10]}:{rt[10:12]}:{rt[12:]}"
        results.append({"ticker": ticker, "triggered_at": triggered,
                          "released_at": released, "vi_type": r["vi_type_cd"]})
    return results


def upsert_vi_events(dsn: str, rows: list[dict]) -> None:
    if not rows:
        return
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """INSERT INTO vi_events (ticker, triggered_at, released_at, vi_type)
                   VALUES (%(ticker)s, %(triggered_at)s, %(released_at)s, %(vi_type)s)
                   ON CONFLICT (ticker, triggered_at) DO UPDATE SET
                     released_at = EXCLUDED.released_at, vi_type = EXCLUDED.vi_type""",
                r,
            )
    conn.commit()
    conn.close()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest data_collection/test_leverage_products.py data_collection/test_backfill_leverage.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: ETN 코드 확인**

KRX 정보데이터시스템(data.krx.co.kr)에서 "TIGER 삼성전자레버리지"/"TIGER SK하이닉스레버리지" 종목코드 조회 → `leverage_products.py`의 `code: None` 두 곳 채우기 → 위 5개 테스트 재실행(PASS 유지 확인) → `save_leverage_products(dsn)` 실행

- [ ] **Step 6: 단위 검증(중요) — 대량 백필 전에 반드시 1종목 1일치로 먼저 확인**

KIS ETF/ETN 현재가+NAV API 응답의 가격·거래량·NAV·AUM 필드가 실제로 어떤 단위(원 vs 백만원 vs 좌 단위 등)로 오는지 이 시점까지 확인된 바 없음 — `daily_valuation`(Task 5)에서 `hts_avls`가 "백만원 단위"라 `×1,000,000`이 필요했던 것과 같은 함정이 이 API에도 있을 수 있음. 대량 백필 전에 반드시 손으로 확인한다.

```python
# 삼성전자 레버리지 ETF 1종(코드 0193W0) 1영업일치(예: 2026-06-01)만 조회
raw = client.request(path="/uapi/etfetn/v1/quotations/inquire-price",  # 정확한 path/tr_id는 KIS Developers 포털에서 재확인
                       tr_id="FHPST02400000", params={...})
print(raw)
```

체크리스트(전부 통과해야 다음 단계 진행):
1. 가격(`close_price`)이 상식적인 ETF 가격대(수백~수만 원)인지 — 원 단위가 아니라 다른 배율로 와있으면 자릿수가 이상하게 튐
2. `nav`가 `close_price`와 비슷한 자릿수인지(NAV와 시장가는 보통 크게 안 벌어짐)
3. `aum`이 `nav × 상장좌수` 근사치와 맞아떨어지는지 — 상장좌수는 종목마스터파일이나 별도 조회로 확인
4. 위 3개 중 하나라도 예상과 다른 자릿수면, 실제 KIS 응답 필드 원본을 그대로 보고 올바른 배율을 역산해서 `parse_leverage_daily_response()`(Step 7에서 구현)에 반영

이 확인 결과(사용한 정확한 배율과 그 근거)를 리포트에 남길 것.

- [ ] **Step 7: 확인된 단위로 파싱 함수 구현 + 실행**

Step 6에서 확인한 배율을 반영해 `parse_leverage_daily_response(raw: dict, code: str) -> dict | None`을 구현(패턴은 Task 5의 `parse_valuation_response`와 동일하게 딕셔너리 반환 + `upsert_leverage_daily` 호출). `run_leverage_backfill.py`로 18종 × 2026-05-27~현재 가격/거래량/AUM/NAV 백필 + 삼성전자·SK하이닉스 VI 이력(KIS VI 현황 API, 2019~현재) 백필 실행(장외 시간)

Expected: `leverage_products` 18행, `leverage_daily` 18종×영업일수, `vi_events`에 2026-05-27 이후 급증 확인. 백필 직후 `SELECT code, avg(close_price), avg(aum) FROM leverage_daily GROUP BY code`로 Step 6 체크리스트 자릿수가 전체 데이터에서도 유지되는지 재확인

- [ ] **Step 8: Commit**

```bash
git add data_collection/leverage_products.py data_collection/backfill_leverage.py data_collection/test_leverage_products.py data_collection/test_backfill_leverage.py
git commit -m "feat: backfill leverage ETF/ETN and VI event data"
```

---

## Task 9: 라벨 생성 + threshold 역산

**Files:**
- Create: `training/label.py`
- Test: `training/test_label.py`

**Interfaces:**
- Produces: `compute_next_day_return(close_prices: list[float]) -> list[Optional[float]]` (마지막 행은 None), `derive_threshold(returns: list[float], target_hold_ratio: float = 0.5) -> float`, `assign_label(returns: list[float], threshold: float) -> list[Optional[int]]` (0=매수, 1=관망, 2=매도), `build_label_rows(ticker: str, trade_dates: list[str], close_prices: list[float], threshold: float) -> list[dict]`, `upsert_labels(dsn: str, rows: list[dict]) -> None` (Task 1의 `labels` 테이블에 저장 — Task 10이 이 테이블을 `feature_pool`에 조인해서 최종 `label` 컬럼을 만든다)

- [ ] **Step 1: 실패하는 테스트 작성**

`training/test_label.py`:

```python
import pytest
from training.label import compute_next_day_return, derive_threshold, assign_label


def test_compute_next_day_return_shifts_forward_one_day():
    closes = [100.0, 110.0, 99.0, None]
    returns = compute_next_day_return(closes)
    assert returns[0] == pytest.approx(0.10)
    assert returns[1] == pytest.approx((99.0 - 110.0) / 110.0)
    assert returns[2] is None  # 마지막 날은 다음날 데이터 없음
    assert returns[3] is None


def test_derive_threshold_targets_hold_ratio():
    # -0.02~0.02 안에 정확히 50%가 들어가도록 구성
    returns = [-0.10, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10, -0.5]
    threshold = derive_threshold(returns, target_hold_ratio=0.5)
    hold_count = sum(1 for r in returns if abs(r) < threshold)
    assert hold_count / len(returns) == pytest.approx(0.5, abs=0.15)


def test_assign_label_buy_hold_sell():
    labels = assign_label([0.02, 0.001, -0.02, None], threshold=0.015)
    assert labels == [0, 1, 2, None]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest training/test_label.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`training/label.py`:

```python
from typing import Optional
import numpy as np


def compute_next_day_return(close_prices: list[float]) -> list[Optional[float]]:
    returns: list[Optional[float]] = []
    for i in range(len(close_prices)):
        if i + 1 >= len(close_prices) or close_prices[i + 1] is None or close_prices[i] is None:
            returns.append(None)
        else:
            returns.append((close_prices[i + 1] - close_prices[i]) / close_prices[i])
    return returns


def derive_threshold(returns: list[float], target_hold_ratio: float = 0.5) -> float:
    """|return| 분포에서, 절댓값 기준으로 target_hold_ratio 분위수를 threshold로 삼는다.
    즉 관망 비율이 정확히 target_hold_ratio가 되는 |r| 값을 반환."""
    abs_returns = np.abs(np.array([r for r in returns if r is not None]))
    return float(np.quantile(abs_returns, target_hold_ratio))


def assign_label(returns: list[Optional[float]], threshold: float) -> list[Optional[int]]:
    labels: list[Optional[int]] = []
    for r in returns:
        if r is None:
            labels.append(None)
        elif r >= threshold:
            labels.append(0)  # 매수
        elif r <= -threshold:
            labels.append(2)  # 매도
        else:
            labels.append(1)  # 관망
    return labels
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest training/test_label.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: 실패하는 테스트 작성 (labels 테이블 upsert)**

`training/test_label.py`에 추가:

```python
from training.label import build_label_rows


def test_build_label_rows_pairs_ticker_dates_with_labels():
    dates = ["2019-01-02", "2019-01-03", "2019-01-04"]
    closes = [100.0, 110.0, 99.0]
    rows = build_label_rows(ticker="005930", trade_dates=dates, close_prices=closes, threshold=0.015)
    assert rows[0] == {"ticker": "005930", "trade_date": "2019-01-02",
                         "next_day_return": pytest.approx(0.10), "label": 0}
    assert rows[2] == {"ticker": "005930", "trade_date": "2019-01-04",
                         "next_day_return": None, "label": None}
```

(파일 상단에 `import pytest` 추가 필요)

- [ ] **Step 6: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest training/test_label.py -v`
Expected: FAIL (`build_label_rows` 미정의)

- [ ] **Step 7: 구현**

`training/label.py`에 추가:

```python
import psycopg2


def build_label_rows(ticker: str, trade_dates: list[str], close_prices: list[float],
                      threshold: float) -> list[dict]:
    returns = compute_next_day_return(close_prices)
    labels = assign_label(returns, threshold)
    return [
        {"ticker": ticker, "trade_date": d, "next_day_return": r, "label": l}
        for d, r, l in zip(trade_dates, returns, labels)
    ]


def upsert_labels(dsn: str, rows: list[dict]) -> None:
    if not rows:
        return
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """INSERT INTO labels (ticker, trade_date, next_day_return, label)
                   VALUES (%(ticker)s, %(trade_date)s, %(next_day_return)s, %(label)s)
                   ON CONFLICT (ticker, trade_date) DO UPDATE SET
                     next_day_return = EXCLUDED.next_day_return, label = EXCLUDED.label""",
                r,
            )
    conn.commit()
    conn.close()
```

- [ ] **Step 8: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest training/test_label.py -v`
Expected: PASS (5 tests)

- [ ] **Step 9: 실제 threshold 계산 + 전체 기간 라벨 저장**

```python
# training/run_derive_threshold.py:
# 1) 1단계 train 구간(2019-01-02~2023-12-31)의 시총200 전체 price_daily.close_price로
#    compute_next_day_return → derive_threshold(target_hold_ratio=0.5) 실행 후 결과를
#    training/threshold.json 에 저장 ({"threshold": <값>, "computed_on": "1단계 train, 2019-2023"})
# 2) 이 threshold로, 시총200 전체 종목의 전체 기간(2019-01-02~현재, 1단계/2단계 공용)에 대해
#    종목별 build_label_rows(ticker, trade_dates, close_prices, threshold) 호출 후
#    upsert_labels(dsn, rows)로 `labels` 테이블에 저장 — Task 10의 feature_pool 조인이 이 테이블을 사용한다
```

Run: `/home/user/miniconda3/envs/kis_collector/bin/python training/run_derive_threshold.py`
Expected: `training/threshold.json` 생성(관망 비율 45~55% 확인), `labels` 테이블에 200종목 × 전체 기간 적재(`SELECT count(*) FROM labels WHERE label IS NOT NULL`로 확인)

- [ ] **Step 10: Commit**

```bash
git add training/label.py training/test_label.py training/run_derive_threshold.py
git commit -m "feat: add next-day return labeling with data-derived threshold, persist to labels table"
```

---

## Task 10: 피처 조인(`feature_pool`) + 레버리지 파생 피처 계산 + 데이터 정합성 검증

**Files:**
- Create: `features/leverage_features.py`
- Create: `features/build_features.py`
- Create: `features/validate_feature_pool.py`
- Test: `features/test_leverage_features.py`
- Test: `features/test_validate_feature_pool.py`

**Interfaces:**
- Consumes: `LEVERAGE_PRODUCTS` (Task 8), `labels` table populated by Task 9's `upsert_labels` (반드시 Task 9가 먼저 완료돼 있어야 함)
- Produces: `check_market_cap_consistency(df, tolerance=0.1) -> list[str]`, `check_value_ranges(df) -> list[str]`, `check_null_rates(df, max_null_ratio=0.3) -> dict[str, float]`, `check_trading_day_gaps(df, expected_min_days) -> list[str]`
- Produces: `compute_rebalancing_flow(prev_aum: float, underlying_return: float, multiple: float) -> float`, `aggregate_leverage_signals(product_rows: list[dict], underlying_return: float, underlying_market_cap: float) -> dict` (returns `lev_total_volume`, `lev_total_aum`, `lev_aum_to_mktcap`, `est_rebalancing_flow`)
- Produces: `build_feature_pool(dsn: str, start_date: str, end_date: str) -> None` (조인 → `feature_pool` 테이블 생성/적재, `label`/`next_day_return` 컬럼 포함)

- [ ] **Step 1: 실패하는 테스트 작성 (계산 로직 — 설계 §6.3 공식)**

`features/test_leverage_features.py`:

```python
import pytest
from features.leverage_features import compute_rebalancing_flow, aggregate_leverage_signals


def test_rebalancing_flow_positive_when_leveraged_fund_tracks_rising_stock():
    # 배율 2x, 전일 AUM 1000억, 기초자산 당일 +2% → 추가 매수 필요(양수)
    flow = compute_rebalancing_flow(prev_aum=100_000_000_000, underlying_return=0.02, multiple=2.0)
    assert flow == pytest.approx(100_000_000_000 * 0.02 * (2.0 - 1))


def test_rebalancing_flow_for_inverse_fund_on_down_day():
    # 배율 -2x, 기초자산 당일 -3% → (배율-1) = -3, return -0.03 → flow 양수(추가 공매도 유사 매도 압력)
    flow = compute_rebalancing_flow(prev_aum=50_000_000_000, underlying_return=-0.03, multiple=-2.0)
    assert flow == pytest.approx(50_000_000_000 * -0.03 * (-2.0 - 1))
    assert flow > 0


def test_aggregate_sums_across_all_products_for_one_underlying():
    product_rows = [
        {"code": "A", "multiple": 2.0, "close_price": 10000, "volume": 100_000, "aum": 200_000_000_000},
        {"code": "B", "multiple": -2.0, "close_price": 5000, "volume": 50_000, "aum": 50_000_000_000},
    ]
    result = aggregate_leverage_signals(product_rows, underlying_return=0.01,
                                          underlying_market_cap=400_000_000_000_000)
    assert result["lev_total_volume"] == pytest.approx(10000 * 100_000 + 5000 * 50_000)
    assert result["lev_total_aum"] == pytest.approx(250_000_000_000)
    assert result["lev_aum_to_mktcap"] == pytest.approx(250_000_000_000 / 400_000_000_000_000)
    expected_flow = (200_000_000_000 * 0.01 * (2.0 - 1)) + (50_000_000_000 * 0.01 * (-2.0 - 1))
    assert result["est_rebalancing_flow"] == pytest.approx(expected_flow)


def test_aggregate_returns_zeros_when_no_products_yet_listed():
    result = aggregate_leverage_signals([], underlying_return=0.01, underlying_market_cap=1e14)
    assert result == {"lev_total_volume": 0.0, "lev_total_aum": 0.0,
                        "lev_aum_to_mktcap": 0.0, "est_rebalancing_flow": 0.0}
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest features/test_leverage_features.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`features/leverage_features.py`:

```python
def compute_rebalancing_flow(prev_aum: float, underlying_return: float, multiple: float) -> float:
    """설계 §6.3: 리밸런싱 금액 = 전일 AUM × 기초자산 당일 수익률 × (배율 - 1)."""
    return prev_aum * underlying_return * (multiple - 1)


def aggregate_leverage_signals(product_rows: list[dict], underlying_return: float,
                                 underlying_market_cap: float) -> dict:
    if not product_rows:
        return {"lev_total_volume": 0.0, "lev_total_aum": 0.0,
                "lev_aum_to_mktcap": 0.0, "est_rebalancing_flow": 0.0}
    total_volume = sum(r["close_price"] * r["volume"] for r in product_rows)
    total_aum = sum(r["aum"] for r in product_rows)
    total_flow = sum(
        compute_rebalancing_flow(r["aum"], underlying_return, r["multiple"])
        for r in product_rows
    )
    return {
        "lev_total_volume": total_volume, "lev_total_aum": total_aum,
        "lev_aum_to_mktcap": total_aum / underlying_market_cap if underlying_market_cap else 0.0,
        "est_rebalancing_flow": total_flow,
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest features/test_leverage_features.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: `feature_pool` 조인 빌드 구현 및 실행**

`features/build_features.py` — `price_daily`(기술적 지표: log_ret, disparity_5/20/60d, rsi_14 등은 pandas로 계산), `daily_valuation`, `investor_flow_daily`, `market_global`, `market_index_daily`, `sector_daily_ohlcv`, `stock_events`, `calendar`, `market_events`를 ticker+trade_date로 조인하고, `leverage_daily`+`leverage_products`를 `aggregate_leverage_signals`로 집계해 붙이고(2026-05-27 이전은 빈 리스트 → 자동 0), `vi_events`에서 `is_vi_triggered`/`vi_count_recent5d`를 계산해 붙이고, **`labels` 테이블(Task 9)을 (ticker, trade_date)로 조인해서 `label`/`next_day_return` 컬럼을 최종 결과에 포함**시켜 `feature_pool` 테이블에 적재하는 스크립트(Task 9가 먼저 완료되어 `labels`가 채워져 있어야 이 조인이 의미 있음 — Task 9 → Task 10 순서 의존성). 여기엔 §7 원칙대로 만들 수 있는 피처를 전부 포함시킨다(추후 ablation에서 subset 선택).

Run: `/home/user/miniconda3/envs/kis_collector/bin/python features/build_features.py --start 2019-01-02 --end <today>`
Expected: `feature_pool` 테이블에 200종목 × 전체 기간 행 적재, 컬럼 수가 기존 55개 + 레버리지 6개 이상

- [ ] **Step 6: 실패하는 테스트 작성 (데이터 정합성 검증 — 순수 함수)**

`features/test_validate_feature_pool.py` (신규 파일):

```python
import pandas as pd
import pytest
from features.validate_feature_pool import (
    check_market_cap_consistency, check_value_ranges, check_null_rates, check_trading_day_gaps,
)


def test_market_cap_cross_check_flags_large_discrepancy():
    # ticker_universe(price*shares)와 daily_valuation(hts_avls*1e6)이 독립적으로 계산된 시총 —
    # 단위 버그가 있으면 이 둘이 자릿수 단위로 어긋난다
    df = pd.DataFrame({
        "ticker": ["005930", "000660"],
        "universe_market_cap": [4.5e14, 9.0e13],
        "valuation_market_cap": [4.51e14, 9.0e7],  # 000660은 1e6배 축소된 버그 상황 가정
    })
    issues = check_market_cap_consistency(df, tolerance=0.1)
    assert issues == ["000660"]


def test_market_cap_cross_check_passes_when_close():
    df = pd.DataFrame({"ticker": ["005930"], "universe_market_cap": [4.5e14],
                         "valuation_market_cap": [4.52e14]})
    assert check_market_cap_consistency(df, tolerance=0.1) == []


def test_check_value_ranges_flags_negative_price_or_volume():
    df = pd.DataFrame({"close_price": [50000.0, -100.0], "volume": [1000, -5]})
    issues = check_value_ranges(df)
    assert "close_price" in issues
    assert "volume" in issues


def test_check_null_rates_flags_columns_over_threshold():
    df = pd.DataFrame({"per": [None, None, None, 1.0], "close_price": [1, 2, 3, 4]})
    issues = check_null_rates(df, max_null_ratio=0.5)
    assert issues == {"per": pytest.approx(0.75)}


def test_check_trading_day_gaps_flags_ticker_with_missing_days():
    df = pd.DataFrame({
        "ticker": ["005930"] * 3 + ["000660"] * 5,
        "trade_date": pd.to_datetime(["2019-01-02", "2019-01-03", "2019-01-04"] +
                                       list(pd.bdate_range("2019-01-02", periods=5))),
    })
    # 삼성전자는 3일치뿐인데 SK하이닉스는 5일치 — 같은 기간 대비 종목별 행 수 편차 검출
    issues = check_trading_day_gaps(df, expected_min_days=5)
    assert issues == ["005930"]
```

- [ ] **Step 7: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest features/test_validate_feature_pool.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 8: 구현**

`features/validate_feature_pool.py` (신규 파일):

```python
import pandas as pd


def check_market_cap_consistency(df: pd.DataFrame, tolerance: float = 0.1) -> list[str]:
    """두 독립 소스(시총 스냅샷 계산 vs KIS 밸류에이션 API)로 구한 시총이 tolerance 이상 어긋나면
    단위 변환 버그일 가능성이 높다 — 티커 목록 반환."""
    flagged = []
    for _, row in df.iterrows():
        a, b = row["universe_market_cap"], row["valuation_market_cap"]
        if a == 0 or b == 0:
            flagged.append(row["ticker"])
            continue
        rel_diff = abs(a - b) / max(a, b)
        if rel_diff > tolerance:
            flagged.append(row["ticker"])
    return flagged


def check_value_ranges(df: pd.DataFrame) -> list[str]:
    """가격은 양수, 거래량은 0 이상이어야 함 — 위반 컬럼명 반환."""
    issues = []
    if "close_price" in df.columns and (df["close_price"] <= 0).any():
        issues.append("close_price")
    if "volume" in df.columns and (df["volume"] < 0).any():
        issues.append("volume")
    return issues


def check_null_rates(df: pd.DataFrame, max_null_ratio: float = 0.3) -> dict[str, float]:
    """컬럼별 결측 비율이 max_null_ratio를 넘으면 {컬럼명: 실제비율} 반환."""
    issues = {}
    for col in df.columns:
        ratio = df[col].isna().mean()
        if ratio > max_null_ratio:
            issues[col] = ratio
    return issues


def check_trading_day_gaps(df: pd.DataFrame, expected_min_days: int) -> list[str]:
    """종목별 행 수가 expected_min_days에 못 미치면 수집 누락 의심 — 티커 목록 반환."""
    counts = df.groupby("ticker").size()
    return counts[counts < expected_min_days].index.tolist()
```

- [ ] **Step 9: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest features/test_validate_feature_pool.py -v`
Expected: PASS (5 tests)

- [ ] **Step 10: 실제 `feature_pool`에 대해 실행**

```python
# features/run_validate_feature_pool.py:
# 1) ticker_universe.market_cap과 daily_valuation.market_cap(같은 ticker, 가장 최근 공통 날짜)을 조인해서
#    check_market_cap_consistency 실행 — 걸린 티커가 있으면 즉시 중단하고 원인(단위 배율) 조사
# 2) feature_pool 전체에 check_value_ranges, check_null_rates(max_null_ratio=0.3, per/pbr/prop_*는 §8
#    원칙대로 별도 허용치 적용), check_trading_day_gaps(expected_min_days= 2019~현재 영업일수의 90%) 실행
# 3) 결과를 features/validation_report.md에 정리(문제 없으면 "이상 없음", 있으면 티커/컬럼 목록)
```

Run: `/home/user/miniconda3/envs/kis_collector/bin/python features/run_validate_feature_pool.py`
Expected: `features/validation_report.md` 생성, 시총 교차검증 불일치 0건(0건이 아니면 Task 8/5의 단위 배율부터 재점검 — Task 11로 넘어가지 말 것)

- [ ] **Step 11: Commit**

```bash
git add features/leverage_features.py features/build_features.py features/test_leverage_features.py features/validate_feature_pool.py features/test_validate_feature_pool.py features/run_validate_feature_pool.py
git commit -m "feat: build joined feature_pool table with leverage-derived features and data-integrity validation"
```

---

## Task 11: train-split 기준 클리핑/스케일링

**Files:**
- Create: `features/clip_scale.py`
- Test: `features/test_clip_scale.py`

**Interfaces:**
- Produces: `fit_clip_bounds(train_values: np.ndarray, lower_q: float = 0.005, upper_q: float = 0.995) -> tuple[float, float]`, `apply_clip(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray`, `fit_scaler(train_df: pd.DataFrame, columns: list[str]) -> StandardScaler`, `save_artifacts(bounds: dict, scaler, path: str) -> None`, `load_artifacts(path: str) -> tuple[dict, StandardScaler]`

- [ ] **Step 1: 실패하는 테스트 작성**

`features/test_clip_scale.py`:

```python
import numpy as np
import pandas as pd
from features.clip_scale import fit_clip_bounds, apply_clip, fit_scaler


def test_fit_clip_bounds_uses_quantiles_not_hardcoded_values():
    train_values = np.array([-100.0] + list(np.linspace(-1, 1, 998)) + [100.0])
    lower, upper = fit_clip_bounds(train_values, lower_q=0.005, upper_q=0.995)
    assert lower > -100.0 and upper < 100.0
    assert lower < 0 < upper


def test_apply_clip_bounds_values():
    result = apply_clip(np.array([-10.0, -0.5, 0.0, 0.5, 10.0]), bounds=(-1.0, 1.0))
    np.testing.assert_array_equal(result, [-1.0, -0.5, 0.0, 0.5, 1.0])


def test_fit_scaler_only_uses_provided_columns():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0], "c": ["x", "y", "z"]})
    scaler = fit_scaler(df, columns=["a", "b"])
    assert list(scaler.feature_names_in_) == ["a", "b"]
    assert scaler.mean_[0] == 2.0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest features/test_clip_scale.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`features/clip_scale.py`:

```python
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_clip_bounds(train_values: np.ndarray, lower_q: float = 0.005,
                     upper_q: float = 0.995) -> tuple[float, float]:
    clean = train_values[~np.isnan(train_values)]
    return float(np.quantile(clean, lower_q)), float(np.quantile(clean, upper_q))


def apply_clip(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return np.clip(values, bounds[0], bounds[1])


def fit_scaler(train_df: pd.DataFrame, columns: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[columns])
    return scaler


def save_artifacts(bounds: dict, scaler: StandardScaler, path: str) -> None:
    with open(f"{path}.bounds.json", "w") as f:
        json.dump(bounds, f)
    with open(f"{path}.scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


def load_artifacts(path: str) -> tuple[dict, StandardScaler]:
    with open(f"{path}.bounds.json") as f:
        bounds = json.load(f)
    with open(f"{path}.scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return bounds, scaler
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest features/test_clip_scale.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: 1단계 train split(2019-2023)로 실제 fit 실행**

```python
# features/run_fit_clip_scale.py:
# feature_pool에서 1단계 train 구간(2019-01-02~2023-12-31)만 로드
# 클리핑 대상 컬럼(log_ret, vol_ratio, disparity_5/20/60, macd_*, prop_*, vix_chg,
#   usd_krw_chg, sector_volume_ratio, est_rebalancing_flow 등 연속형)마다 fit_clip_bounds
# 클리핑 적용 후 StandardScaler는 fit_scaler로 학습
# save_artifacts(bounds, scaler, "features/artifacts/stage1")
```

Run: `/home/user/miniconda3/envs/kis_collector/bin/python features/run_fit_clip_scale.py`
Expected: `features/artifacts/stage1.bounds.json`, `stage1.scaler.pkl` 생성

- [ ] **Step 6: Commit**

```bash
git add features/clip_scale.py features/test_clip_scale.py features/run_fit_clip_scale.py
git commit -m "feat: fit train-split clipping bounds and scaler (no hardcoded constants)"
```

---

## Task 12: PyTorch Dataset (encoder 60일 + 오늘의 미완성 일봉)

**Files:**
- Create: `training/dataset.py`
- Test: `training/test_dataset.py`

**Interfaces:**
- Consumes: `HISTORICAL_COLS`, `KNOWN_FUTURE_COLS`, `STATIC_COLS` (Task 13에서 정의, 여기선 파라미터로 주입)
- Produces: `class TickerDayDataset(torch.utils.data.Dataset)` — `__getitem__` returns `{"historical_ts_numeric": Tensor[60, F], "future_ts_numeric": Tensor[1, K], "static_feats_categorical": Tensor[1, 2], "label": Tensor[1]}`
- Produces: `build_incomplete_today_bar(intraday_5min_rows: list[dict], open_price: float) -> dict` (실시간 서빙용 — 오늘 지금까지의 5분봉으로 미완성 일봉 계산, 오늘 필요 시 encoder 마지막 행을 대체)

- [ ] **Step 1: 실패하는 테스트 작성**

`training/test_dataset.py`:

```python
import numpy as np
import pandas as pd
import torch
from training.dataset import TickerDayDataset, build_incomplete_today_bar

HIST_COLS = ["log_ret", "disparity_20"]
FUT_COLS = ["time_progress", "is_bok"]
STATIC_COLS = ["sector_id", "market_id"]


def make_ticker_df(n_days: int) -> pd.DataFrame:
    return pd.DataFrame({
        "trade_date": pd.date_range("2019-01-02", periods=n_days, freq="B"),
        "log_ret": np.random.randn(n_days) * 0.01,
        "disparity_20": 1.0 + np.random.randn(n_days) * 0.02,
        "time_progress": 1.0, "is_bok": 0,
        "sector_id": 3, "market_id": 1,
        "label": np.random.randint(0, 3, n_days),
    })


def test_returns_none_for_insufficient_history():
    df = make_ticker_df(n_days=30)  # encoder_len=60보다 짧음
    ds = TickerDayDataset({"005930": df}, HIST_COLS, FUT_COLS, STATIC_COLS, encoder_len=60)
    assert len(ds) == 0


def test_produces_correct_tensor_shapes():
    df = make_ticker_df(n_days=65)
    ds = TickerDayDataset({"005930": df}, HIST_COLS, FUT_COLS, STATIC_COLS, encoder_len=60)
    assert len(ds) == 5  # 65 - 60
    sample = ds[0]
    assert sample["historical_ts_numeric"].shape == (60, 2)
    assert sample["future_ts_numeric"].shape == (1, 2)
    assert sample["static_feats_categorical"].shape == (1, 2)
    assert sample["label"].shape == (1,)


def test_static_features_are_int_dtype_for_embedding():
    df = make_ticker_df(n_days=65)
    ds = TickerDayDataset({"005930": df}, HIST_COLS, FUT_COLS, STATIC_COLS, encoder_len=60)
    assert ds[0]["static_feats_categorical"].dtype == torch.long


def test_build_incomplete_today_bar_computes_ohlc_from_intraday():
    rows = [
        {"datetime": "2026-09-08 09:00:00", "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000},
        {"datetime": "2026-09-08 09:05:00", "open": 100.5, "high": 102, "low": 100, "close": 101.5, "volume": 1500},
        {"datetime": "2026-09-08 09:10:00", "open": 101.5, "high": 101.8, "low": 101.0, "close": 101.2, "volume": 800},
    ]
    bar = build_incomplete_today_bar(rows, open_price=100)
    assert bar["open"] == 100
    assert bar["high"] == 102
    assert bar["low"] == 99
    assert bar["close"] == 101.2  # 가장 최근 종가
    assert bar["volume"] == 3300
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest training/test_dataset.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`training/dataset.py`:

```python
import torch
from torch.utils.data import Dataset
import pandas as pd


class TickerDayDataset(Dataset):
    def __init__(self, ticker_dfs: dict[str, pd.DataFrame], historical_cols: list[str],
                 future_cols: list[str], static_cols: list[str], encoder_len: int = 60):
        self.historical_cols = historical_cols
        self.future_cols = future_cols
        self.static_cols = static_cols
        self.encoder_len = encoder_len
        self.index: list[tuple[str, int]] = []
        self.ticker_dfs = ticker_dfs
        for ticker, df in ticker_dfs.items():
            df = df.reset_index(drop=True)
            self.ticker_dfs[ticker] = df
            n_usable = len(df) - encoder_len
            for t in range(max(n_usable, 0)):
                # window [t : t+encoder_len) 은 history, 예측 시점은 t+encoder_len (label 포함)
                if pd.notna(df.loc[t + encoder_len, "label"]):
                    self.index.append((ticker, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        ticker, t = self.index[idx]
        df = self.ticker_dfs[ticker]
        hist = df.loc[t: t + self.encoder_len - 1, self.historical_cols].values.astype("float32")
        target_row = df.loc[t + self.encoder_len]
        fut = target_row[self.future_cols].values.astype("float32").reshape(1, -1)
        static = target_row[self.static_cols].values.astype("int64").reshape(1, -1)
        label = int(target_row["label"])
        return {
            "historical_ts_numeric": torch.tensor(hist),
            "future_ts_numeric": torch.tensor(fut),
            "static_feats_categorical": torch.tensor(static, dtype=torch.long),
            "label": torch.tensor([label]),
        }


def build_incomplete_today_bar(intraday_5min_rows: list[dict], open_price: float) -> dict:
    """설계 §3: '오늘'을 5분봉으로 실시간 계산되는 미완성 일봉으로 표현."""
    if not intraday_5min_rows:
        return {"open": open_price, "high": open_price, "low": open_price,
                "close": open_price, "volume": 0}
    highs = [r["high"] for r in intraday_5min_rows]
    lows = [r["low"] for r in intraday_5min_rows]
    latest = max(intraday_5min_rows, key=lambda r: r["datetime"])
    return {
        "open": open_price, "high": max(highs), "low": min(lows),
        "close": latest["close"], "volume": sum(r["volume"] for r in intraday_5min_rows),
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest training/test_dataset.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add training/dataset.py training/test_dataset.py
git commit -m "feat: add sliding-window Dataset and live incomplete-daily-bar builder"
```

---

## Task 13: TFT config + 학습 루프 (wandb 로깅)

**Files:**
- Create: `training/config.py`
- Create: `training/train.py`
- Test: `training/test_config.py`

**Interfaces:**
- Consumes: `TemporalFusionTransformer` (tft-torch fork, Task complete already in repo), `TickerDayDataset` (Task 12)
- Produces: `HISTORICAL_COLS: list[str]`, `KNOWN_FUTURE_COLS: list[str]`, `STATIC_COLS: list[str]`, `build_tft_config(feature_columns: dict, num_classes: int = 3) -> DictConfig`
- Produces: `train_one_epoch(model, dataloader, optimizer, criterion, device) -> float`, `evaluate_loss(model, dataloader, criterion, device) -> float`, `run_training(config: dict) -> dict` (wandb 로깅 포함, 반환값에 `best_val_loss`, `checkpoint_path`)

- [ ] **Step 1: 실패하는 테스트 작성 (config 빌더 — 순수 함수)**

`training/test_config.py`:

```python
from training.config import build_tft_config


def test_config_sets_three_output_classes():
    cfg = build_tft_config(
        feature_columns={"historical": ["log_ret", "disparity_20"], "future": ["time_progress"],
                          "static_cardinalities": [21, 3]},
        num_classes=3,
    )
    assert cfg.task_type == "classification"
    assert cfg.model.num_classes == 3
    assert cfg.data_props.num_historical_numeric == 2
    assert cfg.data_props.num_future_numeric == 1
    assert cfg.data_props.static_categorical_cardinalities == [21, 3]


def test_config_encoder_state_size_defaults():
    cfg = build_tft_config(
        feature_columns={"historical": ["a"], "future": ["b"], "static_cardinalities": [21, 3]},
        num_classes=3,
    )
    assert cfg.model.state_size > 0
    assert cfg.model.attention_heads > 0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest training/test_config.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`training/config.py`:

```python
from omegaconf import OmegaConf, DictConfig

# 설계 §4, §5: 기존 55개 기반 + 레버리지 6개 등 feature_pool 전체 컬럼 중
# ablation(Task 15)에서 선택된 subset이 여기로 주입된다.
HISTORICAL_COLS_DEFAULT = [
    "rel_close", "rel_high", "rel_low", "log_ret", "disparity_5", "disparity_20", "disparity_60",
    "vol_ratio", "rsi_14", "bb_position", "macd_ratio", "macd_signal_ratio", "macd_hist_ratio",
    "log_ret_1d", "disparity_5d", "disparity_20d", "disparity_60d", "volatility_20d",
    "prop_individual", "prop_foreign", "prop_institution", "per", "pbr", "per_chg_1d", "pbr_chg_1d",
    "kospi_ret", "kosdaq_ret", "snp500_ret", "nasdaq_ret", "phlx_semi_ret", "vix_chg",
    "usd_krw_chg", "us_10y_yield_chg", "rate_spread_us_kr", "wti_ret", "gold_ret",
    "sector_ret_1d", "sector_ret_5d", "sector_ret_20d", "sector_ma_ratio_20d",
    "sector_volatility", "sector_volume_ratio",
    "is_dividend", "is_bonus_issue", "is_rights_offering", "is_split", "is_merger", "is_earnings",
    "day_of_week", "listing_days",
    "lev_total_volume", "lev_total_aum", "lev_aum_to_mktcap", "est_rebalancing_flow",
    "is_vi_triggered", "vi_count_recent5d",
]
KNOWN_FUTURE_COLS = ["time_progress", "is_bok", "is_fomc", "is_witching_kr", "is_witching_us"]
STATIC_COLS = ["sector_id", "market_id"]


def build_tft_config(feature_columns: dict, num_classes: int = 3,
                      state_size: int = 32, attention_heads: int = 4,
                      lstm_layers: int = 1, dropout: float = 0.1) -> DictConfig:
    return OmegaConf.create({
        "task_type": "classification",
        "target_window_start": None,
        "data_props": {
            "num_historical_numeric": len(feature_columns["historical"]),
            "num_historical_categorical": 0, "historical_categorical_cardinalities": [],
            "num_static_numeric": 0, "num_static_categorical": len(feature_columns["static_cardinalities"]),
            "static_categorical_cardinalities": feature_columns["static_cardinalities"],
            "num_future_numeric": len(feature_columns["future"]),
            "num_future_categorical": 0, "future_categorical_cardinalities": [],
        },
        "model": {"state_size": state_size, "attention_heads": attention_heads,
                   "dropout": dropout, "lstm_layers": lstm_layers, "num_classes": num_classes},
    })
```

`training/train.py`:

```python
import sys
import torch
import torch.nn.functional as F
import wandb
sys.path.insert(0, "tft-torch")
from tft_torch.tft import TemporalFusionTransformer


def train_one_epoch(model, dataloader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "label"}
        labels = batch.pop("label", None)
        out = model(batch)
        logits = out["class_logits"].squeeze(1)  # [B, num_classes]
        loss = criterion(logits, labels.squeeze(-1).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * logits.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate_loss(model, dataloader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("label")
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            logits = out["class_logits"].squeeze(1)
            loss = criterion(logits, labels.squeeze(-1).to(device))
            total_loss += loss.item() * logits.size(0)
    return total_loss / len(dataloader.dataset)


def run_training(config: dict) -> dict:
    """config keys: tft_config, class_weights, train_loader, val_loader, epochs, lr,
    device, run_name, checkpoint_dir."""
    device = config["device"]
    model = TemporalFusionTransformer(config["tft_config"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = torch.nn.CrossEntropyLoss(weight=config["class_weights"].to(device))

    wandb.init(project="ai-gentle-viking-re", name=config["run_name"], config=config.get("wandb_config", {}))
    best_val_loss = float("inf")
    checkpoint_path = f"{config['checkpoint_dir']}/{config['run_name']}.pt"
    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(model, config["train_loader"], optimizer, criterion, device)
        val_loss = evaluate_loss(model, config["val_loader"], criterion, device)
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
    wandb.finish()
    return {"best_val_loss": best_val_loss, "checkpoint_path": checkpoint_path}
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest training/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: 통합 스모크 (Task 12 Dataset + 이 config로 forward pass 1회 확인, 소규모 합성 데이터)**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -c "
from training.config import build_tft_config
from training.dataset import TickerDayDataset
from torch.utils.data import DataLoader
import sys; sys.path.insert(0, 'tft-torch')
from tft_torch.tft import TemporalFusionTransformer
import pandas as pd, numpy as np, torch

df = pd.DataFrame({'log_ret': np.random.randn(65)*0.01, 'disparity_20': 1+np.random.randn(65)*0.02,
                    'time_progress': 1.0, 'is_bok': 0, 'sector_id': 3, 'market_id': 1,
                    'label': np.random.randint(0,3,65)})
ds = TickerDayDataset({'005930': df}, ['log_ret','disparity_20'], ['time_progress','is_bok'],
                       ['sector_id','market_id'], encoder_len=60)
loader = DataLoader(ds, batch_size=2)
cfg = build_tft_config({'historical': ['log_ret','disparity_20'], 'future': ['time_progress','is_bok'],
                         'static_cardinalities': [21,3]}, num_classes=3)
model = TemporalFusionTransformer(cfg)
batch = next(iter(loader)); batch.pop('label')
out = model(batch)
assert out['class_logits'].shape == (2, 1, 3)
print('smoke test OK')
"`
Expected: 출력 `smoke test OK`

- [ ] **Step 6: Commit**

```bash
git add training/config.py training/train.py
git commit -m "feat: add TFT config builder and wandb-logged training loop"
```

---

## Task 14: 평가 모듈 (Macro F1 중심, 1단계/2단계 분리 리포트)

**Files:**
- Create: `evaluation/evaluate.py`
- Test: `evaluation/test_evaluate.py`

**Interfaces:**
- Produces: `compute_metrics(y_true: list[int], y_pred: list[int]) -> dict` (`accuracy`, `macro_f1`, `mcc`, `per_class` — `{0: {"precision":..,"recall":..,"f1":..}, 1: {...}, 2: {...}}`, `confusion_matrix`)
- Produces: `split_by_regime(dates: list[str], y_true: list[int], y_pred: list[int], leverage_start: str = "2026-05-27") -> dict` (returns `{"pre_leverage": {...}, "leverage_era": {...}}`, each a `compute_metrics` result)

- [ ] **Step 1: 실패하는 테스트 작성**

`evaluation/test_evaluate.py`:

```python
import pytest
from evaluation.evaluate import compute_metrics, split_by_regime


def test_compute_metrics_perfect_prediction():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["mcc"] == pytest.approx(1.0)
    assert m["per_class"][0]["f1"] == 1.0
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_compute_metrics_detects_class_collapse():
    # 관망(1) 클래스를 전혀 못 맞추는 경우 — macro_f1이 accuracy보다 뚜렷하게 낮아야 함
    y_true = [0, 1, 1, 1, 2]
    y_pred = [0, 0, 0, 0, 2]
    m = compute_metrics(y_true, y_pred)
    assert m["per_class"][1]["recall"] == 0.0
    assert m["macro_f1"] < m["accuracy"]


def test_split_by_regime_separates_pre_and_post_leverage():
    dates = ["2025-01-01", "2025-06-01", "2026-06-01", "2026-07-01"]
    y_true = [0, 1, 2, 0]
    y_pred = [0, 1, 2, 2]
    result = split_by_regime(dates, y_true, y_pred, leverage_start="2026-05-27")
    assert result["pre_leverage"]["accuracy"] == 1.0
    assert result["leverage_era"]["accuracy"] == 0.5
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest evaluation/test_evaluate.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`evaluation/evaluate.py`:

```python
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    matthews_corrcoef, confusion_matrix,
)

CLASS_NAMES = {0: "buy", 1: "hold", 2: "sell"}


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    labels = [0, 1, 2]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    per_class = {
        labels[i]: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
        for i in range(len(labels))
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def split_by_regime(dates: list[str], y_true: list[int], y_pred: list[int],
                     leverage_start: str = "2026-05-27") -> dict:
    pre_idx = [i for i, d in enumerate(dates) if d < leverage_start]
    post_idx = [i for i, d in enumerate(dates) if d >= leverage_start]
    return {
        "pre_leverage": compute_metrics([y_true[i] for i in pre_idx], [y_pred[i] for i in pre_idx]),
        "leverage_era": compute_metrics([y_true[i] for i in post_idx], [y_pred[i] for i in post_idx]),
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest evaluation/test_evaluate.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/evaluate.py evaluation/test_evaluate.py
git commit -m "feat: add evaluation metrics with pre/post-leverage regime split"
```

---

## Task 15: 1단계 학습 실행 + ablation 러너 + Optuna 튜닝

**Files:**
- Create: `training/ablation.py`
- Create: `training/tune.py`
- Create: `docs/model_versions.md`
- Test: `training/test_ablation.py`

**Interfaces:**
- Consumes: `HISTORICAL_COLS_DEFAULT` (Task 13), `run_training` (Task 13), `compute_metrics` (Task 14)
- Produces: `run_ablation(base_columns: list[str], candidate_removals: list[str], train_fn) -> list[dict]` (각 실험의 `{"removed": str, "macro_f1": float}` 리스트), `run_optuna_study(objective_fn, n_trials: int) -> optuna.Study`

- [ ] **Step 1: 실패하는 테스트 작성 (ablation 러너 — train_fn을 주입해서 순수 로직만 테스트)**

`training/test_ablation.py`:

```python
from training.ablation import run_ablation


def test_ablation_tries_removing_each_candidate_once():
    calls = []

    def fake_train_fn(columns: list[str]) -> float:
        calls.append(list(columns))
        return len(columns) * 0.01  # 컬럼 많을수록 점수 높은 가짜 함수

    base = ["a", "b", "c"]
    results = run_ablation(base, candidate_removals=["a", "b"], train_fn=fake_train_fn)

    assert len(calls) == 3  # baseline(전체) + 2개 제거 실험
    assert {"removed": None, "macro_f1": 0.03} in results
    assert {"removed": "a", "macro_f1": 0.02} in results
    assert {"removed": "b", "macro_f1": 0.02} in results


def test_ablation_removed_column_actually_excluded_from_train_call():
    seen_columns = []

    def fake_train_fn(columns: list[str]) -> float:
        seen_columns.append(columns)
        return 0.5

    run_ablation(["x", "y"], candidate_removals=["x"], train_fn=fake_train_fn)
    removal_call = [c for c in seen_columns if "x" not in c]
    assert removal_call == [["y"]]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest training/test_ablation.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`training/ablation.py`:

```python
from typing import Callable


def run_ablation(base_columns: list[str], candidate_removals: list[str],
                  train_fn: Callable[[list[str]], float]) -> list[dict]:
    results = [{"removed": None, "macro_f1": train_fn(base_columns)}]
    for col in candidate_removals:
        reduced = [c for c in base_columns if c != col]
        results.append({"removed": col, "macro_f1": train_fn(reduced)})
    return results
```

`training/tune.py`:

```python
import optuna


def run_optuna_study(objective_fn, n_trials: int = 20) -> optuna.Study:
    """objective_fn(trial) -> macro_f1 (maximize). trial에서 state_size, attention_heads,
    lstm_layers, dropout, lr을 suggest해서 training.train.run_training에 넘기는 방식으로 구성."""
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_fn, n_trials=n_trials)
    return study
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest training/test_ablation.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: 1단계 실제 실행 — Optuna 튜닝(20 trials) → 최적 하이퍼파라미터로 ablation(레버리지 피처 6개 각각 제거 실험 포함) → `docs/model_versions.md`에 v1 기록**

`docs/model_versions.md` 초기 구조:

```markdown
# 모델 버전 기록

| 버전 | 단계 | 피처 구성 | 하이퍼파라미터 | Macro F1 (val) | Macro F1 (test) | 비고 |
|---|---|---|---|---|---|---|
```

Run: 1단계 train(2019-2023)/val(2024)으로 Optuna study 실행 → best trial의 하이퍼파라미터로 ablation 실행 → 결과를 `docs/model_versions.md`에 v1, v2... 행으로 추가(스크립트가 자동 append)
Expected: `docs/model_versions.md`에 여러 버전 행 채워짐, 최고 Macro F1 버전이 표에서 식별 가능

- [ ] **Step 6: Commit**

```bash
git add training/ablation.py training/tune.py training/test_ablation.py docs/model_versions.md
git commit -m "feat: add ablation runner, Optuna tuning, and model version log"
```

---

## Task 16: 1단계 최종 평가 (test=2025) + 아키텍처/피처 조합 선정

**Files:**
- Create: `training/select_stage1_champion.py`
- Test: `training/test_select_stage1_champion.py`

**Interfaces:**
- Consumes: `docs/model_versions.md` 행들(Task 15), `compute_metrics`/`split_by_regime`(Task 14)
- Produces: `select_champion(versions: list[dict]) -> dict` (Macro F1(val) 기준 최고 버전 선택), `evaluate_on_test(model_path: str, test_loader, class_columns: list[str]) -> dict`

- [ ] **Step 1: 실패하는 테스트 작성**

`training/test_select_stage1_champion.py`:

```python
from training.select_stage1_champion import select_champion


def test_selects_highest_val_macro_f1():
    versions = [
        {"version": "v1", "macro_f1_val": 0.41, "columns": ["a", "b"]},
        {"version": "v2", "macro_f1_val": 0.53, "columns": ["a", "b", "c"]},
        {"version": "v3", "macro_f1_val": 0.47, "columns": ["a"]},
    ]
    champion = select_champion(versions)
    assert champion["version"] == "v2"


def test_raises_on_empty_versions():
    import pytest
    with pytest.raises(ValueError, match="no versions"):
        select_champion([])
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest training/test_select_stage1_champion.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`training/select_stage1_champion.py`:

```python
def select_champion(versions: list[dict]) -> dict:
    if not versions:
        raise ValueError("no versions to select from")
    return max(versions, key=lambda v: v["macro_f1_val"])
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest training/test_select_stage1_champion.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: 실제 선정 + test(2025) 평가 실행 — 챔피언 모델을 test 구간에 1회만 적용, `compute_metrics`/`split_by_regime`으로 결과 산출, `docs/model_versions.md`에 test 컬럼 채우기**

Expected: 챔피언 버전의 아키텍처(state_size/attention_heads/lstm_layers/dropout)와 피처 컬럼 리스트가 `training/champion_config.json`으로 저장됨(Task 17에서 그대로 재사용)

- [ ] **Step 6: Commit**

```bash
git add training/select_stage1_champion.py training/test_select_stage1_champion.py training/champion_config.json
git commit -m "feat: select stage-1 champion architecture/feature set and evaluate on 2025 test"
```

---

## Task 17: 2단계 최종 재학습 (2019~현재, 레버리지 국면 포함)

**Files:**
- Create: `training/run_stage2_final.py`
- Test: `training/test_run_stage2_final.py`

**Interfaces:**
- Consumes: `training/champion_config.json`(Task 16), `run_training`(Task 13), `split_by_regime`(Task 14)
- Produces: `build_stage2_config(champion: dict) -> dict` (2단계 날짜 범위로 재구성된 학습 config)

- [ ] **Step 1: 실패하는 테스트 작성**

`training/test_run_stage2_final.py`:

```python
from training.run_stage2_final import build_stage2_config


def test_stage2_uses_full_date_range_and_champion_columns():
    champion = {"version": "v2", "columns": ["log_ret", "est_rebalancing_flow"],
                 "state_size": 64, "attention_heads": 4, "lstm_layers": 2, "dropout": 0.15}
    cfg = build_stage2_config(champion, today="2026-09-08")
    assert cfg["train_start"] == "2019-01-02"
    assert cfg["train_end"] == "2026-09-08"
    assert cfg["columns"] == ["log_ret", "est_rebalancing_flow"]
    assert cfg["state_size"] == 64
    assert cfg["run_name"] == "stage2-final-v2"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest training/test_run_stage2_final.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`training/run_stage2_final.py`:

```python
def build_stage2_config(champion: dict, today: str) -> dict:
    return {
        "train_start": "2019-01-02", "train_end": today,
        "columns": champion["columns"],
        "state_size": champion["state_size"], "attention_heads": champion["attention_heads"],
        "lstm_layers": champion["lstm_layers"], "dropout": champion["dropout"],
        "run_name": f"stage2-final-{champion['version']}",
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest training/test_run_stage2_final.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: 실제 2단계 학습 실행 — `feature_pool` 전체(2019-01-02~현재)로 재학습, `split_by_regime`으로 pre_leverage/leverage_era 성능 분리 리포트, `docs/model_versions.md`에 최종 배포 버전으로 기록, 체크포인트를 `/home/user/AI_Gentle_Viking_RE/serving/best_model_state_dict.pt`로 저장**

Expected: 학습 완료, leverage_era 성능이 pre_leverage보다 낮더라도(§5.2 "변명 가능" 원칙) 기록·보고

- [ ] **Step 6: Commit**

```bash
git add training/run_stage2_final.py training/test_run_stage2_final.py
git commit -m "feat: run stage-2 final retrain on full 2019-present leverage-inclusive data"
```

---

## Task 18: 서빙 통합 (기존 `api_server.py`/crontab 패턴 재사용)

**Files:**
- Create: `serving/inference.py`
- Create: `serving/inference_pipeline.py`
- Create: `serving/api_server.py`
- Test: `serving/test_inference.py`

**Interfaces:**
- Consumes: `build_incomplete_today_bar`(Task 12), `TemporalFusionTransformer`(tft-torch fork), `training/champion_config.json`(Task 16), `serving/best_model_state_dict.pt`(Task 17)
- Produces: `run_inference(ticker: str, encoder_df, today_intraday_rows: list[dict]) -> dict` (`{"pred_label": int, "pred_str": str, "prob_buy": float, "prob_hold": float, "prob_sell": float}`)

- [ ] **Step 1: 실패하는 테스트 작성**

`serving/test_inference.py`:

```python
import numpy as np
import pandas as pd
import torch
from serving.inference import run_inference, softmax_probs


def test_softmax_probs_sums_to_one_and_maps_labels():
    logits = torch.tensor([[2.0, 0.5, 0.1]])
    result = softmax_probs(logits)
    assert result["pred_label"] == 0
    assert result["pred_str"] == "매수"
    assert abs(result["prob_buy"] + result["prob_hold"] + result["prob_sell"] - 1.0) < 1e-6


def test_run_inference_returns_all_required_keys(monkeypatch):
    class FakeModel:
        def eval(self): return self
        def __call__(self, batch):
            return {"class_logits": torch.tensor([[[1.0, 3.0, 0.5]]])}  # 관망이 최댓값

    encoder_df = pd.DataFrame({
        "log_ret": np.random.randn(59) * 0.01, "disparity_20": 1 + np.random.randn(59) * 0.02,
        "time_progress": 1.0, "is_bok": 0, "sector_id": 3, "market_id": 1,
    })
    result = run_inference(
        ticker="005930", encoder_df=encoder_df,
        today_intraday_rows=[{"datetime": "2026-09-08 09:00:00", "open": 100, "high": 101,
                                "low": 99, "close": 100.5, "volume": 1000}],
        model=FakeModel(),
        historical_cols=["log_ret", "disparity_20"], future_cols=["time_progress", "is_bok"],
        static_cols=["sector_id", "market_id"],
    )
    assert set(result.keys()) == {"pred_label", "pred_str", "prob_buy", "prob_hold", "prob_sell"}
    assert result["pred_str"] == "관망"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest serving/test_inference.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

`serving/inference.py`:

```python
import torch
import torch.nn.functional as F
import pandas as pd
from training.dataset import build_incomplete_today_bar

LABEL_MAP = {0: "매수", 1: "관망", 2: "매도"}


def softmax_probs(logits: torch.Tensor) -> dict:
    probs = F.softmax(logits, dim=-1).squeeze().tolist()
    pred_label = int(torch.argmax(logits, dim=-1).item())
    return {
        "pred_label": pred_label, "pred_str": LABEL_MAP[pred_label],
        "prob_buy": probs[0], "prob_hold": probs[1], "prob_sell": probs[2],
    }


def run_inference(ticker: str, encoder_df: pd.DataFrame, today_intraday_rows: list[dict],
                   model, historical_cols: list[str], future_cols: list[str],
                   static_cols: list[str]) -> dict:
    """encoder_df: 과거 59거래일(오늘 제외) 일봉 피처. 오늘 행은 today_intraday_rows로부터
    미완성 일봉으로 계산해 encoder 마지막(60번째) 행으로 붙인다 (설계 §3)."""
    today_bar = build_incomplete_today_bar(today_intraday_rows, open_price=encoder_df.iloc[-1]["close"] if len(encoder_df) else 0)
    today_row = encoder_df.iloc[-1].copy()
    for k in ["open", "high", "low", "close", "volume"]:
        if k in today_row.index:
            today_row[k] = today_bar[k]
    full_encoder = pd.concat([encoder_df, today_row.to_frame().T], ignore_index=True)

    hist = torch.tensor(full_encoder[historical_cols].values.astype("float32")).unsqueeze(0)
    fut = torch.tensor(full_encoder[future_cols].iloc[[-1]].values.astype("float32")).unsqueeze(0)
    static = torch.tensor(full_encoder[static_cols].iloc[[-1]].values.astype("int64")).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        out = model({"historical_ts_numeric": hist, "future_ts_numeric": fut,
                      "static_feats_categorical": static})
    return softmax_probs(out["class_logits"])
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/user/miniconda3/envs/kis_collector/bin/python -m pytest serving/test_inference.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: `inference_pipeline.py`/`api_server.py` 작성 — 기존 원본(`/home/user/inference_pipeline.py`, `/home/user/api_server.py`) 구조를 그대로 계승하되, 신규 모델 로드 경로(`serving/best_model_state_dict.pt`)·`stock_db_v2` 연결·`run_inference` 호출로 교체. 백엔드 통신 인터페이스(`POST /ai/realtime` 등)는 설계 §2 원칙에 따라 그대로 유지. crontab은 원본과 별도 엔트리로 등록(원본 crontab 수정 금지)**

Expected: 로컬에서 `uvicorn serving.api_server:app --port 8001`(원본과 다른 포트) 기동 후 `curl localhost:8001/health` 정상 응답

- [ ] **Step 6: Commit**

```bash
git add serving/inference.py serving/inference_pipeline.py serving/api_server.py serving/test_inference.py
git commit -m "feat: integrate serving pipeline reusing existing api_server/crontab pattern"
```

---

## Self-Review Notes

- **Spec coverage**: §1~§14 전체가 Task 1~18에 매핑됨 — DB/스키마(T1), KIS 클라이언트+격리(T2), 유니버스(T3), 일봉/밸류에이션/수급/지수/매크로/공시/캘린더/섹터(T4-7), 레버리지+VI(T8), 라벨(T9), 피처+레버리지 계산식(T10), 클리핑/스케일링(T11), Dataset+오늘의 미완성 일봉(T12), TFT config+학습루프(T13), 평가(T14), ablation+Optuna+버전관리(T15-16), 2단계 재학습(T17), 서빙(T18)
- **Placeholder scan**: `leverage_products.py`의 ETN `code: None` 2곳은 스펙 §14에 명시된 실제 미해결 항목이며 Task 8 Step 5에서 명시적으로 채우는 단계가 있음 — 방치되는 TODO 아님
- **Type consistency 확인**: `class_logits` 키(T13 train.py, T18 inference.py 동일), `HISTORICAL_COLS`/`KNOWN_FUTURE_COLS`/`STATIC_COLS`(T13에서 정의, T12/T18에서 파라미터로 일관되게 주입), `run_training`의 반환 키(`best_val_loss`, `checkpoint_path`)가 T15/T16/T17에서 일관되게 사용됨
- **범위 밖(향후 과제로 명시, 이 계획엔 미포함)**: 2안(TFT 임베딩+PyCaret), 다른 시계열 모델과의 벤치마크 비교 — 스펙 §13에서 이미 "시간 남으면"으로 낮은 우선순위 처리됨, 9월 마감 압박 고려해 이 계획에서 제외

---

Plan complete and saved to `docs/superpowers/plans/2026-09-08-ai-model-redesign-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** - 태스크마다 새 subagent를 띄워서 처리, 태스크 사이마다 리뷰, 빠른 반복
2. **Inline Execution** - 이 세션에서 직접 태스크를 순서대로 실행, 배치 실행 + 체크포인트마다 확인

어떤 방식으로 진행할까?
