import time
import psycopg2
from data_collection.kis_client import KisClient

# KIS API rate limit: ~18 calls per second
KIS_API_INTERVAL = 0.056

# 20개 "업종지수"(sector_daily_ohlcv, 0005~0026)의 한글명 — sector_id 0~19는 이 리스트의 인덱스와
# 동일한 순서로 대응한다(design brief 확정 매핑, data_collection/run_dart_calendar_sector_backfill.py의
# SECTOR_CODES와 같은 순서). KIS 현재가 조회(FHKST01010100)가 반환하는 bstp_kor_isnm은 이보다 세분화된
# 별도 분류체계라 정확히 일치하지 않는 이름(예: "IT 서비스")이 나올 수 있음 — 그런 경우는 sector_id=20
# ("unclassified", cardinality 21의 마지막 버킷)으로 분류한다.
SECTOR_NAMES = [
    "음식료·담배", "섬유·의류", "종이·목재", "화학", "제약", "비금속", "금속", "기계·장비",
    "전기·전자", "의료·정밀기기", "운송장비·부품", "유통", "전기·가스", "건설", "운송·창고",
    "통신", "금융", "증권", "보험", "일반서비스",
]

UNCLASSIFIED_SECTOR_ID = 20


def match_sector_id(bstp_kor_isnm: str, sector_names: list[str] = SECTOR_NAMES) -> int:
    """KIS bstp_kor_isnm(업종명)을 20개 알려진 sector 이름과 정확히 매칭해서 sector_id(0~19)를
    반환한다. 매칭되지 않으면 20(unclassified, 의도된 정상 결과)을 반환한다."""
    if bstp_kor_isnm in sector_names:
        return sector_names.index(bstp_kor_isnm)
    return UNCLASSIFIED_SECTOR_ID


def load_tickers_from_db(dsn: str) -> list[str]:
    """Load the 200-ticker universe from ticker_universe table."""
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("SELECT ticker FROM ticker_universe ORDER BY ticker")
        tickers = [row[0] for row in cur.fetchall()]
    conn.close()
    return tickers


def fetch_sector_names(client: KisClient, tickers: list[str]) -> dict[str, str]:
    """티커별 KIS 현재가 조회에서 bstp_kor_isnm(업종명)을 가져온다. 개별 티커 실패는
    스킵하고 계속 진행(전체 200건 루프가 한 건 때문에 중단되면 안 됨)."""
    names: dict[str, str] = {}
    for i, ticker in enumerate(tickers):
        try:
            data = client.request(
                path="/uapi/domestic-stock/v1/quotations/inquire-price",
                tr_id="FHKST01010100",
                params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker},
            )
            name = data.get("output", {}).get("bstp_kor_isnm")
            if name:
                names[ticker] = name
            else:
                print(f"  [skip] {ticker}: empty bstp_kor_isnm in response")
        except Exception as exc:
            print(f"  [skip] {ticker}: {exc}")
        finally:
            time.sleep(KIS_API_INTERVAL)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(tickers)}] fetched so far...")
    return names


def update_sector_ids(dsn: str, sector_id_by_ticker: dict[str, int]) -> None:
    if not sector_id_by_ticker:
        return
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        for ticker, sector_id in sector_id_by_ticker.items():
            cur.execute(
                "UPDATE ticker_universe SET sector_id = %s WHERE ticker = %s",
                (sector_id, ticker),
            )
    conn.commit()
    conn.close()


def run_backfill(dsn: str, client: KisClient) -> dict[str, int]:
    """전체 백필 실행: 200 티커 조회 → 매칭 → DB 업데이트. 반환값은 ticker -> sector_id."""
    tickers = load_tickers_from_db(dsn)
    print(f"Loaded {len(tickers)} tickers from ticker_universe")

    names = fetch_sector_names(client, tickers)
    print(f"Fetched sector names for {len(names)}/{len(tickers)} tickers")

    sector_id_by_ticker = {t: match_sector_id(name) for t, name in names.items()}

    matched = sum(1 for sid in sector_id_by_ticker.values() if sid != UNCLASSIFIED_SECTOR_ID)
    unclassified = sum(1 for sid in sector_id_by_ticker.values() if sid == UNCLASSIFIED_SECTOR_ID)
    print(f"Matched to a real sector (0-19): {matched}")
    print(f"Fell back to unclassified (20): {unclassified}")

    update_sector_ids(dsn, sector_id_by_ticker)
    print(f"Updated sector_id for {len(sector_id_by_ticker)} tickers in ticker_universe")

    return sector_id_by_ticker
