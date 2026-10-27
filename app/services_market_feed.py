# app/services_market_feed.py
"""홈 대시보드 "오늘의 뉴스"/"오늘의 공시" 실제 데이터 조회.

이 레포에 이미 네이버 검색 API + DART API 연동 코드가 있었다(app/rec_report.py의
financeDataCollector, AI 리포트 생성용). 거기선 Gemini/KIS까지 같이 붙는 무거운 클래스라
대시보드 카드 두 개 채우자고 그대로 가져다 쓰기엔 과하다 -- 여기선 그 두 API만 가볍게 재사용한다.
지금 AI가 실제로 추론하는 5종목 기준으로 모은다(프론트 KNOWN_STOCK_NAMES와 동일 목록).
"""
import os
import re
import time
from datetime import datetime, timedelta

import requests
import OpenDartReader

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
DART_API_KEY = os.getenv("DART_API_KEY")

# 프론트 lib/stock-list/StockListContext.tsx의 KNOWN_STOCK_NAMES와 맞춘다.
TRACKED_STOCKS = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "005380": "현대차",
    "068270": "셀트리온",
    "207940": "삼성바이오로직스",
}

CACHE_TTL_SEC = 600  # 네이버/DART 둘 다 외부 호출이라 10분 캐시(대시보드 새로고침마다 때리지 않음)
_cache: dict = {"news": None, "news_at": 0.0, "disclosures": None, "disclosures_at": 0.0}

_dart = None
_dart_init_failed = False


def _get_dart():
    global _dart, _dart_init_failed
    if _dart is not None or _dart_init_failed:
        return _dart
    if not DART_API_KEY:
        _dart_init_failed = True
        return None
    try:
        _dart = OpenDartReader(DART_API_KEY)
    except Exception as e:  # noqa: BLE001
        print(f"DART API 연결 실패: {e}")
        _dart_init_failed = True
        _dart = None
    return _dart


def _clean_html(raw: str) -> str:
    if not raw:
        return ""
    return re.sub("<.*?>|&quot;|&amp;|&lt;|&gt;|&#39;", "", raw)


def _parse_naver_date(value: str) -> datetime:
    # 예: "Mon, 27 Oct 2026 14:05:00 +0900"
    try:
        return datetime.strptime(value, "%a, %d %b %Y %H:%M:%S %z")
    except (ValueError, TypeError):
        return datetime.min.replace(tzinfo=None)


def fetch_recent_news(limit: int = 4, force_refresh: bool = False) -> list[dict]:
    """[{title, company, source, time, url}], 최신순. 네이버 API 미설정/실패 시 빈 리스트."""
    now = time.monotonic()
    if not force_refresh and _cache["news"] is not None and now - _cache["news_at"] < CACHE_TTL_SEC:
        return _cache["news"][:limit]

    if not (NAVER_CLIENT_ID and NAVER_CLIENT_SECRET):
        return []

    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
    }

    collected = []
    for code, name in TRACKED_STOCKS.items():
        try:
            resp = requests.get(
                "https://naverapihub.apigw.ntruss.com/search/v1/news",
                headers=headers,
                params={"query": name, "display": 5, "sort": "date"},
                timeout=5,
            )
            items = resp.json().get("items", []) if resp.status_code == 200 else []
        except Exception as e:  # noqa: BLE001
            print(f"네이버 뉴스 조회 실패({name}): {e}")
            items = []

        for item in items:
            pub_dt = _parse_naver_date(item.get("pubDate", ""))
            collected.append({
                "title": _clean_html(item.get("title", "")),
                "company": name,
                "ticker": code,
                "source": "Naver News",
                "time": pub_dt.strftime("%H:%M") if pub_dt != datetime.min.replace(tzinfo=None) else "-",
                "_sort_key": pub_dt,
                "url": item.get("originallink") or item.get("link", ""),
            })

    collected.sort(key=lambda x: x["_sort_key"], reverse=True)
    for item in collected:
        del item["_sort_key"]

    _cache["news"] = collected
    _cache["news_at"] = now
    return collected[:limit]


def fetch_recent_disclosures(limit: int = 4, force_refresh: bool = False) -> list[dict]:
    """[{company, title, time, url}], 최신순. DART API 미설정/실패 시 빈 리스트."""
    now = time.monotonic()
    if not force_refresh and _cache["disclosures"] is not None and now - _cache["disclosures_at"] < CACHE_TTL_SEC:
        return _cache["disclosures"][:limit]

    dart = _get_dart()
    if dart is None:
        return []

    end_dt = datetime.now().strftime("%Y%m%d")
    start_dt = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

    collected = []
    for code, name in TRACKED_STOCKS.items():
        try:
            # 회사명(예: "현대차")이 DART 등록명(예: "현대자동차(주)")과 안 맞는 경우가 있어
            # 이름 대신 종목코드로 조회한다(OpenDartReader가 코드도 그대로 받는다).
            report = dart.list(corp=code, start=start_dt, end=end_dt)
        except Exception as e:  # noqa: BLE001
            print(f"DART 공시 조회 실패({name}): {e}")
            continue
        if report is None or report.empty:
            continue
        for _, row in report.head(5).iterrows():
            rcept_no = str(row.get("rcept_no", "")).strip()
            rcept_dt = str(row.get("rcept_dt", ""))
            collected.append({
                "company": name,
                "ticker": code,
                "title": str(row.get("report_nm", "공시")),
                "time": f"{rcept_dt[:4]}-{rcept_dt[4:6]}-{rcept_dt[6:8]}" if len(rcept_dt) == 8 else rcept_dt,
                "_sort_key": rcept_dt,
                "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}" if rcept_no else "",
            })

    collected.sort(key=lambda x: x["_sort_key"], reverse=True)
    for item in collected:
        del item["_sort_key"]

    _cache["disclosures"] = collected
    _cache["disclosures_at"] = now
    return collected[:limit]
