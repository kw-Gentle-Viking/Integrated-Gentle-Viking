import asyncio
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.demo import demo_mode_enabled
from app.dependencies import get_current_user
from app.models import ManualTradeLock, User

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

router = APIRouter()

BASE_URL      = "https://openapivts.koreainvestment.com:29443"  # 모의투자
REAL_BASE_URL = "https://openapi.koreainvestment.com:9443"        # 실전 (시세조회용)
APP_KEY    = os.getenv("KIS_APP_KEY", "")
APP_SECRET = os.getenv("KIS_APP_SECRET", "")
_ACC_NO = os.getenv("KIS_ACC_NO", "")  # "50158327-01"
CANO, ACNT_PRDT_CD = (_ACC_NO.split("-") + ["01"])[:2]


def _truthy_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "y"}


def _use_real_market_data() -> bool:
    return _truthy_env("KIS_MARKET_DATA_REAL", "true")

# ── 토큰 캐시 (모의 / 실전 분리) ──────────────────────────────────────────────
_token_cache: dict      = {"token": None, "expires_at": None}
_real_token_cache: dict = {"token": None, "expires_at": None}
_token_lock = asyncio.Lock()
_real_token_lock = asyncio.Lock()

DEMO_STOCKS = {
    "005930": {"name": "삼성전자", "price": 81200, "change": 0.61, "sign": "2", "volume": 18234000},
    "000660": {"name": "SK하이닉스", "price": 163000, "change": -1.25, "sign": "5", "volume": 9321000},
    "035420": {"name": "NAVER", "price": 186700, "change": 2.10, "sign": "2", "volume": 1245000},
    "105560": {"name": "KB금융", "price": 78200, "change": 0.34, "sign": "2", "volume": 1642000},
    "005380": {"name": "현대차", "price": 249500, "change": -0.82, "sign": "5", "volume": 824000},
}


def _demo_stock(code: str) -> dict:
    return DEMO_STOCKS.get(code, DEMO_STOCKS["005930"])


def _demo_current_price(code: str) -> dict:
    stock = _demo_stock(code)
    price = stock["price"]
    change_rate = stock["change"]
    previous = price / (1 + change_rate / 100)
    change_amount = int(price - previous)
    return {
        "rt_cd": "0",
        "msg1": "LOCAL_DEMO_MODE",
        "output": {
            "stck_prpr": str(price),
            "prdy_vrss": str(abs(change_amount)),
            "prdy_ctrt": f"{abs(change_rate):.2f}",
            "prdy_vrss_sign": stock["sign"],
            "stck_oprc": str(int(price * 0.985)),
            "stck_hgpr": str(int(price * 1.012)),
            "stck_lwpr": str(int(price * 0.974)),
            "acml_vol": str(stock["volume"]),
            "hts_kor_isnm": stock["name"],
        },
    }


def _demo_daily_chart(code: str, range_key: str) -> dict:
    stock = _demo_stock(code)
    start, end, period = _date_range(range_key)
    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")
    step_days = 1 if period == "D" else 7 if period == "W" else 30
    rows = []
    cursor = start_dt
    index = 0
    while cursor <= end_dt and len(rows) < 140:
        if period != "D" or cursor.weekday() < 5:
            wave = math.sin(index / 5) * 0.018
            drift = (index - 40) * 0.0004
            close = max(1000, int(stock["price"] * (1 + wave + drift)))
            open_price = int(close * (1 - 0.004 + math.sin(index) * 0.002))
            high = int(max(open_price, close) * 1.008)
            low = int(min(open_price, close) * 0.992)
            rows.append(
                {
                    "stck_bsop_date": cursor.strftime("%Y%m%d"),
                    "stck_oprc": str(open_price),
                    "stck_hgpr": str(high),
                    "stck_lwpr": str(low),
                    "stck_clpr": str(close),
                    "acml_vol": str(max(10000, int(stock["volume"] * (0.65 + (index % 9) * 0.05)))),
                }
            )
            index += 1
        cursor += timedelta(days=step_days)
    return {"rt_cd": "0", "msg1": "LOCAL_DEMO_MODE", "output1": {"hts_kor_isnm": stock["name"]}, "output2": rows}


def _demo_intraday_chart(code: str) -> dict:
    stock = _demo_stock(code)
    rows = []
    base = stock["price"]
    current = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    for index in range(78):
        close = max(1000, int(base * (1 + math.sin(index / 8) * 0.007 + (index - 39) * 0.00008)))
        open_price = int(close * (1 - 0.002 + math.sin(index / 3) * 0.001))
        rows.append(
            {
                "stck_cntg_hour": current.strftime("%H%M%S"),
                "stck_prpr": str(close),
                "stck_oprc": str(open_price),
                "stck_hgpr": str(int(max(open_price, close) * 1.004)),
                "stck_lwpr": str(int(min(open_price, close) * 0.996)),
                "cntg_vol": str(max(100, int(stock["volume"] / 78 * (0.5 + (index % 6) * 0.12)))),
            }
        )
        current += timedelta(minutes=5)
    return {"rt_cd": "0", "msg1": "LOCAL_DEMO_MODE", "output2": rows}


def _demo_volume_rank() -> dict:
    output = []
    for rank, (code, stock) in enumerate(DEMO_STOCKS.items(), start=1):
        output.append(
            {
                "data_rank": str(rank),
                "mksc_shrn_iscd": code,
                "hts_kor_isnm": stock["name"],
                "stck_prpr": str(stock["price"]),
                "prdy_vrss_sign": stock["sign"],
                "prdy_ctrt": f"{abs(stock['change']):.2f}",
                "acml_tr_pbmn": str(stock["price"] * stock["volume"]),
            }
        )
    return {"rt_cd": "0", "msg1": "LOCAL_DEMO_MODE", "output": output}


def _demo_market_overview() -> dict:
    return {
        "kospi": {
            "rt_cd": "0",
            "output": {
                "bstp_nmix_prpr": "2868.42",
                "bstp_nmix_prdy_vrss": "18.21",
                "bstp_nmix_prdy_ctrt": "0.64",
                "prdy_vrss_sign": "2",
            },
        },
        "kosdaq": {
            "rt_cd": "0",
            "output": {
                "bstp_nmix_prpr": "917.35",
                "bstp_nmix_prdy_vrss": "4.88",
                "bstp_nmix_prdy_ctrt": "0.53",
                "prdy_vrss_sign": "2",
            },
        },
        "usd": {
            "rt_cd": "0",
            "output": {
                "stck_prpr": "1372.40",
                "prdy_vrss": "2.10",
                "prdy_ctrt": "0.15",
                "prdy_vrss_sign": "5",
            },
        },
    }



async def get_real_access_token() -> str:
    now = datetime.now()
    if _real_token_cache["token"] and _real_token_cache["expires_at"] and _real_token_cache["expires_at"] > now:
        return _real_token_cache["token"]

    async with _real_token_lock:
        now = datetime.now()
        if _real_token_cache["token"] and _real_token_cache["expires_at"] and _real_token_cache["expires_at"] > now:
            return _real_token_cache["token"]

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{REAL_BASE_URL}/oauth2/tokenP",
                json={
                    "grant_type": "client_credentials",
                    "appkey": APP_KEY,
                    "appsecret": APP_SECRET,
                },
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"KIS 실전 토큰 발급 실패: {resp.text}")
            data = resp.json()
            _real_token_cache["token"] = data["access_token"]
            expires_in = int(data.get("expires_in", 86400))
            _real_token_cache["expires_at"] = now + timedelta(seconds=expires_in - 60)
            return _real_token_cache["token"]


async def get_access_token() -> str:
    now = datetime.now()
    if _token_cache["token"] and _token_cache["expires_at"] and _token_cache["expires_at"] > now:
        return _token_cache["token"]

    async with _token_lock:
        now = datetime.now()
        if _token_cache["token"] and _token_cache["expires_at"] and _token_cache["expires_at"] > now:
            return _token_cache["token"]

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BASE_URL}/oauth2/tokenP",
                json={
                    "grant_type": "client_credentials",
                    "appkey": APP_KEY,
                    "appsecret": APP_SECRET,
                },
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"KIS 토큰 발급 실패: {resp.text}")
            data = resp.json()
            _token_cache["token"] = data["access_token"]
            expires_in = int(data.get("expires_in", 86400))
            _token_cache["expires_at"] = now + timedelta(seconds=expires_in - 60)
            return _token_cache["token"]


def _base_headers(token: str, tr_id: str) -> dict:
    return {
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": tr_id,
        "Content-Type": "application/json",
    }


def _date_range(range_key: str) -> tuple[str, str, str]:
    """(시작일 YYYYMMDD, 종료일 YYYYMMDD, 기간구분코드 D/W/M)
    거래일 기준 캔들 수:
      1D  → 약 20개 (달력 30일)
      1W  → 약 40개 (달력 60일)
      3M  → 약 65개 (달력 95일)
      1Y  → 약 52개 주봉 (달력 365일)
      5Y  → 약 60개 월봉 (달력 5년)
      ALL → 약 120개 월봉 (달력 10년)
    """
    today = datetime.now()
    end = today.strftime("%Y%m%d")
    mapping = {
        "1D":  (today - timedelta(days=30),     "D"),
        "1W":  (today - timedelta(days=60),     "D"),
        "3M":  (today - timedelta(days=95),     "D"),
        "1Y":  (today - timedelta(days=365),    "W"),
        "5Y":  (today - timedelta(days=365*5),  "M"),
        "ALL": (today - timedelta(days=365*10), "M"),
    }
    start_dt, period = mapping.get(range_key, (today - timedelta(days=95), "D"))
    return start_dt.strftime("%Y%m%d"), end, period


# ── 현재가 조회 ───────────────────────────────────────────────────────────────
# GET /prices/current?code=005930
# KIS: GET /uapi/domestic-stock/v1/quotations/inquire-price
#      tr_id: VHKST01010100 (모의)
@router.get("/prices/current")
async def get_current_price(code: str):
    if demo_mode_enabled():
        return _demo_current_price(code)
    if _use_real_market_data():
        token = await get_real_access_token()
        base_url = REAL_BASE_URL
        tr_id = "FHKST01010100"
    else:
        token = await get_access_token()
        base_url = BASE_URL
        tr_id = "VHKST01010100"

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{base_url}/uapi/domestic-stock/v1/quotations/inquire-price",
            headers=_base_headers(token, tr_id),
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


# ── 일별 차트 데이터 조회 ──────────────────────────────────────────────────────
# GET /prices/chart?code=005930&range=3M
# KIS: GET /uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice
#      tr_id: FHKST03010100
@router.get("/prices/chart")
async def get_chart(code: str, range: str = "3M"):
    if demo_mode_enabled():
        return _demo_daily_chart(code, range)
    if _use_real_market_data():
        token = await get_real_access_token()
        base_url = REAL_BASE_URL
    else:
        token = await get_access_token()
        base_url = BASE_URL

    start, end, period = _date_range(range)
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            headers=_base_headers(token, "FHKST03010100"),
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": code,
                "FID_INPUT_DATE_1": start,
                "FID_INPUT_DATE_2": end,
                "FID_PERIOD_DIV_CODE": period,
                "FID_ORG_ADJ_PRC": "0",
            },
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


# ── 분봉(당일 intraday) 차트 조회 ────────────────────────────────────────────
# GET /prices/intraday?code=005930
# KIS: GET /uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice
#      tr_id: FHKST03010200
#      1회 30건 반환 → 전체 거래일(09:00~15:30) 커버까지 최대 13회 반복 호출
@router.get("/prices/intraday")
async def get_intraday(code: str):
    if demo_mode_enabled():
        return _demo_intraday_chart(code)
    if _use_real_market_data():
        token = await get_real_access_token()
        base_url = REAL_BASE_URL
    else:
        token = await get_access_token()
        base_url = BASE_URL

    now = datetime.now()

    current_hms = now.strftime("%H%M%S")
    reference_time = current_hms if "090000" <= current_hms <= "153000" else "153000"

    all_items: list = []
    seen_times: set[str] = set()

    async with httpx.AsyncClient(timeout=30.0) as client:
        for _ in range(13):
            resp = await client.get(
                f"{base_url}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
                headers=_base_headers(token, "FHKST03010200"),
                params={
                    "FID_ETC_CLS_CODE": "",
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": code,
                    "FID_INPUT_HOUR_1": reference_time,
                    "FID_PW_DATA_INCU_YN": "N",
                },
            )
            if resp.status_code != 200:
                break

            data = resp.json()
            if data.get("rt_cd") != "0" or not data.get("output2"):
                break

            items = data["output2"]
            for item in items:
                candle_time = item.get("stck_cntg_hour", "")
                if not candle_time or candle_time in seen_times:
                    continue
                seen_times.add(candle_time)
                all_items.append(item)

            last_time = items[-1].get("stck_cntg_hour", "")
            if not last_time or last_time <= "090000":
                break

            reference_dt = datetime.strptime(last_time, "%H%M%S") - timedelta(seconds=1)
            reference_time = reference_dt.strftime("%H%M%S")

    return {"rt_cd": "0", "msg1": "", "output2": all_items}


# ── 현금 매수/매도 주문 ───────────────────────────────────────────────────────
# POST /trade/order
# KIS: POST /uapi/domestic-stock/v1/trading/order-cash
#      tr_id: VTTC0012U (매수 모의), VTTC0011U (매도 모의)
class OrderBody(BaseModel):
    code: str
    side: str        # "buy" | "sell"
    qty: int
    price: int       # 시장가는 0
    price_type: str  # "limit" | "market"


@router.post("/trade/order")
async def place_order(
    body: OrderBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    token = await get_access_token()
    tr_id = "VTTC0012U" if body.side == "buy" else "VTTC0011U"
    ord_dvsn = "01" if body.price_type == "market" else "00"
    ord_unpr = "0" if body.price_type == "market" else str(body.price)

    payload = {
        "CANO": CANO,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "PDNO": body.code,
        "ORD_DVSN": ord_dvsn,
        "ORD_QTY": str(body.qty),
        "ORD_UNPR": ord_unpr,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BASE_URL}/uapi/domestic-stock/v1/trading/order-cash",
            headers=_base_headers(token, tr_id),
            json=payload,
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    if data.get("rt_cd") == "0":
        exists = db.query(ManualTradeLock).filter(
            ManualTradeLock.user_id == current_user.id,
            ManualTradeLock.ticker == body.code,
        ).first()
        if not exists:
            db.add(ManualTradeLock(user_id=current_user.id, ticker=body.code))
            db.commit()
        print(f"[User {current_user.id}] 직접매매 종목 등록: {body.code}")

    return data


# ── 실시간 거래량 순위 ────────────────────────────────────────────────────────
# GET /market/volume-rank
# KIS: GET /uapi/domestic-stock/v1/quotations/volume-rank  (실전 도메인)
#      tr_id: FHPST01710000
@router.get("/market/volume-rank")
async def get_volume_rank():
    if demo_mode_enabled():
        return _demo_volume_rank()
    token = await get_real_access_token()
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{REAL_BASE_URL}/uapi/domestic-stock/v1/quotations/volume-rank",
            headers={
                "authorization": f"Bearer {token}",
                "appkey": APP_KEY,
                "appsecret": APP_SECRET,
                "tr_id": "FHPST01710000",
                "Content-Type": "application/json",
            },
            params={
                "FID_COND_MRKT_DIV_CODE":  "J",
                "FID_COND_SCR_DIV_CODE":   "20171",
                "FID_INPUT_ISCD":          "0000",
                "FID_DIV_CLS_CODE":        "0",
                "FID_BLNG_CLS_CODE":       "0",
                "FID_TRGT_CLS_CODE":       "111111111",
                "FID_TRGT_EXLS_CLS_CODE":  "000000",
                "FID_INPUT_PRICE_1":       "0",
                "FID_INPUT_PRICE_2":       "0",
                "FID_VOL_CNT":             "100000",
                "FID_INPUT_DATE_1":        "",
            },
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


def _market_sign(code: str | None) -> str:
    return "2" if code in {"1", "2", "RISING"} else "5"


def _naver_index_to_kis_shape(data: dict | None) -> dict | None:
    if not data:
        return None
    sign_code = (data.get("compareToPreviousPrice") or {}).get("code")
    return {
        "rt_cd": "0",
        "msg1": "NAVER_INDEX",
        "output": {
            "bstp_nmix_prpr": str(data.get("closePrice") or "0").replace(",", ""),
            "bstp_nmix_prdy_vrss": str(data.get("compareToPreviousClosePrice") or "0").replace(",", ""),
            "bstp_nmix_prdy_ctrt": str(data.get("fluctuationsRatio") or "0").replace(",", ""),
            "prdy_vrss_sign": _market_sign(sign_code),
        },
    }


def _naver_fx_to_kis_shape(data: dict | None) -> dict | None:
    result = (data or {}).get("result") or {}
    if not result:
        return None
    sign_code = (result.get("fluctuationsType") or {}).get("code")
    return {
        "rt_cd": "0",
        "msg1": "NAVER_MARKET_INDEX",
        "output": {
            "stck_prpr": str(result.get("closePrice") or "0").replace(",", ""),
            "prdy_vrss": str(result.get("fluctuations") or "0").replace(",", ""),
            "prdy_ctrt": str(result.get("fluctuationsRatio") or "0").replace(",", ""),
            "prdy_vrss_sign": _market_sign(sign_code),
        },
    }


async def _fetch_naver_index(code: str) -> dict | None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"https://m.stock.naver.com/api/index/{code}/basic")
    if resp.status_code != 200:
        return None
    return _naver_index_to_kis_shape(resp.json())


async def _fetch_naver_usd() -> dict | None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            "https://m.stock.naver.com/front-api/marketIndex/productDetail",
            params={"category": "exchange", "reutersCode": "FX_USDKRW"},
        )
    if resp.status_code != 200:
        return None
    return _naver_fx_to_kis_shape(resp.json())


@router.get("/market/overview")
async def get_market_overview():
    if demo_mode_enabled():
        return _demo_market_overview()

    token = None
    try:
        token = await get_real_access_token()
    except Exception as exc:
        print(f"KIS 시장 토큰 발급 실패, 네이버 지표로 대체: {exc}")

    async def fetch_index(iscd: str, fallback_code: str):
        if token:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"{REAL_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-index-price",
                        headers=_base_headers(token, "FHPUP02100000"),
                        params={"FID_COND_MRKT_DIV_CODE": "U", "FID_INPUT_ISCD": iscd},
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("rt_cd") == "0" and data.get("output"):
                        return data
            except Exception as exc:
                print(f"KIS 지수 조회 실패({iscd}), 네이버 지표로 대체: {exc}")
        return await _fetch_naver_index(fallback_code)

    async def fetch_fx():
        if token:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"{REAL_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price",
                        headers=_base_headers(token, "FHKST01010100"),
                        params={"FID_COND_MRKT_DIV_CODE": "X", "FID_INPUT_ISCD": "USD"},
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("rt_cd") == "0" and data.get("output"):
                        return data
            except Exception as exc:
                print(f"KIS 환율 조회 실패, 네이버 지표로 대체: {exc}")
        return await _fetch_naver_usd()

    kospi, kosdaq, usd = await asyncio.gather(
        fetch_index("0001", "KOSPI"),
        fetch_index("1001", "KOSDAQ"),
        fetch_fx(),
    )

    return {"kospi": kospi, "kosdaq": kosdaq, "usd": usd}


def _to_int(value, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return default


def _to_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def _mask_account_no() -> str:
    if not CANO:
        return "모의투자"
    tail = CANO[-4:] if len(CANO) >= 4 else CANO
    return f"****-{tail}-{ACNT_PRDT_CD}"


def _configured_account_no() -> str:
    if CANO and ACNT_PRDT_CD:
        return f"{CANO}-{ACNT_PRDT_CD}"
    return ""


def _validate_mock_account_config() -> None:
    if not APP_KEY or not APP_SECRET:
        raise HTTPException(status_code=500, detail="KIS 앱 키 설정이 없습니다.")
    if not (CANO.isdigit() and len(CANO) == 8 and ACNT_PRDT_CD.isdigit() and len(ACNT_PRDT_CD) == 2):
        raise HTTPException(status_code=500, detail="KIS 모의투자 계좌번호 형식이 올바르지 않습니다. KIS_ACC_NO는 8자리-2자리 형식이어야 합니다.")


@router.get("/account/assets")
async def get_account_assets(current_user: User = Depends(get_current_user)):
    """KIS 모의투자 계좌 자산을 프론트 자산 화면 형식으로 반환."""
    _validate_mock_account_config()
    token = await get_access_token()
    headers = _base_headers(token, "VTTC8434R")
    headers["tr_cont"] = ""
    headers["custtype"] = "P"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{BASE_URL}/uapi/domestic-stock/v1/trading/inquire-balance",
            headers=headers,
            params={
                "CANO": CANO,
                "ACNT_PRDT_CD": ACNT_PRDT_CD,
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "01",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            },
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    response = resp.json()
    if response.get("rt_cd") != "0":
        raise HTTPException(status_code=502, detail=response.get("msg1") or "KIS 모의투자 잔고 조회 실패")

    output1 = response.get("output1") or []
    output2 = response.get("output2") or []
    summary = output2[0] if output2 else {}

    total_krw = _to_int(summary.get("tot_evlu_amt") or summary.get("nass_amt") or summary.get("dnca_tot_amt"))
    cash_krw = _to_int(summary.get("dnca_tot_amt") or summary.get("prvs_rcdl_excc_amt"))
    orderable_krw = _to_int(summary.get("ord_psbl_cash") or summary.get("prvs_rcdl_excc_amt") or summary.get("dnca_tot_amt"))
    invested_krw = _to_int(summary.get("scts_evlu_amt") or summary.get("evlu_amt_smtl_amt"))
    invested_pnl_krw = _to_int(summary.get("evlu_pfls_smtl_amt"))
    invested_pnl_rate = _to_float(summary.get("asst_icdc_erng_rt"))

    holdings = []
    for item in output1:
        qty = _to_int(item.get("hldg_qty"))
        if qty <= 0:
            continue

        label = item.get("prdt_name") or item.get("hts_kor_isnm") or item.get("pdno") or "국내주식"
        value_krw = _to_int(item.get("evlu_amt") or item.get("pchs_amt"))
        pnl_krw = _to_int(item.get("evlu_pfls_amt"))
        pnl_rate = _to_float(item.get("evlu_pfls_rt"))
        holdings.append({
            "flag": "KR",
            "label": label,
            "valueKRW": value_krw,
            "pnlKRW": pnl_krw,
            "pnlRate": pnl_rate,
        })

    return {
        "broker": "한국투자증권 모의투자",
        "accountNo": _configured_account_no(),
        "totalKRW": total_krw,
        "orderableKRW": orderable_krw,
        "cashKRW": cash_krw,
        "investedKRW": invested_krw,
        "investedPnlKRW": invested_pnl_krw,
        "investedPnlRate": invested_pnl_rate,
        "holdings": holdings,
        "monthly": {
            "totalKRW": 0,
            "saleKRW": 0,
            "dividendKRW": 0,
            "interestKRW": 0,
        },
        "raw": {
            "output1_len": len(output1),
            "output2_len": len(output2),
        },
    }

