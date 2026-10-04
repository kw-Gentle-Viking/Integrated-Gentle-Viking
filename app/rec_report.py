import os
import re
import time
import json
from pathlib import Path
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google import genai
import OpenDartReader
import mojito
import FinanceDataReader as fdr

from sqlalchemy.orm import Session
from app.models import (
    Persona,
    SearchKeyword,
    ValueWatchlist,
    StopWord,
    RecommendationReport,
)


# 환경 설정 및 API 로드
load_dotenv(dotenv_path=".env")

CONFIG = {
    "NAVER": {
        "CLIENT_ID": os.getenv("NAVER_CLIENT_ID"),
        "SECRET": os.getenv("NAVER_CLIENT_SECRET"),
    },
    "GEMINI": {"API_KEY": os.getenv("GEMINI_API_KEY"), "MODEL": os.getenv("GEMINI_MODEL", "gemini-3.8-flash")},
    "DART": {"API_KEY": os.getenv("DART_API_KEY")},
    "KIS": {
        "API_KEY": os.getenv("KIS_APP_KEY"),
        "API_SECRET": os.getenv("KIS_APP_SECRET"),
        "ACC_NO": os.getenv("KIS_ACC_NO"),
        "MOCK": os.getenv("KIS_MOCK", "true").lower() == "true",
    },
}


DEFAULT_UNIVERSE_FILE = (
    Path(__file__).resolve().parent / "data" / "stock_kospi200_kosdaq150.csv"
)
UNIVERSE_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}
PRICE_DISCREPANCY_THRESHOLD = 0.20


def _to_int(value) -> int | None:
    try:
        return int(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _to_float(value) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _fetch_fdr_reference_price(stock_code: str) -> dict | None:
    try:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=14)
        df = fdr.DataReader(stock_code, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        if df is None or df.empty or "Close" not in df.columns:
            return None

        df = df.dropna(subset=["Close"])
        if df.empty:
            return None

        latest = df.iloc[-1]
        close = _to_int(latest["Close"])
        if close is None or close <= 0:
            return None

        change_rate = None
        if len(df) >= 2:
            prev_close = _to_float(df.iloc[-2]["Close"])
            if prev_close and prev_close > 0:
                change_rate = round((close - prev_close) / prev_close * 100, 2)

        return {
            "price": close,
            "change": change_rate,
            "date": str(df.index[-1].date()),
            "source": "FDR",
        }
    except Exception as e:
        print(f"FDR 기준가 조회 실패({stock_code}): {e}")
        return None


def _fetch_naver_reference_price(stock_code: str) -> dict | None:
    try:
        url = f"https://finance.naver.com/item/main.naver?code={stock_code}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        match = re.search(
            r'<p class="no_today">.*?<span class="blind">([0-9,]+)</span>',
            resp.text,
            re.S,
        )
        if not match:
            return None
        price = _to_int(match.group(1))
        if price is None or price <= 0:
            return None
        return {"price": price, "change": None, "date": datetime.now().date().isoformat(), "source": "NAVER"}
    except Exception as e:
        print(f"Naver 기준가 조회 실패({stock_code}): {e}")
        return None


def _fetch_consensus_reference_price(stock_code: str) -> dict | None:
    references = [
        ref
        for ref in (
            _fetch_fdr_reference_price(stock_code),
            _fetch_naver_reference_price(stock_code),
        )
        if ref and ref.get("price")
    ]
    if not references:
        return None

    if len(references) == 1:
        return references[0]

    prices = sorted(ref["price"] for ref in references)
    consensus_price = int(sum(prices) / len(prices))
    source = "+".join(ref["source"] for ref in references)
    change = next((ref.get("change") for ref in references if ref.get("change") is not None), None)
    date = max(ref.get("date", "") for ref in references)
    return {"price": consensus_price, "change": change, "date": date, "source": source}


def _apply_price_guard(stock_code: str, market_data: dict) -> dict:
    kis_price = _to_int(market_data.get("price"))
    if kis_price is None or kis_price <= 0:
        reference = _fetch_consensus_reference_price(stock_code)
        if reference:
            market_data["price"] = str(reference["price"])
            if reference.get("change") is not None:
                market_data["change"] = str(reference["change"])
            market_data["price_source"] = reference["source"]
            market_data["price_warning"] = "KIS 현재가가 비어 있어 보조 기준가로 대체했습니다."
        return market_data

    reference = _fetch_consensus_reference_price(stock_code)
    if not reference:
        market_data["price_source"] = "KIS"
        market_data["price_warning"] = None
        return market_data

    reference_price = reference["price"]
    diff_ratio = abs(kis_price - reference_price) / reference_price
    market_data["reference_price"] = str(reference_price)
    market_data["reference_price_source"] = reference["source"]
    market_data["reference_price_date"] = reference["date"]

    if diff_ratio > PRICE_DISCREPANCY_THRESHOLD:
        market_data["raw_kis_price"] = str(kis_price)
        market_data["price"] = str(reference_price)
        if reference.get("change") is not None:
            market_data["change"] = str(reference["change"])
        market_data["price_source"] = reference["source"]
        market_data["price_warning"] = (
            f"KIS 현재가와 보조 기준가 차이가 {diff_ratio:.1%}라 보조 기준가로 대체했습니다."
        )
        print(
            f"{stock_code} 가격 경고: KIS={kis_price}, "
            f"REF={reference_price}, diff={diff_ratio:.1%}"
        )
    else:
        market_data["price_source"] = "KIS"
        market_data["price_warning"] = None

    return market_data


def _normalize_stock_code(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.extract(r"(\d{6})", expand=False)
        .fillna("")
        .str.zfill(6)
    )


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    normalized = {
        str(col).strip().lower().replace(" ", "").replace("_", ""): col
        for col in df.columns
    }
    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    return None


def _xlsx_first_sheet_to_dataframe(path: Path) -> pd.DataFrame:
    import re as _re
    import xml.etree.ElementTree as _ET
    from zipfile import ZipFile

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    def column_index(cell_ref: str) -> int:
        letters = _re.match(r"[A-Z]+", cell_ref).group(0)
        index = 0
        for letter in letters:
            index = index * 26 + ord(letter) - ord("A") + 1
        return index - 1

    with ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = _ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared_strings.append(
                    "".join(text_node.text or "" for text_node in item.findall(".//m:t", ns))
                )

        sheet = _ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall(".//m:sheetData/m:row", ns):
            values = []
            for cell in row.findall("m:c", ns):
                value_node = cell.find("m:v", ns)
                value = "" if value_node is None else value_node.text or ""
                if cell.get("t") == "s" and value:
                    value = shared_strings[int(value)]
                values.append((column_index(cell.get("r", "A1")), value))

            if not values:
                continue

            row_values = [""] * (max(index for index, _ in values) + 1)
            for index, value in values:
                row_values[index] = value
            rows.append(row_values)

    if not rows:
        return pd.DataFrame()

    max_columns = max(len(row) for row in rows)
    padded_rows = [row + [""] * (max_columns - len(row)) for row in rows]
    return pd.DataFrame(padded_rows[1:], columns=padded_rows[0])


def _read_universe_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        for encoding in ("utf-8-sig", "cp949", "euc-kr"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except ImportError:
            if path.suffix.lower() == ".xlsx":
                return _xlsx_first_sheet_to_dataframe(path)
            raise
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"지원하지 않는 파일 형식: {path.suffix}")


def _extract_stock_map(df: pd.DataFrame) -> dict[str, str]:
    code_col = _pick_column(
        df,
        (
            "종목코드",
            "단축코드",
            "코드",
            "code",
            "ticker",
            "symbol",
            "isu_srt_cd",
        ),
    )
    name_col = _pick_column(
        df,
        (
            "회사명",
            "종목명",
            "종목 이름",
            "한글종목약명",
            "name",
            "ticker_name",
            "corp_name",
            "kor_name",
        ),
    )
    if not code_col or not name_col:
        return {}

    result = df[[name_col, code_col]].copy()
    result[name_col] = result[name_col].astype(str).str.strip()
    result[code_col] = _normalize_stock_code(result[code_col])
    result = result[(result[name_col] != "") & (result[code_col].str.len() == 6)]
    return dict(zip(result[name_col], result[code_col]))


def _local_universe_files() -> list[Path]:
    env_file = os.getenv("STOCK_UNIVERSE_FILE")
    if env_file:
        path = Path(env_file).expanduser()
        if path.is_file():
            return [path]
        print(f"STOCK_UNIVERSE_FILE 경로를 찾을 수 없습니다: {path}")

    if DEFAULT_UNIVERSE_FILE.is_file():
        return [DEFAULT_UNIVERSE_FILE]
    return []


def load_local_stock_universe() -> dict[str, str]:
    stock_map: dict[str, str] = {}
    loaded_files = []
    for path in _local_universe_files():
        try:
            item_map = _extract_stock_map(_read_universe_file(path))
        except Exception as e:
            print(f"로컬 종목 파일 읽기 실패({path}): {e}")
            continue
        if item_map:
            stock_map.update(item_map)
            loaded_files.append(str(path))

    if stock_map:
        print(
            f"로컬 종목 universe 로딩 완료: {len(stock_map)}개 "
            f"(파일 {len(loaded_files)}개)"
        )
    return stock_map


def load_fdr_top_universe() -> dict[str, str]:
    df_kospi = fdr.StockListing("KOSPI")
    df_kosdaq = fdr.StockListing("KOSDAQ")

    df_kospi200 = df_kospi.sort_values("Marcap", ascending=False).head(200)
    df_kosdaq150 = df_kosdaq.sort_values("Marcap", ascending=False).head(150)
    df_total = pd.concat([df_kospi200, df_kosdaq150], ignore_index=True)
    df_total["Code"] = df_total["Code"].astype(str).str.zfill(6)
    return dict(zip(df_total["Name"], df_total["Code"]))


def load_krx_all_stock_map() -> dict[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://kind.krx.co.kr/corpgeneral/corpList.do?method=loadInitPage",
    }
    url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download"
    resp = requests.get(url, headers=headers, timeout=10)
    if resp.status_code != 200:
        raise Exception(f"서버 응답 에러: {resp.status_code}")

    import io

    df_all = pd.read_html(io.BytesIO(resp.content), header=0)[0]
    df_all["종목코드"] = df_all["종목코드"].astype(str).str.zfill(6)
    return dict(zip(df_all["회사명"], df_all["종목코드"]))

# 투자 성향별 페르소나 설정 (회원가입 시 결정될 투자 성향 반영)


def get_persona_config(db: Session, persona_id: int) -> dict:
    persona = db.get(Persona, persona_id)
    if not persona:
        persona = db.get(Persona, 2)  # 기본값 : 안정추구형
    return {
        "name": persona.name,
        "gemini_persona": persona.gemini_persona,
        "criteria": persona.criteria,
        "weights": persona.weights,
        "use_value_scout": persona.use_value_scout,
    }


def get_search_keywords(db: Session) -> dict:
    """카테고리별 키워드 딕셔너리 반환"""
    rows = db.query(SearchKeyword).all()
    result = {}
    for row in rows:
        result.setdefault(row.category, []).append(row.keyword)
    return result


def get_value_watchlist(db: Session) -> list[str]:
    """가치주 watchlist 반환"""
    rows = db.query(ValueWatchlist).all()
    return [row.ticker_name for row in rows]


def get_stop_words(db: Session) -> set[str]:
    """종목명 오탐 방지 단어 반환"""
    rows = db.query(StopWord).all()
    return {row.word for row in rows}


def save_report(db: Session, user_id: int, persona_id: int, report: str):
    """추천 결과 DB 저장"""
    db.add(
        RecommendationReport(
            user_id=user_id,
            persona_id=persona_id,
            report=report,
        )
    )
    db.commit()


# 데이터 수집 클래스
class financeDataCollector:
    def __init__(self, config, stop_words: set):
        self.config = config
        self.stop_words = stop_words
        # Geimini 연결
        try:
            self.gemini_client = genai.Client(api_key=config["GEMINI"]["API_KEY"])
            print("Gemini API 연결")
        except Exception as e:
            print(f"Gemini 연결 실패: {e}")

        # 네이버 검색 API 연결
        if config["NAVER"]["CLIENT_ID"] and config["NAVER"]["SECRET"]:
            print("네이버 검색 API 연결")
        else:
            print("네이버 검색 API 연결 실패: NAVER_CLIENT_ID 또는 NAVER_CLIENT_SECRET 누락")

        # DART 연걸
        try:
            self.dart = OpenDartReader(config["DART"]["API_KEY"])
            print("DART API 연결")
        except:
            self.dart = None

        # KIS 연결
        try:
            self.broker = mojito.KoreaInvestment(
                api_key=config["KIS"]["API_KEY"],
                api_secret=config["KIS"]["API_SECRET"],
                acc_no=config["KIS"]["ACC_NO"],
                mock=config["KIS"]["MOCK"],
            )
            print("한투(KIS) API 연결")
        except Exception as e:
            print(f"한투(KIS) API 연결 실패 : {e}")
            self.broker = None

        # 종목 코드 매핑 로딩: 로컬 KOSPI 200 + KOSDAQ 150 universe 우선 사용
        self.stock_map = load_local_stock_universe()
        if self.stock_map:
            return

        try:
            self.stock_map = load_fdr_top_universe()
            print(f"종목 로딩 성공: {len(self.stock_map)}개 (KOSPI 200 + KOSDAQ 150)")
            return
        except Exception as e:
            print(f"FDR KOSPI 200 + KOSDAQ 150 로딩 실패: {e}")

        try:
            self.stock_map = load_krx_all_stock_map()
            print(f"종목 로딩 성공: {len(self.stock_map)}개 (KRX 전체 fallback)")
        except Exception as e:
            print(f"종목 리스트 로딩 실패: {e}")
            self.stock_map = {}

    # HTML 태그 및 특수문자 제거
    def clean_html(self, raw_html):
        if not raw_html:
            return ""
        return re.sub("<.*?>|&quot;|&amp;|&lt;|&gt;", "", raw_html)

    # 네이버 뉴스 API로 뉴스 수집
    def fetch_naver_news(self, keyword):
        url = "https://naverapihub.apigw.ntruss.com/search/v1/news"

        headers = {
            "X-NCP-APIGW-API-KEY-ID": self.config["NAVER"]["CLIENT_ID"],
            "X-NCP-APIGW-API-KEY": self.config["NAVER"]["SECRET"],
        }

        params = {"query": keyword, "display": 10, "sort": "date"}

        try:
            resp = requests.get(url, headers=headers, params=params)
            return resp.json().get("items", []) if resp.status_code == 200 else []
        except:
            return []

    # 뉴스 제목과 설명에서 종목명과 코드 탐지 (최대 매칭 방식)
    def detect_stock_code(self, title, description):
        if not self.stock_map:
            return None, None

        target_text = f"{title} {description}"
        found_name, found_code = None, None

        for name, code in self.stock_map.items():
            if name in self.stop_words:
                continue
            if name in target_text:
                if found_name is None or len(name) > len(found_name):
                    found_name = name
                    found_code = code
        return found_name, found_code

    # KIS API로 시세 및 재무 데이터 조회
    def get_market_data(self, stock_code):
        if not self.broker:
            return {"status": "Error", "msg": "KIS 미연결"}

        try:
            stock_code = stock_code.zfill(6)  # 종목 코드가 6자리가 되도록 앞에 0 채우기
            resp = self.broker.fetch_price(stock_code)

            # 응답 값 확인을 위한 프린트 추가
            if resp.get("rt_cd") == "0":  # 성공 시 '0' 반환
                print(f"{stock_code} 조회 성공!")
                d = resp["output"]
                market_data = {
                    "price": d["stck_prpr"],
                    "change": d["prdy_ctrt"],
                    "volume": d["acml_vol"],
                    "per": d.get("per", "N/A"),
                    "pbr": d.get("pbr", "NA"),
                    "status": "Success",
                    "price_source": "KIS",
                    "price_warning": None,
                }
                return _apply_price_guard(stock_code, market_data)
            else:
                # 실패 시 메시지 출력
                print(f"{stock_code} 조회 실패: {resp.get('msg1')}")
                return {"status": "Error", "msg": resp.get("msg1")}
        except Exception as e:
            return {"status": "Error", "msg": str(e)}

    # DART API로 최근 공시 정보 조회 (최근 3개월)
    def get_dart_info(self, stock_name):
        if not self.dart:
            return {"summary": "DART 미연결", "sources": []}

        try:
            end_dt = datetime.now().strftime("%Y%m%d")
            start_dt = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")
            report = self.dart.list(corp=stock_name, start=start_dt, end=end_dt)

            if report is not None and not report.empty:
                rows = report.head(3)
                sources = []
                for _, row in rows.iterrows():
                    report_name = str(row.get("report_nm", "공시"))
                    rcept_no = str(row.get("rcept_no", "")).strip()
                    if rcept_no:
                        sources.append({
                            "label": report_name,
                            "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}",
                        })
                return {
                    "summary": " | ".join(rows["report_nm"].astype(str).tolist()),
                    "sources": sources,
                }
            return {"summary": "최근 3개월 공시 없음", "sources": []}
        except Exception as e:
            return {"summary": f"DART 조회 실패: {e}", "sources": []}

    # KIS API로 계좌 잔고 및 예수금 조회
    def get_balance(self):
        if not self.broker:
            return None
        try:
            # 계좌 잔고 및 예수금 조회 API 호출
            resp = self.broker.fetch_balance()

            # 1. 응답 데이터가 정상적으로 왔는지 확인
            if resp and "output2" in resp and len(resp["output2"]) > 0:
                # d2_csh_ast_amt (D+2 예수금) 또는 nrciv_blce (미수 제외 예수금)
                # 모의투자와 실전투자에 따라 필드명이 다를 수 있어 안전하게 get 사용
                data = resp["output2"][0]
                deposit = int(data.get("nrciv_blce", data.get("dnca_tot_amt", 0)))
                return deposit
            return 0
        except Exception as e:
            print(f"계좌 조회 실패: {e}")
            return 0

    # 시장 매크로 이슈 수집 (뉴스 키워드 기반)
    def fetch_macro(self):
        macro_keywords = ["전쟁", "금리", "환율", "유가", "원자재", "GDP", "실업률"]
        macro_news = []

        for kw in macro_keywords:
            news = self.fetch_naver_news(kw)
            if news:
                for item in news[:5]:  # 각 키워드당 최대 5개 뉴스만 처리
                    clean_title = self.clean_html(item["title"])
                    macro_news.append(clean_title)
            time.sleep(0.1)

        return " | ".join(set(macro_news))  # 중복 제거 후 연결


# 뉴스 필터링 (점수 계산, 프롬프트 최적화)
def calculate_news_score(row, weight_map, search_keywords):
    score = 0
    text = (str(row["title"]) + " " + str(row["description"])).replace(" ", "")

    # 1. 키워드 매칭 점수 (성향별 가중치 적용)
    for category, keywords in search_keywords.items():
        current_weight = weight_map.get(category, 0)
        for kw in keywords:
            if kw in text:
                score += current_weight

    # 2. 최신성 가산점 (모든 성향 공통)
    time_diff = (datetime.now() - row["pubDate_clean"]).total_seconds()
    if time_diff < 3600 * 4:  # 4시간 이내
        score += 3

    return score


def _make_source(label: str, url: str) -> dict:
    return {"label": str(label).strip() or "출처", "url": str(url).strip()}


def _source_has_url(source) -> bool:
    if isinstance(source, str):
        return source.startswith("http://") or source.startswith("https://")
    if isinstance(source, dict):
        url = source.get("url") or source.get("href") or ""
        return str(url).startswith("http://") or str(url).startswith("https://")
    return False


def _format_source_lines(sources: list[dict]) -> str:
    if not sources:
        return "- 없음"
    return "\n".join(
        f"- label: {source.get('label', '출처')} | url: {source.get('url', '')}"
        for source in sources
        if source.get("url")
    ) or "- 없음"


def _build_material_flow_sources(ticker: str) -> list[dict]:
    ticker = str(ticker or "").zfill(6)
    if not ticker or ticker == "000000":
        return []
    return [
        _make_source("네이버 금융 종목 시세", f"https://finance.naver.com/item/sise.naver?code={ticker}"),
        _make_source("네이버 금융 투자자별 매매동향", f"https://finance.naver.com/item/frgn.naver?code={ticker}"),
    ]


def _collect_news_sources(group: pd.DataFrame) -> list[dict]:
    sources = []
    for _, row in group.head(3).iterrows():
        link = str(row.get("link", "")).strip()
        title = str(row.get("title", "뉴스 원문")).strip()
        if link.startswith("http://") or link.startswith("https://"):
            sources.append(_make_source(title, link))
    return sources


def _extract_dart_summary(dart_data) -> str:
    if isinstance(dart_data, dict):
        return str(dart_data.get("summary", "데이터 없음"))
    return str(dart_data or "데이터 없음")


def _extract_dart_sources(dart_data) -> list[dict]:
    if isinstance(dart_data, dict):
        return [source for source in dart_data.get("sources", []) if _source_has_url(source)]
    return []



def enrich_report_sources(report_text: str, df_news: pd.DataFrame, scout_data: list[dict]) -> str:
    try:
        report_json = json.loads(report_text)
    except Exception:
        return report_text

    news_sources_by_ticker = {}
    dart_sources_by_ticker = {}

    if "stock_code" in df_news.columns:
        for ticker, group in df_news[df_news["stock_code"].notnull()].groupby("stock_code"):
            ticker = str(ticker).zfill(6)
            news_sources_by_ticker[ticker] = _collect_news_sources(group)
            first_dart = group.iloc[0].get("dart_data")
            dart_sources_by_ticker[ticker] = _extract_dart_sources(first_dart)

    for scout in scout_data:
        ticker = str(scout.get("code", "")).zfill(6)
        if ticker and ticker not in news_sources_by_ticker:
            news_sources_by_ticker[ticker] = []
        if ticker and ticker not in dart_sources_by_ticker:
            dart_sources_by_ticker[ticker] = []

    for item in report_json.get("recommendations", []):
        ticker = str(item.get("ticker", "")).zfill(6)
        reasons = item.setdefault("reasons", {})

        news = reasons.setdefault("news", {})
        if not any(_source_has_url(source) for source in news.get("sources", [])):
            news["sources"] = news_sources_by_ticker.get(ticker, [])

        disclosure = reasons.setdefault("disclosure", {})
        if not any(_source_has_url(source) for source in disclosure.get("sources", [])):
            disclosure["sources"] = dart_sources_by_ticker.get(ticker, [])

        material_flow = reasons.setdefault("materialFlow", {})
        if not any(_source_has_url(source) for source in material_flow.get("sources", [])):
            material_flow["sources"] = _build_material_flow_sources(ticker)

    return json.dumps(report_json, ensure_ascii=False)


# 프롬프트 최적화: 프론트 카드 렌더링에 필요한 근거를 구조화해서 제공
def optimize_prompt(df_news, scout_data, persona_conf):
    context_blocks = []

    news_grouped = df_news[df_news["stock_code"].notnull()].groupby("stock_code")
    for code, group in news_grouped:
        first = group.iloc[0]
        name = first["stock_name"]
        m = first["market_data"]
        d = first["dart_data"]
        disclosure_summary = _extract_dart_summary(d)
        disclosure_source_lines = _format_source_lines(_extract_dart_sources(d))
        material_flow_source_lines = _format_source_lines(_build_material_flow_sources(code))

        if not m or m.get("status") != "Success":
            continue

        news_items = []
        for _, row in group.head(5).iterrows():
            news_items.append(
                f"- title: {row['title']} | score: {row.get('score', 0)} | "
                f"date: {row.get('pubDate', '')} | link: {row.get('link', '')}"
            )

        block = f"""
        [CANDIDATE]
        source_type: news
        ticker: {code}
        name: {name}
        current_price: {m.get('price')}
        change_rate: {m.get('change')}
        volume: {m.get('volume')}
        per: {m.get('per')}
        pbr: {m.get('pbr')}
        price_source: {m.get('price_source')}
        price_warning: {m.get('price_warning')}
        raw_kis_price: {m.get('raw_kis_price')}
        reference_price: {m.get('reference_price')}
        disclosure_recent_3m: {disclosure_summary}
        disclosure_sources:
        {disclosure_source_lines}
        material_flow_sources:
        {material_flow_source_lines}
        related_news:
        {chr(10).join(news_items)}
        [/CANDIDATE]
        """
        context_blocks.append(block)

    for s in scout_data:
        m = s["market_data"]
        if m["status"] != "Success":
            continue

        block = f"""
        [CANDIDATE]
        source_type: value_scout
        ticker: {s.get('code', '')}
        name: {s['name']}
        current_price: {m.get('price')}
        change_rate: {m.get('change')}
        volume: {m.get('volume')}
        per: {m.get('per')}
        pbr: {m.get('pbr')}
        price_source: {m.get('price_source')}
        price_warning: {m.get('price_warning')}
        raw_kis_price: {m.get('raw_kis_price')}
        reference_price: {m.get('reference_price')}
        disclosure_recent_3m: 데이터 없음
        disclosure_sources:
        - 없음
        material_flow_sources:
        {_format_source_lines(_build_material_flow_sources(s.get('code', '')))}
        related_news:
        - title: 특이 뉴스 없음 (재무 스캔 발굴) | score: 0 | date: | link:
        [/CANDIDATE]
        """
        context_blocks.append(block)

    return "\n".join(context_blocks)


def _extract_json_object(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start > end:
        raise ValueError("Gemini 응답에서 JSON 객체를 찾을 수 없습니다.")
    return text[start : end + 1]


def _parse_candidates_from_context(context: str) -> list[dict]:
    candidates = []
    for block in str(context or "").split("[CANDIDATE]")[1:]:
        block = block.split("[/CANDIDATE]", 1)[0]
        item = {}
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in {
                "source_type",
                "ticker",
                "name",
                "current_price",
                "change_rate",
                "volume",
                "per",
                "pbr",
                "price_warning",
            }:
                item[key] = value
        if item.get("ticker") and item.get("name"):
            candidates.append(item)
    return candidates


def _build_fallback_report(
    persona_conf: dict,
    user_deposit: int,
    market_context: str,
    context: str = "",
    error_message: str | None = None,
) -> str:
    candidates = _parse_candidates_from_context(context)
    recommendations = []
    investable_percent = 90 if candidates else 0
    per_stock_percent = max(1, int(investable_percent / min(len(candidates), 5))) if candidates else 0

    for rank, item in enumerate(candidates[:5], start=1):
        price = _to_int(item.get("current_price")) or 0
        change_rate = _to_float(item.get("change_rate")) or 0
        amount = int(user_deposit * per_stock_percent / 100) if price > 0 else 0
        quantity = int(amount / price) if price > 0 else 0
        warning = item.get("price_warning")
        details = (
            f"PER {item.get('per', 'N/A')}, PBR {item.get('pbr', 'N/A')}, "
            f"거래량 {item.get('volume', 'N/A')} 기준으로 후보에 포함됐습니다."
        )
        if warning and warning != "None":
            details += f" {warning}"

        recommendations.append(
            {
                "rank": rank,
                "ticker": str(item.get("ticker", "")).zfill(6),
                "name": item.get("name", "추천 후보"),
                "recommendationScore": max(50, 78 - (rank - 1) * 5),
                "currentPrice": price,
                "changeRate": change_rate,
                "summary": "Gemini 응답 생성에 실패해 수집된 뉴스, 공시, 시세 후보 데이터 기준으로 임시 리포트를 구성했습니다.",
                "allocation": {
                    "weightPercent": per_stock_percent,
                    "amountKrw": amount,
                    "quantity": quantity,
                },
                "strategy": "관망" if quantity == 0 else "매수",
                "risk": error_message or "Gemini 응답을 JSON으로 변환하지 못해 보수적인 임시 판단을 표시합니다.",
                "reasons": {
                    "news": {
                        "title": "뉴스",
                        "headline": "뉴스 기반 후보로 감지되었습니다.",
                        "details": "네이버 검색 API로 수집한 기사에서 종목명이 탐지되어 후보군에 포함됐습니다.",
                        "sources": [],
                        "tags": ["뉴스 후보"],
                    },
                    "disclosure": {
                        "title": "공시",
                        "headline": "공시 데이터는 별도 확인이 필요합니다.",
                        "details": "DART 조회 결과가 있으면 출처가 자동 보강됩니다.",
                        "sources": [],
                        "tags": ["확인 필요"],
                    },
                    "materialFlow": {
                        "title": "재료/수급",
                        "headline": "시세와 밸류 지표를 기준으로 임시 산출했습니다.",
                        "details": details,
                        "sources": _build_material_flow_sources(item.get("ticker", "")),
                        "tags": [item.get("source_type", "candidate")],
                    },
                },
            }
        )

    return json.dumps(
        {
            "personaName": persona_conf.get("name", "AI"),
            "userDepositKrw": user_deposit,
            "cashReservePercent": 100 - sum(
                item["allocation"]["weightPercent"] for item in recommendations
            ),
            "marketSummary": str(market_context or "시장 요약 데이터가 부족합니다."),
            "recommendations": recommendations,
            "fallbackReason": error_message,
        },
        ensure_ascii=False,
    )


# Gemini API로 프론트 카드용 추천 포트폴리오 JSON 생성
def run_gemini(client, context, market_context, persona_conf, user_deposit=0):
    formatted_deposit = f"{user_deposit:,}원"

    prompt = f"""
    당신은 **{persona_conf['gemini_persona']}**입니다.

    [현재 시장 전체 상황]
    {market_context}

    [사용자 투자 가능 현금]
    {formatted_deposit}

    [후보 종목 데이터]
    {context}

    [사용자 투자 성향 기준]
    {persona_conf['criteria']}

    [임무]
    후보 종목 데이터만 사용해서 사용자의 투자 성향에 맞는 Top 5 포트폴리오를 추천하세요.
    프론트엔드는 추천점수/현재가/등락률 카드와 핵심 요약, 뉴스/공시/재료·수급 카드를 렌더링합니다.
    따라서 반드시 아래 JSON 스키마 형식으로 출력하세요.

    [출력 규칙]
    - 마크다운, 코드블록, 설명 문장 없이 유효한 JSON 객체만 출력하세요.
    - 숫자는 문자열이 아니라 number로 출력하세요. 단위(원, %, 주)는 붙이지 마세요.
    - currentPrice는 원화 현재가 number, changeRate는 퍼센트 number입니다. 예: +0.61%는 0.61
    - recommendationScore는 0부터 100까지의 정수입니다.
    - summary는 프로토타입의 핵심 요약 카드에 들어갈 1~2문장입니다.
    - reasons.news/disclosure/materialFlow는 각각 프론트 카드 하나입니다.
    - details는 1~2문장, tags는 1~3개 문자열 배열입니다.
    - sources는 반드시 실제 URL이 있는 출처만 넣으세요. URL이 없는 출처명이나 DART/KRX 홈 링크는 넣지 마세요.
    - disclosure 데이터가 없으면 disclosure 카드에는 "최근 확인된 주요 공시는 없습니다."처럼 솔직히 쓰세요.
    - materialFlow 카드에는 거래량, 등락률, PER/PBR, 업종/재료 추론 중 확인 가능한 내용만 쓰세요.
    - price_warning이 있으면 risk 또는 materialFlow.details에 가격 검증 경고를 짧게 반영하세요.
    - 추천 비중 합계는 100 이하로 하며 현금 보유 비중도 포함하세요.
    - 현재가가 userDepositKrw보다 큰 종목은 추천하지 마세요.

    [JSON 스키마]
    {{
      "personaName": "{persona_conf['name']}",
      "userDepositKrw": {user_deposit},
      "cashReservePercent": 0,
      "marketSummary": "string",
      "recommendations": [
        {{
          "rank": 1,
          "ticker": "후보 종목 데이터의 ticker",
          "name": "후보 종목 데이터의 name",
          "recommendationScore": 0,
          "currentPrice": 0,
          "changeRate": 0,
          "summary": "string",
          "allocation": {{
            "weightPercent": 0,
            "amountKrw": 0,
            "quantity": 0
          }},
          "strategy": "매수 | 관망 | 비중확대 | 제외",
          "risk": "string",
          "reasons": {{
            "news": {{
              "title": "뉴스",
              "headline": "string",
              "details": "string",
              "sources": [{{"label": "원문 제목", "url": "https://..."}}],
              "tags": []
            }},
            "disclosure": {{
              "title": "공시",
              "headline": "string",
              "details": "string",
              "sources": [],
              "tags": []
            }},
            "materialFlow": {{
              "title": "재료/수급",
              "headline": "string",
              "details": "string",
              "sources": [],
              "tags": []
            }}
          }}
        }}
      ]
    }}

    위 스키마의 ticker/name/currentPrice/changeRate 값은 예시값이 아닙니다.
    반드시 [후보 종목 데이터]에 실제로 존재하는 종목의 ticker/name/current_price/change_rate를 변환해 채우세요.
    """
    try:
        try:
            resp = client.models.generate_content(
                model=CONFIG["GEMINI"]["MODEL"],
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
        except TypeError:
            resp = client.models.generate_content(
                model=CONFIG["GEMINI"]["MODEL"], contents=prompt
            )
        report_text = _extract_json_object(resp.text)
        json.loads(report_text)
        return report_text
    except Exception as e:
        print(f"Gemini 리포트 생성 실패: {e}")
        return _build_fallback_report(
            persona_conf,
            user_deposit,
            market_context,
            context,
            error_message=f"Gemini 리포트 생성 실패: {e}",
        )


# 메인 함수: 투자 성향에 따른 맞춤형 포트폴리오 추천 / db버전
def get_ai_recommendation(db: Session, user_id: int, persona_id: int) -> str:
    # DB에서 전부 가져오기
    p_conf = get_persona_config(db, persona_id)

    if os.getenv("LOCAL_DEMO_MODE") == "1":
        demo_context = """
[CANDIDATE]
ticker: 005930
name: 삼성전자
current_price: 81200
change_rate: 0.61
volume: 18234000
per: 18.4
pbr: 1.32
source_type: local_demo

[CANDIDATE]
ticker: 000660
name: SK하이닉스
current_price: 163000
change_rate: -1.25
volume: 9321000
per: 21.8
pbr: 1.71
source_type: local_demo

[CANDIDATE]
ticker: 035420
name: NAVER
current_price: 186700
change_rate: 2.10
volume: 1245000
per: 24.3
pbr: 1.19
source_type: local_demo
"""
        report = _build_fallback_report(
            p_conf,
            10_000_000,
            "로컬 시연 모드: 외부 뉴스/공시/API 호출 없이 고정 후보로 리포트를 생성했습니다.",
            demo_context,
            error_message="LOCAL_DEMO_MODE",
        )
        save_report(db, user_id=user_id, persona_id=persona_id, report=report)
        return report

    search_keywords = get_search_keywords(db)
    value_watchlist = get_value_watchlist(db)
    stop_words = get_stop_words(db)

    collector = financeDataCollector(CONFIG, stop_words)

    market_status = collector.fetch_macro()

    user_cash = collector.get_balance()
    if not user_cash:
        user_cash = 1_000_000

    # [Step 1] 뉴스 수집
    all_news = []
    for _, keywords in search_keywords.items():
        for kw in keywords:
            all_news.extend(collector.fetch_naver_news(kw))

    if not all_news:
        return _build_fallback_report(
            p_conf,
            user_cash,
            market_status,
            error_message="현재 수집된 시장 데이터가 없어 분석을 진행할 수 없습니다.",
        )

    # 데이터프레임 처리
    df = pd.DataFrame(all_news)
    df["title"] = df["title"].apply(collector.clean_html)
    df["description"] = df["description"].apply(collector.clean_html)
    df["pubDate_clean"] = pd.to_datetime(df["pubDate"]).dt.tz_localize(None)
    df.drop_duplicates(subset=["title"], keep="first", inplace=True)
    df["score"] = df.apply(
        lambda row: calculate_news_score(row, p_conf["weights"], search_keywords),
        axis=1,
    )
    df_top = df.nlargest(15, "score").copy()

    # [Step 2] 종목 식별 및 시세 조회
    df_top["stock_name"] = None
    df_top["stock_code"] = None
    df_top["market_data"] = None
    df_top["dart_data"] = None
    df_top = df_top.astype({"market_data": "object", "dart_data": "object"})

    data_cache = {}

    for idx, row in df_top.iterrows():
        name, code = collector.detect_stock_code(row["title"], row["description"])
        df_top.at[idx, "stock_name"] = name
        df_top.at[idx, "stock_code"] = code

        if code and code not in data_cache:
            m_data = collector.get_market_data(code)
            time.sleep(0.5)
            d_data = collector.get_dart_info(name)
            time.sleep(0.5)
            if m_data.get("status") == "Success":
                data_cache[code] = {"market": m_data, "dart": d_data}
            time.sleep(0.7)

        if code in data_cache:
            df_top.at[idx, "market_data"] = data_cache[code].get("market", {})
            df_top.at[idx, "dart_data"] = data_cache[code].get("dart", "데이터 없음")

    # [Step 3] 가치주 스카우팅
    scout_list = []
    if p_conf["use_value_scout"]:
        for v_name in value_watchlist:
            v_code = collector.stock_map.get(v_name)
            if v_code:
                v_data = collector.get_market_data(v_code)
                if v_data.get("status") == "Success":
                    scout_list.append({"name": v_name, "code": v_code, "market_data": v_data})

    # [Step 4] Gemini 분석
    final_context = optimize_prompt(df_top, scout_list, p_conf)
    candidate_count = final_context.count("[CANDIDATE]")
    candidate_lines = [
        line.strip()
        for line in final_context.splitlines()
        if line.strip().startswith(("ticker:", "name:", "current_price:"))
    ]
    print("[AI_REPORT_DEBUG] user_cash:", user_cash)
    print("[AI_REPORT_DEBUG] candidate_count:", candidate_count)
    print("[AI_REPORT_DEBUG] candidates:", " | ".join(candidate_lines))
    report = run_gemini(
        collector.gemini_client, final_context, market_status, p_conf, user_cash
    )
    report = enrich_report_sources(report, df_top, scout_list)

    # [Step 5] 결과 DB 저장
    save_report(db, user_id=user_id, persona_id=persona_id, report=report)

    return report
