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
    # 2026-09-09 백로그 항목에서 해결: KIS 종목마스터파일(kospi_code.mst, 미래에셋 issuer 계열 "Q520xxx")로
    # 정확한 코드/공식명 확인, KIS 일별시세 API로 실거래 데이터까지 검증 완료.
    # 공식명은 "TIGER" 브랜드가 아니라 "미래에셋"(발행사명 그대로) — ETN은 KRX 관행상 ETF와 달리
    # 발행 증권사명을 그대로 브랜드로 씀. 코드가 "Q"+6자리(총 7자)로 다른 16개(6자)보다 길어
    # stock_db_v2의 leverage_products/leverage_daily.code 컬럼을 VARCHAR(6)->VARCHAR(7)로 확장함.
    {"code": "Q520100", "product_name": "미래에셋 레버리지 삼성전자 단일종목 ETN", "underlying_ticker": "005930", "multiple": 2.0, "product_type": "ETN", "issuer": "Mirae Asset", "listed_date": "2026-05-27"},
    {"code": "Q520101", "product_name": "미래에셋 레버리지 SK하이닉스 단일종목ETN", "underlying_ticker": "000660", "multiple": 2.0, "product_type": "ETN", "issuer": "Mirae Asset", "listed_date": "2026-05-27"},
]
