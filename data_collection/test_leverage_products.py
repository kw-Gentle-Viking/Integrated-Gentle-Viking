from data_collection.leverage_products import LEVERAGE_PRODUCTS


def test_has_16_confirmed_etfs_and_2_confirmed_etns():
    etfs = [p for p in LEVERAGE_PRODUCTS if p["product_type"] == "ETF"]
    etns = [p for p in LEVERAGE_PRODUCTS if p["product_type"] == "ETN"]
    assert len(etfs) == 16
    assert len(etns) == 2
    assert all(p["code"] is not None for p in etfs)
    # 2026-09-09 백로그에서 해결: KIS 종목마스터파일 + KIS 일별시세 API로 검증된 실제 코드.
    # code is not None만 확인하면 나중에 실수로 다른(틀린) 코드가 들어가도 못 잡으므로
    # 정확한 값까지 고정해서 회귀를 방지한다(리뷰 2026-09-09 지적 반영).
    assert {p["code"] for p in etns} == {"Q520100", "Q520101"}


def test_samsung_and_hynix_each_have_one_inverse_product():
    samsung_inverse = [p for p in LEVERAGE_PRODUCTS
                        if p["underlying_ticker"] == "005930" and p["multiple"] < 0]
    hynix_inverse = [p for p in LEVERAGE_PRODUCTS
                      if p["underlying_ticker"] == "000660" and p["multiple"] < 0]
    assert len(samsung_inverse) == 1
    assert len(hynix_inverse) == 1


def test_all_products_listed_2026_05_27():
    assert all(p["listed_date"] == "2026-05-27" for p in LEVERAGE_PRODUCTS)
