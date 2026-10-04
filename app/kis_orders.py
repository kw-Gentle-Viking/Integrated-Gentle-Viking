"""KIS 주문 체결 조회 (주식일별주문체결조회, inquire-daily-ccld). mojito 래퍼에 없어서 직접 호출한다."""
import requests


def inquire_order_fill(broker, odno: str, trade_date: str, mock: bool) -> dict | None:
    """주문번호(odno)의 체결 현황. 찾지 못하면 None. trade_date: YYYYMMDD."""
    url = f"{broker.base_url}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
    headers = {
        "content-type": "application/json",
        "authorization": broker.access_token,
        "appKey": broker.api_key,
        "appSecret": broker.api_secret,
        "tr_id": "VTTC0081R" if mock else "TTTC0081R",
        "custtype": "P",
    }
    params = {
        "CANO": broker.acc_no_prefix,
        "ACNT_PRDT_CD": broker.acc_no_postfix,
        "INQR_STRT_DT": trade_date,
        "INQR_END_DT": trade_date,
        "SLL_BUY_DVSN_CD": "00",
        "INQR_DVSN": "00",
        "PDNO": "",
        "CCLD_DVSN": "00",
        "ORD_GNO_BRNO": "",
        "ODNO": odno,
        "INQR_DVSN_3": "00",
        "INQR_DVSN_1": "",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    res = requests.get(url, headers=headers, params=params, timeout=15)
    data = res.json()
    if data.get("rt_cd") != "0":
        raise RuntimeError(data.get("msg1") or f"체결 조회 실패 {res.status_code}")
    for row in data.get("output1") or []:
        if row.get("odno") == odno:
            return {
                "ticker": row.get("pdno"),
                "ordered_qty": int(row.get("ord_qty") or 0),
                "filled_qty": int(row.get("tot_ccld_qty") or 0),
                "avg_price": float(row.get("avg_prvs") or 0),
            }
    return None
