import { apiFetch } from "@/lib/signup/auth";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type OrderSide = "buy" | "sell";
export type PriceType = "market" | "limit";

export interface OrderRequest {
  code: string;
  side: OrderSide;
  qty: number;
  price: number;
  priceType: PriceType;
}

export interface OrderResult {
  orderId: string;
  message: string;
}

// ── KIS API 응답 타입 ──────────────────────────────────────────────────────────
interface KisOrderResponse {
  rt_cd: string;   // "0" = 성공
  msg_cd: string;
  msg1: string;
  output: {
    KRX_FWDG_ORD_ORGNO: string;  // 한국거래소 전송 주문 조직 번호
    ITMNO: string;                 // 종목번호
    RVSE_CNCL_DVSN_NAME: string; // 정정취소구분명
  };
}

/**
 * 현금 매수/매도 주문
 * 백엔드: POST /trade/order
 * KIS:   POST /uapi/domestic-stock/v1/trading/order-cash
 *        tr_id: VTTC0011U (매도 모의), VTTC0012U (매수 모의)
 *        body:
 *          CANO        계좌번호 앞 8자리
 *          ACNT_PRDT_CD 계좌번호 뒤 2자리
 *          PDNO        종목코드
 *          ORD_DVSN    주문구분 00=지정가 01=시장가
 *          ORD_QTY     주문수량
 *          ORD_UNPR    주문단가 (시장가는 "0")
 */
export async function placeOrder(req: OrderRequest): Promise<OrderResult> {
  const res = await apiFetch(`${API_BASE}/trade/order`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      code: req.code,
      side: req.side,
      qty: req.qty,
      price: req.price,
      price_type: req.priceType,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({})) as { detail?: string };
    throw new Error(err.detail ?? `주문 실패 (${res.status})`);
  }

  const data: KisOrderResponse = await res.json();
  if (data.rt_cd !== "0") throw new Error(data.msg1);

  return {
    orderId: data.output?.KRX_FWDG_ORD_ORGNO ?? "",
    message: data.msg1,
  };
}
