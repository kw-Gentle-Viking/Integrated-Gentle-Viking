import type { CandleType } from "@/lib/chart/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── KIS API 응답 타입 (백엔드가 그대로 프록시하는 경우 기준) ──────────────────

interface KisCurrentPriceOutput {
  stck_prpr: string;       // 현재가
  prdy_vrss: string;       // 전일 대비 변동액
  prdy_ctrt: string;       // 전일 대비율 (%)
  prdy_vrss_sign: string;  // 부호: 1=상한/2=상승/3=보합/4=하한/5=하락
  stck_oprc: string;       // 시가
  stck_hgpr: string;       // 고가
  stck_lwpr: string;       // 저가
  acml_vol: string;        // 누적 거래량
  hts_kor_isnm: string;    // 종목명
}

interface KisCurrentPriceResponse {
  rt_cd: string;   // "0" = 성공
  msg1: string;
  output: KisCurrentPriceOutput;
}

interface KisDailyChartItem {
  stck_bsop_date: string;  // 영업일 (YYYYMMDD)
  stck_oprc: string;       // 시가
  stck_hgpr: string;       // 고가
  stck_lwpr: string;       // 저가
  stck_clpr: string;       // 종가
  acml_vol: string;        // 누적 거래량
}

interface KisDailyChartResponse {
  rt_cd: string;
  msg1: string;
  output1: { hts_kor_isnm: string };
  output2: KisDailyChartItem[];
}

// ── 프론트엔드에서 사용할 정규화 타입 ────────────────────────────────────────

export interface StockCurrentPrice {
  code: string;
  name: string;
  priceFormatted: string;   // "70,200원"
  changeFormatted: string;  // "+0.72%" / "-1.45%"
  isUp: boolean;
}

// ── API 호출 함수 ─────────────────────────────────────────────────────────────

/**
 * 단일 종목 현재가 조회
 * 백엔드: GET /prices/current?code={code}
 * KIS:   GET /uapi/domestic-stock/v1/quotations/inquire-price
 *        tr_id: VHKST01010100 (모의), FHKST01010100 (실전)
 *        query: FID_COND_MRKT_DIV_CODE=J, FID_INPUT_ISCD={code}
 */
export async function fetchCurrentPrice(code: string): Promise<StockCurrentPrice> {
  const res = await fetch(`${API_BASE}/prices/current?code=${code}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`현재가 조회 실패 (${res.status})`);

  const data: KisCurrentPriceResponse = await res.json();
  if (data.rt_cd !== "0") throw new Error(`KIS 오류: ${data.msg1}`);

  const { output } = data;
  const price = parseInt(output.stck_prpr, 10);
  const changeRate = parseFloat(output.prdy_ctrt);
  const isUp = output.prdy_vrss_sign === "1" || output.prdy_vrss_sign === "2";

  return {
    code,
    name: output.hts_kor_isnm,
    priceFormatted: `${price.toLocaleString("ko-KR")}원`,
    changeFormatted: `${isUp ? "+" : ""}${changeRate.toFixed(2)}%`,
    isUp,
  };
}

/**
 * 일별 OHLC 차트 데이터 조회
 * 백엔드: GET /prices/chart?code={code}&range={range}
 * KIS:   GET /uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice
 *        tr_id: FHKST03010100
 *        query: FID_COND_MRKT_DIV_CODE=J, FID_INPUT_ISCD={code},
 *               FID_INPUT_DATE_1={from}, FID_INPUT_DATE_2={to},
 *               FID_PERIOD_DIV_CODE=D (일/D, 주/W, 월/M)
 */
export async function fetchChartCandles(
  code: string,
  range: string,
): Promise<CandleType[]> {
  const res = await fetch(
    `${API_BASE}/prices/chart?code=${code}&range=${range}`,
    { cache: "no-store" },
  );
  if (!res.ok) throw new Error(`차트 데이터 조회 실패 (${res.status})`);

  const data: KisDailyChartResponse = await res.json();
  if (data.rt_cd !== "0") throw new Error(`KIS 오류: ${data.msg1}`);

  return data.output2
    .map((item) => {
      const d = item.stck_bsop_date; // YYYYMMDD
      const utc = Date.UTC(
        parseInt(d.slice(0, 4), 10),
        parseInt(d.slice(4, 6), 10) - 1,
        parseInt(d.slice(6, 8), 10),
      );
      return {
        time: utc / 1000,
        open: parseInt(item.stck_oprc, 10),
        high: parseInt(item.stck_hgpr, 10),
        low: parseInt(item.stck_lwpr, 10),
        close: parseInt(item.stck_clpr, 10),
        volume: parseInt(item.acml_vol, 10),
      };
    })
    .sort((a, b) => a.time - b.time);
}

// ── 시장 지수 / 환율 개요 ─────────────────────────────────────────────────────

export interface MarketIndexItem {
  name: string;
  value: string;
  change: string;
  percent: string;
  isUp: boolean;
  color: string;
}

interface KisIndexOutput {
  bstp_nmix_prpr: string;      // 지수 현재가
  bstp_nmix_prdy_vrss: string; // 전일대비
  bstp_nmix_prdy_ctrt: string; // 등락률(%)
  prdy_vrss_sign: string;      // 부호
}

interface KisFxOutput {
  stck_prpr: string;   // 현재 환율
  prdy_vrss: string;   // 전일대비
  prdy_ctrt: string;   // 등락률
  prdy_vrss_sign: string;
}

interface MarketOverviewResponse {
  kospi:  { rt_cd: string; output: KisIndexOutput } | null;
  kosdaq: { rt_cd: string; output: KisIndexOutput } | null;
  usd:    { rt_cd: string; output: KisFxOutput }    | null;
}

function signIsUp(sign: string) {
  return sign === "1" || sign === "2";
}

/**
 * 시장 지수 + 환율 일괄 조회
 * 백엔드: GET /market/overview
 * KIS:
 *   KOSPI/KOSDAQ: inquire-index-price  tr_id: FHPUP02100000
 *   USD/KRW:      inquire-price        tr_id: FHKST01010100  market: X
 */
export async function fetchMarketOverview(): Promise<MarketIndexItem[]> {
  const res = await fetch(`${API_BASE}/market/overview`, { cache: "no-store" });
  if (!res.ok) throw new Error(`시장 개요 조회 실패 (${res.status})`);

  const data: MarketOverviewResponse = await res.json();
  const items: MarketIndexItem[] = [];

  if (data.usd?.rt_cd === "0") {
    const o = data.usd.output;
    const val = parseFloat(o.stck_prpr);
    const chg = parseFloat(o.prdy_vrss);
    const pct = parseFloat(o.prdy_ctrt);
    const isUp = signIsUp(o.prdy_vrss_sign);
    items.push({
      name: "달러 환율",
      value: val.toFixed(2),
      change: `${isUp ? "+" : ""}${chg.toFixed(2)}`,
      percent: `${isUp ? "+" : ""}${pct.toFixed(2)}%`,
      isUp,
      color: "#f43f5e",
    });
  }

  if (data.kospi?.rt_cd === "0") {
    const o = data.kospi.output;
    const val = parseFloat(o.bstp_nmix_prpr);
    const chg = parseFloat(o.bstp_nmix_prdy_vrss);
    const pct = parseFloat(o.bstp_nmix_prdy_ctrt);
    const isUp = signIsUp(o.prdy_vrss_sign);
    items.push({
      name: "코스피",
      value: val.toFixed(2),
      change: `${isUp ? "+" : ""}${chg.toFixed(2)}`,
      percent: `${isUp ? "+" : ""}${pct.toFixed(2)}%`,
      isUp,
      color: "#3b82f6",
    });
  }

  if (data.kosdaq?.rt_cd === "0") {
    const o = data.kosdaq.output;
    const val = parseFloat(o.bstp_nmix_prpr);
    const chg = parseFloat(o.bstp_nmix_prdy_vrss);
    const pct = parseFloat(o.bstp_nmix_prdy_ctrt);
    const isUp = signIsUp(o.prdy_vrss_sign);
    items.push({
      name: "코스닥",
      value: val.toFixed(2),
      change: `${isUp ? "+" : ""}${chg.toFixed(2)}`,
      percent: `${isUp ? "+" : ""}${pct.toFixed(2)}%`,
      isUp,
      color: "#10b981",
    });
  }

  return items;
}

// ── 실시간 거래량 순위 ────────────────────────────────────────────────────────

interface KisVolumeRankItem {
  data_rank: string;
  mksc_shrn_iscd: string;  // 종목코드
  hts_kor_isnm: string;    // 종목명
  stck_prpr: string;       // 현재가
  prdy_vrss_sign: string;  // 부호 (1=상한/2=상승/3=보합/4=하한/5=하락)
  prdy_ctrt: string;       // 등락률
  acml_tr_pbmn: string;    // 누적거래대금
}

interface KisVolumeRankResponse {
  rt_cd: string;
  msg1: string;
  output: KisVolumeRankItem[];
}

export interface RankedStock {
  rank: number;
  code: string;
  name: string;
  price: string;
  change: string;
  isUp: boolean;
  volume: string;
}

function formatVolume(wonStr: string): string {
  const v = parseInt(wonStr, 10);
  if (v >= 100_000_000) return `${Math.round(v / 100_000_000)}억원`;
  if (v >= 10_000) return `${Math.round(v / 10_000)}만원`;
  return `${v}원`;
}

/**
 * 실시간 거래량 순위 조회
 * 백엔드: GET /market/volume-rank
 * KIS:   GET /uapi/domestic-stock/v1/quotations/volume-rank  (실전 도메인)
 *        tr_id: FHPST01710000
 */
export async function fetchVolumeRank(): Promise<RankedStock[]> {
  const res = await fetch(`${API_BASE}/market/volume-rank`, { cache: "no-store" });
  if (!res.ok) throw new Error(`거래량 순위 조회 실패 (${res.status})`);

  const data: KisVolumeRankResponse = await res.json();
  if (data.rt_cd !== "0") throw new Error(`KIS 오류: ${data.msg1}`);

  return data.output.slice(0, 30).map((item, idx) => {
    const price = parseInt(item.stck_prpr, 10);
    const changeRate = parseFloat(item.prdy_ctrt);
    const isUp = item.prdy_vrss_sign === "1" || item.prdy_vrss_sign === "2";
    return {
      rank: idx + 1,
      code: item.mksc_shrn_iscd,
      name: item.hts_kor_isnm,
      price: `${price.toLocaleString("ko-KR")}원`,
      change: `${isUp ? "+" : ""}${changeRate.toFixed(2)}%`,
      isUp,
      volume: formatVolume(item.acml_tr_pbmn),
    };
  });
}

// ── 분봉(당일 intraday) 조회 ──────────────────────────────────────────────────

interface KisIntradayItem {
  stck_cntg_hour: string; // 체결시간 HHMMSS (KST)
  stck_prpr: string;      // 현재가(종가)
  stck_oprc: string;      // 시가
  stck_hgpr: string;      // 고가
  stck_lwpr: string;      // 저가
  cntg_vol: string;       // 체결 거래량
}

interface KisIntradayResponse {
  rt_cd: string;
  msg1: string;
  output2: KisIntradayItem[];
}

/**
 * 당일 분봉 차트 데이터 조회
 * 백엔드: GET /prices/intraday?code={code}
 * KIS:   GET /uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice
 *        tr_id: FHKST03010200  (1회 30건, 최대 13회 호출로 전체 거래일 커버)
 */
export async function fetchIntradayCandles(code: string): Promise<CandleType[]> {
  const res = await fetch(`${API_BASE}/prices/intraday?code=${code}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`분봉 데이터 조회 실패 (${res.status})`);

  const data: KisIntradayResponse = await res.json();
  if (data.rt_cd !== "0") throw new Error(`KIS 오류: ${data.msg1}`);

  const today = new Date();

  return data.output2
    .map((item) => {
      const h = parseInt(item.stck_cntg_hour.slice(0, 2), 10);
      const m = parseInt(item.stck_cntg_hour.slice(2, 4), 10);
      const s = parseInt(item.stck_cntg_hour.slice(4, 6), 10);
      // KST(UTC+9) → UTC 변환
      const utc = Date.UTC(
        today.getFullYear(), today.getMonth(), today.getDate(),
        h - 9, m, s,
      );
      return {
        time: utc / 1000,
        open: parseInt(item.stck_oprc, 10),
        high: parseInt(item.stck_hgpr, 10),
        low: parseInt(item.stck_lwpr, 10),
        close: parseInt(item.stck_prpr, 10),
        volume: parseInt(item.cntg_vol, 10),
      };
    })
    .filter((c) => c.open > 0 && c.close > 0)
    .sort((a, b) => a.time - b.time); // 오래된 순 정렬
}
