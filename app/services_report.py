# app/services_report.py
import os
import time
from google import genai
from app.models import RecommendationReport

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# 일시 장애(503 과부하 등)만 재시도한다. 그 밖의 오류(잘못된 키·모델 404 등)는 즉시 실패.
RETRYABLE_CODES = {429, 500, 503, 504}


def _call_gemini(client, **kwargs):
    attempts = int(os.getenv("GEMINI_RETRY_ATTEMPTS", "3"))
    for i in range(attempts):
        try:
            return client.models.generate_content(**kwargs)
        except Exception as e:
            if i == attempts - 1 or getattr(e, "code", None) not in RETRYABLE_CODES:
                raise
            time.sleep(2 ** i)

SYSTEM_PROMPT = """당신은 한국 주식 시장 AI 예측 분석가입니다.
TFT(Temporal Fusion Transformer) 딥러닝 모델이 생성한 주가 예측 결과와
모델 내부의 피처 중요도·어텐션 데이터를 바탕으로,
투자자가 이해할 수 있는 예측 보고서를 작성합니다.

보고서 작성 원칙:
- 모델이 왜 해당 예측을 내렸는지 근거 중심으로 서술한다.
- 투자 결정을 강요하지 않으며, 분석 근거를 중립적으로 제시한다.
- 기술 용어는 괄호 안에 간단한 설명을 병기한다.
- 수치는 반드시 해석과 함께 제시한다 (숫자만 나열하지 않는다).
- 보고서 말미에는 반드시 투자 유의사항을 포함한다."""

# 피처 해석 매핑. training/champion_config.json의 33개 컬럼과 1:1로 맞춰야 한다 -- 예전엔
# bb_position/vol_ratio/macd_ratio/per/pbr/prop_foreign/prop_individual/prop_institution 등
# 지금 모델에 없는 8개 피처가 매핑돼 있었고, disparity_5d/20d/60d·sector_*·lev_*·snp500_ret 등
# 실제 피처 25개 가량은 매핑이 아예 없어 원본 컬럼명이 그대로 노출되고 있었다(2026-10-02 통합 감사).
FEATURE_DESC = {
    "log_ret": ("직전 봉 로그 수익률", lambda v: "직전 봉 상승" if v > 0 else "직전 봉 하락"),
    "disparity_5d": ("5일 이동평균 이격도", lambda v: "단기 과열" if v > 1.03 else "단기 침체" if v < 0.97 else "중립"),
    "disparity_20d": ("20일 이동평균 이격도", lambda v: "중기 과열" if v > 1.05 else "중기 침체" if v < 0.95 else "이동평균 근접"),
    "disparity_60d": ("60일 이동평균 이격도", lambda v: "장기 과열" if v > 1.1 else "장기 침체" if v < 0.9 else "중립"),
    "rsi_14": ("RSI(14)", lambda v: "과매도 구간" if v < 30 else "과매수 구간" if v > 70 else "중립 구간"),
    "volatility_20d": ("20일 변동성(수익률 표준편차)", lambda v: "고변동성" if v > 0.02 else "저변동성" if v < 0.01 else "보통 변동성"),
    "sector_ret_1d": ("섹터 1일 수익률", lambda v: f"섹터 {'상승' if v > 0 else '하락'} {abs(v)*100:.2f}%"),
    "sector_ret_5d": ("섹터 5일 수익률", lambda v: f"섹터 {'상승' if v > 0 else '하락'} {abs(v)*100:.2f}%"),
    "sector_ret_20d": ("섹터 20일 수익률", lambda v: f"섹터 {'상승' if v > 0 else '하락'} {abs(v)*100:.2f}%"),
    "sector_ma_ratio_20d": ("섹터 20일 이동평균 대비 비율", lambda v: "섹터 과열" if v > 1.05 else "섹터 침체" if v < 0.95 else "중립"),
    "sector_volatility": ("섹터 변동성(고저 range/종가)", lambda v: "섹터 변동성 확대" if v > 0.03 else "섹터 변동성 축소" if v < 0.01 else "보통"),
    "sector_volume_ratio": ("섹터 거래량 비율", lambda v: "섹터 거래량 급증" if v > 2.0 else "섹터 거래량 급감" if v < 0.5 else "보통"),
    "is_dividend": ("배당 이벤트", lambda v: "배당 이벤트 있음" if v else "배당 이벤트 없음"),
    "is_bonus_issue": ("무상증자 이벤트", lambda v: "무상증자 있음" if v else "무상증자 없음"),
    "is_rights_offering": ("유상증자 이벤트", lambda v: "유상증자 있음" if v else "유상증자 없음"),
    "is_split": ("액면분할 이벤트", lambda v: "액면분할 있음" if v else "액면분할 없음"),
    "day_of_week": ("요일(0=월~4=금)", lambda v: {0: "월요일", 1: "화요일", 2: "수요일", 3: "목요일", 4: "금요일"}.get(int(v), f"요일코드 {v}")),
    "lev_total_aum": ("레버리지/인버스 ETF 전체 AUM", lambda v: f"AUM {v:,.0f}"),
    "lev_aum_to_mktcap": ("레버리지 ETF AUM/시장 시총 비율", lambda v: "레버리지 비중 높음" if v > 0.05 else "레버리지 비중 낮음"),
    "est_rebalancing_flow": ("추정 리밸런싱 수급", lambda v: "리밸런싱 매수 추정" if v > 0 else "리밸런싱 매도 추정"),
    "is_vi_triggered": ("VI(변동성완화장치) 발동 여부", lambda v: "VI 발동" if v else "VI 미발동"),
    "vi_count_recent5d": ("최근 5일 VI 발동 횟수", lambda v: f"{int(v)}회"),
    "kospi_ret": ("KOSPI 당일 등락률", lambda v: f"KOSPI {'상승' if v > 0 else '하락'} {abs(v)*100:.2f}%"),
    "kosdaq_ret": ("KOSDAQ 당일 등락률", lambda v: f"KOSDAQ {'상승' if v > 0 else '하락'} {abs(v)*100:.2f}%"),
    "snp500_ret": ("S&P500 당일 등락률", lambda v: f"S&P500 {'상승' if v > 0 else '하락'} {abs(v)*100:.2f}%"),
    "nasdaq_ret": ("NASDAQ 당일 등락률", lambda v: f"NASDAQ {'상승' if v > 0 else '하락'} {abs(v)*100:.2f}%"),
    "phlx_semi_ret": ("필라델피아 반도체지수 당일 등락률", lambda v: f"반도체지수 {'상승' if v > 0 else '하락'} {abs(v)*100:.2f}%"),
    "vix_chg": ("VIX 변동", lambda v: "불확실성 증가" if v > 0 else "시장 안도"),
    "usd_krw_chg": ("원달러 환율 변동", lambda v: "원화 약세" if v > 0 else "원화 강세"),
    "us_10y_yield_chg": ("미국 10년물 금리 변동", lambda v: "금리 상승" if v > 0 else "금리 하락"),
    "rate_spread_us_kr": ("한미 금리 스프레드(미국-한국)", lambda v: f"스프레드 {v:+.2f}%p"),
    "wti_ret": ("WTI 원유 당일 등락률", lambda v: f"WTI {'상승' if v > 0 else '하락'} {abs(v)*100:.2f}%"),
    "gold_ret": ("금 가격 당일 등락률", lambda v: f"금 가격 {'상승' if v > 0 else '하락'} {abs(v)*100:.2f}%"),
}

EVENT_DESC = {
    "time_progress": "장 진행률",
    "is_bok": "금통위",
    "is_fomc": "FOMC",
    "is_witching_kr": "선물만기(한국)",
    "is_witching_us": "선물만기(미국)",
}


def _format_unknown_past(features: list) -> str:
    """unknown_past 피처를 해석 텍스트로 변환"""
    lines = []
    for f in features:
        name = f["feature"]
        importance = f["importance"]
        value = f.get("current_value")

        desc_name, interpreter = FEATURE_DESC.get(name, (name, lambda v: str(v)))

        if value is not None:
            interpretation = interpreter(value)
            lines.append(f"{f['rank']}위. {desc_name} — 현재값: {value} ({interpretation}), 중요도: {importance:.1%}")
        else:
            lines.append(f"{f['rank']}위. {desc_name} — 중요도: {importance:.1%}")

    return "\n".join(lines)


def _format_known_future(features: list) -> str:
    """known_future 이벤트 피처 텍스트 변환"""
    if not features:
        return "특별한 이벤트 없음 (금통위 · FOMC · 선물만기일 해당 없음)"

    events = []
    has_event = False

    for f in features:
        name = f["feature"]
        value = f.get("current_value", 0)
        desc = EVENT_DESC.get(name, name)

        if name == "time_progress":
            pct = value * 100
            minutes = int(value * 390)  # 09:00~15:30 = 390분
            events.append(f"장 진행률: {pct:.1f}% (09:00 이후 약 {minutes}분 경과)")
        elif value == 1:
            has_event = True
            events.append(f"{desc}: 해당")
        else:
            events.append(f"{desc}: 없음")

    if not has_event:
        return events[0] + "\n특별한 이벤트 없음 (금통위 · FOMC · 선물만기일 해당 없음)"

    return "\n".join(events)


def _format_attention(peaks: list) -> str:
    """어텐션 피크 텍스트 변환"""
    parts = [f"{p['minutes_ago']}분 전 봉 (어텐션 {p['attention']*100:.1f}%)" for p in peaks]
    return " — ".join(parts)


def _get_confidence_text(prob: float) -> str:
    if prob >= 0.8:
        return "모델이 높은 확신을 보이고 있습니다"
    elif prob >= 0.6:
        return "모델이 다소 우세한 방향성을 제시합니다"
    else:
        return "방향성이 불명확하여 신중한 접근이 필요합니다"


def _get_direction_emoji(pred_str: str, max_prob: float) -> str:
    if pred_str == "매수" and max_prob > 0.6:
        return "🟢"
    elif pred_str == "매도" and max_prob > 0.6:
        return "🔴"
    else:
        return "🟡"


def build_user_prompt(result: dict, ticker_name: str = "") -> str:
    """AI 추론 결과로 Gemini 프롬프트 생성"""
    ticker = result["ticker"]
    interp = result.get("interpretability") or {}
    top_features = interp.get("top_features", {})

    unknown_past = _format_unknown_past(top_features.get("unknown_past", []))
    known_future = _format_known_future(top_features.get("known_future", []))
    attention = _format_attention(interp.get("attention_peak", []))

    static = top_features.get("static", [])
    sector_desc = ""
    market_desc = ""
    for s in static:
        if s["feature"] == "sector_id":
            sector_desc = f"{s.get('description', '알 수 없음')} (중요도: {s['importance']:.1%})"
        elif s["feature"] == "market_id":
            market_desc = s.get("description", "알 수 없음")

    display_name = f"{ticker_name}({ticker})" if ticker_name else ticker

    return f"""다음 데이터를 바탕으로 {display_name} 예측 보고서를 작성해 주세요.

─────────────────────────────────────────
[예측 기준 시각]
{result.get('trade_datetime', '')}

[모델 예측 결과]
예측: {result['pred_str']}
매수 확률: {result['prob_buy']*100:.1f}%
관망 확률: {result['prob_hold']*100:.1f}%
매도 확률: {result['prob_sell']*100:.1f}%

[모델이 주목한 주요 피처 (상위 5개)]
{unknown_past}

[이벤트 피처]
{known_future}

[모델이 집중한 시점 (어텐션 상위)]
{attention}

[종목 정보]
섹터: {sector_desc}
시장: {market_desc}
─────────────────────────────────────────

아래 보고서 형식에 맞춰 작성해 주세요."""


def generate_report(result: dict, ticker_name: str = "") -> str:
    """Gemini API로 보고서 생성"""
    user_prompt = build_user_prompt(result, ticker_name)

    try:
        # genai.Client(...) 생성 자체가 try 밖에 있어서, GEMINI_API_KEY가 비어있거나(로컬
        # 드라이런 기본값) 잘못되면 여기서 바로 ValueError가 터져 /ai/callback 전체가 500이
        # 됐다(실제 서버에 요청을 보내 재현/확인함, 2026-10-03). generate_content 호출만 감싸던
        # try를 Client 생성까지 넓혔다.
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        resp = _call_gemini(
            client,
            model=GEMINI_MODEL,
            contents=user_prompt,
            config={"system_instruction": SYSTEM_PROMPT},
        )
        return resp.text
    except Exception as e:
        return f"보고서 생성 실패: {e}"


def save_report(db, user_id: int, persona_id: int, report: str):
    """보고서 DB 저장"""
    db.add(RecommendationReport(
        user_id=user_id,
        persona_id=persona_id,
        report=report,
    ))
    db.commit()


def _normalize_direction(value: str) -> str:
    text = (value or "").upper()
    if "BUY" in text or "매수" in value or "비중확대" in value:
        return "BUY"
    if "SELL" in text or "매도" in value or "제외" in value:
        return "SELL"
    if "HOLD" in text or "관망" in value or "보류" in value:
        return "HOLD"
    return "HOLD"


def _signal_label(signal: str) -> str:
    return {"BUY": "매수", "HOLD": "관망", "SELL": "매도"}.get(signal, "관망")


def _agreement_label(recommendation_signal: str, tft_signal: str) -> tuple[str, str]:
    rec = _normalize_direction(recommendation_signal)
    tft = _normalize_direction(tft_signal)
    if rec == tft:
        return "의견 일치", "aligned"
    if "HOLD" in {rec, tft}:
        return "부분 불일치", "partial"
    return "의견 불일치", "diverged"


def _fallback_agreement_analysis(payload: dict) -> dict:
    rec = _normalize_direction(payload.get("recommendation_signal", ""))
    tft = _normalize_direction(payload.get("tft_signal", ""))
    label, level = _agreement_label(rec, tft)
    rec_label = _signal_label(rec)
    tft_label = _signal_label(tft)

    if level == "aligned":
        interpretation = (
            f"추천 리포트와 TFT 모델이 모두 {rec_label} 쪽으로 기울어져 있습니다. "
            "중장기 추천 근거와 단기 시계열 흐름이 같은 방향을 가리키는 상태입니다."
        )
        action_note = "추천 근거와 단기 확률이 함께 우호적이므로, 리스크 한도 안에서 계획된 비중으로 접근할 수 있습니다."
    elif level == "partial":
        interpretation = (
            f"추천 리포트는 {rec_label}, TFT 모델은 {tft_label}로 해석됩니다. "
            "한쪽 모델이 관망을 제시해 방향성은 완전히 충돌하지 않지만, 진입 시점에는 추가 확인이 필요합니다."
        )
        action_note = "즉시 강한 진입보다는 분할 접근 또는 다음 예측 갱신을 확인하는 전략이 적합합니다."
    else:
        interpretation = (
            f"추천 리포트는 {rec_label} 관점이지만 TFT 모델은 {tft_label} 우위입니다. "
            "추천 리포트는 뉴스·공시·재료·수급 등 종목 매력도를, TFT는 최근 가격과 거래량의 단기 패턴을 더 강하게 반영하기 때문에 불일치가 발생할 수 있습니다."
        )
        action_note = "중장기 관심 후보로는 유지하되, 단기 진입은 보류하고 가격 안정 또는 TFT 신호 개선을 확인하는 편이 보수적입니다."

    return {
        "status": "fallback",
        "recommendation_signal": rec_label,
        "tft_signal": tft_label,
        "alignment_label": label,
        "alignment_level": level,
        "summary": f"추천 리포트: {rec_label} / TFT 모델: {tft_label}",
        "interpretation": interpretation,
        "action_note": action_note,
    }


def generate_agreement_analysis(payload: dict) -> dict:
    """추천 리포트와 TFT 예측 결과의 합치성 해석을 Gemini로 생성."""
    rec = _normalize_direction(payload.get("recommendation_signal", ""))
    tft = _normalize_direction(payload.get("tft_signal", ""))
    label, level = _agreement_label(rec, tft)
    fallback = _fallback_agreement_analysis(payload)

    prompt = f"""
다음은 한국 주식 {payload.get('name') or payload.get('ticker')}에 대한 두 AI 판단입니다.

[추천 리포트 판단]
- 방향: {_signal_label(rec)}
- 원문 신호: {payload.get('recommendation_signal', '')}
- 요약: {payload.get('recommendation_summary') or '제공 없음'}
- 추천 근거: {payload.get('recommendation_reasons') or '제공 없음'}

[TFT 시계열 모델 판단]
- 방향: {_signal_label(tft)}
- 확신도: {payload.get('confidence', 0)}%
- 매수 확률: {payload.get('prob_buy', 0)}%
- 관망 확률: {payload.get('prob_hold', 0)}%
- 매도 확률: {payload.get('prob_sell', 0)}%

[판단 상태]
- 합치성: {label}

두 모델의 관점 차이를 투자자가 이해할 수 있게 설명하세요.
추천 리포트는 뉴스·공시·재료·수급·투자성향 기반의 종목 매력도이고,
TFT는 최근 가격·거래량·시계열 패턴 기반의 단기 방향성이라는 점을 반영하세요.

반드시 아래 JSON만 출력하세요.
{{
  "summary": "한 문장 요약",
  "interpretation": "불일치 또는 일치가 발생한 이유를 2~3문장으로 설명",
  "action_note": "투자자가 참고할 대응 관점을 1~2문장으로 설명"
}}
"""

    try:
        import json

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        try:
            resp = _call_gemini(
                client,
                model=GEMINI_MODEL,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
        except TypeError:
            resp = _call_gemini(client, model=GEMINI_MODEL, contents=prompt)

        text = (resp.text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.removeprefix("json").strip()
        parsed = json.loads(text)
        return {
            **fallback,
            "status": "ok",
            "summary": parsed.get("summary") or fallback["summary"],
            "interpretation": parsed.get("interpretation") or fallback["interpretation"],
            "action_note": parsed.get("action_note") or fallback["action_note"],
        }
    except Exception:
        result = dict(fallback)
        result["status"] = "fallback"
        return result

