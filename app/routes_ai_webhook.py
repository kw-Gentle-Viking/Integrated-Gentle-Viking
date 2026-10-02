# app/routes_ai_webhook.py
import os

from fastapi import APIRouter, Depends, Header, HTTPException

from app.ai_history import record_predictions
from app.db import SessionLocal
from app.dependencies import get_current_user
from app.models import LiveCandle, User
from app.schemas import AgreementAnalysisRequest, OnceCallbackPayload, PredictionResult, RealtimePayload, WarmupPayload
from app.services_report import generate_agreement_analysis, generate_report, save_report
from app.ai_client import merge_realtime_prediction, with_staleness
from app.shared_state import SIGNAL_MAP, ai_signal_event, realtime_predictions, warmup_events, warmup_received, warmup_requirements

router = APIRouter()

AI_API_KEY = os.getenv("AI_SERVER_API_KEY", "dev-ai-key")
once_results: dict[str, dict] = {}


def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != AI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")





def build_prediction_analysis(ticker: str, signal: str, confidence: float, prob_buy: float, prob_hold: float, prob_sell: float) -> str:
    signal_text = {"BUY": "매수", "HOLD": "관망", "SELL": "매도"}.get(signal, "관망")
    confidence_pct = confidence * 100 if confidence <= 1 else confidence
    buy_pct = prob_buy * 100 if prob_buy <= 1 else prob_buy
    hold_pct = prob_hold * 100 if prob_hold <= 1 else prob_hold
    sell_pct = prob_sell * 100 if prob_sell <= 1 else prob_sell

    if signal == "BUY":
        direction = "매수 확률이 가장 높아 단기 상승 방향성이 우세한 상태로 해석됩니다."
        action = "다만 관망·매도 확률도 함께 확인해 분할 진입이나 손절 기준을 정해두는 접근이 적합합니다."
    elif signal == "SELL":
        direction = "매도 확률이 가장 높아 단기 하방 위험 또는 과열 해소 가능성이 우세한 상태로 해석됩니다."
        action = "보유 중이라면 비중 축소나 리스크 관리 기준을 우선 확인하는 접근이 적합합니다."
    else:
        direction = "관망 확률이 가장 높아 방향성이 아직 명확하지 않은 중립 구간으로 해석됩니다."
        action = "추가 추세 확인 전까지 신규 진입보다는 가격·거래량 변화를 더 확인하는 접근이 적합합니다."

    return (
        f"{ticker}에 대한 TFT 모델의 현재 판단은 {signal_text} 우위입니다. "
        f"모델 확신도는 {confidence_pct:.1f}%이며, 매수 {buy_pct:.1f}%, "
        f"관망 {hold_pct:.1f}%, 매도 {sell_pct:.1f}%로 산출됐습니다. "
        f"{direction} {action} 이 결과는 로컬 시연용 추론 응답이며, 실제 매매 판단에는 최신 시세와 투자자 위험 성향을 함께 확인해야 합니다."
    )


def build_demo_prediction(ticker: str) -> dict:
    presets = {
        "005930": ("BUY", 0.82, 0.12, 0.06),
        "000660": ("HOLD", 0.22, 0.64, 0.14),
        "035420": ("BUY", 0.67, 0.24, 0.09),
        "105560": ("HOLD", 0.31, 0.58, 0.11),
    }
    signal, prob_buy, prob_hold, prob_sell = presets.get(
        ticker, ("HOLD", 0.34, 0.52, 0.14)
    )
    confidence = max(prob_buy, prob_hold, prob_sell)
    return {
        "ticker": ticker,
        "signal": signal,
        "confidence": confidence,
        "prob_buy": prob_buy,
        "prob_hold": prob_hold,
        "prob_sell": prob_sell,
        "trade_datetime": "2026-06-01T16:45:00+09:00",
        "model_version": "local-demo-fallback",
        "analysis": build_prediction_analysis(ticker, signal, confidence, prob_buy, prob_hold, prob_sell),
    }

def parse_prediction(result: PredictionResult) -> dict:
    signal = SIGNAL_MAP.get(str(result.pred_str).strip(), "HOLD")
    signal = SIGNAL_MAP.get(str(result.pred_label), signal)
    confidence = max(result.prob_buy, result.prob_hold, result.prob_sell)
    return {
        "ticker": result.ticker,
        "signal": signal,
        "confidence": confidence,
        "prob_buy": result.prob_buy,
        "prob_hold": result.prob_hold,
        "prob_sell": result.prob_sell,
        "trade_datetime": result.trade_datetime,
        "model_version": result.model_version,
    }


@router.post("/realtime")
def receive_realtime(
    payload: RealtimePayload,
    x_api_key: str = Header(None),
):
    """5분 자동 추론 결과 수신 (AI 서버 -> 백엔드)"""
    verify_api_key(x_api_key)

    parsed_all = []
    for result in payload.results:
        parsed = parse_prediction(result)
        # 5분 push는 report/analysis/interpretability가 없다 -- ONCE 콜백이 채워둔 값을 지우지 않는다.
        realtime_predictions[result.ticker] = merge_realtime_prediction(
            realtime_predictions.get(result.ticker), parsed)
        parsed_all.append(parsed)
        print(
            f" {result.ticker}: {parsed['signal']} "
            f"(B:{result.prob_buy:.3f} H:{result.prob_hold:.3f} S:{result.prob_sell:.3f})"
        )

    record_predictions(parsed_all)   # 이력 저장 (실패해도 웹훅 응답에는 영향 없음)
    ai_signal_event.set()

    return {"status": "ok", "received": len(payload.results)}


@router.post("/callback")
def receive_once_callback(
    payload: OnceCallbackPayload,
    x_api_key: str = Header(None),
):
    """ONCE 추론 결과 수신 (AI 서버 -> 백엔드)"""
    verify_api_key(x_api_key)

    parsed_results = []
    for result in payload.results:
        parsed = parse_prediction(result)
        parsed["interpretability"] = result.interpretability or {}
        parsed["pred_str"] = result.pred_str
        parsed["prob_buy"] = result.prob_buy
        parsed["prob_hold"] = result.prob_hold
        parsed["prob_sell"] = result.prob_sell
        parsed["trade_datetime"] = result.trade_datetime
        parsed_results.append(parsed)

    db = SessionLocal()
    try:
        for result in parsed_results:
            report = generate_report(result)
            result["analysis"] = report
            result["report"] = report
            realtime_predictions[result["ticker"]] = result
            save_report(db, user_id=int(payload.user_id), persona_id=1, report=report)
            print(f"  {result['ticker']} 보고서 생성 완료")
    finally:
        db.close()

    once_results[payload.job_id] = {
        "user_id": payload.user_id,
        "inference_at": payload.inference_at,
        "results": parsed_results,
    }

    ai_signal_event.set()

    return {
        "status": "ok",
        "job_id": payload.job_id,
        "received": len(parsed_results),
    }

@router.get("/predictions")
def get_predictions():
    """전체 종목 최신 추론 결과 조회"""
    if realtime_predictions:
        return {ticker: with_staleness(pred) for ticker, pred in realtime_predictions.items()}
    if os.getenv("LOCAL_DEMO_MODE") == "1":
        return {
            ticker: build_demo_prediction(ticker)
            for ticker in ["005930", "000660", "035420", "105560"]
        }
    return {}



@router.get("/predictions/{ticker}")
def get_prediction(ticker: str):
    """특정 종목 최신 추론 결과 조회"""
    pred = realtime_predictions.get(ticker)
    if pred:
        if os.getenv("LOCAL_DEMO_MODE") == "1" and not pred.get("analysis"):
            pred = {
                **pred,
                "analysis": build_prediction_analysis(
                    ticker,
                    pred.get("signal", "HOLD"),
                    pred.get("confidence", 0),
                    pred.get("prob_buy", 0),
                    pred.get("prob_hold", 0),
                    pred.get("prob_sell", 0),
                ),
            }
        return with_staleness(pred)
    if os.getenv("LOCAL_DEMO_MODE") == "1":
        return build_demo_prediction(ticker)
    raise HTTPException(status_code=404, detail="추론 결과 없음")


@router.get("/once/{job_id}")
def get_once_result(job_id: str):
    """ONCE 추론 결과 조회"""
    result = once_results.get(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="결과 없음")
    return result


@router.post("/agreement")
def analyze_model_agreement(payload: AgreementAnalysisRequest, current_user: User = Depends(get_current_user)):
    """추천 리포트와 TFT 예측 결과의 합치성/불일치 이유를 해석."""
    return generate_agreement_analysis(payload.model_dump())


@router.post("/warmup")
def receive_warmup(
    payload: WarmupPayload,
    x_api_key: str = Header(None),
):
    verify_api_key(x_api_key)

    db = SessionLocal()
    try:
        for c in payload.candles:
            db.add(
                LiveCandle(
                    ticker=payload.ticker,
                    timeframe="5m",
                    open=c["open"],
                    high=c["high"],
                    low=c["low"],
                    close=c["close"],
                    volume=c["volume"],
                    trade_datetime=c["datetime"],
                )
            )
        db.commit()
    finally:
        db.close()

    received_count = len(payload.candles)
    for user_id, requirements in list(warmup_requirements.items()):
        required_count = requirements.get(payload.ticker)
        if required_count is None:
            continue

        warmup_received.setdefault(user_id, {})[payload.ticker] = received_count
        if all(
            warmup_received.get(user_id, {}).get(ticker, 0) >= count
            for ticker, count in requirements.items()
        ):
            event = warmup_events.get(user_id)
            if event:
                event.set()

    print(f" {payload.ticker}: 워밍업 {received_count}개 수신")
    return {"status": "ok", "received": received_count}
